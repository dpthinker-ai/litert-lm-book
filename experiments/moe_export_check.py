#!/usr/bin/env python3
"""Export tiny MoE modules, inspect serialized weights, and compare CPU results.

This is an exporter compatibility diagnostic, not a full LLM or a benchmark.
The GELU-tanh copies change only the custom attribute and are diagnostic models.
CPU execution reuses the archived single-op C ABI harness without changing it.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import flatbuffers
from flatbuffers import flexbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema

from moe_layer_check import DEFAULT_LIB, reference, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
PIN = 'd592a2f09da4839ea34daaef92e53e638b57090a'
RTOL, ATOL = 2e-5, 2e-6


def read_model(path, directory):
    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    assert graph.OperatorsLength() == 1
    op = graph.Operators(0)
    assert model.OperatorCodes(op.OpcodeIndex()).CustomCode() == b'moe'
    attrs = flexbuffers.Loads(bytes(op.CustomOptionsAsNumpy()))
    tensors, meta = [], []
    for index in op.InputsAsNumpy():
        tensor = graph.Tensors(int(index))
        shape = tuple(int(n) for n in tensor.ShapeAsNumpy())
        raw = model.Buffers(tensor.Buffer()).DataAsNumpy()
        quant = tensor.Quantization()
        meta.append(dict(name=tensor.Name().decode(), shape=shape, type=tensor.Type(),
                         quant_scales=quant.ScaleLength() if quant else 0,
                         zero_points=quant.ZeroPointLength() if quant else 0))
        if len(tensors) < 3:
            position = list(graph.InputsAsNumpy()).index(index)
            raw = (directory/f'input-{position}.bin').read_bytes()
        dtype = {0: '<f4', 2: '<i4', 9: 'i1'}[tensor.Type()]
        tensors.append(np.frombuffer(raw, dtype=dtype).reshape(shape).copy())
    write_json(directory/'model-inspection.json', dict(attributes=attrs, inputs=meta,
               operator_count=1, model_sha256=sha(path)))
    return tensors, attrs


def oracle(tensors, attrs, activation):
    src, weights, indices = tensors[:3]
    if attrs['weight_type'] == 'fp32':
        matrices, scale = tensors[3:6], tensors[6]
    else:
        matrices = [tensors[i].astype(np.float64)*tensors[i+1] for i in (3, 5, 7)]
        scale = tensors[9]
    # Remove only the singleton layout dimension, keeping expert/output axes.
    matrices = [m.reshape(m.shape[0], 3, -1) for m in matrices]
    return reference(src.reshape(-1, 4), weights.reshape(-1, 2),
                     indices.reshape(-1, 2), scale.reshape(3), *matrices, activation)


def diagnostic_copy(original, target):
    obj = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(original.read_bytes(), 0))
    op = obj.subgraphs[0].operators[0]
    attrs = flexbuffers.Loads(bytes(op.customOptions))
    assert attrs['activation'] == 'gelu'
    attrs['activation'] = 'gelu_tanh'
    op.customOptions = np.frombuffer(flexbuffers.Dumps(attrs), dtype=np.uint8)
    builder = flatbuffers.Builder(4096)
    builder.Finish(obj.Pack(builder), file_identifier=b'TFL3')
    target.write_bytes(bytes(builder.Output()))


def compare(actual, expected):
    actual = actual.reshape(expected.shape)
    delta = np.abs(actual-expected)
    return dict(matches=bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
                max_abs_error=float(delta.max()),
                max_relative_error=float((delta/np.maximum(np.abs(expected), 1e-12)).max()))


def export_case(directory, mode, tokens):
    import torch
    import litert_torch
    from litert_torch.generative.layers import moe
    torch.manual_seed(20260913)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, shape in [('gate', (3, 6, 4)), ('up', (3, 6, 4)), ('down', (3, 4, 6))]:
                values = torch.randn(*shape)
                scales = values.abs().amax(dim=-1)/127
                if mode == 'int8':
                    values = torch.round(values/scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
                self.register_buffer(name, moe.flatten_expert_weight(values))
                self.register_buffer(name+'_scale', moe.flatten_expert_scale(scales))
            self.register_buffer('scale', torch.ones(1, 1, 1, 3))

        def forward(self, src, weights, indices):
            kwargs = {} if mode == 'fp32' else dict(ff_gate_scale=self.gate_scale,
                ff1_scale=self.up_scale, linear_scale=self.down_scale)
            return moe.moe_experts(src, weights, indices, self.gate, self.up, self.down,
                self.scale, num_experts=3, num_active_experts=2, model_dim=4, hidden_dim=6,
                weight_type=mode, **kwargs)

    layer = Layer().eval()
    src = torch.randn(1, 2, 4)[:, :tokens].contiguous()
    weights = torch.tensor([[[.7, .3], [.6, .4]]])[:, :tokens].contiguous()
    indices = torch.tensor([[[2, 0], [1, 2]]], dtype=torch.int32)[:, :tokens].contiguous()
    inputs = (src, weights, indices)
    with torch.no_grad():
        np.save(directory/'torch-reference.npy', layer(*inputs).numpy())
    for index, value in enumerate(inputs):
        value.numpy().tofile(directory/f'input-{index}.bin')
    litert_torch.convert(layer, inputs).export(str(directory/'model.tflite'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--library', type=Path, default=DEFAULT_LIB)
    ap.add_argument('--source', type=Path, required=True, help='Clean frozen litert-torch checkout')
    ap.add_argument('--worker-python', type=Path, default=ROOT/'tmp/moe-venv/bin/python')
    ap.add_argument('--export-case', choices=['fp32', 'int8'])
    ap.add_argument('--tokens', type=int, choices=[1, 2])
    args = ap.parse_args()
    args.output = args.output.resolve()
    if args.export_case:
        export_case(args.output, args.export_case, args.tokens)
        return 0

    from litert_torch.generative.layers import moe
    commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == PIN
    assert not subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'])
    assert sha(moe.__file__) == sha(args.source/'litert_torch/generative/layers/moe.py')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=PIN,
        installed_moe_sha256=sha(moe.__file__), script_sha256=sha(__file__),
        worker_sha256=sha(ROOT/'experiments/moe_layer_check.py'), library_sha256=sha(args.library),
        python=sys.version, platform=platform.platform(), argv=sys.argv, rtol=RTOL, atol=ATOL,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        cases=[], limitations=['Synthetic fixed routing; no router, attention, KV cache or generation.',
          'Prebuilt LiteRT-LM 0.17.0 library, not a build of the pinned LiteRT source.',
          'CPU only; no performance, memory, quality or GPU/NPU measurements.',
          'Attribute-modified models are diagnostic copies, not official exporter output.'])
    for mode in ('fp32', 'int8'):
        for tokens in (1, 2):
            name = f'{mode}-t{tokens}'
            directory = args.output/name
            directory.mkdir()
            cmd = [sys.executable, str(Path(__file__).resolve()), '--output', str(directory),
                   '--source', str(args.source), '--export-case', mode, '--tokens', str(tokens)]
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
            (directory/'export.stdout.log').write_bytes(proc.stdout)
            (directory/'export.stderr.log').write_bytes(proc.stderr)
            if proc.returncode:
                manifest['cases'].append(dict(name=name, export_returncode=proc.returncode))
                write_json(args.output/'report.json', manifest)
                raise RuntimeError(f'{name}: export failed; see preserved logs')
            tensors, attrs = read_model(directory/'model.tflite', directory)
            exact = oracle(tensors, attrs, 'gelu')
            tanh = oracle(tensors, attrs, 'gelu_tanh')
            np.save(directory/'oracle-gelu.npy', exact)
            np.save(directory/'oracle-gelu-tanh.npy', tanh)
            for diagnostic in (False, True):
                target = directory
                if diagnostic:
                    target = args.output/(name+'-diagnostic-tanh')
                    target.mkdir()
                    for index in range(3):
                        (target/f'input-{index}.bin').write_bytes((directory/f'input-{index}.bin').read_bytes())
                    diagnostic_copy(directory/'model.tflite', target/'model.tflite')
                    copied, changed = read_model(target/'model.tflite', target)
                    assert all(np.array_equal(a, b) for a, b in zip(tensors, copied))
                    assert changed == dict(attrs, activation='gelu_tanh')
                proc = subprocess.run([str(args.worker_python), str(ROOT/'experiments/moe_layer_check.py'),
                    '--library', str(args.library.resolve()), '--worker', str(target)],
                    capture_output=True, timeout=60)
                (target/'cpu.stdout.log').write_bytes(proc.stdout)
                (target/'cpu.stderr.log').write_bytes(proc.stderr)
                runtime = json.loads((target/'runtime.json').read_text()) if (target/'runtime.json').exists() else {}
                entry = dict(name=target.name, diagnostic=diagnostic, export_origin=name,
                    cpu_returncode=proc.returncode, invoke_completed=runtime.get('invoke_completed', False))
                if proc.returncode == 0 and entry['invoke_completed']:
                    actual = np.load(target/'actual.npy')
                    entry.update(cpu_vs_torch=compare(actual, np.load(directory/'torch-reference.npy')),
                        cpu_vs_exact_gelu=compare(actual, exact), cpu_vs_tanh_gelu=compare(actual, tanh),
                        torch_vs_tanh_oracle=compare(np.load(directory/'torch-reference.npy'), tanh))
                entry['files'] = {p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()}
                manifest['cases'].append(entry)
                write_json(args.output/'report.json', manifest)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
    manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.output/'report.json', manifest)
    # Completion is not a compatibility PASS: see each comparison in report.json.
    return int(any(not c.get('invoke_completed') for c in manifest['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
