#!/usr/bin/env python3
"""Probe INT8 MoE metadata using diagnostic copies of frozen tiny exports.

Never modify the exporter or original artifacts. Compilation acceptance, GPU
coverage, and numerical agreement are separate results, not deployment claims.
"""
import argparse
import copy
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import flatbuffers
from flatbuffers import flexbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from moe_export_check import read_model, oracle, compare
from moe_layer_check import ROOT, DEFAULT_LIB, sha, write_json


def pack(model):
    builder=flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
    return bytes(builder.Output())


def make_copy(original,target,variant):
    raw=original.read_bytes();base=s.ModelT.InitFromObj(s.Model.GetRootAsModel(raw,0))
    model=copy.deepcopy(base);graph=model.subgraphs[0];op=graph.operators[0]
    weights=[int(op.inputs[i]) for i in (3,5,7)]
    scales=[int(op.inputs[i]) for i in (4,6,8)]
    changed=[]
    if variant not in ('original','dequant-fp32'):
        for ti in weights:
            q=s.QuantizationParametersT();q.scale=[.5 if variant=='affine-half' else 1.0]
            q.zeroPoint=[1 if variant=='nonzero-zero-point' else 0];q.quantizedDimension=0
            graph.tensors[ti].quantization=q
            changed.append(f'tensor[{ti}].quantization')
        if variant=='double-linear-scale':
            bi=graph.tensors[scales[2]].buffer
            values=np.frombuffer(model.buffers[bi].data,dtype='<f4')*2.0
            model.buffers[bi].data=np.frombuffer(values.astype('<f4').tobytes(),dtype=np.uint8)
            changed.append(f'buffer[{bi}].data')
    elif variant=='dequant-fp32':
        for ti,si in zip(weights,scales):
            tensor=graph.tensors[ti];scale_tensor=graph.tensors[si]
            integers=np.frombuffer(model.buffers[tensor.buffer].data,dtype='i1').reshape(tensor.shape)
            scale=np.frombuffer(model.buffers[scale_tensor.buffer].data,dtype='<f4').reshape(scale_tensor.shape)
            values=(integers.astype(np.float32)*scale).astype('<f4')
            model.buffers[tensor.buffer].data=np.frombuffer(values.tobytes(),dtype=np.uint8)
            tensor.type=s.TensorType.FLOAT32
            changed.extend([f'tensor[{ti}].type',f'buffer[{tensor.buffer}].data'])
        op.inputs=[int(op.inputs[i]) for i in (0,1,2,3,5,7,9)]
        attrs=flexbuffers.Loads(bytes(op.customOptions));attrs['weight_type']='fp32'
        op.customOptions=np.frombuffer(flexbuffers.Dumps(attrs),dtype=np.uint8)
        changed.extend(['operator.inputs','operator.customOptions.weight_type'])
    target.write_bytes(raw if variant=='original' else pack(model))
    # Normalize only the intended fields, then compare the complete serialization.
    normalized=s.ModelT.InitFromObj(s.Model.GetRootAsModel(target.read_bytes(),0))
    ng=normalized.subgraphs[0];bg=base.subgraphs[0]
    if variant not in ('original','dequant-fp32'):
        for ti in weights:ng.tensors[ti].quantization=copy.deepcopy(bg.tensors[ti].quantization)
        if variant=='double-linear-scale':
            bi=graph.tensors[scales[2]].buffer;normalized.buffers[bi].data=base.buffers[bi].data.copy()
    elif variant=='dequant-fp32':
        for ti in weights:
            ng.tensors[ti].type=bg.tensors[ti].type
            bi=ng.tensors[ti].buffer;normalized.buffers[bi].data=base.buffers[bi].data.copy()
        ng.operators[0].inputs=bg.operators[0].inputs.copy()
        ng.operators[0].customOptions=bg.operators[0].customOptions.copy()
    assert pack(normalized)==pack(base), 'Unexpected model field changes'
    return changed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--exports',type=Path,default=ROOT/'experiments/data/2026-09-13/moe-export')
    ap.add_argument('--worker-python',type=Path,default=ROOT/'tmp/moe-venv/bin/python')
    ap.add_argument('--library',type=Path,default=DEFAULT_LIB)
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    prior=json.loads((args.exports/'report.json').read_text())
    assert sha(args.library)==prior['library_sha256']
    hashes={c['name']:c['files'] for c in prior['cases']}
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        platform=platform.platform(),python=sys.version,library_sha256=sha(args.library),
        script_sha256=sha(__file__),source_report_sha256=sha(args.exports/'report.json'),
        helper_sha256={p:sha(ROOT/'experiments'/p) for p in
                      ['moe_export_check.py','moe_layer_check.py','moe_backend_worker.py']},
        analysis_runtime_commit='9fe5be45564c868408e6514c8aabb83e211a0911',
        packages={p:importlib.metadata.version(p) for p in ['numpy','flatbuffers','ai-edge-litert-nightly']},
        execution_kind='LiteRT-LM 0.17.0 prebuilt dylib; not a local source build',
        atol=2e-6,rtol=2e-5,invoke_completed_is_api_return_only=True,cases=[],limitations=[
            'Post-export diagnostic copies; not a repaired exporter or production recommendation.',
            'Fixed E=3,K=2,D=4,H=6; T=1 is a prefix of T=2, not an independent random sample.',
            'Affine-half and nonzero-zero-point probe metadata interpretation; not equivalent quantized models.',
            'Dequant-fp32 uses floating weights; it cannot prove INT8 execution support.',
            'No full model, task quality, runtime memory, energy or performance measurement.'])
    configs=[('original','webgpu','fp32'),('affine-unit','cpu','default'),
             ('affine-unit','webgpu','fp32'),('affine-unit','webgpu','fp16'),
             ('affine-half','webgpu','fp32'),('double-linear-scale','webgpu','fp32'),
             ('nonzero-zero-point','webgpu','fp32'),('dequant-fp32','webgpu','fp32')]
    for tokens in (1,2):
        origin=f'int8-t{tokens}';source=args.exports/origin
        for variant,backend,precision in configs:
            name=f't{tokens}-{variant}-{backend}-{precision}';d=args.output/name;d.mkdir()
            for filename in ['model.tflite','torch-reference.npy']+[f'input-{i}.bin' for i in range(3)]:
                assert sha(source/filename)==hashes[origin][filename]
            changed=make_copy(source/'model.tflite',d/'model.tflite',variant)
            for file in source.glob('input-*.bin'):shutil.copyfile(file,d/file.name)
            tensors,attrs=read_model(d/'model.tflite',d)
            exact=oracle(tensors,attrs,'gelu');tanh=oracle(tensors,attrs,'gelu_tanh')
            np.save(d/'oracle-exact.npy',exact);np.save(d/'oracle-tanh.npy',tanh)
            shutil.copyfile(source/'torch-reference.npy',d/'original-torch-reference.npy')
            cmd=[str(args.worker_python),str(ROOT/'experiments/moe_backend_worker.py'),
                 '--directory',str(d),'--library',str(args.library.resolve()),'--backend',backend,'--precision',precision]
            try:
                proc=subprocess.run(cmd,capture_output=True,timeout=60)
                code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr
            except subprocess.TimeoutExpired as exc:
                code=None;stdout=exc.stdout or b'';stderr=exc.stderr or b''
            (d/'stdout.log').write_bytes(stdout);(d/'stderr.log').write_bytes(stderr)
            runtime=json.loads((d/'runtime.json').read_text()) if (d/'runtime.json').exists() else {}
            validation_errors=[line for line in stderr.decode(errors='replace').splitlines()
                               if 'Validation error:' in line]
            entry=dict(name=name,source=origin,variant=variant,changed_fields=changed,
                only_intended_fields_changed=True,backend=backend,precision=precision,returncode=code,
                invoke_completed=runtime.get('invoke_completed',False),
                non_cpu_fully_accelerated=runtime.get('non_cpu_fully_accelerated'),
                metal_adapter_logged=b'backend=Metal' in stderr,
                backend_validation_errors=validation_errors,gpu_execution_confirmed=False)
            if code==0 and entry['invoke_completed']:
                actual=np.load(d/'actual.npy')
                entry.update(vs_exact=compare(actual,exact),vs_tanh=compare(actual,tanh),
                    vs_original_torch=compare(actual,np.load(source/'torch-reference.npy')),
                    output_all_zero=bool(np.all(actual==0)))
                expected=entry['vs_exact'] if backend=='cpu' else entry['vs_tanh']
                entry['status']='MATCH' if expected['matches'] else 'NUMERICAL_MISMATCH'
                if backend!='cpu' and not(entry['non_cpu_fully_accelerated'] and entry['metal_adapter_logged']):
                    entry['status']='GPU_COVERAGE_UNCONFIRMED'
                if backend!='cpu':
                    entry['gpu_execution_confirmed']=bool(entry['non_cpu_fully_accelerated'] and
                                                         entry['metal_adapter_logged'] and not validation_errors)
                if validation_errors:entry['status']='BACKEND_VALIDATION_ERROR'
            elif code==2 and any(t['status']!=0 for t in runtime.get('api_trace',[])):
                entry.update(status='API_REJECTED',error=runtime.get('error'))
            else:entry['status']='HARNESS_FAILURE'
            entry['files']={p.name:sha(p) for p in sorted(d.iterdir()) if p.is_file()}
            report['cases'].append(entry);write_json(args.output/'report.json',report)
            print(name,entry['status'],entry.get('output_all_zero'),flush=True)
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    report['counts']={status:sum(c['status']==status for c in report['cases']) for status in
                     ['MATCH','NUMERICAL_MISMATCH','BACKEND_VALIDATION_ERROR','API_REJECTED','GPU_COVERAGE_UNCONFIRMED','HARNESS_FAILURE']}
    write_json(args.output/'report.json',report)
    return int(any(c['status'] in ['HARNESS_FAILURE','GPU_COVERAGE_UNCONFIRMED'] for c in report['cases']))


if __name__=='__main__':raise SystemExit(main())
