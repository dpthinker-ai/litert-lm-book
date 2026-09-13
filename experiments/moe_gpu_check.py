#!/usr/bin/env python3
"""Run frozen MoE artifacts on GPU with explicit precision and coverage checks.

Includes a tiny builtin ADD control. No benchmark or full language model is run.
Raw artifacts and outputs from previous experiments are never overwritten.
"""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from moe_layer_check import DEFAULT_LIB, ROOT, sha, write_json


def control(directory):
    model=s.ModelT();model.version=3
    model.description=b'Synthetic ADD control; not an MoE or language model'
    empty=s.BufferT();constant=s.BufferT()
    constant.data=np.frombuffer(np.array([1.25],dtype='<f4').tobytes(),dtype=np.uint8)
    model.buffers=[empty,constant]
    tensors=[]
    for name,shape,buffer in [('x',[1,2,4],0),('constant',[1],1),('y',[1,2,4],0)]:
        tensor=s.TensorT();tensor.name=name.encode();tensor.shape=shape
        tensor.type=s.TensorType.FLOAT32;tensor.buffer=buffer;tensors.append(tensor)
    opcode=s.OperatorCodeT();opcode.builtinCode=s.BuiltinOperator.ADD
    opcode.deprecatedBuiltinCode=s.BuiltinOperator.ADD;opcode.version=1
    model.operatorCodes=[opcode]
    op=s.OperatorT();op.opcodeIndex=0;op.inputs=[0,1];op.outputs=[2]
    op.builtinOptionsType=s.BuiltinOptions.AddOptions;op.builtinOptions=s.AddOptionsT()
    graph=s.SubGraphT();graph.name=b'add_control';graph.inputs=[0];graph.outputs=[2]
    graph.tensors=tensors;graph.operators=[op];model.subgraphs=[graph]
    sig=s.SignatureDefT();sig.signatureKey=b'serving_default';sig.subgraphIndex=0
    inp=s.TensorMapT();inp.name=b'x';inp.tensorIndex=0
    out=s.TensorMapT();out.name=b'y';out.tensorIndex=2
    sig.inputs=[inp];sig.outputs=[out];model.signatureDefs=[sig]
    builder=flatbuffers.Builder(1024);builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
    (directory/'model.tflite').write_bytes(bytes(builder.Output()))
    values=np.arange(8,dtype='<f4').reshape(1,2,4)/4-1
    values.tofile(directory/'input-0.bin');np.save(directory/'expected.npy',values+1.25)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--exports',type=Path,default=ROOT/'experiments/data/2026-09-13/moe-export')
    ap.add_argument('--worker-python',type=Path,default=ROOT/'tmp/moe-venv/bin/python')
    ap.add_argument('--library',type=Path,default=DEFAULT_LIB)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    prior=json.loads((args.exports/'report.json').read_text())
    expected_hashes={c['name']:c['files'] for c in prior['cases']}
    cases=[(f'fp32-t{t}',b,p) for t in (1,2) for b,p in
           [('cpu','default'),('webgpu','default'),('webgpu','fp16'),('webgpu','fp32')]]
    cases += [(f'int8-t{t}','webgpu','fp32') for t in (1,2)]
    cases += [('fp32-t2','gpu-auto','default'),('fp32-t2-diagnostic-tanh','webgpu','fp32'),
              ('add-control','webgpu','fp32')]
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        platform=platform.platform(),python=sys.version,library_sha256=sha(args.library),
        script_sha256=sha(__file__),worker_sha256=sha(ROOT/'experiments/moe_backend_worker.py'),
        imported_helper_sha256=sha(ROOT/'experiments/moe_layer_check.py'),
        source_report_sha256=sha(args.exports/'report.json'),
        analysis_runtime_commit='9fe5be45564c868408e6514c8aabb83e211a0911',
        execution_kind='LiteRT-LM 0.17.0 prebuilt library, not a local pinned-source build',
        packages={p:importlib.metadata.version(p) for p in ['numpy','flatbuffers','ai-edge-litert-nightly']},
        rtol=2e-5,atol=2e-6,cases=[],limitations=[
            'Tiny fixed-routing expert modules; no full model, quality, memory, power or timing benchmark.',
            'Requested precision and API coverage do not capture individual shader instruction precision.',
            'T=1 is the prefix of T=2; two shapes, not independent random samples.',
            'A rejection proves only the tested artifact/configuration is unsupported.'])
    for source,backend,precision in cases:
        name=f'{source}-{backend}-{precision}';directory=args.output/name;directory.mkdir()
        if source=='add-control':
            control(directory);expected=np.load(directory/'expected.npy')
        else:
            for file in (args.exports/source).iterdir():
                if file.name=='model.tflite' or file.name.startswith('input-'):
                    assert sha(file)==expected_hashes[source][file.name]
                    shutil.copyfile(file,directory/file.name)
            original=source.replace('-diagnostic-tanh','')
            ref=args.exports/original/'torch-reference.npy'
            assert sha(ref)==expected_hashes[original]['torch-reference.npy']
            expected=np.load(ref);np.save(directory/'expected.npy',expected)
        cmd=[str(args.worker_python),str(ROOT/'experiments/moe_backend_worker.py'),
             '--directory',str(directory.resolve()),'--library',str(args.library.resolve()),
             '--backend',backend,'--precision',precision]
        try:
            proc=subprocess.run(cmd,capture_output=True,timeout=60)
            code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr
        except subprocess.TimeoutExpired as exc:
            code=None;stdout=exc.stdout or b'';stderr=exc.stderr or b''
        (directory/'stdout.log').write_bytes(stdout);(directory/'stderr.log').write_bytes(stderr)
        runtime=json.loads((directory/'runtime.json').read_text()) if (directory/'runtime.json').exists() else {}
        entry=dict(name=name,source=source,backend=backend,precision=precision,returncode=code,
            invoke_completed=runtime.get('invoke_completed',False),
            non_cpu_fully_accelerated=runtime.get('non_cpu_fully_accelerated'),
            metal_adapter_logged=b'backend=Metal' in stderr)
        if code==0 and entry['invoke_completed']:
            actual=np.load(directory/'actual.npy').reshape(expected.shape);delta=np.abs(actual-expected)
            matches=bool(np.isfinite(actual).all() and np.allclose(actual,expected,rtol=2e-5,atol=2e-6))
            entry.update(numerical_match=matches,max_abs_error=float(delta.max()),
                max_relative_error=float((delta/np.maximum(np.abs(expected),1e-12)).max()))
            entry['status']='MATCH' if matches else 'NUMERICAL_MISMATCH'
            if backend!='cpu' and not(entry['non_cpu_fully_accelerated'] and entry['metal_adapter_logged']):
                entry['status']='GPU_COVERAGE_UNCONFIRMED'
        elif code==2 and any(t['status']!=0 for t in runtime.get('api_trace',[])):
            entry['status']='API_REJECTED';entry['error']=runtime.get('error')
        else:entry['status']='HARNESS_FAILURE'
        entry['files']={p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
        report['cases'].append(entry);write_json(args.output/'report.json',report)
        print(name,entry['status'],entry.get('max_abs_error',entry.get('error')),flush=True)
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    report['counts']={key:sum(c['status']==key for c in report['cases']) for key in
                      ['MATCH','NUMERICAL_MISMATCH','API_REJECTED','GPU_COVERAGE_UNCONFIRMED','HARNESS_FAILURE']}
    write_json(args.output/'report.json',report)
    # Exit code concerns collection integrity, never universal GPU support.
    return int(any(c['status'] in ['HARNESS_FAILURE','GPU_COVERAGE_UNCONFIRMED'] for c in report['cases']))


if __name__=='__main__':raise SystemExit(main())
