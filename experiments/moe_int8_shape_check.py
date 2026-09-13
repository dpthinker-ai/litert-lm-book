#!/usr/bin/env python3
"""Distinguish INT8 GPU failures from hidden-channel alignment effects.

Crop H=6 to H=4 or zero-pad to H=8 in copies of prior affine-unit fixtures.
H=8 preserves the original function; H=4 has its own recomputed reference.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from flatbuffers import flexbuffers
from ai_edge_litert import schema_py_generated as s
from moe_int8_gpu_check import pack, make_copy
from moe_export_check import read_model, oracle, compare
from moe_layer_check import ROOT, DEFAULT_LIB, sha, write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    origin=ROOT/'experiments/data/2026-09-13/moe-int8-gpu'
    prior=json.loads((origin/'report.json').read_text())
    assert sha(DEFAULT_LIB)==prior['library_sha256']
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
                library_sha256=sha(DEFAULT_LIB),source_report_sha256=sha(origin/'report.json'),
                helpers={n:sha(ROOT/'experiments'/n) for n in ['moe_int8_gpu_check.py','moe_export_check.py','moe_backend_worker.py','moe_layer_check.py']},
                cases=[],limitations=['Synthetic fixed routes, E=3,K=2,D=4; no full language model.',
                                      'CPU reference uses exact GELU; GPU reference uses tanh GELU.',
                                      'FP32 control executes dequantized weights, not an INT8 kernel.',
                                      'Log correlation with a source branch is not a patched-runtime proof.'])
    for tokens in (1,2):
        source=origin/f't{tokens}-affine-unit-webgpu-fp32'
        hashes=next(c['files'] for c in prior['cases'] if c['name']==source.name)
        for hidden in (4,8):
            fixture=out/f't{tokens}-h{hidden}';fixture.mkdir()
            assert sha(source/'model.tflite')==hashes['model.tflite']
            model=s.ModelT.InitFromObj(s.Model.GetRootAsModel((source/'model.tflite').read_bytes(),0))
            graph=model.subgraphs[0];op=graph.operators[0]
            for position in (3,4,5,6,7):
                tensor=graph.tensors[int(op.inputs[position])]
                dtype='i1' if position in (3,5,7) else '<f4'
                values=np.frombuffer(model.buffers[tensor.buffer].data,dtype=dtype).reshape(tensor.shape)
                axis=3 if position==7 else 0
                shape=list(values.shape);shape[axis]=hidden
                resized=np.ones(shape,dtype=dtype) if position in (4,6) else np.zeros(shape,dtype=dtype)
                slices=[slice(None)]*4;slices[axis]=slice(0,min(hidden,6))
                resized[tuple(slices)]=values[tuple(slices)]
                tensor.shape=shape;model.buffers[tensor.buffer].data=np.frombuffer(resized.tobytes(),dtype='u1')
            attrs=flexbuffers.Loads(bytes(op.customOptions));attrs['hidden_dim']=hidden
            op.customOptions=np.frombuffer(flexbuffers.Dumps(attrs),dtype='u1')
            (fixture/'model.tflite').write_bytes(pack(model))
            for i in range(3):
                assert sha(source/f'input-{i}.bin')==hashes[f'input-{i}.bin']
                shutil.copyfile(source/f'input-{i}.bin',fixture/f'input-{i}.bin')
            tensors,attrs=read_model(fixture/'model.tflite',fixture)
            exact=oracle(tensors,attrs,'gelu');tanh=oracle(tensors,attrs,'gelu_tanh')
            if hidden==8:
                assert compare(tanh,np.load(source/'oracle-tanh.npy'))['matches']
            for backend,mode in [('cpu','int8'),('webgpu','int8'),('webgpu','dequant-fp32')]:
                directory=fixture/f'{backend}-{mode}';directory.mkdir()
                if mode=='int8':shutil.copyfile(fixture/'model.tflite',directory/'model.tflite')
                else:make_copy(fixture/'model.tflite',directory/'model.tflite','dequant-fp32')
                for i in range(3):shutil.copyfile(fixture/f'input-{i}.bin',directory/f'input-{i}.bin')
                expected=exact if backend=='cpu' else tanh;np.save(directory/'expected.npy',expected)
                command=[str(ROOT/'tmp/moe-venv/bin/python'),str(ROOT/'experiments/moe_backend_worker.py'),
                         '--directory',str(directory),'--backend',backend,'--precision','fp32' if backend=='webgpu' else 'default']
                proc=subprocess.run(command,capture_output=True,timeout=60)
                (directory/'stdout.log').write_bytes(proc.stdout);(directory/'stderr.log').write_bytes(proc.stderr)
                runtime=json.loads((directory/'runtime.json').read_text())
                errors=[line for line in proc.stderr.decode(errors='replace').splitlines() if 'Validation error:' in line]
                comparison=compare(np.load(directory/'actual.npy'),expected) if (directory/'actual.npy').exists() else None
                status='MATCH' if proc.returncode==0 and comparison and comparison['matches'] else 'FAILED'
                if errors:status='BACKEND_VALIDATION_ERROR'
                if backend=='webgpu' and status=='MATCH' and not(runtime.get('non_cpu_fully_accelerated') and b'backend=Metal' in proc.stderr):status='GPU_COVERAGE_UNCONFIRMED'
                entry=dict(name=str(directory.relative_to(out)),T=tokens,H=hidden,backend=backend,mode=mode,status=status,
                           returncode=proc.returncode,comparison=comparison,
                           missing_buffer_errors=sum('Binding entry buffer not set' in line for line in errors),
                           undefined_scale_errors=sum("unresolved value 'scale'" in line for line in errors),
                           validation_errors=errors,files={p.name:sha(p) for p in directory.iterdir()})
                report['cases'].append(entry);write_json(out/'report.json',report)
                print(entry['name'],status,entry['missing_buffer_errors'],entry['undefined_scale_errors'],flush=True)
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    write_json(out/'report.json',report)


if __name__=='__main__':main()
