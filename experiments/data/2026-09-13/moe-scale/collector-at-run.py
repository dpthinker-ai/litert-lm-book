#!/usr/bin/env python3
"""Reproducible synthetic MoE scale/routing experiment, not an LLM benchmark.

Generated FP32 model files are intentionally excluded from Git. Recreate them
with the recorded script, NumPy version and seeds; their SHA256s are archived.
CPU uses gelu_tanh and GPU uses gelu (the latter implements tanh GELU).
"""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tflite
from moe_layer_check import ROOT, DEFAULT_LIB, make_model, reference, sha, write_json


def fixture(directory, experts, active, tokens, dim, routing):
    directory.mkdir(parents=True, exist_ok=False)
    hidden=2*dim
    rng=np.random.default_rng(20260914)
    # Fixed maximum arrays make E and T changes select prefixes of the same data.
    src=rng.normal(0,.4,(64,dim)).astype('<f4')[:tokens].copy()
    matrices=[]
    for rows,cols in [(hidden,dim),(hidden,dim),(dim,hidden)]:
        values=rng.normal(0,1/math.sqrt(cols),(16,rows,cols)).astype('<f4')
        matrices.append(np.ascontiguousarray(values[:experts].transpose(1,0,2)))
    weights=np.full((tokens,active),1/active,dtype='<f4')
    indices=np.tile(np.arange(active,dtype='<i4'),(tokens,1))
    if routing=='distributed':indices=(indices+np.arange(tokens,dtype='<i4')[:,None]*active)%experts
    expert_scale=np.ones(experts,dtype='<f4')
    expected=reference(src,weights,indices,expert_scale,*[a.astype('float64') for a in matrices],'gelu_tanh')
    np.save(directory/'expected.npy',expected)
    arrays=[src,weights,indices]
    for i,array in enumerate(arrays):array.tofile(directory/f'input-{i}.bin')
    tensors=[dict(name=n,shape=shape,type=ty,raw=None) for n,shape,ty in
             [('src',[1,tokens,dim],tflite.TensorType.FLOAT32),
              ('top_weights',[1,tokens,active],tflite.TensorType.FLOAT32),
              ('top_indices',[1,tokens,active],tflite.TensorType.INT32)]]
    for name,array in zip(['gate','up','down'],matrices):
        tensors.append(dict(name=name,shape=[array.shape[0],experts,1,array.shape[2]],
                            type=tflite.TensorType.FLOAT32,raw=array.tobytes()))
    tensors.extend([dict(name='expert_scale',shape=[1,1,1,experts],type=tflite.TensorType.FLOAT32,raw=expert_scale.tobytes()),
                    dict(name='output',shape=[1,tokens,dim],type=tflite.TensorType.FLOAT32,raw=None)])
    for backend,activation in [('cpu','gelu_tanh'),('webgpu','gelu')]:
        attrs=dict(num_experts=experts,num_active_experts=active,model_dim=dim,hidden_dim=hidden,
                   weight_type='fp32',activation=activation,renormalized_top_weights=True)
        (directory/f'{backend}.tflite').write_bytes(make_model(tensors,attrs))
    meta=dict(E=experts,K=active,T=tokens,D=dim,H=hidden,routing=routing,seed=20260914,
              selected_experts=np.unique(indices).tolist(),routes_per_expert=np.bincount(indices.ravel(),minlength=experts).tolist(),
              projection_weight_bytes=3*experts*dim*hidden*4,
              selected_projection_weight_bytes=3*len(np.unique(indices))*dim*hidden*4,
              projection_flops=6*tokens*active*dim*hidden,
              predicted_gpu_branch='packed_groups' if tokens*active>experts else 'batch_ids_fc',
              byte_counts_are_logical_sizes_not_measured_traffic=True,
              files={p.name:sha(p) for p in sorted(directory.iterdir())})
    write_json(directory/'fixture.json',meta)
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    # E, K, T, D, route pattern. Change one factor around each baseline.
    configs=[(e,2,1,128,'concentrated') for e in (4,8,16)]
    configs += [(8,k,1,128,'concentrated') for k in (1,4)]
    configs += [(8,2,t,128,r) for t in (4,5,16,64) for r in ('concentrated','distributed')]
    configs += [(8,2,t,512,r) for t,r in [(1,'concentrated'),(16,'concentrated'),(16,'distributed')]]
    if args.pilot:configs=[configs[1],configs[7]]
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,platform=platform.platform(),python=sys.version,
                machine=dict(chip='Apple M5 Pro',memory_bytes=25769803776),
                packages={p:importlib.metadata.version(p) for p in ('numpy','flatbuffers','tflite')},
                library_sha256=sha(DEFAULT_LIB),script_sha256=sha(__file__),
                worker_sha256=sha(ROOT/'experiments/moe_timing_worker.py'),
                helper_sha256=sha(ROOT/'experiments/moe_layer_check.py'),
                runtime='LiteRT-LM 0.17.0 prebuilt; not a local source build',
                cpu_activation='gelu_tanh',gpu_activation='gelu implements tanh GELU',
                rtol=2e-4,atol=2e-5,warmups_per_process=3,timed_runs_per_process=12,
                worker_environment_overrides=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'),
                process_repetitions=1 if args.pilot else 3,fixtures={},cases=[],limitations=[
                    'Synthetic single expert layer with supplied routes; excludes router, attention, KV cache and sampling.',
                    'Synchronized host-observed wall time, not GPU kernel time or end-to-end language-model throughput.',
                    'Process maximum RSS includes compilation and host memory; it is not GPU memory allocation.',
                    'Logical selected weight size is not measured DRAM traffic or a measured working set.',
                    'Other applications, clock frequency and thermal state are not isolated.',
                    'All timed iterations repeat the same input and routes; no cold expert-cache or route-transition measurement.',
                    'Timing includes Python/ctypes dispatch and output readback overhead; CPU thread count is not explicitly configured.'])
    jobs=[]
    for config in configs:
        e,k,t,d,r=config;name=f'e{e}-k{k}-t{t}-d{d}-{r}'
        report['fixtures'][name]=fixture(out/name,*config)
        for backend in ('cpu','webgpu'):
            for rep in range(report['process_repetitions']):jobs.append((name,backend,rep))
    random.Random(20260914).shuffle(jobs)
    write_json(out/'report.json',report)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for name,backend,rep in jobs:
        pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
        if pressure&4:raise RuntimeError('Critical system memory pressure; no new worker started')
        source=out/name;directory=source/f'{backend}-{rep}';directory.mkdir()
        os.link(source/f'{backend}.tflite',directory/'model.tflite')
        for filename in ['expected.npy']+[f'input-{i}.bin' for i in range(3)]:shutil.copyfile(source/filename,directory/filename)
        command=[sys.executable,str(ROOT/'experiments/moe_timing_worker.py'),'--directory',str(directory),
                 '--backend',backend,'--precision','fp32' if backend=='webgpu' else 'default']
        try:
            proc=subprocess.run(command,env=env,capture_output=True,timeout=90)
            code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr
        except subprocess.TimeoutExpired as exc:code=None;stdout=exc.stdout or b'';stderr=exc.stderr or b''
        (directory/'stdout.log').write_bytes(stdout);(directory/'stderr.log').write_bytes(stderr)
        runtime=json.loads((directory/'runtime.json').read_text()) if (directory/'runtime.json').exists() else {}
        errors=[line for line in stderr.decode(errors='replace').splitlines() if 'Validation error:' in line]
        valid=code==0 and runtime.get('invoke_completed') and not errors
        if backend=='webgpu':valid=valid and runtime.get('non_cpu_fully_accelerated') and b'backend=Metal' in stderr
        entry=dict(fixture=name,backend=backend,repetition=rep,returncode=code,valid=bool(valid),
                   pressure_before=pressure,validation_errors=errors,
                   median_ms=runtime.get('median_ms'),compile_ms=runtime.get('compile_ms'),
                   process_max_rss_bytes=runtime.get('process_max_rss_bytes'),
                   files={p.name:sha(p) for p in sorted(directory.iterdir())})
        report['cases'].append(entry);write_json(out/'report.json',report)
        print(name,backend,rep,'VALID' if valid else 'FAILED',entry['median_ms'],flush=True)
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    report['valid_cases']=sum(c['valid'] for c in report['cases'])
    write_json(out/'report.json',report)
    return int(report['valid_cases']!=len(report['cases']))


if __name__=='__main__':raise SystemExit(main())
