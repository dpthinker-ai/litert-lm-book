#!/usr/bin/env python3
"""Synchronized single-layer timing probe derived from moe_backend_worker.py.

Execute in a child process; raw native errors and acceleration coverage matter.
Only static FP32/INT32 inputs and a single FP32 output are accepted.
"""
import argparse
import ctypes as C
import json
import math
import time
import resource
from pathlib import Path
import numpy as np
from moe_layer_check import Layout, RankedType, DEFAULT_LIB, write_json

def worker(directory, libpath, backend, precision, warmups, repeats):
    assert C.sizeof(Layout)==68 and C.sizeof(RankedType)==72
    lib=C.CDLL(str(libpath)); P=C.c_void_p; PP=C.POINTER(P); Z=C.c_size_t; I=C.c_int
    specs={
        'LiteRtCreateEnvironment':[I,P,PP], 'LiteRtCreateOptions':[PP],
        'LiteRtSetOptionsHardwareAccelerators':[P,I], 'LiteRtCreateModelFromFile':[P,C.c_char_p,PP],
        'LiteRtCreateCompiledModel':[P,P,P,PP], 'LiteRtGetModelSignature':[P,Z,PP],
        'LiteRtGetSignatureInputTensorByIndex':[P,Z,PP], 'LiteRtGetSignatureOutputTensorByIndex':[P,Z,PP],
        'LiteRtGetRankedTensorType':[P,C.POINTER(RankedType)],
        'LiteRtGetCompiledModelInputBufferRequirements':[P,Z,Z,PP],
        'LiteRtGetCompiledModelOutputBufferRequirements':[P,Z,Z,PP],
        'LiteRtCreateManagedTensorBufferFromRequirements':[P,C.POINTER(RankedType),P,PP],
        'LiteRtLockTensorBuffer':[P,PP,I], 'LiteRtUnlockTensorBuffer':[P],
        'LiteRtRunCompiledModel':[P,Z,Z,PP,Z,PP],
        'LiteRtGetNumSignatureInputs':[P,C.POINTER(Z)],
        'LiteRtGetNumSignatureOutputs':[P,C.POINTER(Z)],
        'LiteRtCompiledModelIsNonCpuFullyAccelerated':[P,C.POINTER(C.c_bool)],
        'LiteRtGetTensorBufferType':[P,C.POINTER(I)],
        'LrtCreateGpuOptions':[PP], 'LrtSetGpuOptionsGpuBackend':[P,I],
        'LrtSetGpuAcceleratorCompilationOptionsPrecision':[P,I],
        'LrtGetOpaqueGpuOptionsData':[P,C.POINTER(C.c_char_p),PP,PP],
        'LiteRtCreateOpaqueOptions':[C.c_char_p,P,P,PP],
        'LiteRtAddOpaqueOptions':[P,P],
    }
    for name,args in specs.items():
        f=getattr(lib,name);f.argtypes=args;f.restype=I
    trace=[]
    def call(name,*args):
        rc=int(getattr(lib,name)(*args));trace.append(dict(api=name,status=rc))
        if rc:raise RuntimeError(f'{name} returned {rc}')
    owned=[]
    def own(handle,destroy):owned.append((handle,destroy))
    result={'requested_backend':backend,'requested_precision':precision}
    try:
        env=P();call('LiteRtCreateEnvironment',0,None,C.byref(env));own(env,'LiteRtDestroyEnvironment')
        opts=P();call('LiteRtCreateOptions',C.byref(opts));own(opts,'LiteRtDestroyOptions')
        call('LiteRtSetOptionsHardwareAccelerators',opts,1 if backend=='cpu' else 2)
        if backend=='webgpu':
            gpu=P();call('LrtCreateGpuOptions',C.byref(gpu));own(gpu,'LrtDestroyGpuOptions')
            call('LrtSetGpuOptionsGpuBackend',gpu,2)
            if precision!='default':
                call('LrtSetGpuAcceleratorCompilationOptionsPrecision',gpu,2 if precision=='fp32' else 1)
            identifier=C.c_char_p();payload=P();deleter=P()
            call('LrtGetOpaqueGpuOptionsData',gpu,C.byref(identifier),C.byref(payload),C.byref(deleter))
            opaque=P();call('LiteRtCreateOpaqueOptions',identifier,payload,deleter,C.byref(opaque))
            own(opaque,'LiteRtDestroyOpaqueOptions')
            call('LiteRtAddOpaqueOptions',opts,opaque)
            owned.pop() # Ownership transferred to opts.
        model=P();call('LiteRtCreateModelFromFile',env,str(directory/'model.tflite').encode(),C.byref(model));own(model,'LiteRtDestroyModel')
        started=time.perf_counter_ns()
        compiled=P();call('LiteRtCreateCompiledModel',env,model,opts,C.byref(compiled));own(compiled,'LiteRtDestroyCompiledModel')
        result['compile_ms']=(time.perf_counter_ns()-started)/1e6
        accelerated=C.c_bool();call('LiteRtCompiledModelIsNonCpuFullyAccelerated',compiled,C.byref(accelerated))
        result['non_cpu_fully_accelerated']=accelerated.value
        sig=P();call('LiteRtGetModelSignature',model,0,C.byref(sig))
        nin=Z();nout=Z();call('LiteRtGetNumSignatureInputs',sig,C.byref(nin))
        call('LiteRtGetNumSignatureOutputs',sig,C.byref(nout))
        assert nout.value==1, 'This probe supports one FP32 output'
        input_count=nin.value
        result['buffers']=[]
        bufs=[]
        for i in range(input_count+1):
            tensor=P();kind='Input' if i<input_count else 'Output';index=i if i<input_count else 0
            call(f'LiteRtGetSignature{kind}TensorByIndex',sig,index,C.byref(tensor))
            ty=RankedType();call('LiteRtGetRankedTensorType',tensor,C.byref(ty))
            req=P();call(f'LiteRtGetCompiledModel{kind}BufferRequirements',compiled,0,index,C.byref(req))
            buf=P();call('LiteRtCreateManagedTensorBufferFromRequirements',env,C.byref(ty),req,C.byref(buf));own(buf,'LiteRtDestroyTensorBuffer');bufs.append(buf)
            shape=list(ty.layout.dimensions[:ty.layout.rank])
            assert all(d>0 for d in shape)
            count=math.prod(shape)
            assert ty.element_type in (1,2), f'Unsupported element type {ty.element_type}'
            if kind=='Output': assert ty.element_type==1
            btype=I();call('LiteRtGetTensorBufferType',buf,C.byref(btype))
            result['buffers'].append(dict(kind=kind,index=index,shape=shape,type=btype.value))
            raw=(directory/f'input-{i}.bin').read_bytes() if i<input_count else bytes(count*4)
            assert len(raw)==count*4, 'Input byte count must match buffer tensor size'
            address=P();call('LiteRtLockTensorBuffer',buf,C.byref(address),1)
            C.memmove(address,raw,len(raw))
            call('LiteRtUnlockTensorBuffer',buf)
        expected=np.load(directory/'expected.npy').reshape(-1)
        measurements=[];errors=[]
        inputs=(P*input_count)(*bufs[:input_count]);outputs=(P*1)(bufs[-1])
        for step in range(warmups+repeats):
            started=time.perf_counter_ns()
            call('LiteRtRunCompiledModel',compiled,0,input_count,inputs,1,outputs)
            address=P();call('LiteRtLockTensorBuffer',bufs[-1],C.byref(address),0)
            output=np.ctypeslib.as_array(C.cast(address,C.POINTER(C.c_float)),shape=(count,)).copy()
            call('LiteRtUnlockTensorBuffer',bufs[-1])
            elapsed=(time.perf_counter_ns()-started)/1e6
            # Every warmup and timed result is checked; correctness is outside timing.
            matches=bool(np.isfinite(output).all() and np.allclose(output,expected,rtol=2e-4,atol=2e-5))
            errors.append(dict(step=step,matches=matches,max_abs_error=float(np.max(np.abs(output-expected)))))
            if step>=warmups:measurements.append(elapsed)
            if not matches:raise RuntimeError('Output does not match float64 tanh-GELU reference')
        np.save(directory/'actual.npy',output)
        result.update(samples_ms=measurements,checks=errors,warmups=warmups,
                      median_ms=float(np.median(measurements)),
                      process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      timing_scope='Run plus output lock/wait, host copy and unlock; excludes compile/input upload/reference check')
        result['invoke_completed']=True
    except Exception as exc:
        result.update(invoke_completed=False,error=str(exc))
    finally:
        for h,name in reversed(owned):
            f=getattr(lib,name);f.argtypes=[P];f.restype=None;f(h)
    result['api_trace']=trace;write_json(directory/'runtime.json',result)
    print(json.dumps(result));return 0 if result['invoke_completed'] else 2


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--library',type=Path,default=DEFAULT_LIB)
    ap.add_argument('--backend',choices=['cpu','gpu-auto','webgpu'],required=True)
    ap.add_argument('--precision',choices=['default','fp16','fp32'],default='default')
    ap.add_argument('--warmups',type=int,default=3)
    ap.add_argument('--repeats',type=int,default=12)
    args=ap.parse_args()
    if args.warmups<0 or args.repeats<1:ap.error('Invalid repeat counts')
    if args.backend!='webgpu' and args.precision!='default':ap.error('Precision requires explicit webgpu backend')
    raise SystemExit(worker(args.directory.resolve(),args.library.resolve(),args.backend,args.precision,args.warmups,args.repeats))
