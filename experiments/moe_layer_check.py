#!/usr/bin/env python3
"""Small synthetic CPU MoE correctness checks using LiteRT's public C ABI.

This loads a prebuilt LiteRT-LM library. It does not build/validate the pinned
LiteRT source tree, measure performance, or run a complete language model.
Run in a venv containing numpy, flatbuffers and tflite schema bindings.
Each fixture executes in a separate process; native errors/crashes are retained.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import flatbuffers
from flatbuffers import flexbuffers
import numpy as np
import tflite

ROOT = Path(__file__).resolve().parents[1]
PIN = '9fe5be45564c868408e6514c8aabb83e211a0911'
DEFAULT_LIB = ROOT / 'tmp/upgrade-v0.17.0-venv/lib/python3.12/site-packages/litert_lm/liblitert-lm.dylib'
DEFAULT_OUT = ROOT / 'experiments/data/2026-09-13/moe-layer'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def make_model(tensors, attrs):
    """Build one custom op; only src/top_weights/top_indices are graph inputs."""
    b = flatbuffers.Builder(4096)
    def vec(vals, start, dtype='offset'):
        start(b, len(vals))
        for v in reversed(vals):
            (b.PrependUOffsetTRelative if dtype == 'offset' else b.PrependInt32)(v)
        return b.EndVector()
    buffers = []
    for raw in [b''] + [t['raw'] for t in tensors if t['raw'] is not None]:
        data = b.CreateByteVector(raw) if raw else None
        tflite.BufferStart(b)
        if data is not None: tflite.BufferAddData(b, data)
        buffers.append(tflite.BufferEnd(b))
    ts = []; buf_idx = 1
    for t in tensors:
        name = b.CreateString(t['name'])
        shape = vec(t['shape'], tflite.TensorStartShapeVector, 'int')
        tflite.TensorStart(b); tflite.TensorAddName(b, name)
        tflite.TensorAddShape(b, shape); tflite.TensorAddType(b, t['type'])
        if t['raw'] is not None:
            tflite.TensorAddBuffer(b, buf_idx); buf_idx += 1
        ts.append(tflite.TensorEnd(b))
    custom = b.CreateString('moe')
    tflite.OperatorCodeStart(b)
    tflite.OperatorCodeAddBuiltinCode(b, tflite.BuiltinOperator.CUSTOM)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(b, tflite.BuiltinOperator.CUSTOM)
    tflite.OperatorCodeAddCustomCode(b, custom)
    op_code = tflite.OperatorCodeEnd(b)
    op_inputs = vec(list(range(len(tensors)-1)), tflite.OperatorStartInputsVector, 'int')
    op_outputs = vec([len(tensors)-1], tflite.OperatorStartOutputsVector, 'int')
    options = b.CreateByteVector(flexbuffers.Dumps(attrs))
    tflite.OperatorStart(b); tflite.OperatorAddOpcodeIndex(b, 0)
    tflite.OperatorAddInputs(b, op_inputs); tflite.OperatorAddOutputs(b, op_outputs)
    tflite.OperatorAddCustomOptions(b, options)
    tflite.OperatorAddCustomOptionsFormat(b, tflite.CustomOptionsFormat.FLEXBUFFERS)
    op = tflite.OperatorEnd(b)
    tv = vec(ts, tflite.SubGraphStartTensorsVector)
    iv = vec([0, 1, 2], tflite.SubGraphStartInputsVector, 'int')
    ov = vec([len(tensors)-1], tflite.SubGraphStartOutputsVector, 'int')
    ops = vec([op], tflite.SubGraphStartOperatorsVector)
    name = b.CreateString('moe_layer')
    tflite.SubGraphStart(b); tflite.SubGraphAddTensors(b, tv)
    tflite.SubGraphAddInputs(b, iv); tflite.SubGraphAddOutputs(b, ov)
    tflite.SubGraphAddOperators(b, ops); tflite.SubGraphAddName(b, name)
    sg = tflite.SubGraphEnd(b)
    def tensor_map(name, index):
        n = b.CreateString(name); tflite.TensorMapStart(b)
        tflite.TensorMapAddName(b, n); tflite.TensorMapAddTensorIndex(b, index)
        return tflite.TensorMapEnd(b)
    im = [tensor_map(tensors[i]['name'], i) for i in range(3)]
    om = [tensor_map('output', len(tensors)-1)]
    siv = vec(im, tflite.SignatureDefStartInputsVector)
    sov = vec(om, tflite.SignatureDefStartOutputsVector)
    key = b.CreateString('serving_default')
    tflite.SignatureDefStart(b); tflite.SignatureDefAddInputs(b, siv)
    tflite.SignatureDefAddOutputs(b, sov); tflite.SignatureDefAddSignatureKey(b, key)
    tflite.SignatureDefAddSubgraphIndex(b, 0); sig = tflite.SignatureDefEnd(b)
    sgvec = vec([sg], tflite.ModelStartSubgraphsVector)
    bv = vec(buffers, tflite.ModelStartBuffersVector)
    cv = vec([op_code], tflite.ModelStartOperatorCodesVector)
    sv = vec([sig], tflite.ModelStartSignatureDefsVector)
    desc = b.CreateString('Synthetic MoE correctness fixture; not an LLM')
    tflite.ModelStart(b); tflite.ModelAddVersion(b, 3)
    tflite.ModelAddSubgraphs(b, sgvec); tflite.ModelAddBuffers(b, bv)
    tflite.ModelAddOperatorCodes(b, cv); tflite.ModelAddSignatureDefs(b, sv)
    tflite.ModelAddDescription(b, desc)
    model = tflite.ModelEnd(b); b.Finish(model, file_identifier=b'TFL3')
    return bytes(b.Output())


def reference(src, route_weights, indices, expert_scale, gate, up, down, activation):
    """Float64 direct per-route formula; no expert batching or packed addressing."""
    result = np.zeros(src.shape, dtype=np.float64)
    for token, x in enumerate(src.astype(np.float64)):
        for route, raw_expert in enumerate(indices[token]):
            expert = int(raw_expert) % len(expert_scale)
            g = gate[:, expert, :] @ x
            if activation == 'gelu':
                activated = .5*g*(1 + np.array([math.erf(float(v)/math.sqrt(2)) for v in g]))
            else:
                activated = .5*g*(1 + np.tanh(math.sqrt(2/math.pi)*(g+.044715*g**3)))
            hidden = activated * (up[:, expert, :] @ x)
            result[token] += route_weights[token, route]*expert_scale[expert]*(down[:, expert, :] @ hidden)
    return result


def fixture(directory, mode, tokens, activation, scenario='standard'):
    directory.mkdir(parents=True, exist_ok=False)
    E, K, D, H = 3, 2, 4, 6
    rng = np.random.default_rng(20260913 + tokens)
    src = rng.uniform(-.8, .8, (tokens, D)).astype('<f4')
    route_weights = rng.uniform(.1, .9, (tokens, K)).astype('<f4')
    route_weights /= route_weights.sum(axis=1, keepdims=True)
    indices = np.array([[2, 0], [1, 2], [0, 1], [2, 1]], dtype='<i4')
    indices = np.tile(indices, (math.ceil(tokens/4), 1))[:tokens].copy()
    if scenario == 'duplicates': indices[:] = [1, 1]
    if scenario == 'negative_indices': indices -= E
    if scenario == 'zero_routes': route_weights[:, 1] = 0
    if scenario == 'swapped_routes':
        indices = indices[:, ::-1].copy(); route_weights = route_weights[:, ::-1].copy()
    if scenario == 'invalid_high': indices[0, 0] = E
    if scenario == 'invalid_low': indices[0, 0] = -E-1
    expert_scale = np.array([.7, 1.1, 1.3], dtype='<f4')
    tensors = [dict(name=n, shape=list(a.shape), type=ty, raw=None) for n,a,ty in
               [('src',src,tflite.TensorType.FLOAT32),('top_weights',route_weights,tflite.TensorType.FLOAT32),
                ('top_indices',indices,tflite.TensorType.INT32)]]
    effective = []; archive = dict(src=src, top_weights=route_weights, top_indices=indices, expert_scale=expert_scale)
    for name, shape in [('gate', (H,E,D)), ('up',(H,E,D)), ('down',(D,E,H))]:
        if mode == 'fp32':
            values = rng.uniform(-.45,.45,shape).astype('<f4')
            ty=tflite.TensorType.FLOAT32; raw=values.tobytes(); storage_shape=list(shape)
            effective.append(values.astype(np.float64))
        else:
            values = rng.integers(-7,8,shape,dtype=np.int8)
            groups = 1 if mode == 'int8' else 2
            scales = rng.uniform(.02,.07,(*shape[:2],groups)).astype('<f4')
            # Oracle expands from original signed values, not packed nibbles.
            full_scales = np.repeat(scales, shape[-1]//groups, axis=-1)
            effective.append(values.astype(np.float64)*full_scales.astype(np.float64))
            if mode == 'int8':
                raw=values.tobytes(); ty=tflite.TensorType.INT8; storage_shape=list(shape)
            else:
                nibbles=values.reshape(-1).astype(np.int16)&15
                raw=(nibbles[::2] | (nibbles[1::2]<<4)).astype(np.uint8).tobytes()
                ty=tflite.TensorType.INT4; storage_shape=list(shape)
                if scenario == 'int8_storage': ty=tflite.TensorType.INT8; storage_shape=[len(raw)]
            archive[name+'_scale']=scales
        archive[name+'_stored_values']=values
        tensors.append(dict(name=name+'_weight',shape=storage_shape,type=ty,raw=raw))
        if mode != 'fp32':
            tensors.append(dict(name=name+'_scale',shape=list(scales.shape),type=tflite.TensorType.FLOAT32,raw=scales.tobytes()))
    tensors.append(dict(name='expert_scale',shape=[E],type=tflite.TensorType.FLOAT32,raw=expert_scale.tobytes()))
    tensors.append(dict(name='output',shape=[tokens,D],type=tflite.TensorType.FLOAT32,raw=None))
    attrs=dict(num_experts=E,num_active_experts=K,model_dim=D,hidden_dim=H,weight_type=mode,activation=activation)
    if scenario == 'unsupported_activation': attrs['activation']='silu'
    model=make_model(tensors,attrs); (directory/'model.tflite').write_bytes(model)
    np.savez(directory/'inputs-and-weights.npz',**archive)
    for i,a in enumerate((src,route_weights,indices)):a.tofile(directory/f'input-{i}.bin')
    invalid=scenario in ('invalid_high','invalid_low','unsupported_activation')
    if not invalid:
        expected=reference(src,route_weights,indices,expert_scale,*effective,activation)
        np.save(directory/'expected.npy',expected)
    meta=dict(weight_type=mode,tokens=tokens,activation=activation,scenario=scenario,attrs=attrs,
              expected_behavior='runtime_error' if invalid else 'numerical_match',
              oracle='float64 direct per-token/per-route GEMM, erf GELU or tanh GELU; signed quant values expanded independently',
              rtol=2e-5,atol=2e-6)
    write_json(directory/'fixture.json',meta)
    return meta


class Layout(C.Structure):
    _fields_=[('rank',C.c_uint,7),('has_strides',C.c_uint,1),('dimensions',C.c_int32*8),('strides',C.c_uint32*8)]
class RankedType(C.Structure):
    _fields_=[('element_type',C.c_int),('layout',Layout)]


def worker(directory, libpath):
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
    }
    for name,args in specs.items():
        f=getattr(lib,name);f.argtypes=args;f.restype=I
    trace=[]
    def call(name,*args):
        rc=int(getattr(lib,name)(*args));trace.append(dict(api=name,status=rc))
        if rc:raise RuntimeError(f'{name} returned {rc}')
    owned=[]
    def own(handle,destroy):owned.append((handle,destroy))
    result={}
    try:
        env=P();call('LiteRtCreateEnvironment',0,None,C.byref(env));own(env,'LiteRtDestroyEnvironment')
        opts=P();call('LiteRtCreateOptions',C.byref(opts));own(opts,'LiteRtDestroyOptions')
        call('LiteRtSetOptionsHardwareAccelerators',opts,1)
        model=P();call('LiteRtCreateModelFromFile',env,str(directory/'model.tflite').encode(),C.byref(model));own(model,'LiteRtDestroyModel')
        compiled=P();call('LiteRtCreateCompiledModel',env,model,opts,C.byref(compiled));own(compiled,'LiteRtDestroyCompiledModel')
        sig=P();call('LiteRtGetModelSignature',model,0,C.byref(sig))
        bufs=[]
        for i in range(4):
            tensor=P();kind='Input' if i<3 else 'Output';index=i if i<3 else 0
            call(f'LiteRtGetSignature{kind}TensorByIndex',sig,index,C.byref(tensor))
            ty=RankedType();call('LiteRtGetRankedTensorType',tensor,C.byref(ty))
            req=P();call(f'LiteRtGetCompiledModel{kind}BufferRequirements',compiled,0,index,C.byref(req))
            buf=P();call('LiteRtCreateManagedTensorBufferFromRequirements',env,C.byref(ty),req,C.byref(buf));own(buf,'LiteRtDestroyTensorBuffer');bufs.append(buf)
            address=P();call('LiteRtLockTensorBuffer',buf,C.byref(address),1)
            if i<3:
                raw=(directory/f'input-{i}.bin').read_bytes();C.memmove(address,raw,len(raw))
            else:
                count=math.prod(ty.layout.dimensions[:ty.layout.rank]);C.memset(address,0,count*4)
            call('LiteRtUnlockTensorBuffer',buf)
        call('LiteRtRunCompiledModel',compiled,0,3,(P*3)(*bufs[:3]),1,(P*1)(bufs[3]))
        address=P();call('LiteRtLockTensorBuffer',bufs[3],C.byref(address),0)
        output=np.ctypeslib.as_array(C.cast(address,C.POINTER(C.c_float)),shape=(count,)).copy()
        call('LiteRtUnlockTensorBuffer',bufs[3]);np.save(directory/'actual.npy',output)
        result['invoke_completed']=True
    except Exception as exc:
        result.update(invoke_completed=False,error=str(exc))
    finally:
        for h,name in reversed(owned):
            f=getattr(lib,name);f.argtypes=[P];f.restype=None;f(h)
    result['api_trace']=trace;write_json(directory/'runtime.json',result)
    print(json.dumps(result));return 0 if result['invoke_completed'] else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,default=DEFAULT_LIB)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--worker',type=Path)
    ap.add_argument('--quick',action='store_true',help='only the first FP32 T=1 fixture')
    args=ap.parse_args();args.library=args.library.resolve()
    if args.worker:return worker(args.worker.resolve(),args.library)
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[(m,t,a,'standard') for m in ('fp32','int8','int4') for t in (1,2,8) for a in ('gelu','gelu_tanh')]
    cases += [('fp32',2,'gelu',s) for s in ('duplicates','negative_indices','zero_routes','swapped_routes','invalid_high','invalid_low','unsupported_activation')]
    cases += [('int4',2,'gelu','int8_storage')]
    if args.quick:cases=cases[:1]
    manifest=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),library=str(args.library),library_sha256=sha(args.library),
                  analysis_source_commit=PIN,execution_kind='prebuilt LiteRT-LM 0.17.0 C ABI; not a local build of the pinned source',
                  python=sys.version,platform=platform.platform(),machine=platform.machine(),
                  packages={p:importlib.metadata.version(p) for p in ('numpy','flatbuffers','tflite')},
                  script_sha256=sha(__file__),argv=sys.argv,cases=[],limitations=[
                      'No full language model, router, quality, performance, peak memory, GPU or NPU test.',
                      'Default CPU compilation options; no custom MoE opt-in flag or custom kernel was registered.',
                      'INT4 tests use even input dimensions and two equally sized groups per row.',
                      'The prebuilt binary is not proven to be built from the pinned analysis commit.'])
    for idx,(mode,tokens,activation,scenario) in enumerate(cases):
        name=f'{idx:02d}-{mode}-t{tokens}-{activation}-{scenario}';d=args.output/name
        meta=fixture(d,mode,tokens,activation,scenario)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--library',str(args.library),'--worker',str(d)]
        proc=subprocess.run(cmd,capture_output=True,timeout=60)
        (d/'stdout.log').write_bytes(proc.stdout);(d/'stderr.log').write_bytes(proc.stderr)
        runtime=json.loads((d/'runtime.json').read_text()) if (d/'runtime.json').exists() else {}
        entry=dict(name=name,returncode=proc.returncode,expected_behavior=meta['expected_behavior'],invoke_completed=runtime.get('invoke_completed',False))
        if meta['expected_behavior']=='numerical_match' and entry['invoke_completed']:
            expected=np.load(d/'expected.npy');actual=np.load(d/'actual.npy').reshape(expected.shape)
            delta=np.abs(actual-expected);entry.update(max_abs_error=float(delta.max()),max_relative_error=float((delta/np.maximum(np.abs(expected),1e-12)).max()),
                status='PASS' if np.allclose(actual,expected,rtol=meta['rtol'],atol=meta['atol']) and np.isfinite(actual).all() else 'FAIL')
        elif meta['expected_behavior']=='runtime_error':
            # A signal or harness failure is not evidence of correct rejection.
            trace=runtime.get('api_trace',[])
            rejected=any(x['status']!=0 and x['api'] in ('LiteRtCreateCompiledModel','LiteRtRunCompiledModel') for x in trace)
            entry['status']='EXPECTED_REJECTION' if rejected and proc.returncode==2 else 'FAIL'
        else:entry['status']='FAIL'
        entry['files']={p.name:sha(p) for p in sorted(d.iterdir()) if p.is_file()}
        manifest['cases'].append(entry);write_json(args.output/'report.json',manifest)
        print(name,entry['status'],entry.get('max_abs_error',runtime.get('error','')),flush=True)
    manifest['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    manifest['counts']={s:sum(c['status']==s for c in manifest['cases']) for s in ('PASS','EXPECTED_REJECTION','FAIL')}
    write_json(args.output/'report.json',manifest)
    return int(manifest['counts']['FAIL']>0)

if __name__=='__main__':raise SystemExit(main())
