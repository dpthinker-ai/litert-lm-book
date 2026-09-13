#!/usr/bin/env python3
"""Phase-aligned macOS process memory for a fixed full MoE workload.

New collector derived from the immutable moe_generation_check.py. Each process
creates one engine and fresh conversations; guard conditions are unchanged.
Callback arrival is a client-visible event, not a per-token timestamp.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def stop(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()


def make_notes_prompt(engine, target):
    prefix='Reference notes. The access code is AMBER-42.\n'
    question='\nUse the notes to state the access code and explain in one short sentence what RAM stores.'
    records=[]
    while True:
        row=f'Record {len(records):03d}: RAM stores temporary data needed by active programs.\n'
        candidate=prefix+''.join(records+[row])+question
        if len(engine.tokenize(candidate))>target:break
        records.append(row)
    prompt=prefix+''.join(records)+question
    tokens=engine.tokenize(prompt)
    if not records or len(tokens)>target:raise ValueError('Input token budget too small')
    return prompt,dict(raw_token_budget=target,raw_token_count=len(tokens),record_count=len(records),
                       marker='AMBER-42',tokens=tokens)



class RusageInfoV0(ctypes.Structure):
    # macOS SDK sys/resource.h:202-214, RUSAGE_INFO_V0.
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups',
            'pageins', 'wired_size', 'resident_size', 'phys_footprint',
            'proc_start_abstime', 'proc_exit_abstime')]


def process_memory(pid):
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    call = library.proc_pid_rusage
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    call.restype = ctypes.c_int
    usage = RusageInfoV0()
    if call(pid, 0, ctypes.byref(usage)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return {name: getattr(usage, name) for name in (
        'resident_size', 'phys_footprint', 'wired_size', 'pageins')}

def worker(args):
    import dataclasses
    import queue
    import litert_lm as lm
    from litert_lm._ffi import STREAM_CALLBACK_TYPE
    lm.set_min_log_severity(lm.LogSeverity.INFO)
    began=time.monotonic()
    result=dict(phase='engine_creation',engine_created=False,generation_completed=False,
                prompt=args.prompt,generations=[],memory_stages=[])
    def checkpoint():save(args.output/'worker.json',result)
    def snapshot(stage, mapping=False):
        entry = dict(stage=stage, elapsed_seconds=time.monotonic()-began)
        try:
            entry.update(process_memory(os.getpid()))
        except OSError as exc:
            entry['capture_error'] = 'libproc: ' + str(exc)
            result['memory_stages'].append(entry)
            checkpoint()
            return
        result['memory_stages'].append(entry)
        checkpoint()
        if mapping:
            entry['vmmap_started_seconds'] = time.monotonic()-began
            try:
                capture = subprocess.run(['vmmap', '-summary', str(os.getpid())],
                                         capture_output=True, text=True, timeout=15)
                (args.output / (stage + '-vmmap.txt')).write_text(capture.stdout + capture.stderr)
                entry['vmmap_returncode'] = capture.returncode
                if capture.returncode:
                    entry['capture_error'] = 'vmmap returned nonzero status'
            except (OSError, subprocess.SubprocessError) as exc:
                entry['capture_error'] = str(exc)
            entry['vmmap_finished_seconds'] = time.monotonic()-began
            checkpoint()
    try:
        snapshot('before_engine')
        engine_started=time.monotonic()
        with lm.Engine(str(args.model),backend=lm.Backend.GPU(),max_num_tokens=args.context,
                       cache_dir=str(args.cache),enable_speculative_decoding=False,enable_benchmark=True) as engine:
            result.update(engine_created=True,engine_create_seconds=time.monotonic()-engine_started)
            snapshot('engine_created', mapping=True)
            if args.input_tokens:
                args.prompt,fixture=make_notes_prompt(engine,args.input_tokens)
                result['prompt']=args.prompt
                save(args.output/'prompt-tokenization.json',fixture)
                (args.output/'prompt.txt').write_text(args.prompt)
                checkpoint()
            notes_prompt = args.prompt
            for index in range(args.repeats):
                args.prompt = ('In one short sentence, explain what RAM stores.'
                               if index == 0 else notes_prompt)
                stage = 'short' if index == 0 else 'long'
                result['phase']='conversation_creation';checkpoint()
                with engine.create_conversation(
                    thinking_config=lm.ThinkingConfig(enable_thinking=False,thinking_token_budget=0),
                    sampler_config=lm.SamplerConfig(top_k=1,temperature=0,seed=42),
                    max_output_tokens=args.output_tokens) as conversation:
                    item=dict(index=index,prompt=args.prompt,events=[],stream_final=False,stream_error=None)
                    result['generations'].append(item)
                    snapshot(stage + '_conversation_created')
                    result['phase']='generation';checkpoint()
                    q=queue.Queue();lib=conversation._lib;started=time.monotonic()
                    def receive(unused,chunk):
                        # Copy borrowed C strings inside the callback, then queue Python values.
                        error=lib.litert_lm_stream_chunk_get_error(chunk)
                        content=lib.litert_lm_stream_chunk_get_text(chunk)
                        q.put(dict(elapsed_seconds=time.monotonic()-started,
                                   error=error.decode('utf-8') if error else None,
                                   message=content.decode('utf-8') if content else '',
                                   is_final=bool(lib.litert_lm_stream_chunk_is_final(chunk))))
                    callback=STREAM_CALLBACK_TYPE(receive)
                    # The frozen C API inherits sampler/thinking/output limits from this conversation.
                    payload=json.dumps(dict(role='user',content=[dict(type='text',text=args.prompt)]))
                    rc=lib.litert_lm_conversation_send_message_stream(
                        conversation._ptr,payload,'{}',None,callback,None)
                    if rc:raise RuntimeError(f'stream request failed: {rc}')
                    text_parts=[]
                    while True:
                        event=q.get();item['events'].append(event)
                        if event['error']:
                            item['stream_error']=event['error'];checkpoint()
                            raise RuntimeError(event['error'])
                        if event['message']:
                            message=json.loads(event['message'])
                            for content in message.get('content',[]):
                                if content.get('type')=='text' and content.get('text'):
                                    text_parts.append(content['text'])
                                    item.setdefault('first_text_callback_seconds',event['elapsed_seconds'])
                        item['text']=''.join(text_parts)
                        item['stream_final']=event['is_final'];checkpoint()
                        if event['is_final']:break
                    item['generation_wall_seconds']=time.monotonic()-started
                    result['phase']='metrics';checkpoint()
                    item['benchmark']=dataclasses.asdict(conversation.get_benchmark_info())
                    item['conversation_token_count']=conversation.token_count
                    if not item['text'] or item['benchmark']['last_decode_token_count']<2:
                        raise RuntimeError('No verified multi-token response')
                    if index == 1:item['answer_contains_marker']='AMBER-42' in item['text'].upper()
                    item['generation_completed']=True;checkpoint()
                    snapshot(stage + '_generation_finished', mapping=True)
                item['conversation_closed']=True;checkpoint()
                snapshot(stage + '_conversation_closed')
            result.update(generation_completed=True,phase='cleanup');checkpoint()
        result['engine_closed']=True
        snapshot('engine_closed', mapping=True)
        result.update(phase='completed',engine_closed=True)
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
    checkpoint()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['phase']=='completed' else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--context',type=int,default=4096)
    ap.add_argument('--output-tokens',type=int,default=64)
    ap.add_argument('--prompt',default='Reply with only the word OK.')
    ap.add_argument('--timeout',type=float,default=180)
    ap.add_argument('--max-rss-gib',type=float,default=18)
    ap.add_argument('--input-tokens',type=int,default=3968)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    args.model=args.model.resolve();args.output=args.output.resolve();args.cache=args.cache.resolve()
    if args.input_tokens<0 or (args.input_tokens and args.input_tokens+args.output_tokens+32>args.context):
        ap.error('Reserve 32 tokens for chat formatting, in addition to input and output budgets')
    if (args.context,args.input_tokens,args.output_tokens,args.repeats)!=(4096,3968,64,2):
        ap.error('Fixed experiment: capacity 4096, notes budget 3968, output cap 64, two requests')
    if args.worker:return worker(args)
    if sys.platform!='darwin':ap.error('This collector uses macOS memory-pressure guards')
    if min(args.context,args.output_tokens,args.timeout,args.max_rss_gib,args.repeats)<=0:ap.error('Limits must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    args.cache.mkdir(parents=True,exist_ok=False)
    digest=sha(args.model)
    if digest!=args.sha256:raise ValueError('Full model hash mismatch; engine not started')
    import litert_lm
    library=Path(litert_lm.__file__).parent/'liblitert-lm.dylib'
    assert importlib.metadata.version('litert-lm-api')=='0.17.0'
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--model',str(args.model),
             '--sha256',digest,'--output',str(args.output),'--cache',str(args.cache),
             '--context',str(args.context),'--output-tokens',str(args.output_tokens),'--prompt',args.prompt,
             '--repeats',str(args.repeats),'--input-tokens',str(args.input_tokens)]
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in
                                            ['litert-lm-api','litert-lm','litert-lm-builder']},
        model=dict(path=str(args.model),bytes=args.model.stat().st_size,sha256=digest),
        library_sha256=sha(library),script_sha256=sha(__file__),worker_command=command,
        requested_backend='GPU',context_capacity=args.context,max_output_tokens=args.output_tokens,
        repetitions_in_engine=args.repeats,benchmark_enabled=True,input_raw_token_budget=args.input_tokens,
        machine=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                     memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize']))),
        before_run=dict(vm_stat=subprocess.check_output(['vm_stat'],text=True),
                        swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True)),
        thinking=False,speculative_decoding=False,top_k=1,temperature=0,seed=42,
        timeout_seconds=args.timeout,max_process_rss_gib=args.max_rss_gib,
        limitations=['Phase snapshots and vmmap interrupt execution; timing is not a performance comparison.',
            'libproc RSS and physical footprint are separate overlapping process accounting views, not GPU allocation.',
            'Snapshots after generation include both prefill and decode; no isolated prefill completion event.',
            'Prebuilt package; not a local build of frozen release source.',
            'Fresh conversations share one engine; later repetitions reuse prepared model state.',
            'Notes are built and tokenized before generation timing; raw tokenizer count excludes chat formatting.',
            'Marker retrieval is a single synthetic task check, not a general quality evaluation.',
            'Raw stream-final and runtime decode counts verify multi-token completion.',
            'Client timestamps are taken inside the native callback before queue insertion; callback count is not a token count.',
            'Runtime benchmark fields retained verbatim; not yet a sustained-performance or quality benchmark.',
            'Sampled process RSS and system pressure are stop guards and can miss short-lived peaks.'])
    save(args.output/'report.json',report)
    started=time.monotonic();samples=[];stop_reason=None
    with (args.output/'stdout.log').open('wb') as stdout,(args.output/'stderr.log').open('wb') as stderr:
        proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            while proc.poll() is None:
                try:
                    rss=subprocess.run(['ps','-o','rss=','-p',str(proc.pid)],
                                       capture_output=True,text=True,timeout=2)
                    pressure=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],
                                            capture_output=True,text=True,timeout=2,check=True)
                    rss_kib=int(rss.stdout.strip()) if rss.stdout.strip() else None
                    pressure_level=int(pressure.stdout.strip())
                    if rss.returncode and proc.poll() is None:raise ValueError('Cannot read live worker RSS')
                except (subprocess.SubprocessError,ValueError) as exc:
                    stop_reason='resource_monitor_error'
                    samples.append(dict(elapsed_seconds=time.monotonic()-started,error=str(exc)))
                    stop(proc)
                    break
                elapsed=time.monotonic()-started
                sample=dict(elapsed_seconds=elapsed,rss_kib=rss_kib,pressure=pressure_level)
                try:sample.update(process_memory(proc.pid))
                except OSError as exc:
                    if proc.poll() is None:
                        stop_reason='resource_monitor_error'
                        sample['rusage_error']=str(exc)
                        samples.append(sample)
                        stop(proc)
                        break
                    sample['rusage_exit_race']=str(exc)
                samples.append(sample)
                if elapsed>args.timeout:stop_reason='timeout'
                elif sample['pressure'] is not None and sample['pressure']&4:stop_reason='critical_system_memory_pressure'
                elif sample['rss_kib'] is not None and sample['rss_kib']*1024>args.max_rss_gib*2**30:
                    stop_reason='process_rss_limit'
                if stop_reason:
                    stop(proc)
                    break
                time.sleep(.25)
        finally:
            if proc.poll() is None:stop(proc)
    save(args.output/'resource-guard.json',samples)
    try:result=json.loads((args.output/'worker.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        result=dict(record_error=str(exc))
    status='GENERATION_COMPLETED' if proc.returncode==0 and result.get('phase')=='completed' else 'INCOMPLETE'
    if stop_reason:status='STOPPED_BY_GUARD'
    elif proc.returncode==2 and result.get('error'):status='RUNTIME_ERROR'
    elif any(s.get('capture_error') for s in result.get('memory_stages',[])):
        status='MEMORY_CAPTURE_INCOMPLETE'
    native=(args.output/'stderr.log').read_text(errors='replace')
    backend_errors=[line for line in native.splitlines() if any(marker in line for marker in
                    ['Validation error:', 'Binding entry buffer not set', "unresolved value"])]
    if backend_errors and not stop_reason:status='BACKEND_VALIDATION_ERROR'
    report['backend_errors']=backend_errors
    report.update(returncode=proc.returncode,status=status,stop_reason=stop_reason,worker_result=result,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    report['files']={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='report.json'}
    save(args.output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if status=='GENERATION_COMPLETED' else 2


if __name__=='__main__':raise SystemExit(main())
