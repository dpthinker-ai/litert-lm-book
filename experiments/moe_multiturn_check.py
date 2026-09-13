#!/usr/bin/env python3
"""Full MoE multi-turn state tracking with raw stream events and runtime metrics.

Derived from immutable moe_generation_check.py. Each process creates one engine
and three independent six-turn histories; guard conditions are unchanged.
Callback arrival is a client-visible event, not a per-token timestamp.
"""
import argparse
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


def history_fixture(history):
    states=[('ORION','NANJING','AMBER-42','COBALT-73','SUZHOU'),
            ('LYRA','HANGZHOU','JADE-51','RUBY-86','WUXI'),
            ('VEGA','CHENGDU','IVORY-64','TEAL-29','NINGBO')]
    project,city,code,new_code,new_city=states[history]
    expected=dict(project=project,city=city,code=code)
    turns=[]
    for turn in range(6):
        if turn==0:
            update=f'For this conversation, project is {project}, city is {city}, and code is {code}.'
        elif turn==2:
            update=f'Update only the code to {new_code}. Preserve the project and city.'
            expected['code']=new_code
        elif turn==4:
            update=f'Update only the city to {new_city}. Preserve the project and code.'
            expected['city']=new_city
        else:
            update='Keep all project fields unchanged.'
        notes=''.join(f'Note {turn*20+j:03d}: RAM stores temporary data used by active programs.\n'
                      for j in range(20))
        prompt=(update+'\nBackground notes, not project updates:\n'+notes+
                '\nUsing the latest state from this conversation, return only a JSON object with '
                'exactly the keys project, city, code. Do not include markdown or commentary.')
        turns.append(dict(prompt=prompt,expected=expected.copy()))
    return turns


def worker(args):
    import dataclasses
    import queue
    import litert_lm as lm
    from litert_lm._ffi import STREAM_CALLBACK_TYPE
    lm.set_min_log_severity(lm.LogSeverity.INFO)
    began=time.monotonic()
    result=dict(phase='engine_creation',engine_created=False,generation_completed=False,
                scenario='six_turn_project_state',generations=[])
    def checkpoint():save(args.output/'worker.json',result)
    checkpoint()
    try:
        with lm.Engine(str(args.model),backend=lm.Backend.GPU(),max_num_tokens=args.context,
                       cache_dir=str(args.cache),enable_speculative_decoding=False,enable_benchmark=True) as engine:
            result.update(engine_created=True,engine_create_seconds=time.monotonic()-began)
            for index in range(args.repeats):
                result['phase']='conversation_creation';checkpoint()
                with engine.create_conversation(
                    thinking_config=lm.ThinkingConfig(enable_thinking=False,thinking_token_budget=0),
                    sampler_config=lm.SamplerConfig(top_k=1,temperature=0,seed=42),
                    max_output_tokens=args.output_tokens) as conversation:
                    for turn,fixture in enumerate(history_fixture(index)):
                        prompt=fixture['prompt']
                        raw_count=len(engine.tokenize(prompt))
                        before=conversation.token_count
                        if before+raw_count+args.output_tokens+32>args.context:
                            raise ValueError('Insufficient capacity for next turn plus output and formatting reserve')
                        item=dict(index=len(result['generations']),history=index,turn=turn,
                                  prompt=prompt,expected=fixture['expected'],input_raw_token_count=raw_count,
                                  conversation_token_count_before=before,events=[],stream_final=False,stream_error=None)
                        result['generations'].append(item)
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
                        payload=json.dumps(dict(role='user',content=[dict(type='text',text=prompt)]))
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
                        try:
                            item['parsed_answer']=json.loads(item['text'])
                        except json.JSONDecodeError:
                            item['parsed_answer']=None
                        item['state_matches']=item['parsed_answer']==fixture['expected']
                        item['generation_completed']=True;checkpoint()
                for item in result['generations']:
                    if item['history']==index:item['conversation_closed']=True
                checkpoint()
            result.update(generation_completed=True,phase='cleanup');checkpoint()
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
    ap.add_argument('--timeout',type=float,default=180)
    ap.add_argument('--max-rss-gib',type=float,default=18)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    args.model=args.model.resolve();args.output=args.output.resolve();args.cache=args.cache.resolve()
    if args.repeats!=3:ap.error('This fixed fixture requires exactly three histories')
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
             '--context',str(args.context),'--output-tokens',str(args.output_tokens),
             '--repeats',str(args.repeats)]
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in
                                            ['litert-lm-api','litert-lm','litert-lm-builder']},
        model=dict(path=str(args.model),bytes=args.model.stat().st_size,sha256=digest),
        library_sha256=sha(library),script_sha256=sha(__file__),worker_command=command,
        requested_backend='GPU',context_capacity=args.context,max_output_tokens=args.output_tokens,
        repetitions_in_engine=args.repeats,turns_per_history=6,benchmark_enabled=True,
        machine=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                     memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize']))),
        before_run=dict(vm_stat=subprocess.check_output(['vm_stat'],text=True),
                        swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True)),
        thinking=False,speculative_decoding=False,top_k=1,temperature=0,seed=42,
        timeout_seconds=args.timeout,max_process_rss_gib=args.max_rss_gib,
        limitations=['Prebuilt package; not a local build of frozen release source.',
            'Three histories share one engine; six sequential requests within each history reuse one conversation.',
            'JSON state equality is a fixed synthetic task check, separate from generation completion.',
            'A 32-token formatting reserve is checked before each request; actual session step counts are recorded.',
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
