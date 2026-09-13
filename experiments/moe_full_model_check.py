#!/usr/bin/env python3
"""Bounded full-model generation probe for the pinned LiteRT-LM release on macOS.

Hash the complete local model before launching a fresh worker. Preserve the
creation/generation phase and native logs, including failures before inference.
Resource samples are stop guards, not a GPU memory or latency benchmark.
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


def worker(args):
    import litert_lm as lm
    lm.set_min_log_severity(lm.LogSeverity.INFO)
    result=dict(phase='engine_creation',engine_created=False,conversation_created=False,
                generation_completed=False,prompt=args.prompt)
    save(args.output/'worker.json',result)
    try:
        with lm.Engine(str(args.model),backend=lm.Backend.GPU(),max_num_tokens=args.context,
                       cache_dir=str(args.cache),enable_speculative_decoding=False) as engine:
            result.update(engine_created=True,phase='conversation_creation')
            save(args.output/'worker.json',result)
            with engine.create_conversation(
                thinking_config=lm.ThinkingConfig(enable_thinking=False,thinking_token_budget=0),
                sampler_config=lm.SamplerConfig(top_k=1,temperature=0,seed=42),
                max_output_tokens=args.output_tokens) as conversation:
                result.update(conversation_created=True,phase='generation')
                save(args.output/'worker.json',result)
                response=conversation.send_message(args.prompt)
                result.update(generation_completed=True,response=response,phase='cleanup')
                save(args.output/'worker.json',result)
        result['phase']='completed'
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
    save(args.output/'worker.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['phase']=='completed' else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--context',type=int,default=1024)
    ap.add_argument('--output-tokens',type=int,default=32)
    ap.add_argument('--prompt',default='Reply with only the word OK.')
    ap.add_argument('--timeout',type=float,default=180)
    ap.add_argument('--max-rss-gib',type=float,default=18)
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    args.model=args.model.resolve();args.output=args.output.resolve();args.cache=args.cache.resolve()
    if args.worker:return worker(args)
    if sys.platform!='darwin':ap.error('This collector uses macOS memory-pressure guards')
    if min(args.context,args.output_tokens,args.timeout,args.max_rss_gib)<=0:ap.error('Limits must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    args.cache.mkdir(parents=True,exist_ok=False)
    digest=sha(args.model)
    if digest!=args.sha256:raise ValueError('Full model hash mismatch; engine not started')
    import litert_lm
    library=Path(litert_lm.__file__).parent/'liblitert-lm.dylib'
    assert importlib.metadata.version('litert-lm-api')=='0.17.0'
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--model',str(args.model),
             '--sha256',digest,'--output',str(args.output),'--cache',str(args.cache),
             '--context',str(args.context),'--output-tokens',str(args.output_tokens),'--prompt',args.prompt]
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in
                                            ['litert-lm-api','litert-lm','litert-lm-builder']},
        model=dict(path=str(args.model),bytes=args.model.stat().st_size,sha256=digest),
        library_sha256=sha(library),script_sha256=sha(__file__),worker_command=command,
        requested_backend='GPU',context_capacity=args.context,max_output_tokens=args.output_tokens,
        thinking=False,speculative_decoding=False,top_k=1,temperature=0,seed=42,
        timeout_seconds=args.timeout,max_process_rss_gib=args.max_rss_gib,
        limitations=['Prebuilt package; not a local build of frozen release source.',
            'At most one short generation; not a quality, throughput, energy or peak GPU memory benchmark.',
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
    report.update(returncode=proc.returncode,status=status,stop_reason=stop_reason,worker_result=result,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    report['files']={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='report.json'}
    save(args.output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if status=='GENERATION_COMPLETED' else 2


if __name__=='__main__':raise SystemExit(main())
