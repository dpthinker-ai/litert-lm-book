#!/usr/bin/env python3
"""Guard a pinned official-demo browser run on a verified local model.

Browser process-tree RSS is a conservative stop guard, not a GPU allocation
measurement: mappings may be counted more than once. Never closes user apps.
"""
import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from moe_full_model_check import sha, save, stop


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--site',type=Path,required=True)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--node',required=True)
    ap.add_argument('--playwright-module',required=True)
    ap.add_argument('--timeout',type=float,default=180)
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.model=args.model.resolve();args.site=args.site.resolve()
    assert sha(args.model)==args.sha256,'Model hash mismatch'
    assets=json.loads(args.assets.read_text())
    for name,entry in assets['files'].items():assert sha(args.site/name)==entry['sha256']
    root=Path(__file__).resolve().parents[1]
    command=[args.node,str(root/'experiments/moe_web_worker.mjs'),str(args.site),str(args.model),str(args.output)]
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                machine=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                             memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize']))),
                model=dict(path=str(args.model),sha256=args.sha256,bytes=args.model.stat().st_size),
                assets_manifest_sha256=sha(args.assets),demo_revision=assets['revision'],command=command,
                script_sha256=sha(__file__),worker_sha256=sha(root/'experiments/moe_web_worker.mjs'),
                helper_sha256=sha(root/'experiments/moe_full_model_check.py'),
                timeout_seconds=args.timeout,browser_tree_rss_guard_bytes=20*2**30,
                limitations=['Pinned demo binary, not a build or test of LiteRT-LM v0.17.0 C++ sources.',
                    'Uses original demo local-file entry with a 128-token context and a short prompt.',
                    'Only localhost requests are permitted; no remote inference or prompt upload.',
                    'Process-tree RSS and global pressure are protective signals, not model GPU memory measurements.',
                    'Other applications are not isolated; one run is not a performance or quality benchmark.'])
    save(args.output/'report.json',report)
    env=dict(os.environ,PLAYWRIGHT_MODULE=args.playwright_module)
    started=time.monotonic();samples=[];reason=None;warn_start=None
    with (args.output/'stdout.log').open('wb') as stdout,(args.output/'stderr.log').open('wb') as stderr:
        proc=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            while proc.poll() is None:
                pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],timeout=2))
                rows=[list(map(int,line.split())) for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True,timeout=2).splitlines()]
                ids={proc.pid}
                while True:
                    new=ids|{pid for pid,parent,rss in rows if parent in ids}
                    if new==ids:break
                    ids=new
                members=[dict(pid=pid,ppid=parent,rss_kib=rss) for pid,parent,rss in rows if pid in ids]
                total=sum(p['rss_kib']*1024 for p in members);elapsed=time.monotonic()-started
                samples.append(dict(elapsed_seconds=elapsed,pressure=pressure,tree_rss_bytes=total,processes=members))
                if pressure&2:warn_start=warn_start if warn_start is not None else elapsed
                else:warn_start=None
                if pressure&4:reason='critical_system_memory_pressure'
                elif warn_start is not None and elapsed-warn_start>5:reason='persistent_warning_memory_pressure'
                elif total>20*2**30:reason='browser_process_tree_rss_limit'
                elif elapsed>args.timeout:reason='timeout'
                if reason:stop(proc);break
                time.sleep(.5)
        except Exception as exc:
            reason='monitor_error';report['monitor_error']=str(exc);stop(proc)
        finally:
            if proc.poll() is None:stop(proc)
    save(args.output/'resource-guard.json',samples)
    worker=json.loads((args.output/'worker.json').read_text()) if (args.output/'worker.json').exists() else {}
    log=(args.output/'console.jsonl').read_text() if (args.output/'console.jsonl').exists() else ''
    page_errors=(args.output/'page-errors.log').read_text() if (args.output/'page-errors.log').exists() else ''
    errors=[json.loads(line) for line in log.splitlines() if json.loads(line)['type']=='error']
    valid=proc.returncode==0 and worker.get('generation_completed') and worker.get('engine_closed') and not errors and not page_errors
    status='GENERATION_COMPLETED' if valid else 'INCOMPLETE'
    if reason:status='STOPPED_BY_GUARD'
    elif worker.get('error') or errors or page_errors:status='RUNTIME_ERROR'
    report.update(status=status,stop_reason=reason,returncode=proc.returncode,worker_result=worker,
                  console_errors=errors,finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  files={p.name:sha(p) for p in args.output.iterdir() if p.name!='report.json'})
    save(args.output/'report.json',report)
    print(json.dumps(dict(status=status,stop_reason=reason,worker=worker),ensure_ascii=False,indent=2))
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main())
