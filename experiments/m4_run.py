#!/usr/bin/env python3
"""Run an isolated device-side M4 vision trial and preserve raw evidence, including failures."""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from m3_preflight import Recorder, MODEL_SHA256, SOURCE_COMMIT


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def thermal_status(raw):
    m = re.search(r'^Thermal Status: (\d+)', raw, re.M)
    return int(m[1]) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', required=True)
    p.add_argument('--source',type=Path,default=Path(os.environ.get('LITERT_LM_SOURCE','/Users/dpthinker/workspace/LiteRT-LM-v0.13.1')))
    p.add_argument('--backend', choices=['cpu','gpu'], default='cpu')
    p.add_argument('--timeout', type=int, default=600, help='Host deadline in seconds, including engine loading')
    p.add_argument('--sample-ms', type=int, default=500)
    p.add_argument('--label', required=True)
    p.add_argument('--environment-report',type=Path)
    p.add_argument('--bundle', type=Path, default=Path('tmp/m4-build/bundle'))
    p.add_argument('--fixture', type=Path, default=Path('experiments/fixtures/m4'))
    p.add_argument('--vision-backend',choices=['cpu','gpu'],default='gpu')
    p.add_argument('--pilot',action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', args.label):
        p.error('label must use lowercase letters, digits and hyphens')
    stamp = dt.datetime.now(dt.timezone.utc)
    out = Path('experiments/data') / stamp.strftime('%Y-%m-%d') / ('m4-'+stamp.strftime('%H%M%S-%fZ')+'-'+args.label)
    out.mkdir(parents=True, exist_ok=False)
    print(out, flush=True)
    rec = Recorder(out)
    adb = ['adb', '-s', args.serial]
    remote = '/data/local/tmp/litertlm/' + out.name
    def shell(name, command, timeout=30):
        return rec.run(name, adb+['shell', command], timeout)
    def require(result):
        if result['returncode'] != 0:
            raise RuntimeError(result)
        return result['stdout']
    manifest = {'started_utc':stamp.isoformat(), 'parameters':vars(args).copy(),
                'source_commit':SOURCE_COMMIT, 'model_expected_sha256':MODEL_SHA256,
                'remote_dir':remote, 'clock':'device CLOCK_MONOTONIC nanoseconds',
                'environment_report':None,
                'max_context_tokens':4096, 'max_output_tokens':256,
                'sampler':{'type':'top_p','top_k':1,'top_p':1.0,'temperature':1.0,'seed':42}, 'mtp':False,
                'vision_backend':args.vision_backend, 'cpu_threads':4, 'cache_dir':'/data/local/tmp/litertlm/m4-cache',
                'process_exit':None, 'stop_reason':None}
    manifest['parameters'] = {k:str(v) if isinstance(v,Path) else v for k,v in manifest['parameters'].items()}
    proc = None
    try:
        if args.environment_report:
            manifest['environment_report']=json.loads(args.environment_report.read_text())
            (out/'environment.json').write_bytes(args.environment_report.read_bytes())
        source_dir=args.source
        observed=require(rec.run('source_commit',['git','-C',str(source_dir),'rev-parse','HEAD'])).strip()
        dirty=require(rec.run('source_status',['git','-C',str(source_dir),'status','--porcelain'])).strip()
        if observed!=SOURCE_COMMIT or dirty: raise RuntimeError('source identity mismatch')
        model=require(shell('model_sha256','sha256sum /data/local/tmp/litertlm/model.litertlm',180)).split()[0]
        if model!=MODEL_SHA256: raise RuntimeError('model identity mismatch')
        manifest['model_observed_sha256']=model
        require(shell('create_run_dir', 'mkdir '+shlex.quote(remote)))
        require(shell('create_cache_dir', 'mkdir -p /data/local/tmp/litertlm/m4-cache'))
        require(rec.run('push_bundle', adb+['push', str(args.bundle)+'/.' ,remote],120))
        cases=[('budget70-1',70,'shapes.png')] if args.pilot else [
            ('budget70-1',70,'shapes.png'),('budget280-1',280,'shapes.png'),
            ('budget280-2',280,'shapes.png'),('budget70-2',70,'shapes.png'),
            ('budget70-3',70,'shapes.png'),('budget280-3',280,'shapes.png'),
            ('invalid-image',70,'invalid.png'),('invalid-budget',0,'shapes.png'),
            ('recovery',70,'shapes.png')]
        fixture_dir=out/'inputs'; fixture_dir.mkdir()
        for f in args.fixture.iterdir():
            if f.is_file(): (fixture_dir/f.name).write_bytes(f.read_bytes())
        prompt='Name the three colored shapes from left to right. For each shape give its color and shape. Answer in one short sentence.'
        lines=[]
        for id,budget,filename in cases:
            message={'role':'user','content':[{'type':'image','path':remote+'/'+filename},{'type':'text','text':prompt}]}
            (fixture_dir/(id+'.json')).write_text(json.dumps(message)+'\n')
            lines.append(id+'\t'+str(budget)+'\t'+remote+'/'+id+'.json')
        (fixture_dir/'cases.tsv').write_text('\n'.join(lines)+'\n')
        require(rec.run('push_inputs',adb+['push',str(fixture_dir)+'/.',remote]))
        manifest['cases']=cases
        manifest['bundle_sha256']={f.name:digest(f) for f in sorted(args.bundle.iterdir()) if f.is_file()}
        manifest['observer_source_sha256']=digest(Path('experiments/m4_observe.cc'))
        manifest['inputs_sha256']={f.name:digest(f) for f in fixture_dir.iterdir() if f.is_file()}
        for name,command in [('device_hashes','sha256sum '+remote+'/*'),
                             ('properties','getprop ro.build.fingerprint; getprop ro.product.model; getprop ro.soc.model'),
                             ('system_memory','cat /proc/meminfo'),('battery_before','dumpsys battery'),
                             ('power_before','dumpsys power'),('thermal_before','dumpsys thermalservice'),
                             ('low_power','settings get global low_power'),
                             ('graphics_driver',"dumpsys SurfaceFlinger | grep -E 'GLES:|GL_RENDERER|GL_VERSION'")]:
            result=shell(name,command)
            if name=='device_hashes':
                observed_hashes={Path(line.split()[-1]).name:line.split()[0] for line in require(result).splitlines()}
                for library,expected in manifest['bundle_sha256'].items():
                    if observed_hashes.get(library)!=expected: raise RuntimeError('device binary hash mismatch: '+library)
            if name=='thermal_before':
                status=thermal_status(result['stdout'])
                if result['returncode'] or status is None or status>=3:
                    raise RuntimeError('thermal status unavailable or >= 3')
        clockcmd='LD_PRELOAD='+shlex.quote(remote+'/libLiteRt.so')+' LD_LIBRARY_PATH='+shlex.quote(remote)+' '+shlex.quote(remote+'/m4_observe')+' --clock'
        command='cd '+shlex.quote(remote)+' && LD_PRELOAD='+shlex.quote(remote+'/libLiteRt.so')+' LD_LIBRARY_PATH='+shlex.quote(remote)+' ./m4_observe /data/local/tmp/litertlm/model.litertlm '+shlex.join([args.backend,args.vision_backend,remote+'/cases.tsv',remote+'/events.jsonl',str(args.sample_ms)])
        manifest['command']=command
        with (out/'stdout.txt').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
            proc=subprocess.Popen(adb+['shell',command],stdout=stdout,stderr=stderr)
            start=time.monotonic()
            stopped=False
            while proc.poll() is None:
                before=time.monotonic_ns()
                result=shell('resource_sample',clockcmd+'; dumpsys thermalservice; dumpsys battery',20)
                record={'host_start_ns':before,'host_end_ns':time.monotonic_ns(),**result}
                with (out/'resources.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
                status=thermal_status(result['stdout'])
                reason = ('resource_capture_failed' if result['returncode'] or status is None else
                          'thermal_status_ge_3' if status>=3 else
                          'host_deadline' if time.monotonic()-start>args.timeout else None)
                if reason and not stopped:
                    manifest['stop_reason']=reason
                    # Signal only the PID identified by this run's own event log.
                    first=shell('read_owned_pid','head -n 1 '+shlex.quote(remote+'/events.jsonl'))
                    try:
                        event=json.loads(first['stdout'])
                        pid=int(event['pid'])
                        if event['event']!='process_start' or pid<=1: raise ValueError('bad pid')
                        require(shell('stop_owned_process','kill -TERM '+str(pid)))
                        stopped=True
                    except Exception:
                        manifest['stop_reason']+='; device_stop_unconfirmed'
                        proc.terminate()
                        break
                if stopped and time.monotonic()-start>args.timeout+90:
                    raise RuntimeError('device did not stop after cancellation')
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
            manifest['process_exit']=proc.wait(timeout=30)
        require(rec.run('pull_events',adb+['pull',remote+'/events.jsonl',str(out/'events.jsonl')],60))
        for name,command in [('battery_after','dumpsys battery'),('thermal_after','dumpsys thermalservice')]:
            shell(name,command)
        manifest['stop_reason']=manifest['stop_reason'] or ('completed' if manifest['process_exit']==0 else 'client_error')
    except Exception as error:
        if proc is not None and proc.poll() is None:
            try:
                owned=json.loads(shell('read_owned_pid_on_error','head -n 1 '+shlex.quote(remote+'/events.jsonl'))['stdout'])
                pid=int(owned['pid'])
                if owned['event']!='process_start' or pid<=1: raise ValueError('invalid owned pid')
                shell('stop_owned_on_error','kill -TERM '+str(pid))
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    shell('kill_owned_after_grace','kill -KILL '+str(pid))
                    proc.wait(timeout=10)
                manifest['process_exit']=proc.returncode
            except Exception as stop_error:
                manifest['device_stop_error']=str(stop_error)
                proc.terminate()
        manifest['exception']=str(error)
        manifest['stop_reason']=manifest['stop_reason'] or 'setup_or_capture_failed'
    finally:
        manifest['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        manifest['raw_sha256']={str(f.relative_to(out)):digest(f) for f in sorted(out.rglob('*')) if f.is_file()}
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'directory':str(out),'exit':manifest['process_exit'],'reason':manifest['stop_reason']},ensure_ascii=False),flush=True)
    return 0 if manifest['process_exit']==0 and manifest['stop_reason']=='completed' else 2

if __name__=='__main__':
    raise SystemExit(main())
