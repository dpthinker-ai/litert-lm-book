#!/usr/bin/env python3
"""Recompute M3 measurements from device events. Never replace absent fields with zero."""
import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


def field(raw, name):
    m = re.search(r'^'+re.escape(name)+r':\s+(\d+) kB$', raw, re.M)
    return int(m[1]) if m else None


def span(events, start, end):
    a = next((e['ns'] for e in events if e['event']==start),None)
    b = next((e['ns'] for e in events if e['event']==end),None)
    return (b-a)/1e6 if a is not None and b is not None else None


def describe(values):
    return {'count':len(values), 'min':min(values), 'median':statistics.median(values), 'max':max(values)} if values else {'count':0,'min':None,'median':None,'max':None}


def summarize(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in manifest['raw_sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Raw evidence hash mismatch: '+name)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    result={'directory':directory.name, 'parameters':manifest['parameters'], 'stop_reason':manifest['stop_reason'],
            'process_exit':manifest['process_exit'], 'load_ms':span(events,'load_start','load_end'),
            'process_to_first_text_ms':span(events,'process_start','first_text'),
            'requests':[], 'boundaries':[], 'stages':{}}
    # Log writes from separate threads may arrive out of timestamp order.
    events.sort(key=lambda e:e['ns'])
    for rid in sorted({e['request'] for e in events if e['request']>=0}):
        subset=[e for e in events if e['request']==rid]
        texts=[e for e in subset if e['event']=='callback' and e['text']]
        end=next((e for e in subset if e['event']=='request_end'),None)
        decode=next((e for e in subset if e['event']=='benchmark_turn' and e['kind']=='decode'),None)
        prefill=next((e for e in subset if e['event']=='benchmark_turn' and e['kind']=='prefill'),None)
        row={'request':rid,'start_ns':next((e['ns'] for e in subset if e['event']=='request_start'),None),
             'first_text_ms':span(subset,'request_start','first_text'),
             'input_preparation_ms':span(subset,'request_start','input_prepared'),
             'prefill_api_ms':span(subset,'prefill_api_start','prefill_api_end'),
             'decode_api_ms':span(subset,'decode_api_start','decode_api_end'),
             'request_ms':span(subset,'request_start','request_end'),
             'input_tokens':prefill['tokens'] if prefill else None,
             'decode_tokens':decode['tokens'] if decode else None,
             'decode_benchmark_tokens_s':decode['tokens_per_second'] if decode else None,
             'text_callbacks':len(texts),'output_utf8_bytes':sum(len(e['text'].encode()) for e in texts),
             'output_sha256':hashlib.sha256(''.join(e['text'] for e in texts).encode()).hexdigest(),
             'callback_gap_ms':describe([(b['ns']-a['ns'])/1e6 for a,b in zip(texts,texts[1:])]),
             'error':end['error'] if end else 'missing_request_end'}
        row['decode_wall_tokens_s']=row['decode_tokens']/(row['decode_api_ms']/1000) if row['decode_tokens'] is not None and row['decode_api_ms'] else None
        result['requests'].append(row)
    stage=None
    stage_samples={}
    durations=[]
    for e in events:
        name=e['event']
        if name=='load_start': stage='load'
        elif name=='load_end': stage='loaded'
        elif name=='prefill_api_start': stage='prefill'
        elif name=='prefill_api_end': stage='between_prefill_decode'
        elif name=='decode_api_start': stage='decode'
        elif name=='decode_api_end': stage='after_decode'
        elif name=='engine_release_start': stage='release'
        elif name=='engine_release_end': stage='post_release'
        if name=='memory_sample':
            durations.append((e['read_end_ns']-e['ns'])/1e6)
            stage_samples.setdefault(stage or 'before_load',[]).append(e)
        if name=='memory_boundary':
            result['boundaries'].append({'request':e['request'],'boundary':e['boundary'],
                'ns':e['ns'],'read_ms':(e['read_end_ns']-e['ns'])/1e6,
                **{k+'_kib':field(e['smaps_raw'],k) for k in ['Rss','Pss','Private_Clean','Private_Dirty','Swap','SwapPss']}})
    for stage,samples in stage_samples.items():
        result['stages'][stage]={'samples':len(samples)}
        for key in ['VmRSS','VmHWM','VmSwap']:
            values=[v for e in samples if (v:=field(e['status_raw'],key)) is not None]
            result['stages'][stage][key+'_max_kib']=max(values) if values else None
    result['status_read_ms']=describe(durations)
    result['thermal_samples']=[]
    for line in (directory/'resources.jsonl').read_text().splitlines():
        r=json.loads(line); raw=r['stdout']
        clock=re.match(r'^(\d+)\s*\n',raw)
        thermal=re.search(r'^Thermal Status: (\d+)',raw,re.M)
        # Use current HAL readings only, never the service's cached temperatures.
        current=raw.split('Current temperatures from HAL:',1)[-1].split('Current cooling devices',1)[0] if 'Current temperatures from HAL:' in raw else ''
        temps=[{'c':float(c),'type':int(t),'name':n} for c,t,n in re.findall(r'mValue=([\d.]+), mType=(\d+), mName=([^,}]+)',current)]
        result['thermal_samples'].append({'ns':int(clock[1]) if clock else None,
            'status':int(thermal[1]) if thermal else None,'temperatures':temps})
    result['valid_complete_run']=manifest['process_exit']==0 and manifest['stop_reason']=='completed' and bool(result['requests']) and all(r['error']=='' and r['text_callbacks']>0 and r['decode_tokens'] is not None for r in result['requests']) and events[-1]['event']=='process_end'
    (directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=[{k:v for k,v in r.items() if k!='callback_gap_ms'} for r in result['requests']]
    if rows:
        with (directory/'requests.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path)
    for directory in p.parse_args().directories:
        s=summarize(directory)
        print(json.dumps({'directory':str(directory),'valid':s['valid_complete_run'],'requests':len(s['requests']),'load_ms':s['load_ms']},ensure_ascii=False))
