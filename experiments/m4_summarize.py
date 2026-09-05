#!/usr/bin/env python3
"""Recompute Conversation text timing, memory and expected input errors."""
import argparse,csv,hashlib,json,re
from pathlib import Path
from m3_summarize import field,span,describe


def callback_text(raw):
    if not raw:return ''
    message=json.loads(raw)
    content=message.get('content',[])
    if isinstance(content,str):return content
    return ''.join(part.get('text','') for part in content if part.get('type')=='text')


def summarize(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in manifest['raw_sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Evidence hash mismatch: '+name)
    events=sorted([json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()],key=lambda e:e['ns'])
    requests=[]
    for start in [e for e in events if e['event']=='request_start']:
        rid=start['request']; selected=[e for e in events if e['request']==rid]
        end=next((e for e in selected if e['event']=='request_end'),None)
        send=next((e for e in selected if e['event']=='send_start'),None)
        chunks=[(e,callback_text(e['text'])) for e in selected if e['event']=='callback']
        nonempty=[(e,t) for e,t in chunks if t]
        text=''.join(t for e,t in nonempty)
        first=nonempty[0][0]['ns'] if nonempty else None
        row={'case':start['case'],'request':rid,'budget':start['visual_token_budget'],
             'status':end['status'] if end else None,'callback_error':end['error'] if end else 'missing_end',
             'callback_final':end['callback_final'] if end else False,
             'request_to_first_text_ms':(first-start['ns'])/1e6 if first else None,
             'send_to_first_text_ms':(first-send['ns'])/1e6 if first and send else None,
             'prepare_ms':span(selected,'request_start','send_start'),
             'total_request_ms':span(selected,'request_start','request_end'),
             'text_callbacks':len(nonempty),'output':text,'output_sha256':hashlib.sha256(text.encode()).hexdigest()}
        for kind in ['prefill','decode']:
            turns=[e for e in selected if e['event']=='benchmark_turn' and e['kind']==kind]
            row[kind+'_tokens']=sum(e['tokens'] for e in turns) if turns else None
            row[kind+'_ms']=sum(e['tokens']/e['tokens_per_second']*1000 for e in turns) if turns and all(e['tokens_per_second']>0 for e in turns) else None
        values=[field(e['status_raw'],'VmRSS') for e in events if e['event']=='memory_sample' and end and start['ns']<=e['ns']<=end['ns']]
        row['rss_periodic_samples']=len(values)
        row['rss_periodic_max_kib']=max((v for v in values if v is not None),default=None)
        boundary=next((e for e in selected if e['event']=='memory_boundary' and e['boundary']=='after_request'),None)
        for key in ['Rss','Pss','Swap']:
            row['after_'+key+'_kib']=field(boundary['smaps_raw'],key) if boundary else None
        expected_error=start['case'].startswith('invalid-')
        row['expected_error']=expected_error
        row['completed_as_expected']=(end is not None and end['status']!=0 and not nonempty and not end['callback_final']) if expected_error else (end is not None and end['status']==0 and end['error']=='' and end['callback_final'] and bool(text) and row['prefill_tokens'] is not None and row['decode_tokens'] is not None)
        # A narrow fixture check, not a general model quality score. Preserve exact output for human review.
        row['fixture_phrase_check']=bool(re.search(r'red square.*blue circle.*green triangle',text.lower())) if text else None
        requests.append(row)
    boundaries=[{'stage':e['boundary'],'request':e['request'],**{k+'_kib':field(e['smaps_raw'],k) for k in ['Rss','Pss','Swap']}} for e in events if e['event']=='memory_boundary']
    stderr=(directory/'stderr.txt').read_text()
    resize=[dict(zip(['source_width','source_height','width','height','patches','patch_limit'],map(int,m))) for m in re.findall(r'Resize image from (\d+)x(\d+) to (\d+)x(\d+) which will result in (\d+) patches to fit the max_num_patches: (\d+)',stderr)]
    thermal=[]
    for line in (directory/'resources.jsonl').read_text().splitlines():
        raw=json.loads(line)['stdout']; status=re.search(r'^Thermal Status: (\d+)',raw,re.M)
        current=raw.split('Current temperatures from HAL:',1)[-1].split('Current cooling devices',1)[0] if 'Current temperatures from HAL:' in raw else ''
        skin=re.search(r'mValue=([\d.]+), mType=3, mName=skin[,}]',current)
        thermal.append({'status':int(status[1]) if status else None,'skin_c':float(skin[1]) if skin else None})
    hwm=[field(e['status_raw'],'VmHWM') for e in events if e['event']=='memory_sample']
    result={'directory':directory.name,'load_ms':span(events,'load_start','load_end'),'requests':requests,'boundaries':boundaries,'resize_log':resize,'thermal':thermal,
            'lifetime_sampled_vmhwm_kib':max((x for x in hwm if x is not None),default=None),
            'expected_input_error_log':[line for line in stderr.splitlines() if 'Failed to decode image' in line or 'Visual token budget must be positive' in line],
            'status_read_ms':describe([(e['read_end_ns']-e['ns'])/1e6 for e in events if e['event']=='memory_sample']),
            'valid':manifest['process_exit']==0 and manifest['stop_reason']=='completed' and len(requests)==len(manifest['cases']) and all(r['completed_as_expected'] for r in requests)}
    (directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if requests:
        with (directory/'requests.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(requests[0]));w.writeheader();w.writerows(requests)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    s=summarize(a.directory);print(json.dumps(s,ensure_ascii=False,indent=2))
    raise SystemExit(0 if s['valid'] else 2)
