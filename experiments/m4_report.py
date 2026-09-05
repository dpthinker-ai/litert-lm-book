#!/usr/bin/env python3
"""Verify selected M4 records and derive effective visual positions indirectly."""
import argparse,hashlib,json
from pathlib import Path
from m4_summarize import summarize
from m3_summarize import describe


def report(index):
    root=index.parent; selected=json.loads(index.read_text())['E05']
    pilot=summarize(root/selected['pilot']); series=summarize(root/selected['series'])
    if not pilot['valid'] or not series['valid']:raise ValueError('incomplete trial')
    audit=root/selected['token_audit'];manifest=json.loads((audit/'manifest.json').read_text())
    for name,sha in manifest['raw_sha256'].items():
        if hashlib.sha256((audit/name).read_bytes()).hexdigest()!=sha:raise ValueError('audit hash mismatch')
    run=json.loads((audit/'result.json').read_text())
    if run['returncode']!=0:raise ValueError('audit failed')
    counts=json.loads(next(line for line in run['stdout'].splitlines() if line.startswith('{')))
    nonvisual=sum(counts['counts'])+counts['bos_positions']+counts['image_end_positions']
    if nonvisual!=counts['nonvisual_positions']:raise ValueError('token sum mismatch')
    shapes=json.loads((root/selected['model_shapes']).read_text())
    budget_rows=[]
    for budget in [70,280]:
        requests=[r for r in series['requests'] if r['case'].startswith('budget'+str(budget)+'-')]
        later=[r for r in requests if r['request']!=0]
        visual={r['prefill_tokens']-nonvisual for r in requests}
        if len(visual)!=1:raise ValueError('inconsistent visual count')
        effective=visual.pop()
        resized=[r for r in series['resize_log'] if r['patch_limit']==budget*9]
        if not resized or any(r['patches']!=effective*9 for r in resized):raise ValueError('patch crosscheck failed')
        signature=next(s for s in shapes['signatures'] if s['name']=='vision_'+str(budget))
        if not any(x['signature_name']=='mask' for x in signature['outputs']):raise ValueError('unexpected model output')
        budget_rows.append({'budget':budget,'requests':len(requests),'later_requests':len(later),'effective_visual_positions_indirect':effective,
                            'combined_prefill_tokens':requests[0]['prefill_tokens'],'resized':resized[0],
                            'later_first_text_ms':describe([r['request_to_first_text_ms'] for r in later]),
                            'later_prefill_ms':describe([r['prefill_ms'] for r in later]),
                            'later_rss_periodic_max_kib':describe([r['rss_periodic_max_kib'] for r in later]),
                            'all_fixture_phrase_checks_pass':all(r['fixture_phrase_check'] for r in requests)})
    result={'index':str(index),'E04':'not_completed','E05':'completed','load_ms':series['load_ms'],'nonvisual_position_audit':counts,
            'budget_rows':budget_rows,'successful_series_requests':sum(not r['expected_error'] for r in series['requests']),
            'expected_error_cases':sum(r['expected_error'] for r in series['requests']),
            'first_request':series['requests'][0],'recovery_request':series['requests'][-1],
            'boundaries':series['boundaries'],'thermal':series['thermal'],
            'visual_count_method':'Indirect: observed combined prefill counts minus separately tokenized text plus BOS and ImageEnd; raw encoder mask not captured.'}
    (root/'m4-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path);a=p.parse_args()
    r=report(a.index); print(json.dumps({k:r[k] for k in ['E04','E05','budget_rows']},ensure_ascii=False,indent=2))
