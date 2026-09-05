#!/usr/bin/env python3
"""Rebuild the selected M3 case report and SVG from preserved raw observations."""
import argparse
import json
from pathlib import Path
import statistics
from m3_summarize import summarize, describe


def window(rows):
    return {'requests':len(rows),'tokens':sum(r['decode_tokens'] for r in rows),
            'decode_s':sum(r['decode_api_ms'] for r in rows)/1000,
            'decode_tokens_s':sum(r['decode_tokens'] for r in rows)/(sum(r['decode_api_ms'] for r in rows)/1000)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index',type=Path);p.add_argument('--plot',action='store_true')
    a=p.parse_args();index=json.loads(a.index.read_text());base=a.index.parent
    runs={k:summarize(base/v) for k,v in index['runs'].items()}
    if not all(r['valid_complete_run'] for r in runs.values()):
        raise ValueError('Selected report run did not finish successfully; report the failure separately.')
    report={'runs':{},'index':index}
    for label,s in runs.items():
        rows=s['requests']; warm=rows[1:]
        report['runs'][label]={'load_ms':s['load_ms'],'process_to_first_text_ms':s['process_to_first_text_ms'],
            'first_request':rows[0],'requests':len(rows),'loaded_first_text_ms':describe([r['first_text_ms'] for r in warm]),
            'loaded_decode_tokens_s':describe([r['decode_wall_tokens_s'] for r in warm]),
            'all_output_hashes':sorted({r['output_sha256'] for r in rows}),
            'input_tokens':sorted({r['input_tokens'] for r in rows}),
            'decode_tokens':sorted({r['decode_tokens'] for r in rows}),
            'text_callback_counts':sorted({r['text_callbacks'] for r in rows})}
    continuous=runs['continuous'];rows=continuous['requests'];t0=rows[0]['start_ns']
    duration=(rows[-1]['start_ns']-t0)/1e9+rows[-1]['request_ms']/1000
    first=[r for r in rows if (r['start_ns']-t0)/1e9<120]
    last=[r for r in rows if (r['start_ns']-t0)/1e9>=duration-120]
    report['continuous']={'request_span_s':duration,'first_120s_by_start':window(first),'last_120s_by_start':window(last),'full_run':window(rows),'stages':continuous['stages'],'boundaries':continuous['boundaries'],'status_read_ms':continuous['status_read_ms']}
    thermal=[r for r in continuous['thermal_samples'] if r['ns'] is not None and t0<=r['ns']<=t0+duration*1e9]
    report['continuous']['thermal_statuses']=sorted({r['status'] for r in thermal if r['status'] is not None})
    for name in ['skin','battery']:
        values=[t['c'] for r in thermal for t in r['temperatures'] if t['name']==name]
        report['continuous'][name+'_c']={'first':values[0] if values else None,'last':values[-1] if values else None,**describe(values)}
    gpu=[max([t['c'] for t in r['temperatures'] if t['type']==1],default=None) for r in thermal]
    gpu=[v for v in gpu if v is not None]
    report['continuous']['gpu_sensor_max_c']={'first':gpu[0] if gpu else None,'last':gpu[-1] if gpu else None,**describe(gpu)}
    (base/'m3-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font=Path('/System/Library/Fonts/STHeiti Light.ttc')
        if font.exists(): font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'litert-lm-m3-2026-09-05','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True,layout='constrained')
        x=[(r['start_ns']-t0)/60e9 for r in rows]
        axes[0].plot(x,[r['decode_wall_tokens_s'] for r in rows],color='#3877b7',marker='.',linewidth=1)
        axes[0].set_ylabel('decode（tokens/s）');axes[0].set_ylim(bottom=0)
        for typ,name,color in [(3,'skin','#a54b30'),(1,'GPU 传感器最大值','#3877b7')]:
            points=[((r['ns']-t0)/60e9,max(t['c'] for t in r['temperatures'] if t['type']==typ)) for r in thermal if any(t['type']==typ for t in r['temperatures'])]
            axes[1].plot([p[0] for p in points],[p[1] for p in points],label=name,color=color)
        axes[1].set_ylabel('温度（°C）');axes[1].legend(loc='upper right',frameon=False)
        axes[2].step([(r['ns']-t0)/60e9 for r in thermal],[r['status'] for r in thermal],where='post',color='#805c9b')
        axes[2].set_ylabel('系统热状态');axes[2].set_yticks([0,1,2,3],['0 无','1 轻度','2 中度','3 严重']);axes[2].set_ylim(-.2,3.3)
        axes[2].set_xlabel('首次请求开始后的时间（min）')
        for ax in axes: ax.grid(axis='y',alpha=.25)
        dest=Path('appendix/figs/m3-continuous.svg');dest.parent.mkdir(exist_ok=True)
        fig.savefig(dest,format='svg',metadata={'Date':None})
        svg=dest.read_text().replace('<svg ', '<svg role="img" aria-label="HONOR MEP-AN00 持续运行中的吞吐、温度和系统热状态" ',1)
        # Standalone fallbacks also permit the book theme to override line colors.
        svg=svg.replace('#3877b7','var(--m3-blue, #3877b7)').replace('#a54b30','var(--m3-red, #a54b30)').replace('#805c9b','var(--m3-purple, #805c9b)')
        dest.write_text(svg)
        print(dest)
    print(base/'m3-report.json')

if __name__=='__main__':main()
