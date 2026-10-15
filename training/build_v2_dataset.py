"""Add explicit reviewed labels without overwriting the frozen v1 benchmark."""
import bisect
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'training/data';OUT=BASE/'v2'


def part(t):
    return 'train' if t<500 else 'validation' if 510<=t<620 else 'test' if 630<=t<787 else None


def main():
    annotations=json.loads((ROOT/'training/annotations_v2.json').read_text())
    observations=[json.loads(x) for x in (OUT/'observations.jsonl').read_text().splitlines()]
    times=[x['time'] for x in observations];sets={p:{} for p in ('train','validation','test')}
    corrections=[]
    for p in sets:
        original=[json.loads(x) for x in (BASE/f'{p}.jsonl').read_text().splitlines()]
        for r in original:
            label=r['expected']['action'];i=max(0,bisect.bisect_right(times,r['time'])-1)
            if label in ('attack','retreat') and p!='test':continue
            if label=='hold' and observations[i]['state'].get('health') and p!='test':continue
            if r['time']>=759.5:
                corrections.append({'id':r['id'],'time':r['time'],'old_label':label,'audited_label':'hold','reason':'dead spectating'})
                label='hold'
            sets[p][i]={'index':i,'time':times[i],'label':label,'weight':r['weight'],
                         'evidence':r['provenance']['evidence'],'episode':f'baseline-{int(r["time"]//10)}','baseline_id':r['id']}
    for a in annotations['intervals']:
        indices=[i for i,t in enumerate(times) if a['start']<=t<=a['end'] and part(t)]
        # Downsample long dead/overlay intervals; do not swamp combat labels.
        if a['label']=='hold':indices=indices[::4]
        for i in indices:
            p=part(times[i])
            if any(abs(e['time']-times[i])<.30 and e['label'] in ('skill1','skill2','ultimate') for e in sets[p].values()) and a['label']!='hold':continue
            sets[p][i]={'index':i,'time':times[i],'label':a['label'],'weight':a['confidence'],
                        'evidence':'explicit_assistant_interval','episode':a['episode']}
    report={'source_sha256':json.loads((BASE/'source/metadata.json').read_text())['sha256'],
            'annotation_sha256':hashlib.sha256((ROOT/'training/annotations_v2.json').read_bytes()).hexdigest(),
            'warnings':['One edited match only. Dense labels are correlated frames, not new independent recordings.',
                        'Explicit labels received assistant visual review, not human expert confirmation.',
                        'Frozen 94-example/24-cast benchmark retained unchanged and scored separately.',
                        'Retreat label at 766.293 is actually death. A safe policy cannot score 0.80 retreat F1 on that single invalid example.'],
            'corrections':corrections,'splits':{}}
    for p,by_index in sets.items():
        rows=sorted(by_index.values(),key=lambda r:r['time']);(OUT/f'{p}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report['splits'][p]={'rows':len(rows),'labels':dict(Counter(r['label'] for r in rows)),
                            'explicit_episodes':{label:len({r['episode'] for r in rows if r['label']==label and r['evidence']=='explicit_assistant_interval'}) for label in ('attack','retreat')}}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
