"""Versioned, auditable video imitation labels. No generated game-rule targets.

Cooldown and joystick labels are observations with imperfect CV; the few attack
annotations below were visually reviewed by the assistant, not by the user.
"""
import bisect
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arlott_schema import QUESTION, model_state, state_features

DATA = ROOT / 'training/data'

# Specific pre-action timestamps inspected in the contact sheets. Confidence is
# intentionally below 1: attacks and spell animation can be ambiguous in a video.
REVIEWED_ATTACKS = [58,58.5,59.5,60,90,90.5,91,145,145.5,197,197.5,
                    238,238.5,311,311.5,467,467.5,607,716,716.5,717]
REVIEWED_ROTATIONS = [(220,222),(256,258),(352,354),(393,395),(510,512),
                      (558,559.5),(580,582),(649,650),(742,744)]


def read(name):
    return [json.loads(line) for line in (DATA/name).read_text(encoding='utf-8').splitlines()]


def split(t):
    # Ten-second gaps prevent an event and adjacent frames straddling a split.
    return 'train' if t < 500 else 'validation' if 510 <= t < 620 else 'test' if 630 <= t < 787 else None


def main():
    observations = read('observations.jsonl')
    times = [x['time'] for x in observations]
    skills = [x for x in read('candidate_events.jsonl') if x['evidence']=='observed_ready_to_cooldown']
    candidates = []
    for e in skills:
        candidates.append(dict(time=e['time'], label=e['label'], confidence=.80,
            evidence=e['evidence'], label_post_time=e['post_time'], review_post_frame=e['source_frame']))

    def close_cast(t, radius=.7):
        return any(abs(e['time']-t) < radius for e in skills)

    last_move = -99
    for o in observations:
        t, s, thumb = o['time'], o['state'], o.get('thumb')
        if t<35 or not s['in_battle'] or not thumb or math.hypot(*thumb)<.45 or t-last_move<1.8 or close_cast(t):
            continue
        label='reposition'
        if s.get('target'):
            dx=s['target'][0]-.5; dy=(s['target'][1]-.53)*.45
            dot=(dx*thumb[0]+dy*thumb[1])/max(.001,math.hypot(dx,dy)*math.hypot(*thumb))
            if dot>.55: label='approach'
            elif dot<-.55 and s.get('health',1)<.6: label='retreat'
        candidates.append(dict(time=t,label=label,confidence=.45,evidence='joystick_direction_weak'))
        last_move=t

    for t in REVIEWED_ATTACKS:
        if not close_cast(t,.35):
            candidates.append(dict(time=t,label='attack',confidence=.70,evidence='assistant_visual_review'))
    for a,b in REVIEWED_ROTATIONS:
        for t in (a,a+.75,b):
            if not close_cast(t):
                candidates.append(dict(time=t,label='reposition',confidence=.75,evidence='assistant_visual_review'))
    last_hold=-99
    for o in observations:
        if o['time']>35 and not o['state']['in_battle'] and o['time']-last_hold>7:
            candidates.append(dict(time=o['time'],label='hold',confidence=.7,evidence='hud_absent_no_control'))
            last_hold=o['time']

    rows={name:[] for name in ('train','validation','test')}
    for e in sorted(candidates,key=lambda x:(x['time'], -x['confidence'])):
        part=split(e['time'])
        if not part: continue
        idx=max(0,bisect.bisect_right(times,e['time'])-1)
        o=observations[idx]
        # Past casts only. Post-transition frames and joystick are never features.
        prior=[x for x in skills if x['post_time'] < o['time']]
        history={'previous_action':prior[-1]['label'],'previous_action_age':o['time']-prior[-1]['post_time']} if prior else None
        features=state_features(o['state'],history)
        row={'id':f"{part}-{len(rows[part]):04d}", 'time':o['time'],
             'state':model_state(features),'features':features,'questions':QUESTION,
             'expected':{'action':e['label']}, 'weight':e['confidence'],
             'provenance':{k:v for k,v in e.items() if k not in ('label','confidence')},
             'raw_state':o['state']}
        if rows[part] and abs(rows[part][-1]['time']-row['time']) < .2:
            if rows[part][-1]['weight'] >= row['weight']: continue
            rows[part].pop()
        rows[part].append(row)

    report={'source':json.loads((DATA/'source/metadata.json').read_text()),
            'schema':'arlott-v1', 'split_policy':'train <500s; validation [510,620); test [630,787); 10s gaps',
            'label_warning':'CV cooldown and joystick labels are noisy; visual review is assistant review. This is one edited match, not independent matches.',
            'features_exclude':['joystick','future frames','timestamps','post-action cooldown state','label evidence'],
            'splits':{}}
    for part,data in rows.items():
        payload=''.join(json.dumps(x)+'\n' for x in data)
        (DATA/f'{part}.jsonl').write_text(payload,encoding='utf-8')
        report['splits'][part]={'rows':len(data),'labels':dict(Counter(x['expected']['action'] for x in data)),
                               'evidence':dict(Counter(x['provenance']['evidence'] for x in data)),
                               'sha256':hashlib.sha256(payload.encode()).hexdigest()}
    train_inputs={r['state'] for r in rows['train']}
    for part in ('validation','test'):
        report['splits'][part]['identical_state_strings_in_train']=sum(r['state'] in train_inputs for r in rows[part])
        assert not {r['time'] for r in rows['train']} & {r['time'] for r in rows[part]}
    (DATA/'dataset_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
