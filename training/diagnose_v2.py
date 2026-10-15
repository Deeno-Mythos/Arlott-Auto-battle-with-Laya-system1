"""Audit the baseline labels and lock its replay before improving the policy."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data';OUT=ROOT/'training/reports/v2'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sets={p:[json.loads(x) for x in (DATA/f'{p}.jsonl').read_text().splitlines()] for p in ('train','validation','test')}
    evaluation=json.loads((ROOT/'training/reports/evaluation.json').read_text())
    audit={}
    for part,rows in sets.items():
        groups=defaultdict(Counter)
        per=defaultdict(list)
        for r in rows:
            label=r['expected']['action'];groups[r['state']][label]+=1;per[label].append(r)
        audit[part]={'labels':dict(Counter(r['expected']['action'] for r in rows)),
                     'identical_state_label_conflicts':sum(sum(c.values()) for c in groups.values() if len(c)>1),
                     'best_possible_accuracy_for_identical_static_features':sum(max(c.values()) for c in groups.values())/len(rows),
                     'by_action':{}}
        for label,data in per.items():
            audit[part]['by_action'][label]={'n':len(data),'target_missing':sum(not x['raw_state'].get('target') for x in data),
               'health_missing_or_zero':sum(not x['raw_state'].get('health') for x in data),
               'evidence':dict(Counter(x['provenance']['evidence'] for x in data)),
               'times':[x['time'] for x in data]}
    locked={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ('train.jsonl','validation.jsonl','test.jsonl','observations.jsonl')}
    lockfile=OUT/'baseline_lock.json'
    if lockfile.exists():
        assert json.loads(lockfile.read_text())==locked, 'Baseline data changed'
    lockfile.write_text(json.dumps(locked,indent=2))
    report={'dataset_audit':audit,'baseline_test':evaluation['trained']['test'],
            'baseline_skill_casts':evaluation['test_cast_events'],
            'latency_design_limits_ms':{'watchdog_poll':50,'combat_interval':120,'blocking_tap_hold':55,'full_laya_p95':453},
            'hardware':'Intel i3-1215U / Intel UHD; no NVIDIA GPU',
            'locked':locked,
            'causes':['Attack/retreat have 14/4 train examples; validation has 1/0.',
                      'Retreat heuristic requires target, joystick opposite target AND health below 0.6; high-health retreats become reposition.',
                      'Instantaneous state cannot observe damage trend, target motion, repeated attacks or skill transitions.',
                      'Cooldown transitions are weak labels, not expert-verified skill casts.',
                      'Action-event labels and movement-state labels have different durations; one-label sampling hides simultaneous movement/attack.',
                      'The 50 ms poll, 120 ms cadence, 55 ms blocking tap and 453 ms transformer exceed requested latency budgets independently.'],
            'measurement_limit':'Phone capture/USB/Android touch registration cannot be certified using a local replay. One test retreat cannot substantiate reliable 0.80 F1.'}
    (OUT/'diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'audit':audit,'baseline_confusion':report['baseline_test']['confusion']},indent=2))


if __name__=='__main__':main()
