"""Reload saved Laya weights and replay held-out states with no Android access."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gameplay import LayaPolicy


def main():
    config=json.loads((ROOT/'config.json').read_text())
    config.update(model='models/arlott-video-v1',policy_schema='arlott-v1')
    policy=LayaPolicy(config)
    rows=[json.loads(x) for x in (ROOT/'training/data/test.jsonl').read_text().splitlines()]
    expected={r['id']:r for r in [json.loads(x) for x in (ROOT/'training/reports/test_predictions.jsonl').read_text().splitlines()]}
    results=[];adapter_parity=[];adapter_latencies=[]
    for row in rows:
        history={'previous_action':row['features']['previous_action'],
                 'previous_action_age':.5 if row['features']['previous_action_age']=='recent' else 2.0}
        vote=policy.decide(row['raw_state'],history)
        rapid=policy.refresh(row['raw_state'],vote,history)
        adapter_parity.append(rapid['action']==vote['action'])
        adapter_latencies.append(rapid.get('adapter_latency_ms',0))
        results.append({'id':row['id'],'time':row['time'],'expected':row['expected']['action'],**vote,
                        'matches_training_export':vote['action']==expected[row['id']]['trained'],
                        'confidence_error':abs(vote['confidence']-expected[row['id']]['confidence'])})
        if len(results)%20==0: print(json.dumps({'replayed':len(results),'total':len(rows)}),flush=True)
    report={'model':config['model'],'phone_access':False,'rows':len(results),
            'reload_prediction_agreement':sum(r['matches_training_export'] for r in results)/len(results),
            'accuracy':sum(r['action']==r['expected'] for r in results)/len(results),
            'confidence_max_error':max(r['confidence_error'] for r in results),
            'fast_refresh_same_state_agreement':sum(adapter_parity)/len(adapter_parity),
            'fast_refresh_median_ms':statistics.median(adapter_latencies),
            'latency_median_ms':statistics.median(r['latency_ms'] for r in results[1:]),
            'latency_p95_ms':sorted(r['latency_ms'] for r in results[1:])[int((len(results)-1)*.95)],
            'weights_sha256':hashlib.file_digest((ROOT/config['model']/'model.safetensors').open('rb'),'sha256').hexdigest()}
    (ROOT/'training/reports/reload_verification.json').write_text(json.dumps(report,indent=2))
    (ROOT/'training/reports/replay.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    if report['reload_prediction_agreement']<.99: raise RuntimeError(f'Reload disagreement: {report}')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
