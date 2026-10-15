"""Measure CPU configurations using validation states only; never touch a phone."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gameplay import LayaPolicy
from arlott_adapter import vectorize


def main():
    import torch
    config=json.loads((ROOT/'config.json').read_text())
    policy=LayaPolicy(config)
    data=[json.loads(x) for x in (ROOT/'training/data/validation.jsonl').read_text().splitlines()][::10][:6]
    results={}
    for threads in (1,2,4):
        torch.set_num_threads(threads);times=[]
        for row in data:
            history={'previous_action':row['features']['previous_action'],
                     'previous_action_age':.5 if row['features']['previous_action_age']=='recent' else 2}
            vote=policy.decide(row['raw_state'],history);times.append(vote['latency_ms'])
        results[str(threads)]={'median_ms':statistics.median(times),'samples_ms':times}
        print(json.dumps({'threads':threads,**results[str(threads)]}),flush=True)
    best=min(results,key=lambda k:results[k]['median_ms'])
    neutral={a:1/8 for a in policy.adapter.config['actions']};feature=data[0]['features']
    policy.adapter.predict(feature,neutral);start=time.perf_counter()
    for _ in range(1000):policy.adapter.predict(feature,neutral)
    report={'thread_benchmark':results,'recommended_threads':int(best),'adapter_only_mean_ms':(time.perf_counter()-start),'phone_access':False}
    (ROOT/'training/reports/performance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
