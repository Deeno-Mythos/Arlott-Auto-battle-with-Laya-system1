"""Measure runtime and quantization error on validation data, with one CPU thread."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arlott_rt import ArlottRTPolicy


def main():
    data=np.load(ROOT/'training/data/v2/export_validation.npz',allow_pickle=False)
    x=data['inputs'];gold=data['expected_logits'];models={};outputs={}
    for backend in ('fp32','int8'):
        policy=ArlottRTPolicy(ROOT/'models/arlott-rt-v2',backend)
        for row in x[:20]:policy.session.run(None,{'context':row.reshape(1,-1)})
        samples=[];z=[]
        for i in range(2000):
            row=x[i%len(x)].reshape(1,-1);start=time.perf_counter_ns()
            result=policy.session.run(None,{'context':row})[0][0]
            samples.append((time.perf_counter_ns()-start)/1e6)
            if i<len(x):z.append(result)
        logits=np.stack(z);outputs[backend]=logits
        models[backend]={'p50_ms':float(np.percentile(samples,50)),'p95_ms':float(np.percentile(samples,95)),
                         'p99_ms':float(np.percentile(samples,99)),
                         'max_logit_error_vs_pytorch':float(abs(logits-gold).max()),
                         'argmax_agreement_vs_pytorch':float((logits.argmax(1)==gold.argmax(1)).mean()),
                         'bytes':(ROOT/f'models/arlott-rt-v2/policy-{backend}.onnx').stat().st_size}
    # Prefer fp32 unless quantization is faster AND preserves validation decisions.
    selected='int8' if models['int8']['p95_ms']<models['fp32']['p95_ms'] and models['int8']['argmax_agreement_vs_pytorch']>=.99 else 'fp32'
    report={'samples_per_backend':2000,'models':models,'selected':selected,'scope':'Model session only; excludes perception, capture, USB and phone touch registration.'}
    (ROOT/'training/reports/v2/onnx_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
