"""Extract causal features at original observation times, preserving v1 files."""
import json
import sys
import time
from pathlib import Path
import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arlott_rt import RTObserver, CausalContext
from training.extract_demonstrations import normalize


def main():
    cv2.setNumThreads(1);out=ROOT/'training/data/v2';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'training/data/video_config.json').read_text())
    config.update(dynamic_hp=True,bar_border=.30)
    (out/'perception.json').write_text(json.dumps(config,indent=2))
    source=json.loads((ROOT/'training/data/source/metadata.json').read_text())
    original=[json.loads(x) for x in (ROOT/'training/data/observations.jsonl').read_text().splitlines()]
    targets=[x['time'] for x in original];index=0;observer=RTObserver(config);contexts=CausalContext()
    x=[];rows=[];start=time.monotonic()
    with av.open(source['path']) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        for frame in container.decode(stream):
            if index>=len(targets):break
            if frame.time is None or frame.time+0.0006<targets[index]:continue
            image=normalize(frame.to_ndarray(format='bgr24'))
            state,features=observer.read(image,targets[index]);context=contexts.push(targets[index],features)
            x.append(context);rows.append({'time':targets[index],'state':state})
            index+=1
            if index%750==0:print(json.dumps({'frames':index,'total':len(targets),'seconds':round(time.monotonic()-start,1)}),flush=True)
    assert index==len(targets)
    np.savez_compressed(out/'contexts.npz',contexts=np.stack(x),times=np.asarray(targets))
    (out/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps({'saved':len(x),'shape':list(np.stack(x).shape),'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
