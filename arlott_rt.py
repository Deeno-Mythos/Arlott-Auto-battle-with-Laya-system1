"""Standalone causal Arlott policy for the per-frame CPU path; no transformer.

Training and runtime share visual features and time-based sequence resampling.
Joystick/aim HUD and future cooldown transitions are not feature inputs.
"""
from collections import deque
from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
from arlott_schema import ACTION_NAMES, state_features
from arlott_adapter import vectorize
from gameplay import observe

CONTEXT_TIMES = (.875,.750,.625,.500,.375,.250,.125,0.)


def dynamic_health(frame, config):
    """Search for the own HP fill near its expected anchor; return uncertainty.

    Green/yellow fill alone is not a death detector. An absent bar stays unknown.
    The full track width is calibrated, not estimated from remaining fill.
    """
    h,w=frame.shape[:2];r=config['hp_rect'];expected=np.array([r[0]*w,(r[1]+r[3])*.5*h])
    x1=max(0,int(expected[0]-.13*w));x2=min(w,int(expected[0]+.18*w))
    y1=max(0,int(expected[1]-.12*h));y2=min(h,int(expected[1]+.13*h))
    hsv=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
    low,high=config.get('player_hp_hue',[22,45])
    mask=cv2.inRange(hsv,np.array([low,115,110]),np.array([high,255,255]))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    full=(r[2]-r[0])*w;candidates=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if not (2<=bh<=h*.022 and 3<=bw<=full*1.15 and bw/max(1,bh)>=2.5):continue
        density=np.mean(mask[y:y+bh,x:x+bw]>0)
        if density<.7:continue
        ax,ay=x+x1,y+y1
        above=frame[max(0,ay-3):ay,ax:ax+bw]
        if not above.size or np.mean(above.max(-1)<150)<.3:continue
        error=abs(ax-expected[0])/w+abs(ay-expected[1])/h
        candidates.append((error,bw/full,[ax,ay,bw,bh]))
    if not candidates:return None,0.,None
    error,value,bar=min(candidates)
    return float(np.clip(value,0,1)),float(max(0,1-error*4)),bar


class RTObserver:
    def __init__(self, config):
        self.config=dict(config)
        self.last_skills={};self.last_cast={};self.last_health=None;self.last_time=None

    def read(self,frame,stamp):
        state=observe(frame,self.config)
        if self.config.get('dynamic_hp',False) and state.get('in_battle'):
            hp,confidence,bar=dynamic_health(frame,self.config)
            state.update(health=hp,health_confidence=confidence,own_health_bar=bar)
        for name in ('skill1','skill2','ultimate'):
            s=state.get('skills',{}).get(name,{})
            if self.last_skills.get(name,{}).get('ready') and s.get('cooldown_visible'):
                self.last_cast[name]=stamp
        self.last_skills=state.get('skills',{})
        previous=max(self.last_cast,key=self.last_cast.get) if self.last_cast else None
        history={'previous_action':previous,'previous_action_age':stamp-self.last_cast[previous]} if previous else None
        f=vectorize(state_features(state,history))
        h,w=frame.shape[:2]
        # World patch only: excludes joystick, skill buttons, minimap and captions.
        world=frame[int(.18*h):int(.67*h),int(.30*w):int(.70*w)]
        small=cv2.resize(world,(6,3),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255
        hsv=cv2.cvtColor(world,cv2.COLOR_BGR2HSV)
        # Exact same integer thresholds as training, fused in native OpenCV.
        bounds=(((0,126,0),(11,255,255)),((36,101,0),(89,255,255)),
                ((96,101,0),(129,255,255)),((131,101,0),(179,255,255)))
        colors=np.array([cv2.countNonZero(cv2.inRange(hsv,lo,hi))/hsv.shape[0]/hsv.shape[1]
                         for lo,hi in bounds],dtype=np.float32)
        hp=state.get('health');dt=stamp-self.last_time if self.last_time is not None else 0
        hp_rate=(hp-self.last_health)/dt if hp is not None and self.last_health is not None and 0<dt<.5 else 0
        extra=np.array([state.get('health_confidence',1 if hp is not None else 0),np.clip(hp_rate,-2,2)/2,
                        *[min(5,stamp-self.last_cast.get(k,stamp-5))/5 for k in ('skill1','skill2','ultimate')]],dtype=np.float32)
        self.last_health=hp;self.last_time=stamp
        return state,np.concatenate([f,small,colors,extra]).astype(np.float32)


class CausalContext:
    def __init__(self):self.frames=deque(maxlen=160)
    def push(self,stamp,features):
        if self.frames and stamp<=self.frames[-1][0]:
            if stamp==self.frames[-1][0]:return self.current()
            self.frames.clear()
        self.frames.append((stamp,features));return self.current()
    def current(self):
        now=self.frames[-1][0];values=list(self.frames);result=[]
        for age in CONTEXT_TIMES:
            eligible=[v for t,v in values if t<=now-age+1e-6]
            result.append(eligible[-1] if eligible else values[0][1])
        return np.stack(result).astype(np.float32)


class ArlottRTPolicy:
    realtime=True
    def __init__(self,directory,backend='fp32'):
        import onnxruntime as ort
        directory=Path(directory);self.config=json.loads((directory/'policy.json').read_text())
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(directory/f'policy-{backend}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.observer=None;self.context=CausalContext();self.current_context=None;self.stamp=0
    def observe(self,frame,config,stamp):
        if self.observer is None:self.observer=RTObserver(config)
        state,features=self.observer.read(frame,stamp)
        self.current_context=self.context.push(stamp,features);self.stamp=stamp
        return state
    def refresh(self,state,decision,history=None):
        if self.current_context is None:return {'action':'hold','confidence':0,'latency_ms':0}
        return {**self.predict(self.current_context),'observed_at':self.stamp}
    def predict(self,context):
        start=time.perf_counter();x=context.reshape(1,-1).astype(np.float32)
        z=self.session.run(None,{'context':x})[0][0]/self.config['temperature']
        p=np.exp(z-z.max());p/=p.sum();choice=int(p.argmax())
        return {'action':ACTION_NAMES[choice],'confidence':float(p[choice]),
                'probabilities':dict(zip(ACTION_NAMES,map(float,p))),
                'latency_ms':(time.perf_counter()-start)*1000,'source':'Arlott ONNX sequence policy'}
