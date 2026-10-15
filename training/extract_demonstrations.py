"""Decode the supplied local recording and identify observed input transitions.

Skill labels come from post-frame cooldown transitions, but model inputs use only
the PRE-transition observation. Movement/attack weak labels are marked separately.
No phone, cloud upload, or game session is used.
"""
import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arlott_schema import state_features
from gameplay import observe, crop

DATA = ROOT / 'training' / 'data'


def video_config():
    config = json.loads((ROOT / 'config.json').read_text())
    config.update(frame_size=[1600, 720], player_hp_hue=[35, 95],
                  hp_rect=[401/854, (193-42)/396, 468/854, (202-42)/396],
                  player_center=[.50, .53])
    coords = {'skill1':(620,386),'skill2':(656,315),'ultimate':(728,275),
              'attack':(722,376),'regen':(489,391),'spell':(545,391),'recall':(437,391),
              'joystick':(153,374)}
    config['points'] = {name:[x/854,(y-42)/396] for name,(x,y) in coords.items()}
    p = config['points']['attack']
    config['battle_anchor'] = {'file':'training/data/video-anchor.png', 'rect':[p[0]-.028,p[1]-.062,p[0]+.028,p[1]+.062]}
    return config


def normalize(frame):
    return cv2.resize(frame[42:438, :], (1600,720), interpolation=cv2.INTER_LINEAR)


def joystick(frame):
    # The recording's visible thumb is evidence for observed movement, not a model input.
    roi = frame[260:430, 50:240]
    grey = cv2.GaussianBlur(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(5,5),1)
    circles = cv2.HoughCircles(grey,cv2.HOUGH_GRADIENT,1,20,param1=80,param2=17,minRadius=16,maxRadius=26)
    if circles is None:
        return None
    candidates = []
    for x,y,r in circles[0]:
        px,py = x+50,y+260
        if np.hypot(px-153,py-374) < 75:
            candidates.append((abs(float(r)-21),float(px-153),float(py-374)))
    if not candidates:
        return None
    _,dx,dy=min(candidates)
    return [round(dx/60,2),round(dy/60,2)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--fps',type=float,default=8)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'event_frames').mkdir(exist_ok=True)
    config=video_config()
    reference=cv2.imread(str(DATA/'source'/'frame-0600.jpg'))
    cv2.imwrite(str(DATA/'video-anchor.png'),crop(normalize(reference),config['battle_anchor']['rect']))
    (DATA/'video_config.json').write_text(json.dumps(config,indent=2))
    observations=[]; events=[]; recent=deque(maxlen=5); last_event={}; last_action='unknown';last_action_time=-99
    next_sample=0; last_weak=-99; started=time.monotonic()
    with av.open(args.video) as container:
        stream=container.streams.video[0]; stream.thread_type='AUTO'
        for decoded in container.decode(stream):
            seconds=decoded.time
            if seconds is None or seconds<next_sample:continue
            next_sample=seconds+1/args.fps
            original=decoded.to_ndarray(format='bgr24')
            frame=normalize(original)
            state=observe(frame,config)
            thumb=joystick(original) if state['in_battle'] else None
            record={'time':round(seconds,3),'state':state,'thumb':thumb}
            observations.append(record)
            if not state['in_battle']:
                recent.clear();continue
            if recent:
                prior=recent[-1]
                for skill in ('skill1','skill2','ultimate'):
                    before=prior['state'].get('skills',{}).get(skill,{})
                    after=state.get('skills',{}).get(skill,{})
                    if before.get('ready') and after.get('cooldown_visible') and seconds-last_event.get(skill,-99)>1.0:
                        event={'time':prior['time'],'label':skill,'confidence':.85,'evidence':'observed_ready_to_cooldown',
                               'post_time':round(seconds,3),'features':state_features(prior['state'],{'previous_action':last_action,'previous_action_age':prior['time']-last_action_time}),
                               'state':prior['state'],'source_frame':f'event_frames/{len(events):04d}.jpg'}
                        events.append(event);cv2.imwrite(str(DATA/event['source_frame']),frame)
                        last_event[skill]=seconds;last_action=skill;last_action_time=seconds
                if thumb and np.hypot(*thumb)>.35 and prior.get('thumb') and np.hypot(*prior['thumb'])<.22 and seconds-last_weak>2:
                    label='reposition'
                    target=prior['state'].get('target')
                    if target:
                        vector=np.array(target)-np.array(config['player_center']); vector[1]*=.45
                        dot=float(np.dot(vector,np.array(thumb))/max(.001,np.linalg.norm(vector)*np.linalg.norm(thumb)))
                        if dot>.4:label='approach'
                        elif dot<-.4 and prior['state']['health']<.6:label='retreat'
                    event={'time':prior['time'],'label':label,'confidence':.55,'evidence':'observed_joystick_start_weak',
                           'features':state_features(prior['state'],{'previous_action':last_action,'previous_action_age':prior['time']-last_action_time}),
                           'state':prior['state'],'source_frame':f'event_frames/{len(events):04d}.jpg'}
                    events.append(event);cv2.imwrite(str(DATA/event['source_frame']),frame)
                    last_weak=seconds;last_action=label;last_action_time=seconds
            recent.append(record)
            if len(observations)%500==0:
                print(json.dumps({'samples':len(observations),'time':round(seconds,1),'events':len(events),'elapsed':round(time.monotonic()-started,1)}),flush=True)
    for name,rows in [('observations.jsonl',observations),('candidate_events.jsonl',events)]:
        (DATA/name).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    report={'samples':len(observations),'battle_samples':sum(x['state']['in_battle'] for x in observations),
            'events':len(events),'labels':dict(Counter(x['label'] for x in events)),'seconds':round(time.monotonic()-started,2)}
    (DATA/'extraction_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
