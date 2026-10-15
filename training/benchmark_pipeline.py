"""Paced offline replay through the actual controller and loopback control socket.

This DOES NOT measure game-event-to-phone-touch. No phone/ADB/game is used.
"""
import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av
import cv2
import numpy as np
from controller import Controller
from scrcpy_transport import ScrcpySession
from arlott_rt import ArlottRTPolicy
from training.extract_demonstrations import normalize


class ReplaySession(ScrcpySession):
    def __init__(self):
        super().__init__(serial='OFFLINE-NO-PHONE')
        self.width=1600;self.height=720;self.control,self.peer=socket.socketpair()
        self.sent=[];self.origin=0
        self.drain=threading.Thread(target=self._drain,daemon=True);self.drain.start()
    def _drain(self):
        try:
            while self.peer.recv(4096):pass
        except OSError:pass
    def publish(self,frame):
        with self.frame_condition:
            self.frame=frame;self.frame_time=time.perf_counter();self.frame_sequence+=1
            self.frame_condition.notify_all()
    def touch(self,action,pointer,point):
        before=getattr(self,'last_control_sent_at',0);origin=self.origin
        super().touch(action,pointer,point)
        sent=getattr(self,'last_control_sent_at',0)
        if sent!=before and action in (0,2) and origin:
            self.sent.append({'action':action,'decoded_to_socket_sent_ms':(sent-origin)*1000})
    def close(self):
        super().close();self.peer.close();self.drain.join(timeout=.2)


class ReplayController(Controller):
    def _decision_loop(self):return
    def _execute(self,action,state):
        self.session.origin=self.last_processed_frame or self.session.frame_time
        super()._execute(action,state)


def stats(values):
    return {'n':len(values),'p50_ms':float(np.percentile(values,50)),
            'p95_ms':float(np.percentile(values,95)),'p99_ms':float(np.percentile(values,99))} if values else {'n':0}


def main():
    cv2.setNumThreads(2)
    video=json.loads((ROOT/'training/data/source/metadata.json').read_text())['path']
    frames=[];decode=[];conversion=[]
    with av.open(video) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        container.seek(int(410/stream.time_base),stream=stream,backward=True)
        iterator=container.decode(stream)
        while len(frames)<120:
            t=time.perf_counter_ns();frame=next(iterator);d=time.perf_counter_ns()
            if frame.time<410:continue
            image=normalize(frame.to_ndarray(format='bgr24'));end=time.perf_counter_ns()
            frames.append(image);decode.append((d-t)/1e6);conversion.append((end-d)/1e6)
    config=json.loads((ROOT/'training/data/v2/perception.json').read_text())
    config.update(model='models/arlott-rt-v2',policy_engine='onnx',policy_schema='arlott-rt-v2',onnx_backend='fp32',auto_combo_fallback=False)
    config_file=ROOT/'training/reports/v2/offline-benchmark-config.json'
    config_file.write_text(json.dumps(config,indent=2))
    results={}
    for fps in (30,60):
        session=ReplaySession();controller=None
        try:
            session.publish(frames[0])
            controller=ReplayController(session,load_model=False,config_path=config_file)
            controller.policy=ArlottRTPolicy(ROOT/'models/arlott-rt-v2','fp32');controller.model_status='ready'
            controller.command({'kind':'mode','mode':'auto'})
            # Warm the actual observer, sequence resampler, ONNX session and scheduler.
            for frame in frames[:20]:
                controller.heartbeat=time.perf_counter();session.publish(frame);time.sleep(1/fps)
            with controller.lock:controller.timing_samples.clear();session.sent.clear()
            sequence=session.frame_sequence;started=time.perf_counter();deadline=started
            for i in range(300):
                delay=deadline-time.perf_counter()
                if delay>0:time.sleep(delay)
                controller.heartbeat=time.perf_counter();session.publish(frames[i%len(frames)]);deadline+=1/fps
            time.sleep(.08)
            with controller.lock:samples=list(controller.timing_samples);sent=list(session.sent)
            if controller.error:raise RuntimeError(controller.error)
            results[str(fps)]={'published_frames':300,'processed_frames':len(samples),'frames_not_processed':300-len(samples),
                'wall_seconds':time.perf_counter()-started,
                'stages':{name:stats([r[name] for r in samples]) for name in ('queue_ms','perception_ms','policy_guard_ms','dispatch_ms','decoded_to_dispatch_ms')},
                'actual_loopback_down_or_move_sent':stats([r['decoded_to_socket_sent_ms'] for r in sent]),
                'neural_source':controller.control_source,'host_processing_target_met':float(np.percentile([r['decoded_to_dispatch_ms'] for r in samples],95))<=(33 if fps==30 else 16)}
            print(json.dumps({'fps':fps,**results[str(fps)]}),flush=True)
        finally:
            if controller:controller.close()
            session.close()
    report={'scope':'OFFLINE replay: decoded/published host frame to real loopback socket send. Not game event to Android touch.',
            'phone_access':False,'models':'candidate ONNX FP32; candidate not promoted because accuracy gates failed',
            'source_video_fps':29.952,'sixty_fps_warning':'60 Hz is synthetic pacing/reuse of 30 fps source frames; no native 60 fps phone capture was tested.',
            'excluded':['phone display/capture/encoding','USB input/output','Android input registration','browser JPEG display load'],
            'configuration':'event-driven latest frame; two OpenCV threads, one ONNX thread; no background Laya; fixed fallback combos disabled',
            'recorded_file_decode':stats(decode[5:]),'bgr_conversion_and_normalization':stats(conversion[5:]),
            'rates':results,'requested_true_event_touch_p95':{'30fps_33ms':'UNVERIFIED','60fps_16ms':'UNVERIFIED'}}
    (ROOT/'training/reports/v2/pipeline_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'saved':'training/reports/v2/pipeline_benchmark.json','true_event_touch_targets':'unverified'}),flush=True)


if __name__=='__main__':main()
