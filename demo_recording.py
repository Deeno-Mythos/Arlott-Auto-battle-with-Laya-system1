"""Opt-in local manual-input journal. JPEG writing never blocks control dispatch."""
import json
import queue
import threading
from datetime import datetime,timezone,timedelta
from pathlib import Path
import cv2


class ManualRecorder:
    def __init__(self,root):
        name=datetime.now(timezone(timedelta(hours=8))).strftime('manual-%Y%m%d-%H%M%S')
        self.directory=Path(root)/name;self.directory.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=64);self.dropped=0;self.last_frame=-99
        self.thread=threading.Thread(target=self._write,name='manual-demo-journal',daemon=True);self.thread.start()
    def submit(self,frame,stamp,kind,**data):
        if frame is None:return
        if kind=='input' and stamp>data.get('input_time',stamp):
            raise ValueError('Demonstration frame is newer than the input label')
        if kind=='frame':
            if stamp-self.last_frame<.125:return
            self.last_frame=stamp
        row={'kind':kind,'frame_decoded_at':stamp,'clock':'QueryPerformanceCounter / perf_counter',
             'timezone':'Asia/Shanghai','outcome':'input intent only; game acceptance unconfirmed',**data}
        try:self.queue.put_nowait((frame,row))
        except queue.Full:self.dropped+=1
    def _write(self):
        with (self.directory/'events.jsonl').open('w',encoding='utf-8') as journal:
            index=0
            while True:
                item=self.queue.get()
                if item is None:self.queue.task_done();break
                frame,row=item;name=f'{index:06d}.jpg';index+=1
                if cv2.imwrite(str(self.directory/name),frame,[cv2.IMWRITE_JPEG_QUALITY,88]):
                    row['frame']=name;journal.write(json.dumps(row)+'\n');journal.flush()
                else:self.dropped+=1
                self.queue.task_done()
    def close(self):
        self.queue.put(None);self.thread.join(timeout=5)
        (self.directory/'recording.json').write_text(json.dumps({'dropped':self.dropped,'flushed':not self.thread.is_alive(),
            'labels':'Only explicitly manual inputs are recorded as intent; move direction requires separate retreat/approach annotation.'},indent=2))
