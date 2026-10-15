import socket
import struct
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from controller import Controller
from test_gameplay import FakeSession, FIXTURE_CONFIG

from scrcpy_transport import ScrcpySession, recv_exact


class TransportTiming(unittest.TestCase):
    def setUp(self):
        self.session=ScrcpySession(serial='offline-test')
        self.session.width=1600;self.session.height=720
        self.session.control,self.peer=socket.socketpair();self.peer.settimeout(.3)
    def tearDown(self):
        self.session.close();self.peer.close()
    def packet(self):return struct.unpack('>BBQiiHHHII',recv_exact(self.peer,32))
    def test_tap_sends_down_without_sleep_and_stop_cancels_release_timer(self):
        with patch('scrcpy_transport.time.sleep',side_effect=AssertionError('blocking control sleep')):
            self.session.tap((.5,.5),pointer=2)
            self.assertEqual(self.packet()[1],0)
            self.session.release_all()
            self.assertEqual(self.packet()[1],1)
        time.sleep(.08);self.peer.settimeout(.02)
        with self.assertRaises(socket.timeout):self.peer.recv(32)
    def test_scheduled_release_and_pointer_reuse(self):
        self.session.tap((.5,.5),pointer=2)
        self.assertEqual(self.packet()[1],0)
        self.session.tap((.6,.5),pointer=2)
        self.assertEqual(self.packet()[1],1)
        self.assertEqual(self.packet()[1],0)
        self.assertEqual(self.packet()[1],1)
        self.assertFalse(self.session.pointers)
    def test_frame_notification_retains_latest_frame(self):
        with self.session.frame_condition:
            self.session.frame_sequence=9
            self.session.frame_condition.notify_all()
        self.assertEqual(self.session.wait_frame(8),9)


class RecordingContract(unittest.TestCase):
    def test_manual_intent_is_causal_and_macro_inputs_are_excluded(self):
        events=[]
        with patch.object(Controller,'_worker',return_value=None):
            controller=Controller(FakeSession(),load_model=False,config_path=FIXTURE_CONFIG)
        controller.recorder=SimpleNamespace(submit=lambda frame,stamp,kind,**data:events.append((stamp,data)),close=lambda:None)
        try:
            controller._tap('skill1')
            self.assertFalse(events)
            controller._tap('skill1',origin='manual')
            self.assertEqual(len(events),1)
            self.assertEqual(events[0][1]['action'],'skill1')
            self.assertLessEqual(events[0][0],events[0][1]['input_time'])
        finally:controller.close()


if __name__=='__main__':unittest.main()
