import json
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from controller import Controller
from gameplay import ROOT, allowed_action, learned_action, observe

FIXTURES = ROOT / "tests" / "fixtures"
FIXTURE_CONFIG = FIXTURES / "config.json"

class FakeSession:
    width = 1600
    height = 720
    error = None

    def __init__(self):
        self.frame_time = time.perf_counter()
        self.pointers = {}
        self.events = []

    def latest(self):
        self.frame_time = time.perf_counter()
        return np.zeros((720, 1600, 3), dtype=np.uint8), self.frame_time

    def touch(self, action, pointer, point):
        self.events.append((action, pointer, point))
        if action == 1:
            self.pointers.pop(pointer, None)
        else:
            self.pointers[pointer] = point

    def tap(self, point, pointer=1):
        self.events.append(("tap", point))

    def release_all(self):
        self.pointers.clear()


class CombatRegression(unittest.TestCase):
    def setUp(self):
        config_patch = patch("controller.CONFIG", FIXTURE_CONFIG)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.config = json.loads(FIXTURE_CONFIG.read_text())
        self.nearby = {"in_battle": True, "health": 0.9, "target": [0.55, 0.53],
                       "enemy_in_melee_range": True}

    def test_top_edge_enemy_bar_and_empty_base(self):
        frame = cv2.imread(str(FIXTURES / "attack-debug.jpg"))
        state = observe(frame, self.config)
        self.assertTrue(state["enemy_visible"])
        x, y, _, _ = state["target_bar"]
        self.assertTrue(1100 < x < 1300 and y < 100)
        for filename in ("battle-initial.jpg", "movement-test.jpg", "practice-leave.jpg"):
            with self.subTest(filename=filename):
                empty = observe(cv2.imread(str(FIXTURES / filename)), self.config)
                self.assertFalse(empty["enemy_visible"])

    def test_bad_model_vote_does_not_block_basic_combat(self):
        vote = {"action": "retreat", "confidence": 0.55}
        self.assertEqual(allowed_action(vote, self.nearby, self.config), "attack")
        self.assertEqual(allowed_action(vote, {**self.nearby, "enemy_in_melee_range": False}, self.config), "approach")
        self.assertEqual(allowed_action(vote, {**self.nearby, "health": 0.2}, self.config), "retreat")
        self.assertEqual(allowed_action(vote, {**self.nearby, "target": None}, self.config), "hold")
        self.assertEqual(allowed_action(vote, {**self.nearby, "in_battle": False}, self.config), "hold")

    def test_glowing_training_target_bar(self):
        state = observe(cv2.imread(str(FIXTURES / "attack-fixed.jpg")), self.config)
        self.assertTrue(state["enemy_visible"])
        x, y, _, _ = state["target_bar"]
        self.assertTrue(1000 < x < 1200 and 240 < y < 290)

    def test_one_hp_training_target_stays_visible(self):
        state = observe(cv2.imread(str(FIXTURES / "combo-before.jpg")), self.config)
        self.assertTrue(state["enemy_visible"])
        self.assertTrue(state["enemy_in_melee_range"])

    def test_auto_attacks_and_focus_change_preserves_auto(self):
        session = FakeSession()
        with patch("controller.observe", return_value=self.nearby):
            controller = Controller(session, load_model=False)
            try:
                controller.config["auto_use_skills"] = False
                controller.model_status = "ready"
                controller.decision = {"action": "retreat", "confidence": 0.55}
                controller.command({"kind": "mode", "mode": "auto"})
                deadline = time.perf_counter() + 1
                while not any(event[0] == "tap" for event in session.events) and time.perf_counter() < deadline:
                    time.sleep(0.02)
                self.assertTrue(any(event[0] == "tap" for event in session.events))
                self.assertEqual(controller.applied_action, "attack")
                self.assertEqual(controller.control_source, "combat fallback")
                controller.command({"kind": "release_manual"})
                self.assertEqual(controller.mode, "auto")
                controller.heartbeat = time.perf_counter() - 2
                time.sleep(0.15)
                self.assertEqual(controller.mode, "manual")
                self.assertFalse(session.pointers)
            finally:
                controller.close()

    def test_combo_order_ultimate_spacing_and_stop(self):
        session = FakeSession()
        with patch.object(Controller, "_worker", return_value=None):
            controller = Controller(session, load_model=False)
        controller.mode = "auto"
        try:
            with patch("controller.time.perf_counter", return_value=1000) as timer:
                self.assertTrue(controller._begin_combo(self.nearby))
                for delay in (0.4, 0.85, 1.1, 1.65, 2.1):
                    timer.return_value = 1000 + delay
                    controller._advance_combo(self.nearby)
                point_names = {tuple(v): k for k, v in controller.config["points"].items()}
                casts = [point_names[tuple(event[1])] for event in session.events if event[0] == "tap"]
                self.assertEqual(casts, ["skill1", "skill2", "attack", "ultimate", "skill2", "attack"])
                self.assertEqual(controller.auto_attack_count, 2)
                self.assertFalse(controller.combo_queue)
                timer.return_value = 1005
                self.assertTrue(controller._begin_combo(self.nearby))
                self.assertNotIn("ultimate", [name for _, name in controller.combo_queue])
                before = len(session.events)
                controller.command({"kind": "stop"})
                timer.return_value = 1010
                controller._advance_combo(self.nearby)
                self.assertFalse(controller.combo_queue)
                self.assertEqual(len(session.events), before)
                self.assertEqual(controller.applied_action, "hold")
        finally:
            controller.close()

    def test_trained_vote_selects_skill_and_respects_cooldown(self):
        config = {**self.config, "policy_schema": "arlott-v1"}
        state = {**self.nearby, "target_distance": .08,
                 "skills": {"skill2": {"ready": True}}}
        vote = {"action": "skill2", "confidence": .9}
        self.assertEqual(allowed_action(vote, state, config), "skill2")
        state["skills"]["skill2"]["ready"] = False
        self.assertEqual(allowed_action(vote, state, config), "attack")
        self.assertIsNone(learned_action(vote, state, config))
        self.assertEqual(allowed_action({"action":"ultimate","confidence":.9},
                                       {**state,"health":0}, config), "hold")

    def test_live_loop_applies_learned_cast_without_starting_fixed_combo(self):
        session = FakeSession()
        state = {**self.nearby, "target_distance": .08,
                 "skills": {"skill1": {"ready": True}}}
        with patch("controller.observe", return_value=state):
            controller = Controller(session, load_model=False)
            try:
                controller.config["policy_schema"] = "arlott-v1"
                controller.model_status = "ready"
                controller.command({"kind":"mode","mode":"auto"})
                controller.decision = {"action":"skill1", "confidence":.9,"observed_at":time.perf_counter()}
                deadline=time.perf_counter()+.6
                while not any(e[0]=="tap" for e in session.events) and time.perf_counter()<deadline:
                    time.sleep(.01)
                self.assertEqual([e[1] for e in session.events if e[0]=="tap"],
                                 [controller.config["points"]["skill1"]])
                self.assertEqual(controller.control_source, "video-trained Laya")
                self.assertFalse(controller.combo_queue)
                controller.command({"kind":"stop"})
                self.assertFalse(session.pointers)
            finally:
                controller.close()

    def test_trained_hold_and_reposition_do_not_invent_navigation(self):
        config={**self.config,"policy_schema":"arlott-v1"}
        self.assertEqual(allowed_action({"action":"hold","confidence":.9},self.nearby,config),"hold")
        self.assertEqual(allowed_action({"action":"reposition","confidence":.9},self.nearby,config),"attack")
        self.assertEqual(allowed_action({"action":"reposition","confidence":.9},
                                       {**self.nearby,"target":None},config),"hold")

    def test_stale_laya_context_is_not_refreshed_or_used_for_casting(self):
        class ExpiredPolicy:
            def refresh(self,*args):
                raise AssertionError("Expired transformer context reached the adapter")
        session=FakeSession()
        with patch("controller.observe",return_value={**self.nearby,"target_distance":.08}), \
                patch.object(Controller,"_begin_combo",return_value=False):
            controller=Controller(session,load_model=False)
            try:
                controller.config['policy_schema']='arlott-v1'
                controller.model_status='ready'
                controller.command({'kind':'mode','mode':'auto'})
                controller.policy=ExpiredPolicy()
                controller.decision={'action':'skill2','confidence':.99,'observed_at':time.perf_counter()-2}
                deadline=time.perf_counter()+.3
                while not any(e[0]=='tap' for e in session.events) and time.perf_counter()<deadline:
                    time.sleep(.01)
                self.assertEqual([e[1] for e in session.events if e[0]=='tap'],[controller.config['points']['attack']])
                self.assertEqual(controller.control_source,'combat fallback')
                self.assertIsNone(controller.error)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
