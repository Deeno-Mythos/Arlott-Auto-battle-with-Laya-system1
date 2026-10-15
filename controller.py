"""Local browser controller: manual multi-touch and an optional Laya loop."""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from gameplay import LayaPolicy, allowed_action, combat_action, crop, learned_action, observe
from scrcpy_transport import ROOT, ScrcpySession
from tactics import TacticalMemory

CONFIG = ROOT / "config.json"
MOVEMENT_KEYS = {"KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"}
BUTTON_KEYS = {"Space": "attack", "Digit1": "skill1", "Digit2": "skill2", "Digit3": "ultimate",
               "KeyB": "spell", "KeyE": "regen", "KeyR": "recall"}


class Controller:
    def __init__(self, session, load_model=True, config_path=None, record_demos=False):
        self.session = session
        self.config_path = Path(config_path) if config_path else CONFIG
        self.config = json.loads(self.config_path.read_text())
        (ROOT / "logs").mkdir(exist_ok=True)
        cv2.setNumThreads(int(self.config.get('opencv_threads',2)))
        self.recorder=None
        if record_demos:
            from demo_recording import ManualRecorder
            self.recorder=ManualRecorder(ROOT/'logs'/'demonstrations')
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.mode = "manual"
        self.epoch = 0
        self.keys = set()
        self.heartbeat = time.perf_counter()
        self.mouse_down = False
        self.state = {}
        self.decision = {"action": "hold", "confidence": 0, "latency_ms": 0}
        self.model_status = "loading" if load_model else "disabled"
        self.policy = None
        self.error = None
        self.action_deadline = 0.0
        self.next_attack = 0.0
        self.next_combat = 0.0
        self.applied_action = "hold"
        self.control_source = "manual"
        self.auto_attack_count = 0
        self.combo_queue = deque()
        self.combo_count = 0
        self.next_combo = 0.0
        self.combo_missing_since = None
        self.skill_input_counts = {name: 0 for name in ("skill1", "skill2", "ultimate")}
        self.skill_last_used = {}
        self.tactics = TacticalMemory()
        self.combo_wait_since = None
        self.skipped_skill_inputs = 0
        self.workers = []
        self.last_processed_frame = None
        self.timing_samples = deque(maxlen=600)
        if load_model:
            self._worker(self._load_policy)
        self._worker(self._watchdog)
        self._worker(self._decision_loop)

    def _worker(self, target):
        thread = threading.Thread(target=target, daemon=True)
        self.workers.append(thread)
        thread.start()

    def _load_policy(self):
        try:
            if self.config.get('policy_engine')=='onnx':
                from arlott_rt import ArlottRTPolicy
                self.policy=ArlottRTPolicy(ROOT/self.config['model'],self.config.get('onnx_backend','fp32'))
            else:
                self.policy = LayaPolicy(self.config)
            self.model_status = "ready"
        except Exception as exc:
            self.model_status = "failed"
            self.error = f"Laya: {type(exc).__name__}: {exc}"

    def _stop_inputs(self):
        self.epoch += 1
        self.keys.clear()
        self.mouse_down = False
        self.action_deadline = 0
        self.last_processed_frame = None
        self.decision["observed_at"] = 0
        self.applied_action = "hold"
        self.control_source = "stopped"
        self.combo_queue.clear()
        self.combo_missing_since = None
        self.combo_wait_since = None
        self.tactics = TacticalMemory()
        self.session.release_all()

    def _watchdog(self):
        from realtime_scheduling import prioritize_control_thread
        self.scheduling_status=prioritize_control_thread()
        sequence = 0
        while not self.closed.is_set():
            if hasattr(self.session,'wait_frame'):
                sequence=self.session.wait_frame(sequence,timeout=.01)
            elif self.closed.wait(.005):
                break
            try:
                with self.lock:
                    stale = time.perf_counter() - self.session.frame_time > 1.5
                    inactive = time.perf_counter() - self.heartbeat > 1
                    if self.recorder and self.mode=='manual':
                        frame,stamp=self.session.latest()
                        self.recorder.submit(frame,stamp,'frame',source='manual')
                    if stale or inactive or self.session.error:
                        self._stop_inputs()
                        self.mode = "manual"
                    elif self.combo_queue:
                        frame, _ = self.session.latest()
                        state = self._observe(frame)
                        self.state = state
                        self._advance_combo(state)
                    elif self.mode == "auto" and self.config.get("combat_fallback") and time.perf_counter() >= self.next_combat:
                        frame, frame_stamp = self.session.latest()
                        if self.last_processed_frame == frame_stamp:
                            continue
                        self.last_processed_frame = frame_stamp
                        processing_started=time.perf_counter()
                        state = self._observe(frame,frame_stamp)
                        observed_at=time.perf_counter()
                        self.state = state
                        trained = self.config.get("policy_schema") in ("arlott-v1","arlott-rt-v2")
                        realtime = self.policy is not None and getattr(self.policy,'realtime',False)
                        recent_vote = time.perf_counter() - self.decision.get("observed_at", 0) <= self.config["max_decision_age"]
                        if trained and (recent_vote or realtime) and self.policy is not None:
                            self.decision = self.policy.refresh(state, self.decision, self._policy_history())
                            recent_vote=time.perf_counter()-self.decision.get('observed_at',0)<=self.config['max_decision_age']
                        learned = learned_action(self.decision, state, self.config) if trained and recent_vote else None
                        action = learned if learned is not None else combat_action(state, self.config)
                        decided_at=time.perf_counter()
                        self.applied_action = action
                        self.control_source = (self.decision.get('source','video-trained Laya') if trained and learned is not None else "combat fallback")
                        if state.get("recovering") and self._skill_ready(state, "regen") and time.perf_counter() - self.skill_last_used.get("regen", float("-inf")) > 40:
                            self._tap("regen")
                        self._execute(action, state)
                        sent_at=time.perf_counter()
                        self.timing_samples.append({'decoded_to_dispatch_ms':(sent_at-frame_stamp)*1000,
                            'queue_ms':(processing_started-frame_stamp)*1000,
                            'perception_ms':(observed_at-processing_started)*1000,
                            'policy_guard_ms':(decided_at-observed_at)*1000,
                            'dispatch_ms':(sent_at-decided_at)*1000,
                            'scope':'host decoded frame to dispatch; phone registration not measured'})
                        self.next_combat = 0
                    elif self.mode == "auto" and time.perf_counter() > self.action_deadline:
                        self.session.release_all()
                    elif self.mode != "auto" and "Space" in self.keys and time.perf_counter() >= self.next_attack:
                        self._tap("attack",origin='manual-repeat')
                        self.next_attack = time.perf_counter() + 0.3
            except Exception as exc:
                self.error = f"Control watchdog: {exc}"
                self.mode = "manual"

    def move(self, dx, dy):
        point = self.config["points"].get("joystick")
        if not point or (dx == 0 and dy == 0):
            self.session.touch(1, 0, (0, 0))
            return
        norm = math.hypot(dx, dy)
        radius = self.config["joystick_radius"]
        aspect = self.session.width / self.session.height
        target = [point[0] + dx / norm * radius, point[1] + dy / norm * radius * aspect]
        if 0 not in self.session.pointers:
            self.session.touch(0, 0, point)
        self.session.touch(2, 0, target)

    def _keyboard_move(self):
        dx = int(bool(self.keys & {"KeyD", "ArrowRight"})) - int(bool(self.keys & {"KeyA", "ArrowLeft"}))
        dy = int(bool(self.keys & {"KeyS", "ArrowDown"})) - int(bool(self.keys & {"KeyW", "ArrowUp"}))
        if self.recorder:
            frame,stamp=self.session.latest()
            self.recorder.submit(frame,stamp,'input',action='move',direction=[dx,dy],source='manual',input_time=time.perf_counter())
        self.move(dx, dy)

    def _tap(self, name, origin=None):
        point = self.config["points"].get(name)
        if point:
            if self.recorder and origin in ('manual','manual-repeat'):
                frame,stamp=self.session.latest()
                self.recorder.submit(frame,stamp,'input',action=name,source=origin,input_time=time.perf_counter())
            self.session.tap(point, pointer=2)
            self.skill_last_used[name] = time.perf_counter()
            if name in self.skill_input_counts:
                self.skill_input_counts[name] += 1

    def _observe(self, frame, stamp=None):
        if frame is not None and self.policy is not None and getattr(self.policy,'realtime',False):
            state=self.policy.observe(frame,self.config,self.session.frame_time if stamp is None else stamp)
        else:
            state = observe(frame, self.config) if frame is not None else {"in_battle": False}
        return self.tactics.update(state, self.config)

    def _skill_ready(self, state, name):
        reading = state.get("skills", {}).get(name)
        elapsed = time.perf_counter() - self.skill_last_used.get(name, float("-inf"))
        if reading is not None:
            return reading["ready"] and elapsed >= (0.25 if name == "skill2" else 0.8)
        # Older/manual observations without a skill reader retain conservative spacing.
        return elapsed >= (self.config.get("ultimate_interval", 30) if name == "ultimate" else 0.25)

    def _policy_history(self):
        casts = {k:v for k,v in list(self.skill_last_used.items()) if k in self.skill_input_counts}
        previous = max(casts,key=casts.get) if casts else None
        return {"previous_action":previous,"previous_action_age":time.perf_counter()-casts[previous]} if previous else None

    def _begin_combo(self, state, manual=False):
        if (self.combo_queue or not state.get("in_battle") or not state.get("target") or
                not state.get("enemy_in_melee_range") or not state.get("health") or
                state["health"] < self.config["retreat_health"] or state.get("tactical_action") == "retreat"):
            return False
        now = time.perf_counter()
        skills = state.get("skills", {})
        use_ultimate = self._skill_ready(state, "ultimate") and (
            manual or state.get("target_health") is None or state["target_health"] > 0.15 or
            state.get("nearby_hero_candidates", 0) >= 2)
        if self._skill_ready(state, "skill1"):
            steps = [(0.0, "skill1"), (0.40, "skill2"), (0.85, "attack")]
            if use_ultimate:
                steps += [(1.10, "ultimate"), (1.65, "skill2"), (2.10, "attack")]
            else:
                steps += [(1.20, "attack")]
        elif self._skill_ready(state, "skill2") and skills.get("skill2", {}).get("mark_hint"):
            steps = [(0.0, "skill2"), (0.50, "attack")]
        elif use_ultimate:
            steps = [(0.0, "ultimate"), (0.55, "skill2"), (1.05, "attack")]
        else:
            return False
        self.move(0, 0)
        self.combo_queue.extend((now + delay, name) for delay, name in steps)
        self.next_combo = now + (1.0 if skills else self.config.get("combo_interval", 9))
        self.combo_missing_since = None
        self.combo_count += 1
        self.applied_action = "combo"
        self.control_source = "Arlott adaptive combo"
        self._advance_combo(state)
        return True

    def _advance_combo(self, state):
        if not self.combo_queue:
            return
        now = time.perf_counter()
        if state.get("tactical_action") == "retreat" or (state.get("health") and state["health"] < self.config["retreat_health"]):
            self.combo_queue.clear()
            self.applied_action = "hold"
            return
        if not state.get("in_battle") or not state.get("health") or not state.get("target"):
            # Dashes can briefly move the health HUD or obscure a target. Pause
            # inputs for that transition, then cancel if it does not recover.
            if self.combo_missing_since is None:
                self.combo_missing_since = now
            if now - self.combo_missing_since > 0.6:
                self.combo_queue.clear()
                self.session.release_all()
                self.applied_action = "hold"
            return
        self.combo_missing_since = None
        self.applied_action = "combo"
        self.control_source = "Arlott adaptive combo"
        if self.combo_queue and now >= self.combo_queue[0][0]:
            name = self.combo_queue[0][1]
            if name in self.skill_input_counts and not self._skill_ready(state, name):
                if self.combo_wait_since is None:
                    self.combo_wait_since = now
                if now - self.combo_wait_since > 0.5:
                    self.combo_queue.popleft()
                    self.combo_wait_since = None
                    self.skipped_skill_inputs += 1
                return
            self.combo_wait_since = None
            _, name = self.combo_queue.popleft()
            self._tap(name)
            if name == "attack" and self.mode == "auto":
                self.auto_attack_count += 1
            self.action_deadline = time.perf_counter() + 0.8

    def _decision_loop(self):
        if self.config.get('policy_engine')=='onnx':
            return  # Per-frame ONNX path owns decisions; no transformer or polling.
        log = (ROOT / "logs" / "decisions.jsonl").open("a", encoding="utf-8")
        try:
            while not self.closed.wait(self.config["decision_interval"]):
                frame, stamp = self.session.latest()
                if frame is None:
                    continue
                with self.lock:
                    config = json.loads(json.dumps(self.config))
                    generation = self.mode
                    epoch = self.epoch
                state = observe(frame, config)
                if self.mode != "auto" or not config.get("combat_fallback"):
                    self.state = state
                if self.mode == "manual" or self.policy is None or not state["in_battle"]:
                    continue
                try:
                    history = self._policy_history()
                    decision = self.policy.decide(state, history)
                    decision["observed_at"] = stamp
                    self.decision = decision
                    action = allowed_action(decision, state, config)
                    age = time.perf_counter() - stamp
                    entry = {"time": time.time(), "state": state, "decision": decision,
                             "applied_action": self.applied_action if generation == "auto" else None,
                             "control_source": self.control_source,
                             "frame_age_seconds": round(age, 3), "mode": generation}
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    if config.get("combat_fallback"):
                        continue  # Fast loop applies fresh learned choices with current HUD checks.
                    with self.lock:
                        # Use a fresh observation as a second check after potentially slow CPU inference.
                        fresh, fresh_stamp = self.session.latest()
                        current = observe(fresh, self.config) if fresh is not None else {"in_battle": False}
                        if (generation != self.mode or epoch != self.epoch or self.mode != "auto" or
                                age > config["max_decision_age"] or not current["in_battle"] or
                                time.perf_counter() - fresh_stamp > 1 or
                                time.perf_counter() - self.heartbeat > 1):
                            continue
                        if allowed_action(decision, current, config) != action:
                            action = "hold"
                        self.applied_action = action
                        self.control_source = decision.get("source", "Laya")
                        self._execute(action, current)
                except Exception as exc:
                    self.error = f"Decision loop: {type(exc).__name__}: {exc}"
                    with self.lock:
                        self.mode = "manual"
                        self._stop_inputs()
        finally:
            log.close()

    def _execute(self, action, state):
        self.action_deadline = time.perf_counter() + min(0.8, self.config["decision_interval"] + 0.2)
        target = state.get("target")
        if action == "retreat" and state.get("retreat_direction"):
            dx, dy = state["retreat_direction"]
            self.move(dx, dy * self.session.height / self.session.width)
        elif action in ("approach", "retreat") and target:
            player = self.config["player_center"]
            sign = -1 if action == "retreat" else 1
            self.move(sign * (target[0] - player[0]), sign * (target[1] - player[1]) * self.session.height / self.session.width)
        else:
            self.move(0, 0)
        if action in self.skill_input_counts and self._skill_ready(state, action):
            self._tap(action)
        elif action == "attack" and time.perf_counter() >= self.next_attack:
            if (self.control_source not in ("video-trained Laya","Arlott ONNX sequence policy") and self.config.get("auto_use_skills") and
                    self.config.get('auto_combo_fallback',True) and
                    time.perf_counter() >= self.next_combo and self._begin_combo(state)):
                return
            self._tap("attack")
            self.auto_attack_count += 1
            self.next_attack = time.perf_counter() + 0.3

    def status(self):
        with self.lock:
            return {"mode": self.mode, "model": self.model_status, "hero": self.config["hero"],
                    "state": self.state, "decision": self.decision,
                    "applied_action": self.applied_action, "control_source": self.control_source,
                    "auto_attack_count": self.auto_attack_count,
                    "combo_count": self.combo_count,
                    "combo_steps_remaining": [name for _, name in self.combo_queue],
                    "skill_input_counts": self.skill_input_counts,
                    "skipped_skill_inputs": self.skipped_skill_inputs,
                    "timing":self.timing_samples[-1] if self.timing_samples else None,
                    "scheduling":getattr(self,'scheduling_status',{}),
                    "recording":str(self.recorder.directory) if self.recorder else None,
                    "error": self.error or self.session.error, "points": self.config["points"],
                    "hp_rect": self.config["hp_rect"], "frame_size": [self.session.width, self.session.height],
                    "stream_age_ms": round((time.perf_counter() - self.session.frame_time) * 1000)}

    def command(self, body):
        with self.lock:
            kind = body["kind"]
            if kind == "heartbeat":
                self.heartbeat = time.perf_counter()
                return
            if kind == "stop":
                self.mode = "manual"
                self._stop_inputs()
            elif kind == "release_manual":
                if self.mode != "auto":
                    self._stop_inputs()
            elif kind == "mode":
                mode = body["mode"]
                if mode not in ("manual", "coach", "auto"):
                    raise ValueError("Unknown controller mode")
                if mode != "manual" and self.model_status != "ready":
                    raise ValueError("Laya is still loading or failed. Manual controls are available.")
                if mode == "auto" and (not self.config["battle_anchor"] or not self.config["hp_rect"]):
                    raise ValueError("Calibrate the battle HUD and full health bar first.")
                self._stop_inputs()
                self.mode = mode
                self.heartbeat = time.perf_counter()
            elif kind in ("key", "tap", "touch", "combo"):
                if self.mode == "auto":
                    self.mode = "manual"
                    self._stop_inputs()
                self.heartbeat = time.perf_counter()
                if kind == "key":
                    key = body["code"]
                    down = bool(body["down"])
                    repeated = key in self.keys
                    self.keys.add(key) if down else self.keys.discard(key)
                    if key in MOVEMENT_KEYS:
                        self._keyboard_move()
                    elif key == "KeyC" and down and not repeated:
                        frame, _ = self.session.latest()
                        if not self._begin_combo(self._observe(frame), manual=True):
                            raise ValueError("Move within melee range of a detected target to start the combo.")
                    elif down and not repeated and key in BUTTON_KEYS:
                        self._tap(BUTTON_KEYS[key],origin='manual')
                elif kind == "combo":
                    frame, _ = self.session.latest()
                    if not self._begin_combo(self._observe(frame), manual=True):
                        raise ValueError("Move within melee range of a detected target to start the combo.")
                elif kind == "tap":
                    self._tap(body["name"],origin='manual')
                else:
                    action = int(body["action"])
                    if self.recorder:
                        frame,stamp=self.session.latest()
                        self.recorder.submit(frame,stamp,'input',action='touch',touch_action=action,
                                             point=body['point'],source='manual',input_time=time.perf_counter())
                    self.session.touch(action, 1, validated_point(body["point"]))
            elif kind == "calibrate_point":
                name = body["name"]
                if name not in {"joystick", "attack", "skill1", "skill2", "ultimate", "spell", "regen", "recall"}:
                    raise ValueError("Unknown control")
                self.mode = "manual"
                self._stop_inputs()
                self.config["points"][name] = validated_point(body["point"])
                self.config["frame_size"] = [self.session.width, self.session.height]
                self.save_config()
            elif kind == "calibrate_hp":
                self.mode = "manual"
                self._stop_inputs()
                a, b = validated_point(body["a"]), validated_point(body["b"])
                rect = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
                if rect[2] - rect[0] < 0.01 or rect[3] - rect[1] < 0.002:
                    raise ValueError("Select both corners of the full green health bar.")
                self.config["hp_rect"] = rect
                self.save_config()
            elif kind == "calibrate_hud":
                self.mode = "manual"
                self._stop_inputs()
                point = self.config["points"].get("attack")
                if not point:
                    raise ValueError("Calibrate the attack button first.")
                frame, _ = self.session.latest()
                # Capture the button's ring rather than an animation in its center.
                rect = [max(0, point[0] - 0.028), max(0, point[1] - 0.062),
                        min(1, point[0] + 0.028), min(1, point[1] + 0.062)]
                cv2.imwrite(str(ROOT / "logs" / "battle-anchor.png"), crop(frame, rect))
                self.config["battle_anchor"] = {"file": "logs/battle-anchor.png", "rect": rect}
                self.config["frame_size"] = [self.session.width, self.session.height]
                self.save_config()
            else:
                raise ValueError("Unknown command")

    def save_config(self):
        temp = self.config_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.config, indent=2) + "\n")
        temp.replace(self.config_path)

    def close(self):
        self.closed.set()
        with self.lock:
            self._stop_inputs()
        for thread in self.workers:
            thread.join(timeout=1)
        if self.recorder:self.recorder.close()


def validated_point(point):
    if len(point) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in point):
        raise ValueError("Invalid screen coordinates")
    return [float(x) for x in point]


def handler_for(controller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/":
                self.reply(200, (ROOT / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/status":
                self.reply(200, json.dumps(controller.status()).encode())
            elif route == "/frame":
                frame, _ = controller.session.latest()
                if frame is None:
                    self.reply(503, b'{}')
                else:
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.reply(200 if ok else 503, jpg.tobytes(), "image/jpeg")
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            # Only the local page may submit controls; prevent cross-origin device input.
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host != f"127.0.0.1:{self.server.server_port}" or (origin and origin != f"http://{host}"):
                self.reply(403, b'{"error":"Local controller only"}')
                return
            if self.path != "/command":
                self.reply(404, b'{}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                controller.command(json.loads(self.rfile.read(length)))
                self.reply(200, b'{"ok":true}')
            except Exception as exc:
                self.reply(400, json.dumps({"error": str(exc)}).encode())
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--fps',type=int,choices=(30,60),default=30)
    parser.add_argument('--record-demos',action='store_true')
    parser.add_argument('--tap-ms',type=float,default=55)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with ScrcpySession(serial=config.get("serial"),max_fps=args.fps,tap_hold_ms=args.tap_ms) as session:
        controller = Controller(session, load_model=not args.no_model,config_path=args.config,record_demos=args.record_demos)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(controller))
        server.daemon_threads = True
        print(f"Arlott controller ready: http://127.0.0.1:{args.port}", flush=True)
        print("Starts in manual mode. Close the page or press Escape to release all touches.", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            controller.close()
            server.server_close()


if __name__ == "__main__":
    main()
