"""Screen readiness checks and the explicit stateful combat fallback."""
import math
import time

import cv2
import numpy as np


def read_skills(frame, config):
    height, width = frame.shape[:2]
    result = {}
    for name in ("skill1", "skill2", "ultimate", "regen"):
        point = config["points"].get(name)
        if not point:
            continue
        cx, cy = round(point[0] * width), round(point[1] * height)
        sx, sy = max(0, cx - 28), max(0, cy - 22)
        patch = frame[sy:min(height, cy + 22), sx:min(width, cx + 28)]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([180, 80, 255]))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        _, _, stats, centers = cv2.connectedComponentsWithStats(white)
        # Cooldown numerals occupy the center. Icon artwork is wider, shorter,
        # or off-center. Read presence rather than inventing exact seconds.
        timer = any(3 <= w <= 25 and 18 <= h <= 32 and 35 <= area <= 500 and
                    abs(center[0] - (cx - sx)) <= 14 and abs(center[1] - (cy - sy)) <= 12
                    for (x, y, w, h, area), center in zip(stats[1:], centers[1:]))
        lit = float(np.mean(hsv[:, :, 2])) > 70 and float(np.mean(hsv[:, :, 1])) > 35
        result[name] = {"ready": bool(lit and not timer), "cooldown_visible": bool(timer)}
        if name == "skill2":
            radius = 56
            ring = frame[max(0, cy - radius):min(height, cy + radius + 1),
                         max(0, cx - radius):min(width, cx + radius + 1)]
            if ring.shape[:2] == (113, 113):
                ring_hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
                yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
                annulus = ((xx * xx + yy * yy) >= 42 ** 2) & ((xx * xx + yy * yy) <= 54 ** 2)
                pink = ((ring_hsv[:, :, 0] >= 125) & (ring_hsv[:, :, 1] > 90) & (ring_hsv[:, :, 2] > 160))
                result[name]["mark_hint"] = bool(np.mean(pink[annulus]) > 0.035)
    return result


class TacticalMemory:
    def __init__(self):
        self.target = None
        self.target_seen = 0.0
        self.retreating = False

    def update(self, state, config):
        now = time.perf_counter()
        targets = state.get("targets", [])
        if targets and self.target:
            near = min(targets, key=lambda item: math.hypot(item["position"][0] - self.target[0],
                                                          item["position"][1] - self.target[1]))
            drift = math.hypot(near["position"][0] - self.target[0], near["position"][1] - self.target[1])
            chosen_health = state.get("target_health")
            if drift < 0.08 and not (chosen_health is not None and chosen_health < 0.20 and
                                    (near.get("health") or 1) > 0.45):
                state.update(target=near["position"], target_distance=near["distance"],
                             target_bar=near["bar"], target_health=near["health"], target_kind=near["kind"],
                             enemy_in_melee_range=near["distance"] <= config["melee_range"])
        if state.get("target"):
            self.target = list(state["target"])
            self.target_seen = now
        health = state.get("health")
        nearby = sum(item["kind"] == "hero" and item["distance"] < 0.18 for item in targets)
        state["nearby_hero_candidates"] = nearby
        state["recovering"] = False
        if not state.get("in_battle") or not health:
            state["tactical_action"] = "hold"
            state["tactical_reason"] = "Health or battle HUD unavailable."
            return state
        if health < config["retreat_health"] or (nearby >= 2 and health < 0.55):
            self.retreating = True
        if self.retreating and health >= config.get("resume_health", 0.50) and nearby < 2:
            self.retreating = False
        if self.retreating:
            threat = state.get("target") or (self.target if now - self.target_seen < 0.7 else None)
            state["tactical_action"] = "retreat" if threat else "hold"
            state["recovering"] = not bool(threat)
            state["tactical_reason"] = "Recover health before re-engaging."
            if threat:
                player = config["player_center"]
                state["retreat_direction"] = [player[0] - threat[0], player[1] - threat[1]]
        elif not state.get("target"):
            state["tactical_action"] = "hold"
            state["tactical_reason"] = "No target detected."
        elif state.get("target_distance") is not None and state["target_distance"] > config.get("max_chase_distance", 0.30):
            state["tactical_action"] = "hold"
            state["tactical_reason"] = "Target is beyond the chase limit."
        else:
            state["tactical_action"] = "attack" if state["enemy_in_melee_range"] else "approach"
            state["tactical_reason"] = "Keep the target; finish low-health targets first."
        return state
