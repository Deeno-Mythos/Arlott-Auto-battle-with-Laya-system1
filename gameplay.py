"""Calibrated HUD heuristics and Laya policy. This is an experimental controller."""
from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from tactics import read_skills
from arlott_schema import QUESTION as ARLOTT_QUESTION, model_state, state_features

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def _hud_template(path, modified):
    return cv2.imread(path)
QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Choose the immediate action from the observed Mobile Legends state. Retreat when health is below 30%. Otherwise attack a target in melee range, approach a visible target outside melee range, or hold when no target is visible.",
        "criteria": {
            "retreat": "Health below 30 percent. Move away from the enemy to survive.",
            "attack": "Health at least 30 percent and an enemy is in melee range. Basic attack.",
            "approach": "Health at least 30 percent and enemy visible outside melee range. Move toward enemy.",
            "hold": "No enemy visible or game state unknown. Release joystick and wait.",
        },
    }
}


def pixel_rect(rect, width, height):
    x1, y1, x2, y2 = rect
    return (max(0, round(x1 * width)), max(0, round(y1 * height)),
            min(width, round(x2 * width)), min(height, round(y2 * height)))


def crop(frame, rect):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = pixel_rect(rect, width, height)
    return frame[y1:y2, x1:x2]


def observe(frame, config):
    height, width = frame.shape[:2]
    state = {"hero": config["hero"], "in_battle": False, "health": None,
             "enemy_visible": False, "enemy_in_melee_range": False,
             "target": None, "target_distance": None}
    if [width, height] != config["frame_size"]:
        state["reason"] = "Screen dimensions changed; recalibrate."
        return state
    anchor = config.get("battle_anchor")
    if not anchor or not config.get("hp_rect"):
        state["reason"] = "Calibrate controls and the full health bar in a battle."
        return state
    template_path = ROOT / anchor["file"]
    template = _hud_template(str(template_path), template_path.stat().st_mtime_ns) if template_path.exists() else None
    patch = crop(frame, anchor["rect"])
    if template is None or template.shape != patch.shape:
        state["reason"] = "Battle HUD template is missing or dimensions changed."
        return state
    match = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    gray_match = float(cv2.matchTemplate(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                                      cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)[0, 0])
    state["hud_match"] = round(match, 3)
    state["hud_gray_match"] = round(gray_match, 3)
    if match < 0.65 and gray_match < 0.60:
        state["reason"] = "Battle HUD was not detected."
        return state
    hp = crop(frame, config["hp_rect"])
    hsv = cv2.cvtColor(hp, cv2.COLOR_BGR2HSV)
    hue_low, hue_high = config.get("player_hp_hue", [22, 45])
    mask = cv2.inRange(hsv, np.array([hue_low, 100, 100]), np.array([hue_high, 255, 255]))
    # Look for a contiguous horizontal fill, not a count of all green pixels.
    columns = np.mean(mask > 0, axis=0) > 0.2
    filled = np.flatnonzero(columns)
    health = float((filled[-1] + 1) / len(columns)) if len(filled) else 0.0
    state["health"] = round(health, 2)
    state["in_battle"] = True
    state["skills"] = read_skills(frame, config)
    # Red horizontal health bars are candidate targets; this does not identify heroes,
    # towers, Arlott's marks, walls or bushes. Exclude the minimap and skill HUD.
    rect = config["enemy_region"]
    x1, y1, x2, y2 = pixel_rect(rect, width, height)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, np.array([0, 130, 100]), np.array([10, 255, 255])) |
           cv2.inRange(hsv, np.array([170, 130, 100]), np.array([179, 255, 255])))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((2, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player = config["player_center"]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.01 or h < 2 or h > height * 0.025 or w / h < 4:
            continue
        # A health bar is a dense red rectangle with a dark border above it.
        # This excludes the hero's cape, spell effects, and most red HUD text.
        density = float(np.mean(red[y:y + h, x:x + w] > 0))
        above = frame[max(0, y1 + y - 3):y1 + y, x1 + x:x1 + x + w]
        dark_border = float(np.mean(np.max(above, axis=2) < 110)) if above.size else 0
        below = hsv[y + h:y + h + 16, x:x + w]
        mana = float(np.mean((below[:, :, 0] >= 90) & (below[:, :, 0] <= 135) &
                             (below[:, :, 1] > 100) & (below[:, :, 2] > 100))) if below.size else 0
        # Targeted training bots have a bright outline instead of a dark one;
        # their blue mana strip still identifies the health HUD reliably.
        if density < config.get("bar_density", .70) or (dark_border < config.get("bar_border", .50) and mana < .20):
            continue
        kind = "hero" if mana >= 0.20 or w > width * 0.055 else "unit"
        full_width = max(w, round(width * 0.075)) if kind == "hero" else w
        target_health = round(min(1.0, w / full_width), 2) if kind == "hero" else None
        point = [(x1 + x + full_width / 2) / width, (y1 + y + h + height * 0.06) / height]
        distance = math.hypot(point[0] - player[0], (point[1] - player[1]) * height / width)
        candidates.append((distance, point, [x1 + x, y1 + y, w, h], target_health, kind))
    # Immortal Practice targets remain alive at 1 HP with no visible red fill.
    # Recover their HUD from a blue mana strip below a mostly grey health track.
    blue = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([135, 255, 255]))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    mana_bars, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in mana_bars:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.025 or not 4 <= h <= 10 or w / h < 8 or y < 13:
            continue
        if float(np.mean(blue[y:y + h, x:x + w] > 0)) < 0.80:
            continue
        above = hsv[y - 13:y - 2, x:x + w]
        grey_track = float(np.mean((above[:, :, 1] < 100) & (above[:, :, 2] < 140)))
        if grey_track < 0.80:
            continue
        point = [(x1 + x + w / 2) / width, (y1 + y + h + height * 0.06) / height]
        distance = math.hypot(point[0] - player[0], (point[1] - player[1]) * height / width)
        if not any(math.hypot(point[0] - item[1][0], point[1] - item[1][1]) < 0.025 for item in candidates):
            candidates.append((distance, point, [x1 + x, y1 + y - 12, w, h + 12], 0.01, "hero"))
    if candidates:
        candidates.sort(key=lambda item: (item[0] > config.get("max_chase_distance", 0.30),
                                          item[4] != "hero", item[0] > config["melee_range"],
                                          item[3] if item[3] is not None else 1, item[0]))
        distance, point, bounds, target_health, kind = candidates[0]
        state.update(enemy_visible=True, target=point, target_distance=round(distance, 3),
                     enemy_in_melee_range=distance <= config["melee_range"], target_bar=bounds,
                     target_health=target_health, target_kind=kind)
    state["targets"] = [{"position": item[1], "distance": round(item[0], 3), "bar": item[2],
                         "health": item[3], "kind": item[4]} for item in candidates]
    state["reason"] = "HUD heuristics; target identity and Arlott marks are unknown."
    return state


def state_text(state):
    health = "unknown" if state["health"] is None else f"{state['health'] * 100:.0f}%"
    return (f"Hero {state['hero']}. Health {health}. "
            f"Enemy visible: {'yes' if state['enemy_visible'] else 'no'}. "
            f"Enemy in melee range: {'yes' if state['enemy_in_melee_range'] else 'no'}. "
            f"Battle HUD detected: {'yes' if state['in_battle'] else 'no'}.")


class LayaPolicy:
    def __init__(self, config):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import laya
        import torch
        torch.set_num_threads(config["torch_threads"])
        local = ROOT / config["model"]
        self.agent = laya.load(str(local) if local.is_dir() else config["model"], device=config["model_device"])
        self.trained = config.get("policy_schema") == "arlott-v1"
        self.adapter = None
        if self.trained:
            from arlott_adapter import ArlottAdapter, scorer_fingerprint
            self.adapter = ArlottAdapter(local)
            expected = self.adapter.config.get("base_scorer_sha256")
            if expected and scorer_fingerprint(self.agent.model.scorer.state_dict()) != expected:
                raise ValueError("Arlott adapter belongs to different Laya weights. Rerun training/train_adapter.py.")

    def decide(self, state, history=None):
        started = time.perf_counter()
        features = state_features(state, history) if self.trained else None
        result = self.agent.predict(model_state(features) if self.trained else state_text(state),
                                    ARLOTT_QUESTION if self.trained else QUESTION)
        answer = result["answers"]["action"]
        probabilities = self.adapter.predict(features, answer["probabilities"]) if self.adapter else answer.get("probabilities", {})
        action = max(probabilities, key=probabilities.get) if self.adapter else answer["choice"]
        return {"action": action,
                "confidence": float(probabilities[action]) if self.adapter else float(answer.get("answer_confidence", 0)),
                "probabilities": probabilities,
                "base_action": answer["choice"],
                "base_probabilities": answer.get("probabilities", {}),
                "component": "Laya + video-trained neural adapter" if self.adapter else "Laya",
                "source": "video-trained Laya" if self.trained else "stock Laya",
                "latency_ms": round((time.perf_counter() - started) * 1000)}

    def refresh(self, state, decision, history=None):
        """Refresh the small state adapter using cached, age-limited Laya context.

        Controller enforces max_decision_age before calling this. The transformer
        remains on its background thread; cooldown and HP changes reach this head
        immediately without waiting for another CPU transformer pass.
        """
        if self.adapter is None or not decision.get("base_probabilities"):
            return decision
        start=time.perf_counter()
        probabilities=self.adapter.predict(state_features(state,history),decision['base_probabilities'])
        action=max(probabilities,key=probabilities.get)
        return {**decision,"action":action,"confidence":probabilities[action],"probabilities":probabilities,
                "adapter_latency_ms":round((time.perf_counter()-start)*1000,3)}


def allowed_action(decision, state, config):
    """No guessed inputs when the HUD/health is missing or the model is uncertain."""
    if config.get("policy_schema") in ("arlott-v1","arlott-rt-v2"):
        learned = learned_action(decision, state, config)
        if learned is not None:
            return learned
        return combat_action(state, config) if config.get("combat_fallback") else "hold"
    if config.get("combat_fallback", False):
        return combat_action(state, config)
    if not state["in_battle"] or state["health"] is None or state["health"] <= 0:
        return "hold"
    if decision["confidence"] < config["min_answer_confidence"]:
        return "hold"
    action = decision["action"]
    if action in ("attack", "approach", "retreat") and not state["target"]:
        return "hold"
    if action == "attack" and not state["enemy_in_melee_range"]:
        return "hold"
    if state["health"] < config["retreat_health"] and action != "retreat":
        return "hold"
    return action if action in ("attack", "approach", "retreat", "hold") else "hold"


def learned_action(decision, state, config):
    """None means abstain/fallback. A valid hold remains a deliberate model choice.

    Reposition has no learned map route in this dataset, so it releases movement.
    Readiness/range checks constrain inputs; they are separate from neural scores.
    """
    if not state.get("in_battle") or not state.get("health"):
        return "hold"
    if decision.get("confidence", 0) < config["min_answer_confidence"]:
        return None
    action = decision.get("action")
    if (state["health"] < config["retreat_health"] or state.get("tactical_action") == "retreat") and action != "retreat":
        return None
    if action == "reposition":
        return None if state.get("target") else "hold"
    if action == "hold":
        return "hold"
    if not state.get("target"):
        return None
    distance = state.get("target_distance")
    if action == "retreat":
        return action
    if distance is not None and distance > config.get("max_chase_distance", .30):
        return None
    if action == "approach":
        return action
    if action == "attack":
        return action if state.get("enemy_in_melee_range") else None
    if action in ("skill1", "skill2", "ultimate"):
        if not config.get("auto_use_skills") or not state.get("skills", {}).get(action, {}).get("ready"):
            return None
        reach = {"skill1": .17, "skill2": .20, "ultimate": .20}[action]
        return action if distance is not None and distance <= reach else None
    return None


def combat_action(state, config):
    """Explicit rule fallback, not a prediction from the untrained Laya model."""
    if state.get("tactical_action") in ("hold", "approach", "attack", "retreat"):
        return state["tactical_action"]
    if not state.get("in_battle") or not state.get("health") or not state.get("target"):
        return "hold"
    if state["health"] < config["retreat_health"]:
        return "retreat"
    return "attack" if state["enemy_in_melee_range"] else "approach"
