"""Versioned state/question schema shared by offline training and runtime."""
import json

ACTION_NAMES = ("hold", "approach", "retreat", "attack", "skill1", "skill2", "ultimate", "reposition")
QUESTION = {"action": {"type": "choice", "instructions": "Choose Arlott's next demonstrated action from the current combat state.",
    "criteria": {"hold": "Wait without advancing.", "approach": "Move toward an enemy.",
                 "retreat": "Move away from danger.", "attack": "Use a basic attack.",
                 "skill1": "Cast Dauntless to stun.", "skill2": "Cast Vengeance dash.",
                 "ultimate": "Cast Final Slash.", "reposition": "Move or rotate without attacking."}}}


def state_features(state, history=None):
    history = history or {}
    hp = state.get("health")
    distance = state.get("target_distance")
    target_hp = state.get("target_health")
    skills = state.get("skills", {})
    return {
        "hero": "Arlott", "battle": bool(state.get("in_battle")),
        "hp": "unknown" if hp is None else "critical" if hp < .30 else "low" if hp < .55 else "healthy",
        "hp_fraction": None if hp is None else round(float(hp), 2),
        "enemy_range": "none" if not state.get("target") else "melee" if distance is not None and distance <= .10 else "near" if distance is not None and distance <= .25 else "far",
        "enemy_hp": "unknown" if target_hp is None else "finisher" if target_hp < .20 else "healthy",
        "enemy_count": min(4, len(state.get("targets", []))),
        "enemy_distance": None if distance is None else round(float(distance), 2),
        "target_kind": state.get("target_kind", "unknown"),
        "s1": "ready" if skills.get("skill1", {}).get("ready") else "cooldown",
        "s2": "ready" if skills.get("skill2", {}).get("ready") else "cooldown",
        "ult": "ready" if skills.get("ultimate", {}).get("ready") else "cooldown",
        "mark_hint": bool(skills.get("skill2", {}).get("mark_hint")),
        "previous_action": history.get("previous_action", "unknown"),
        "previous_action_age": "recent" if history.get("previous_action_age", 99) < 1 else "older",
    }


def model_state(features):
    return json.dumps(features, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
