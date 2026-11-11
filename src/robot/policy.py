"""Safety policy layer applied after reasoning and before robot execution.

Rules (safe by default):
- Low battery → force STOP
- PICK / APPROACH require a high-confidence detection of the target
- MOVE_TO coordinates are clamped to the workspace [0, 1]
- Decision confidence below floor → force WAIT (or STOP if critical)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reasoning.bedrock_client import RobotDecision
from src.vision.adapter import SceneContext


@dataclass
class PolicyConfig:
    min_battery: float = 0.15
    min_pick_confidence: float = 0.75
    min_approach_confidence: float = 0.55
    min_decision_confidence: float = 0.35
    workspace_min: float = 0.0
    workspace_max: float = 1.0


@dataclass
class PolicyResult:
    decision: RobotDecision
    applied: bool = False
    rules_fired: list[str] = field(default_factory=list)
    original_command: str | None = None


def _detection_confidence(scene: SceneContext, label: str | None) -> float:
    if not label:
        return 0.0
    label_l = label.lower()
    best = 0.0
    for d in scene.detections:
        if d.label.lower() == label_l:
            best = max(best, d.confidence)
    return best


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _battery_level(robot_state: dict[str, Any], scene: SceneContext) -> float | None:
    """Prefer live robot state; fall back to scene.robot_state."""
    for source in (robot_state, scene.robot_state):
        if not source:
            continue
        if "battery" in source:
            raw = source["battery"]
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            # Accept 0–1 or 0–100 scales
            if val > 1.0:
                val = val / 100.0
            return _clamp(val, 0.0, 1.0)
    return None


def apply_safety_policy(
    decision: RobotDecision,
    scene: SceneContext,
    robot_state: dict[str, Any] | None = None,
    config: PolicyConfig | None = None,
) -> PolicyResult:
    """Return a possibly rewritten decision plus which rules fired."""
    cfg = config or PolicyConfig()
    robot_state = robot_state or {}
    rules: list[str] = []
    original = decision.command.upper().strip()

    command = original
    params = dict(decision.params)
    confidence = decision.confidence
    reasoning = decision.reasoning

    # 1) Battery gate
    battery = _battery_level(robot_state, scene)
    if battery is not None and battery < cfg.min_battery:
        if command not in {"STOP", "WAIT"}:
            rules.append(
                f"low_battery:{battery:.2f}<{cfg.min_battery:.2f}→STOP"
            )
            command = "STOP"
            params = {}
            reasoning = (
                f"[policy] Battery {battery:.0%} below threshold "
                f"({cfg.min_battery:.0%}). Forcing STOP. Original: {original}. "
                f"{reasoning}"
            )

    # 2) Decision confidence floor
    if command not in {"STOP", "WAIT"} and confidence < cfg.min_decision_confidence:
        rules.append(
            f"low_decision_confidence:{confidence:.2f}<{cfg.min_decision_confidence:.2f}→WAIT"
        )
        command = "WAIT"
        params = {"seconds": 1}
        reasoning = (
            f"[policy] Decision confidence {confidence:.2f} too low. "
            f"Forcing WAIT. Original: {original}. {reasoning}"
        )

    # 3) Target confidence for PICK / APPROACH
    if command in {"PICK", "APPROACH"}:
        target = params.get("target_label")
        det_conf = _detection_confidence(scene, target if isinstance(target, str) else None)
        threshold = (
            cfg.min_pick_confidence if command == "PICK" else cfg.min_approach_confidence
        )
        if det_conf < threshold:
            rules.append(
                f"weak_target:{command}:{target}:{det_conf:.2f}<{threshold:.2f}→INSPECT|STOP"
            )
            if det_conf > 0 and target:
                command = "INSPECT"
                params = {"target_label": target}
                reasoning = (
                    f"[policy] Target '{target}' confidence {det_conf:.2f} "
                    f"below {threshold:.2f}. Downgrading to INSPECT. {reasoning}"
                )
            else:
                command = "STOP"
                params = {}
                reasoning = (
                    f"[policy] Missing or weak target for {original}. "
                    f"Forcing STOP. {reasoning}"
                )

    # 4) Workspace clamp for MOVE_TO
    if command == "MOVE_TO":
        try:
            x = float(params.get("x", 0.5))
            y = float(params.get("y", 0.5))
        except (TypeError, ValueError):
            rules.append("invalid_move_to_coords→STOP")
            command = "STOP"
            params = {}
            reasoning = f"[policy] Invalid MOVE_TO coordinates. Forcing STOP. {reasoning}"
        else:
            cx = _clamp(x, cfg.workspace_min, cfg.workspace_max)
            cy = _clamp(y, cfg.workspace_min, cfg.workspace_max)
            if cx != x or cy != y:
                rules.append(f"workspace_clamp:({x},{y})→({cx},{cy})")
                params = {**params, "x": cx, "y": cy}
                reasoning = (
                    f"[policy] Clamped MOVE_TO to workspace "
                    f"[{cfg.workspace_min}, {cfg.workspace_max}]. {reasoning}"
                )

    applied = len(rules) > 0
    new_decision = decision.model_copy(
        update={
            "command": command,
            "params": params,
            "reasoning": reasoning,
        }
    )
    return PolicyResult(
        decision=new_decision,
        applied=applied,
        rules_fired=rules,
        original_command=original if applied else None,
    )
