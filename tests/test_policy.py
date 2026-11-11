"""Safety policy unit tests."""

from src.reasoning.bedrock_client import RobotDecision
from src.robot.policy import PolicyConfig, apply_safety_policy
from src.vision.adapter import Detection, SceneContext


def _scene(detections=None, battery=None):
    robot_state = {}
    if battery is not None:
        robot_state["battery"] = battery
    return SceneContext(
        timestamp="2026-01-01T00:00:00Z",
        frame_id="t1",
        detections=detections or [],
        robot_state=robot_state,
    )


def _decision(command, params=None, confidence=0.9):
    return RobotDecision(
        command=command,
        params=params or {},
        confidence=confidence,
        reasoning="test",
        model_id="mock",
    )


def test_low_battery_forces_stop():
    scene = _scene(
        detections=[Detection(label="cup", confidence=0.95)],
        battery=0.05,
    )
    result = apply_safety_policy(
        _decision("PICK", {"target_label": "cup"}),
        scene,
        robot_state={"battery": 0.05},
    )
    assert result.applied
    assert result.decision.command == "STOP"
    assert any("low_battery" in r for r in result.rules_fired)


def test_weak_pick_target_downgrades_to_inspect():
    scene = _scene(detections=[Detection(label="cup", confidence=0.4)])
    result = apply_safety_policy(
        _decision("PICK", {"target_label": "cup"}, confidence=0.9),
        scene,
    )
    assert result.applied
    assert result.decision.command == "INSPECT"


def test_missing_pick_target_stops():
    scene = _scene(detections=[])
    result = apply_safety_policy(
        _decision("PICK", {"target_label": "cup"}, confidence=0.9),
        scene,
    )
    assert result.applied
    assert result.decision.command == "STOP"


def test_move_to_clamped():
    scene = _scene()
    result = apply_safety_policy(
        _decision("MOVE_TO", {"x": 1.5, "y": -0.2}, confidence=0.9),
        scene,
    )
    assert result.applied
    assert result.decision.command == "MOVE_TO"
    assert result.decision.params["x"] == 1.0
    assert result.decision.params["y"] == 0.0


def test_low_decision_confidence_waits():
    scene = _scene(detections=[Detection(label="box", confidence=0.9)])
    result = apply_safety_policy(
        _decision("APPROACH", {"target_label": "box"}, confidence=0.1),
        scene,
        config=PolicyConfig(min_decision_confidence=0.35),
    )
    assert result.applied
    assert result.decision.command == "WAIT"


def test_safe_command_passes():
    scene = _scene(
        detections=[Detection(label="cup", confidence=0.95)],
        battery=0.9,
    )
    result = apply_safety_policy(
        _decision("APPROACH", {"target_label": "cup"}, confidence=0.85),
        scene,
        robot_state={"battery": 0.9},
    )
    assert not result.applied
    assert result.decision.command == "APPROACH"
