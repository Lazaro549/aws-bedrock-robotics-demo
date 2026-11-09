"""Smoke tests that require no AWS credentials."""

from src.metrics.collector import MetricsCollector
from src.reasoning.bedrock_client import MockReasoner
from src.robot.commands import RobotCommand, validate_command
from src.robot.simulator import RobotSimulator
from src.vision.simulator import SimulatedVision


def test_vision_produces_scene():
    vision = SimulatedVision(seed=42)
    scene = vision.capture()
    assert scene.frame_id
    assert isinstance(scene.detections, list)


def test_mock_reasoner_returns_valid_command():
    vision = SimulatedVision(seed=1)
    reasoner = MockReasoner()
    scene = vision.capture()
    decision = reasoner.decide(scene)
    ok, err = validate_command(decision.command, decision.params)
    assert ok, err
    assert 0.0 <= decision.confidence <= 1.0


def test_robot_executes_stop():
    robot = RobotSimulator()
    result = robot.execute(RobotCommand(name="STOP", params={}))
    assert result.success


def test_full_mock_loop():
    vision = SimulatedVision(seed=7)
    reasoner = MockReasoner()
    robot = RobotSimulator(seed=7)
    metrics = MetricsCollector()

    scene = vision.capture()
    decision = reasoner.decide(scene)
    cmd = RobotCommand(name=decision.command, params=decision.params)
    result = robot.execute(cmd)

    assert result is not None
    assert decision.command in {"MOVE_TO", "APPROACH", "PICK", "PLACE", "INSPECT", "STOP", "WAIT"}
