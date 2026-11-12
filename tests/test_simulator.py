"""Regression tests for simulator state consistency."""
from src.robot.commands import RobotCommand
from src.robot.simulator import RobotSimulator


def _command(name, params=None):
    return RobotCommand(name=name, params=params or {}, confidence=0.9)


def test_pick_consumes_battery_and_updates_holding():
    robot = RobotSimulator(seed=7)
    before = robot.state["battery"]
    result = robot.execute(_command("PICK", {"target_label": "cup"}))
    assert result.success
    assert result.final_state["battery"] < before
    assert result.final_state["holding"] == "cup"
    assert result.final_state["gripper"] == "closed"


def test_result_state_is_deep_snapshot():
    robot = RobotSimulator(seed=7)
    result = robot.execute(_command("MOVE_TO", {"x": 0.8, "y": 0.2}))
    robot.execute(_command("MOVE_TO", {"x": 0.1, "y": 0.1}))
    assert result.final_state["position"] == {"x": 0.8, "y": 0.2}


def test_stop_state_is_preserved():
    robot = RobotSimulator(seed=7)
    result = robot.execute(_command("STOP"))
    assert result.success
    assert result.final_state["status"] == "STOPPED"


def test_invalid_command_does_not_mutate_position():
    robot = RobotSimulator(seed=7)
    before = dict(robot.state["position"])
    result = robot.execute(_command("MOVE_TO", {"x": 0.5}))
    assert not result.success
    assert robot.state["position"] == before
