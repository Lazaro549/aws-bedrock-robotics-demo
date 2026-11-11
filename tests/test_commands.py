"""Unit tests for robot command validation."""

from src.robot.commands import validate_command


def test_stop_ok():
    ok, err = validate_command("STOP", {})
    assert ok
    assert err == ""


def test_unknown_command():
    ok, err = validate_command("FLY", {})
    assert not ok
    assert "Unknown" in err


def test_move_to_requires_xy():
    ok, _ = validate_command("MOVE_TO", {"x": 0.5, "y": 0.2})
    assert ok
    ok, err = validate_command("MOVE_TO", {"x": 0.5})
    assert not ok
    assert "x" in err.lower() or "y" in err.lower()


def test_approach_requires_target():
    ok, _ = validate_command("APPROACH", {"target_label": "cup"})
    assert ok
    ok, err = validate_command("APPROACH", {})
    assert not ok
    assert "target_label" in err
