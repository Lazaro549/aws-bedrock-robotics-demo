"""Unit tests for strict robot command validation."""
import pytest

from src.robot.commands import validate_command


@pytest.mark.parametrize("name,params", [
    ("STOP", {}), ("PLACE", {}),
    ("MOVE_TO", {"x": 0.5, "y": 0.2}),
    ("APPROACH", {"target_label": "cup"}),
    ("PICK", {"target_label": "cup"}),
    ("INSPECT", {"target_label": "cup"}),
    ("WAIT", {}), ("WAIT", {"seconds": 0.5}),
])
def test_valid_commands(name, params):
    assert validate_command(name, params) == (True, "")


@pytest.mark.parametrize("name,params", [
    ("FLY", {}),
    ("MOVE_TO", {"x": 0.5}),
    ("MOVE_TO", {"x": float("nan"), "y": 0.5}),
    ("MOVE_TO", {"x": float("inf"), "y": 0.5}),
    ("MOVE_TO", {"x": 1.1, "y": 0.5}),
    ("MOVE_TO", {"x": True, "y": 0.5}),
    ("APPROACH", {"target_label": ""}),
    ("PICK", {"target_label": "cup", "force": 100}),
    ("WAIT", {"seconds": -1}),
    ("WAIT", {"seconds": 11}),
    ("WAIT", {"seconds": float("nan")}),
    ("STOP", {"unexpected": True}),
    ("PLACE", {"target_label": "cup"}),
])
def test_invalid_commands_are_rejected(name, params):
    ok, error = validate_command(name, params)
    assert not ok
    assert error


def test_command_name_must_be_string():
    ok, error = validate_command(None, {})
    assert not ok
    assert "string" in error
