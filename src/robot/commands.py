"""Robot command schema and validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

ALLOWED_COMMANDS = {
    "MOVE_TO",
    "APPROACH",
    "PICK",
    "PLACE",
    "INSPECT",
    "STOP",
    "WAIT",
}


class RobotCommand(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""


def validate_command(name: str, params: dict) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    name = name.upper().strip()
    if name not in ALLOWED_COMMANDS:
        return False, f"Unknown command '{name}'. Allowed: {sorted(ALLOWED_COMMANDS)}"

    if name == "MOVE_TO":
        if "x" not in params or "y" not in params:
            return False, "MOVE_TO requires params x and y"
    if name in {"APPROACH", "PICK", "INSPECT"}:
        if "target_label" not in params:
            return False, f"{name} requires param target_label"
    if name == "WAIT":
        if "seconds" in params and not isinstance(params["seconds"], (int, float)):
            return False, "WAIT seconds must be numeric"

    return True, ""
