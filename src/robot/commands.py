"""Strict robot command schemas and validation.

LLM output is untrusted input. Validate it again at the execution boundary.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ALLOWED_COMMANDS = {"MOVE_TO", "APPROACH", "PICK", "PLACE", "INSPECT", "STOP", "WAIT"}


class MoveToParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TargetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_label: str = Field(min_length=1, max_length=80)


class WaitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float = Field(default=1.0, gt=0.0, le=10.0, allow_inf_nan=False, strict=True)


class EmptyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


PARAM_SCHEMAS = {
    "MOVE_TO": MoveToParams,
    "APPROACH": TargetParams,
    "PICK": TargetParams,
    "INSPECT": TargetParams,
    "WAIT": WaitParams,
    "PLACE": EmptyParams,
    "STOP": EmptyParams,
}


class RobotCommand(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    reasoning: str = ""


def validate_command(name: str, params: dict) -> tuple[bool, str]:
    """Return (ok, error_message), rejecting unknown or unsafe parameters."""
    if not isinstance(name, str):
        return False, "Command name must be a string"
    normalized = name.upper().strip()
    if normalized not in ALLOWED_COMMANDS:
        return False, f"Unknown command '{normalized}'. Allowed: {sorted(ALLOWED_COMMANDS)}"
    if not isinstance(params, dict):
        return False, "Command params must be an object"
    try:
        PARAM_SCHEMAS[normalized].model_validate(params)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return False, f"Invalid {normalized} params at '{location}': {first['msg']}"
    return True, ""
