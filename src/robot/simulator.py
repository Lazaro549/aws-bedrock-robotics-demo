"""Simulated robot that executes validated commands."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

from .commands import RobotCommand, validate_command


@dataclass
class ExecutionResult:
    success: bool
    message: str
    duration_s: float
    final_state: dict = field(default_factory=dict)


class RobotSimulator:
    """Lightweight deterministic simulator; not a physical safety controller."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.state = {
            "position": {"x": 0.5, "y": 0.5},
            "gripper": "open",
            "holding": None,
            "battery": 1.0,
            "status": "IDLE",
        }

    def _result(self, success: bool, message: str, start: float) -> ExecutionResult:
        duration = time.perf_counter() - start
        self.state["status"] = "IDLE" if success else "ERROR"
        return ExecutionResult(
            success=success,
            message=message,
            duration_s=round(duration, 4),
            final_state=copy.deepcopy(self.state),
        )

    def execute(self, command: RobotCommand) -> ExecutionResult:
        ok, err = validate_command(command.name, command.params)
        if not ok:
            self.state["status"] = "ERROR"
            return ExecutionResult(False, err, 0.0, copy.deepcopy(self.state))

        start = time.perf_counter()
        name = command.name.upper().strip()
        params = command.params
        self.state["status"] = "EXECUTING"

        if name == "STOP":
            self.state["status"] = "STOPPED"
            return self._result(True, "Robot halted.", start)
        if name == "WAIT":
            seconds = float(params.get("seconds", 1.0))
            time.sleep(min(seconds, 0.3))
            msg = f"Waited {seconds:g}s."
        elif name == "MOVE_TO":
            self.state["position"] = {"x": float(params["x"]), "y": float(params["y"])}
            msg = f"Moved to ({params['x']}, {params['y']})."
        elif name == "APPROACH":
            self.state["position"]["x"] = round(min(0.95, self.state["position"]["x"] + 0.1), 2)
            msg = f"Approaching target '{params['target_label']}'."
        elif name == "PICK":
            if self.state["gripper"] == "closed":
                return self._result(False, "Gripper already closed.", start)
            self.state["gripper"] = "closed"
            self.state["holding"] = params["target_label"]
            msg = f"Picked '{params['target_label']}'."
        elif name == "PLACE":
            held = self.state["holding"]
            self.state["gripper"] = "open"
            self.state["holding"] = None
            msg = f"Placed '{held}'." if held else "Gripper opened (nothing was held)."
        elif name == "INSPECT":
            msg = f"Inspecting '{params['target_label']}'."
        else:  # Defensive: validate_command should make this unreachable.
            return self._result(False, f"Unsupported command '{name}'.", start)

        # Every successful non-STOP action, including PICK, consumes energy.
        self.state["battery"] = round(
            max(0.0, self.state["battery"] - self._rng.uniform(0.001, 0.01)), 3
        )
        return self._result(True, msg, start)
