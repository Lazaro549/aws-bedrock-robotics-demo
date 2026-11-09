"""Simulated robot that executes validated commands."""

from __future__ import annotations

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
    """Very lightweight robot state machine."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.state = {
            "position": {"x": 0.5, "y": 0.5},
            "gripper": "open",
            "holding": None,
            "battery": 1.0,
        }

    def execute(self, command: RobotCommand) -> ExecutionResult:
        ok, err = validate_command(command.name, command.params)
        if not ok:
            return ExecutionResult(success=False, message=err, duration_s=0.0, final_state=self.state.copy())

        start = time.perf_counter()
        name = command.name.upper()
        params = command.params

        if name == "STOP":
            msg = "Robot halted."
        elif name == "WAIT":
            seconds = float(params.get("seconds", 1))
            time.sleep(min(seconds, 0.3))  # don't really block long in demo
            msg = f"Waited {seconds}s."
        elif name == "MOVE_TO":
            self.state["position"] = {"x": float(params["x"]), "y": float(params["y"])}
            msg = f"Moved to ({params['x']}, {params['y']})."
        elif name == "APPROACH":
            # Nudge toward a fictional object position
            self.state["position"]["x"] = round(min(0.95, self.state["position"]["x"] + 0.1), 2)
            msg = f"Approaching target '{params.get('target_label')}'."
        elif name == "PICK":
            if self.state["gripper"] == "closed":
                msg = "Gripper already closed."
                success = False
            else:
                self.state["gripper"] = "closed"
                self.state["holding"] = params.get("target_label")
                msg = f"Picked '{params.get('target_label')}'."
                success = True
            duration = time.perf_counter() - start
            return ExecutionResult(success=success, message=msg, duration_s=round(duration, 4), final_state=self.state.copy())
        elif name == "PLACE":
            self.state["gripper"] = "open"
            held = self.state["holding"]
            self.state["holding"] = None
            msg = f"Placed '{held}'." if held else "Gripper opened (nothing was held)."
        elif name == "INSPECT":
            msg = f"Inspecting '{params.get('target_label')}'."
        else:
            msg = f"Executed {name}."

        # Small simulated cost
        self.state["battery"] = round(max(0.0, self.state["battery"] - self._rng.uniform(0.001, 0.01)), 3)
        duration = time.perf_counter() - start
        return ExecutionResult(
            success=True,
            message=msg,
            duration_s=round(duration, 4),
            final_state=self.state.copy(),
        )
