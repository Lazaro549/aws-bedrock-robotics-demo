"""Vision layer contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single object detection."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 normalized
    attributes: dict = Field(default_factory=dict)


class SceneContext(BaseModel):
    """Structured scene description ready for the reasoning layer."""

    timestamp: str
    frame_id: str
    detections: List[Detection]
    robot_state: dict = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)

    def to_prompt_block(self) -> str:
        lines = [
            f"Frame: {self.frame_id} @ {self.timestamp}",
            f"Robot state: {self.robot_state}",
            f"Environment: {self.environment}",
            "Detections:",
        ]
        for d in self.detections:
            bbox = f" bbox={d.bbox}" if d.bbox else ""
            lines.append(f"  - {d.label} (conf={d.confidence:.2f}){bbox}")
        return "\n".join(lines)


class VisionAdapter(ABC):
    """Abstract vision source. Swap simulator for a real model or ROS topic."""

    @abstractmethod
    def capture(self) -> SceneContext:
        """Return the current scene context."""
        ...
