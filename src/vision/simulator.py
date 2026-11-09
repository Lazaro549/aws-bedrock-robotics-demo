"""Simulated vision source that produces realistic scenes."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from .adapter import Detection, SceneContext, VisionAdapter

# Simple scenarios the robot might encounter
SCENARIOS = [
    {
        "name": "kitchen_table",
        "detections": [
            ("table", 0.96),
            ("cup", 0.91),
            ("bottle", 0.88),
            ("person", 0.75),
        ],
        "environment": {"location": "kitchen", "lighting": "indoor"},
    },
    {
        "name": "workshop_floor",
        "detections": [
            ("toolbox", 0.93),
            ("screwdriver", 0.87),
            ("box", 0.90),
            ("person", 0.60),
        ],
        "environment": {"location": "workshop", "lighting": "industrial"},
    },
    {
        "name": "empty_corridor",
        "detections": [
            ("door", 0.94),
            ("wall", 0.99),
        ],
        "environment": {"location": "corridor", "lighting": "indoor"},
    },
    {
        "name": "cluttered_desk",
        "detections": [
            ("laptop", 0.95),
            ("mouse", 0.89),
            ("cup", 0.84),
            ("notebook", 0.81),
            ("person", 0.70),
        ],
        "environment": {"location": "office", "lighting": "indoor"},
    },
]


class SimulatedVision(VisionAdapter):
    """Generates synthetic but realistic scene contexts."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._frame_counter = 0

    def capture(self) -> SceneContext:
        self._frame_counter += 1
        scenario = self._rng.choice(SCENARIOS)

        detections = []
        for label, base_conf in scenario["detections"]:
            conf = max(0.5, min(0.99, base_conf + self._rng.uniform(-0.05, 0.05)))
            # Simple normalized bbox
            x1 = self._rng.uniform(0.1, 0.6)
            y1 = self._rng.uniform(0.1, 0.6)
            x2 = min(0.95, x1 + self._rng.uniform(0.1, 0.3))
            y2 = min(0.95, y1 + self._rng.uniform(0.1, 0.3))
            detections.append(
                Detection(
                    label=label,
                    confidence=round(conf, 3),
                    bbox=(round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)),
                )
            )

        return SceneContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_id=f"sim-{self._frame_counter:04d}-{uuid.uuid4().hex[:6]}",
            detections=detections,
            robot_state={
                "position": {"x": round(self._rng.uniform(0, 1), 2), "y": round(self._rng.uniform(0, 1), 2)},
                "gripper": self._rng.choice(["open", "closed"]),
                "battery": round(self._rng.uniform(0.4, 1.0), 2),
            },
            environment=scenario["environment"],
        )
