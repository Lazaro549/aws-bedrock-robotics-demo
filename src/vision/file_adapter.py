"""Replay vision scenes from JSON or JSONL files.

Supports:
- .jsonl — one SceneContext (or bridge payload) per line
- .json  — a single scene object or a list of scenes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.vision.adapter import Detection, SceneContext, VisionAdapter

_SAMPLE_HINT = "fixtures/scenes/sample.jsonl"
_POLICY_HINT = "fixtures/scenes/policy.jsonl"


def _missing_scenes_message(path: Path) -> str:
    return (
        f"Scenes file not found: {path}\n\n"
        "Hints:\n"
        f"  • Try the sample fixture:  python -m src.pipeline --mode mock --scenes {_SAMPLE_HINT}\n"
        f"  • Policy stress fixtures:  python -m src.pipeline --mode mock --scenes {_POLICY_HINT}\n"
        "  • Record from the bridge:  python -m src.bridge_server --record logs/scenes.jsonl\n"
        "    then replay with:         python -m src.pipeline --mode mock --scenes logs/scenes.jsonl\n"
    )


def _parse_detection(raw: dict[str, Any]) -> Detection:
    bbox = raw.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        bbox_t = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    else:
        bbox_t = None
    attrs = dict(raw.get("attributes") or {})
    if raw.get("track_id") is not None:
        attrs.setdefault("track_id", raw["track_id"])
    return Detection(
        label=str(raw.get("label") or raw.get("class") or "object"),
        confidence=float(raw.get("confidence", raw.get("score", 0.5))),
        bbox=bbox_t,
        attributes=attrs,
    )


def scene_from_dict(raw: dict[str, Any]) -> SceneContext:
    detections_raw = raw.get("detections") or []
    detections = [_parse_detection(d) for d in detections_raw if isinstance(d, dict)]
    return SceneContext(
        timestamp=str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        frame_id=str(raw.get("frame_id") or raw.get("id") or "file-scene"),
        detections=detections,
        robot_state=dict(raw.get("robot_state") or {}),
        environment=dict(raw.get("environment") or {"source": "file"}),
    )


def load_scenes(path: str | Path) -> list[SceneContext]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(_missing_scenes_message(path))

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    scenes: list[SceneContext] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no} of {path} must be a JSON object")
            scenes.append(scene_from_dict(raw))
        return scenes

    data = json.loads(text)
    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} in {path} must be a JSON object")
            scenes.append(scene_from_dict(item))
    elif isinstance(data, dict):
        scenes.append(scene_from_dict(data))
    else:
        raise ValueError(f"Unsupported JSON root type in {path}")
    return scenes


def append_scene(path: str | Path, scene: SceneContext) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(scene.model_dump_json() + "\n")


class FileVision(VisionAdapter):
    def __init__(self, path: str | Path, loop: bool = False):
        self.path = Path(path)
        self.loop = loop
        self._scenes = load_scenes(self.path)
        if not self._scenes:
            raise ValueError(
                f"No scenes found in {self.path}\n"
                f"Add at least one JSON object, or use {_SAMPLE_HINT}"
            )
        self._index = 0

    def __len__(self) -> int:
        return len(self._scenes)

    @property
    def remaining(self) -> int:
        if self.loop:
            return -1
        return max(0, len(self._scenes) - self._index)

    def capture(self) -> SceneContext:
        if self._index >= len(self._scenes):
            if self.loop:
                self._index = 0
            else:
                raise StopIteration(f"Exhausted {len(self._scenes)} scenes from {self.path}")
        scene = self._scenes[self._index]
        self._index += 1
        return scene

    def iter_scenes(self) -> Iterator[SceneContext]:
        for scene in self._scenes:
            yield scene
