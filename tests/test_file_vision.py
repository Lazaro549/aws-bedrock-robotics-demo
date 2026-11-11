"""File / JSONL vision adapter tests."""

from pathlib import Path

import pytest

from src.vision.adapter import Detection, SceneContext
from src.vision.file_adapter import FileVision, append_scene, load_scenes, scene_from_dict

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "scenes" / "sample.jsonl"
POLICY = Path(__file__).resolve().parents[1] / "fixtures" / "scenes" / "policy.jsonl"


def test_load_sample_jsonl():
    scenes = load_scenes(SAMPLE)
    assert len(scenes) >= 5
    assert scenes[0].frame_id == "sample-001"
    assert scenes[0].detections[0].label == "cup"


def test_load_policy_jsonl():
    scenes = load_scenes(POLICY)
    assert any(s.frame_id == "policy-low-battery" for s in scenes)


def test_file_vision_sequential():
    vision = FileVision(SAMPLE, loop=False)
    n = len(vision)
    vision.capture()
    for _ in range(n - 1):
        vision.capture()
    with pytest.raises(StopIteration):
        vision.capture()


def test_file_vision_loop():
    vision = FileVision(SAMPLE, loop=True)
    first = vision.capture().frame_id
    for _ in range(len(vision) - 1):
        vision.capture()
    assert vision.capture().frame_id == first


def test_scene_from_bridge_shaped_payload():
    raw = {
        "frame_id": "dash-1",
        "detections": [
            {"label": "bottle", "confidence": 0.8, "bbox": [10, 20, 30, 40], "track_id": 9}
        ],
        "robot_state": {"battery": 88},
    }
    scene = scene_from_dict(raw)
    assert scene.detections[0].attributes.get("track_id") == 9


def test_append_scene(tmp_path):
    path = tmp_path / "out.jsonl"
    scene = SceneContext(
        timestamp="2026-01-01T00:00:00Z",
        frame_id="a1",
        detections=[Detection(label="cup", confidence=0.9)],
    )
    append_scene(path, scene)
    append_scene(path, scene)
    loaded = load_scenes(path)
    assert len(loaded) == 2


def test_load_json_array(tmp_path):
    path = tmp_path / "scenes.json"
    path.write_text(
        '[{"frame_id": "j1", "detections": []}, {"frame_id": "j2", "detections": []}]',
        encoding="utf-8",
    )
    scenes = load_scenes(path)
    assert [s.frame_id for s in scenes] == ["j1", "j2"]


def test_missing_file_has_hints():
    with pytest.raises(FileNotFoundError) as ei:
        load_scenes("logs/missing-scenes.jsonl")
    assert "sample.jsonl" in str(ei.value)
