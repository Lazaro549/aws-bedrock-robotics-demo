"""Evaluator unit tests (mock vs mock, no AWS)."""

from pathlib import Path

from src.evaluate import compare_scenes
from src.reasoning.bedrock_client import MockReasoner
from src.vision.file_adapter import load_scenes

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "scenes" / "sample.jsonl"
POLICY = Path(__file__).resolve().parents[1] / "fixtures" / "scenes" / "policy.jsonl"


def test_mock_vs_mock_agreement():
    scenes = load_scenes(SAMPLE)
    report = compare_scenes(
        scenes,
        MockReasoner(),
        MockReasoner(),
        left_name="mock",
        right_name="mock",
        apply_policy=True,
    )
    assert report["summary"]["scenes"] == len(scenes)
    assert report["summary"]["agreement_rate"] == 1.0
    assert report["summary"]["disagreements"] == 0


def test_policy_fixtures_load():
    scenes = load_scenes(POLICY)
    assert len(scenes) >= 6
    ids = {s.frame_id for s in scenes}
    assert "policy-low-battery" in ids
    assert "policy-weak-pick-target" in ids


def test_missing_file_message():
    from src.vision.file_adapter import load_scenes
    import pytest

    with pytest.raises(FileNotFoundError) as ei:
        load_scenes("logs/this-file-does-not-exist.jsonl")
    msg = str(ei.value)
    assert "fixtures/scenes/sample.jsonl" in msg
    assert "bridge" in msg.lower() or "record" in msg.lower()
