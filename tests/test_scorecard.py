"""Session scorecard unit tests."""

from src.metrics.collector import MetricsCollector, PipelineMetrics
from src.metrics.scorecard import build_scorecard, write_scorecard_json


def _run(latency: float, success: bool = True, command: str = "STOP") -> PipelineMetrics:
    return PipelineMetrics(
        timestamp="2026-01-01T00:00:00Z",
        mode="mock",
        vision_latency_s=0.01,
        reasoning_latency_s=latency * 0.5,
        robot_latency_s=0.01,
        total_latency_s=latency,
        command=command,
        confidence=0.8,
        success=success,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.001,
    )


def test_empty_scorecard():
    c = MetricsCollector()
    card = build_scorecard(c)
    assert card["runs"] == 0


def test_scorecard_aggregates():
    c = MetricsCollector()
    for lat in (0.1, 0.2, 0.3, 0.4, 1.0):
        c.record(_run(lat, success=lat < 0.9, command="APPROACH" if lat < 0.5 else "STOP"))

    card = build_scorecard(c, session_label="test")
    assert card["runs"] == 5
    assert card["successes"] == 4
    assert 0.7 < card["success_rate"] < 0.9
    assert card["latency_s"]["total"]["p50"] > 0
    assert card["latency_s"]["total"]["p95"] >= card["latency_s"]["total"]["p50"]
    assert card["tokens"]["input_total"] == 500
    assert card["estimated_cost_usd"]["total"] > 0
    assert "APPROACH" in card["commands"]
    assert "STOP" in card["commands"]


def test_write_scorecard_json(tmp_path):
    c = MetricsCollector()
    c.record(_run(0.25))
    card = build_scorecard(c)
    path = write_scorecard_json(card, tmp_path / "scorecard.json")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "success_rate" in text
