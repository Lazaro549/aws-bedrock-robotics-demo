"""Pipeline metrics collection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from src.metrics.cost import estimate_cost_usd


@dataclass
class PipelineMetrics:
    timestamp: str
    mode: str
    vision_latency_s: float
    reasoning_latency_s: float
    robot_latency_s: float
    total_latency_s: float
    command: str
    confidence: float
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_id: str = ""
    notes: str = ""


class MetricsCollector:
    def __init__(self, log_path: str | Path | None = None):
        self.runs: List[PipelineMetrics] = []
        self.log_path = Path(log_path) if log_path else None

    def record(self, metrics: PipelineMetrics) -> None:
        if metrics.estimated_cost_usd == 0.0 and (metrics.input_tokens or metrics.output_tokens):
            metrics.estimated_cost_usd = estimate_cost_usd(
                metrics.input_tokens, metrics.output_tokens
            )
        self.runs.append(metrics)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")

    def summary(self) -> dict:
        """Backward-compatible flat summary (prefer scorecard() for full view)."""
        from src.metrics.scorecard import build_scorecard

        card = build_scorecard(self)
        if card.get("runs", 0) == 0:
            return {}
        return {
            "runs": card["runs"],
            "success_rate": card["success_rate"],
            "avg_total_latency_s": card["latency_s"]["total"]["avg"],
            "p50_total_latency_s": card["latency_s"]["total"]["p50"],
            "p95_total_latency_s": card["latency_s"]["total"]["p95"],
            "avg_reasoning_latency_s": card["latency_s"]["reasoning"]["avg"],
            "total_input_tokens": card["tokens"]["input_total"],
            "total_output_tokens": card["tokens"]["output_total"],
            "total_estimated_cost_usd": card["estimated_cost_usd"]["total"],
        }

    def scorecard(self, session_label: str = "session") -> dict:
        from src.metrics.scorecard import build_scorecard

        return build_scorecard(self, session_label=session_label)
