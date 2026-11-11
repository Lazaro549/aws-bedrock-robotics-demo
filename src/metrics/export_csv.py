"""CSV export helpers for scorecards and evaluation reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_scorecard_csv(scorecard: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lat = (scorecard.get("latency_s") or {}).get("total") or {}
    rlat = (scorecard.get("latency_s") or {}).get("reasoning") or {}
    tok = scorecard.get("tokens") or {}
    cost = scorecard.get("estimated_cost_usd") or {}
    conf = scorecard.get("confidence") or {}
    commands = scorecard.get("commands") or {}

    row = {
        "session": scorecard.get("session", ""),
        "generated_at": scorecard.get("generated_at", ""),
        "runs": scorecard.get("runs", 0),
        "success_rate": scorecard.get("success_rate", ""),
        "successes": scorecard.get("successes", ""),
        "failures": scorecard.get("failures", ""),
        "latency_avg_s": lat.get("avg", ""),
        "latency_p50_s": lat.get("p50", ""),
        "latency_p95_s": lat.get("p95", ""),
        "reasoning_avg_s": rlat.get("avg", ""),
        "tokens_in": tok.get("input_total", ""),
        "tokens_out": tok.get("output_total", ""),
        "cost_usd_total": cost.get("total", ""),
        "cost_usd_avg": cost.get("avg_per_run", ""),
        "confidence_avg": conf.get("avg", ""),
        "commands": ";".join(f"{k}:{v}" for k, v in commands.items()),
    }

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def write_eval_csv(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = report.get("summary") or {}
    left = summary.get("left", "left")
    right = summary.get("right", "right")

    fieldnames = [
        "frame_id",
        "detections",
        "agree",
        f"{left}_command",
        f"{right}_command",
        f"{left}_confidence",
        f"{right}_confidence",
        f"{left}_latency_s",
        f"{right}_latency_s",
        f"{left}_cost_usd",
        f"{right}_cost_usd",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("rows") or []:
            writer.writerow(
                {
                    "frame_id": row.get("frame_id", ""),
                    "detections": ",".join(row.get("detections") or []),
                    "agree": row.get("agree", ""),
                    f"{left}_command": (row.get(left) or {}).get("command", ""),
                    f"{right}_command": (row.get(right) or {}).get("command", ""),
                    f"{left}_confidence": (row.get(left) or {}).get("confidence", ""),
                    f"{right}_confidence": (row.get(right) or {}).get("confidence", ""),
                    f"{left}_latency_s": (row.get(left) or {}).get("latency_s", ""),
                    f"{right}_latency_s": (row.get(right) or {}).get("latency_s", ""),
                    f"{left}_cost_usd": (row.get(left) or {}).get("estimated_cost_usd", ""),
                    f"{right}_cost_usd": (row.get(right) or {}).get("estimated_cost_usd", ""),
                }
            )
    return path
