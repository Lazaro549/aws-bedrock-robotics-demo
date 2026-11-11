"""Session scorecard: aggregate metrics with percentiles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from src.metrics.collector import MetricsCollector, PipelineMetrics


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile for p in [0, 100]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p = max(0.0, min(100.0, p))
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def build_scorecard(
    collector: "MetricsCollector",
    *,
    session_label: str = "session",
) -> dict:
    """Build a structured scorecard from recorded runs."""
    runs: list[PipelineMetrics] = list(collector.runs)
    if not runs:
        return {
            "session": session_label,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs": 0,
            "message": "No runs recorded yet.",
        }

    n = len(runs)
    total_lat = sorted(r.total_latency_s for r in runs)
    reason_lat = sorted(r.reasoning_latency_s for r in runs)
    successes = sum(1 for r in runs if r.success)
    commands: dict[str, int] = {}
    for r in runs:
        commands[r.command] = commands.get(r.command, 0) + 1

    confidences = [r.confidence for r in runs]

    return {
        "session": session_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": n,
        "success_rate": round(successes / n, 4),
        "successes": successes,
        "failures": n - successes,
        "latency_s": {
            "total": {
                "avg": round(sum(total_lat) / n, 4),
                "p50": round(_percentile(total_lat, 50), 4),
                "p95": round(_percentile(total_lat, 95), 4),
                "min": round(total_lat[0], 4),
                "max": round(total_lat[-1], 4),
            },
            "reasoning": {
                "avg": round(sum(reason_lat) / n, 4),
                "p50": round(_percentile(reason_lat, 50), 4),
                "p95": round(_percentile(reason_lat, 95), 4),
            },
        },
        "tokens": {
            "input_total": sum(r.input_tokens for r in runs),
            "output_total": sum(r.output_tokens for r in runs),
            "input_avg": round(sum(r.input_tokens for r in runs) / n, 2),
            "output_avg": round(sum(r.output_tokens for r in runs) / n, 2),
        },
        "estimated_cost_usd": {
            "total": round(sum(r.estimated_cost_usd for r in runs), 8),
            "avg_per_run": round(sum(r.estimated_cost_usd for r in runs) / n, 8),
        },
        "confidence": {
            "avg": round(sum(confidences) / n, 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
        },
        "commands": dict(sorted(commands.items(), key=lambda kv: (-kv[1], kv[0]))),
        "modes": sorted({r.mode for r in runs}),
    }


def write_scorecard_json(scorecard: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    return path


def print_scorecard(scorecard: dict, console=None) -> None:
    """Pretty-print scorecard to a Rich console or stdout."""
    if console is None:
        from rich.console import Console

        console = Console()

    if scorecard.get("runs", 0) == 0:
        console.print("[dim]Scorecard: no runs recorded.[/]")
        return

    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Runs", str(scorecard["runs"]))
    table.add_row(
        "Success rate",
        f"{scorecard['success_rate'] * 100:.1f}% "
        f"({scorecard['successes']}/{scorecard['runs']})",
    )
    lat = scorecard["latency_s"]["total"]
    table.add_row(
        "Latency total",
        f"avg {lat['avg']:.3f}s · p50 {lat['p50']:.3f}s · p95 {lat['p95']:.3f}s",
    )
    rlat = scorecard["latency_s"]["reasoning"]
    table.add_row(
        "Latency reasoning",
        f"avg {rlat['avg']:.3f}s · p50 {rlat['p50']:.3f}s · p95 {rlat['p95']:.3f}s",
    )
    tok = scorecard["tokens"]
    table.add_row(
        "Tokens",
        f"in {tok['input_total']} · out {tok['output_total']}",
    )
    cost = scorecard["estimated_cost_usd"]
    table.add_row(
        "Est. cost (USD)",
        f"total ${cost['total']:.6f} · avg ${cost['avg_per_run']:.6f}/run",
    )
    conf = scorecard["confidence"]
    table.add_row(
        "Confidence",
        f"avg {conf['avg']:.2f} · min {conf['min']:.2f} · max {conf['max']:.2f}",
    )
    cmds = ", ".join(f"{k}×{v}" for k, v in scorecard["commands"].items())
    table.add_row("Commands", cmds or "—")

    title = f"Scorecard · {scorecard.get('session', 'session')}"
    console.print(Panel(table, title=title, border_style="green"))
