"""Compare two reasoners on the same scene set (Mock vs Bedrock by default).

Usage:
    python -m src.evaluate --scenes fixtures/scenes/sample.jsonl
    python -m src.evaluate --scenes fixtures/scenes/policy.jsonl --left mock --right mock
    python -m src.evaluate --scenes fixtures/scenes/sample.jsonl --left mock --right bedrock
    bedrock-robotics-evaluate --scenes fixtures/scenes/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.metrics.cost import estimate_cost_usd
from src.reasoning.bedrock_client import BedrockReasoner, MockReasoner, Reasoner, RobotDecision
from src.robot.policy import apply_safety_policy
from src.vision.adapter import SceneContext
from src.vision.file_adapter import load_scenes

console = Console()


def _make_reasoner(name: str) -> Reasoner:
    name = name.lower().strip()
    if name == "mock":
        return MockReasoner()
    if name == "bedrock":
        return BedrockReasoner()
    raise ValueError(f"Unknown reasoner '{name}'. Use: mock | bedrock")


def _run_one(
    reasoner: Reasoner,
    scene: SceneContext,
    *,
    apply_policy: bool,
    robot_state: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = reasoner.decide(scene)
    policy_rules: list[str] = []
    original = decision.command
    if apply_policy:
        result = apply_safety_policy(decision, scene, robot_state=robot_state or scene.robot_state)
        if result.applied:
            policy_rules = result.rules_fired
            decision = result.decision
    cost = estimate_cost_usd(decision.input_tokens, decision.output_tokens)
    return {
        "command": decision.command,
        "params": decision.params,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "latency_s": decision.latency_s,
        "input_tokens": decision.input_tokens,
        "output_tokens": decision.output_tokens,
        "estimated_cost_usd": cost,
        "model_id": decision.model_id,
        "policy_rules": policy_rules,
        "original_command": original if policy_rules else None,
    }


def compare_scenes(
    scenes: list[SceneContext],
    left: Reasoner,
    right: Reasoner,
    *,
    left_name: str,
    right_name: str,
    apply_policy: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    agree = 0

    for scene in scenes:
        left_out = _run_one(left, scene, apply_policy=apply_policy, robot_state=scene.robot_state)
        right_out = _run_one(right, scene, apply_policy=apply_policy, robot_state=scene.robot_state)
        same = left_out["command"] == right_out["command"]
        if same:
            agree += 1
        rows.append(
            {
                "frame_id": scene.frame_id,
                "detections": [d.label for d in scene.detections],
                "agree": same,
                left_name: left_out,
                right_name: right_out,
            }
        )

    n = len(rows) or 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenes": len(rows),
        "left": left_name,
        "right": right_name,
        "policy_applied": apply_policy,
        "agreement_rate": round(agree / n, 4) if rows else 0.0,
        "agreements": agree,
        "disagreements": len(rows) - agree,
        "left_total_cost_usd": round(sum(r[left_name]["estimated_cost_usd"] for r in rows), 8),
        "right_total_cost_usd": round(sum(r[right_name]["estimated_cost_usd"] for r in rows), 8),
        "left_avg_latency_s": round(sum(r[left_name]["latency_s"] for r in rows) / n, 4) if rows else 0.0,
        "right_avg_latency_s": round(sum(r[right_name]["latency_s"] for r in rows) / n, 4) if rows else 0.0,
    }
    return {"summary": summary, "rows": rows}


def _print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    left = s["left"]
    right = s["right"]

    table = Table(title="Reasoner comparison", show_lines=True)
    table.add_column("frame", style="cyan")
    table.add_column("detections")
    table.add_column(left)
    table.add_column(right)
    table.add_column("agree")

    for row in report["rows"]:
        dets = ", ".join(row["detections"]) or "—"
        mark = "[green]yes[/]" if row["agree"] else "[red]no[/]"
        table.add_row(
            str(row["frame_id"]),
            dets,
            str(row[left]["command"]),
            str(row[right]["command"]),
            mark,
        )

    console.print(table)

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Scenes", str(s["scenes"]))
    summary.add_row("Agreement", f"{s['agreement_rate'] * 100:.1f}% ({s['agreements']}/{s['scenes']})")
    summary.add_row(f"{left} avg latency", f"{s['left_avg_latency_s']:.3f}s")
    summary.add_row(f"{right} avg latency", f"{s['right_avg_latency_s']:.3f}s")
    summary.add_row(f"{left} est. cost", f"${s['left_total_cost_usd']:.6f}")
    summary.add_row(f"{right} est. cost", f"${s['right_total_cost_usd']:.6f}")
    summary.add_row("Policy", "on" if s["policy_applied"] else "off")
    console.print(Panel(summary, title="Summary", border_style="green"))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Compare two reasoners on the same scenes")
    parser.add_argument(
        "--scenes",
        default="fixtures/scenes/sample.jsonl",
        help="Path to .json / .jsonl scenes",
    )
    parser.add_argument("--left", default="mock", choices=["mock", "bedrock"])
    parser.add_argument("--right", default="bedrock", choices=["mock", "bedrock"])
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Disable safety policy for both reasoners",
    )
    parser.add_argument(
        "--out",
        default="logs/eval-report.json",
        help="Write full JSON report (empty string to skip)",
    )
    args = parser.parse_args()

    try:
        scenes = load_scenes(args.scenes)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1) from exc

    if not scenes:
        console.print("[red]No scenes loaded.[/]")
        raise SystemExit(1)

    console.print(
        Panel.fit(
            f"[bold]Reasoner eval[/]\n{args.left} vs {args.right} · {len(scenes)} scene(s)\n{args.scenes}",
            border_style="blue",
        )
    )

    left = _make_reasoner(args.left)
    right = _make_reasoner(args.right)

    report = compare_scenes(
        scenes,
        left,
        right,
        left_name=args.left,
        right_name=args.right,
        apply_policy=not args.no_policy,
    )
    _print_report(report)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        console.print(f"[dim]Report written to {path}[/]")


if __name__ == "__main__":
    main()
