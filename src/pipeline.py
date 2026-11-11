"""
Main orchestrator: Vision → Reasoning → Policy → Robot → Metrics
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.metrics.collector import MetricsCollector, PipelineMetrics
from src.metrics.cost import estimate_cost_usd
from src.metrics.export_csv import write_scorecard_csv
from src.metrics.scorecard import print_scorecard, write_scorecard_json
from src.reasoning.bedrock_client import BedrockReasoner, MockReasoner
from src.robot.commands import RobotCommand
from src.robot.policy import apply_safety_policy
from src.robot.simulator import RobotSimulator
from src.vision.file_adapter import FileVision
from src.vision.simulator import SimulatedVision

console = Console()


def run_once(
    mode: str,
    vision,
    reasoner,
    robot,
    metrics: MetricsCollector,
    *,
    apply_policy: bool = True,
) -> bool:
    t0 = time.perf_counter()

    t_vis = time.perf_counter()
    try:
        scene = vision.capture()
    except StopIteration:
        console.print("[yellow]No more scenes in file.[/]")
        return False
    vision_latency = time.perf_counter() - t_vis

    console.print(
        f"[bold cyan][VISION][/] {scene.frame_id} · {len(scene.detections)} objects — "
        + ", ".join(f"{d.label}({d.confidence:.2f})" for d in scene.detections)
    )

    t_rea = time.perf_counter()
    decision = reasoner.decide(scene)
    reasoning_latency = decision.latency_s or (time.perf_counter() - t_rea)
    cost = estimate_cost_usd(decision.input_tokens, decision.output_tokens)

    console.print(
        f"[bold magenta][REASON][/] {decision.command} {decision.params}  "
        f"(conf={decision.confidence:.2f}, tokens={decision.input_tokens}/{decision.output_tokens}, "
        f"cost≈${cost:.6f}, latency={reasoning_latency:.2f}s)"
    )
    console.print(f"         {decision.reasoning}")

    policy_applied = False
    policy_rules: list[str] = []
    if apply_policy:
        policy = apply_safety_policy(decision, scene, robot_state=robot.state)
        policy_applied = policy.applied
        policy_rules = policy.rules_fired
        if policy.applied:
            console.print(
                f"[bold yellow][POLICY][/] {policy.original_command} → {policy.decision.command}  "
                f"({', '.join(policy.rules_fired)})"
            )
            decision = policy.decision
    else:
        console.print("[dim][POLICY] disabled[/]")

    cmd = RobotCommand(
        name=decision.command,
        params=decision.params,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
    )
    t_rob = time.perf_counter()
    result = robot.execute(cmd)
    robot_latency = time.perf_counter() - t_rob

    status_color = "green" if result.success else "red"
    console.print(
        f"[bold {status_color}][ROBOT][/] {result.message} "
        f"(exec={result.duration_s:.3f}s)"
    )

    total = time.perf_counter() - t0
    notes = result.message
    if policy_applied:
        notes = f"[policy:{','.join(policy_rules)}] {notes}"

    m = PipelineMetrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode if apply_policy else f"{mode}-nopolicy",
        vision_latency_s=round(vision_latency, 4),
        reasoning_latency_s=round(reasoning_latency, 4),
        robot_latency_s=round(robot_latency, 4),
        total_latency_s=round(total, 4),
        command=decision.command,
        confidence=decision.confidence,
        success=result.success,
        input_tokens=decision.input_tokens,
        output_tokens=decision.output_tokens,
        estimated_cost_usd=cost,
        model_id=decision.model_id,
        notes=notes,
    )
    metrics.record(m)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Total latency", f"{total:.3f}s")
    table.add_row("Est. cost (USD)", f"${cost:.6f}")
    table.add_row("Policy", "applied" if policy_applied else ("pass" if apply_policy else "off"))
    table.add_row("Success", str(result.success))
    console.print(Panel(table, title="[METRIC]", border_style="dim"))
    return True


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Vision → Bedrock → Policy → Robot demo pipeline")
    parser.add_argument("--mode", choices=["mock", "bedrock"], default=os.getenv("PIPELINE_MODE", "mock"))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--no-policy", action="store_true", help="Disable safety policy")
    parser.add_argument("--scorecard", default="logs/scorecard-pipeline.json")
    parser.add_argument("--csv", default="logs/scorecard-pipeline.csv", help="CSV export path (empty to skip)")
    args = parser.parse_args()

    robot = RobotSimulator(seed=args.seed)
    metrics = MetricsCollector(log_path="logs/pipeline.jsonl")
    apply_policy = not args.no_policy

    if args.scenes:
        try:
            vision = FileVision(args.scenes, loop=args.loop)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise SystemExit(1) from exc
        console.print(f"[bold]Vision source:[/] file ({args.scenes}) · {len(vision)} scene(s)")
    else:
        vision = SimulatedVision(seed=args.seed)
        console.print("[bold]Vision source:[/] simulator")

    if args.mode == "bedrock":
        console.print("[bold]Using Amazon Bedrock (Claude)[/]")
        reasoner = BedrockReasoner()
    else:
        console.print("[bold]Using MockReasoner (no AWS calls)[/]")
        reasoner = MockReasoner()

    console.print(
        Panel.fit(
            "[bold]aws-bedrock-robotics-demo[/]\n"
            f"Vision → Reasoning → {'Policy → ' if apply_policy else ''}Action → Measure",
            border_style="green",
        )
    )

    try:
        if args.scenes and not args.loop:
            while run_once(args.mode, vision, reasoner, robot, metrics, apply_policy=apply_policy):
                console.print()
        elif args.loop:
            while True:
                ok = run_once(args.mode, vision, reasoner, robot, metrics, apply_policy=apply_policy)
                if not ok:
                    break
                console.print()
                time.sleep(args.interval)
        else:
            run_once(args.mode, vision, reasoner, robot, metrics, apply_policy=apply_policy)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user[/]")

    label = f"pipeline-{args.mode}"
    if args.scenes:
        label += "-file"
    if not apply_policy:
        label += "-nopolicy"
    card = metrics.scorecard(session_label=label)
    print_scorecard(card, console=console)
    if args.scorecard:
        path = write_scorecard_json(card, args.scorecard)
        console.print(f"[dim]Scorecard JSON → {path}[/]")
    if args.csv:
        path = write_scorecard_csv(card, args.csv)
        console.print(f"[dim]Scorecard CSV  → {path}[/]")


if __name__ == "__main__":
    main()
