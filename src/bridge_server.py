"""
Vision bridge for robot-vision-dashboard.

Receives real COCO-SSD detections from the browser dashboard,
runs the reasoning + robot layers, and returns the decision.

Usage:
    python -m src.bridge_server
    bedrock-robotics-bridge --mode mock --port 8787

Dashboard posts to: POST http://127.0.0.1:8787/scene
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console

from src.metrics.collector import MetricsCollector, PipelineMetrics
from src.metrics.cost import estimate_cost_usd
from src.metrics.scorecard import build_scorecard, print_scorecard, write_scorecard_json
from src.reasoning.bedrock_client import BedrockReasoner, MockReasoner, RobotDecision
from src.robot.commands import RobotCommand
from src.robot.simulator import RobotSimulator
from src.vision.adapter import Detection, SceneContext

load_dotenv()
console = Console()

_reasoner: MockReasoner | BedrockReasoner | None = None
_robot = RobotSimulator()
_metrics = MetricsCollector(log_path="logs/bridge.jsonl")
_last_scene: SceneContext | None = None
_last_decision: RobotDecision | None = None
_last_result: dict[str, Any] | None = None
_mode = "mock"


def _init_reasoner() -> None:
    global _reasoner, _mode
    _mode = os.getenv("PIPELINE_MODE", "mock")
    if _mode == "bedrock":
        console.print("[bold green]Bridge reasoner: Amazon Bedrock (Claude)[/]")
        _reasoner = BedrockReasoner()
    else:
        console.print("[bold cyan]Bridge reasoner: MockReasoner[/]")
        _reasoner = MockReasoner()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_reasoner()
    yield
    # Shutdown: emit session scorecard
    card = build_scorecard(_metrics, session_label=f"bridge-{_mode}")
    print_scorecard(card, console=console)
    if card.get("runs", 0) > 0:
        path = write_scorecard_json(card, "logs/scorecard-bridge.json")
        console.print(f"[dim]Scorecard written to {path}[/]")


app = FastAPI(
    title="aws-bedrock-robotics-demo bridge",
    description="Receives vision scenes from robot-vision-dashboard and returns robot decisions.",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IncomingDetection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = None
    track_id: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class IncomingScene(BaseModel):
    timestamp: str | None = None
    frame_id: str | None = None
    detections: list[IncomingDetection] = Field(default_factory=list)
    robot_state: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    frame_width: float | None = None
    frame_height: float | None = None


class DecisionResponse(BaseModel):
    command: str
    params: dict[str, Any]
    confidence: float
    reasoning: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    model_id: str
    robot_success: bool
    robot_message: str
    robot_state: dict[str, Any]
    total_latency_s: float


def _normalize_bbox(
    bbox: list[float] | None,
    frame_width: float | None,
    frame_height: float | None,
) -> tuple[float, float, float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if frame_width and frame_height and max(x1, y1, x2, y2) > 1.5:
        x1, x2 = x1 / frame_width, x2 / frame_width
        y1, y2 = y1 / frame_height, y2 / frame_height
    return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))


def _to_scene(payload: IncomingScene) -> SceneContext:
    detections = []
    for d in payload.detections:
        attrs = dict(d.attributes)
        if d.track_id is not None:
            attrs["track_id"] = d.track_id
        detections.append(
            Detection(
                label=d.label,
                confidence=d.confidence,
                bbox=_normalize_bbox(d.bbox, payload.frame_width, payload.frame_height),
                attributes=attrs,
            )
        )

    return SceneContext(
        timestamp=payload.timestamp or datetime.now(timezone.utc).isoformat(),
        frame_id=payload.frame_id or f"dash-{int(time.time() * 1000)}",
        detections=detections,
        robot_state=payload.robot_state or {"source": "robot-vision-dashboard"},
        environment=payload.environment or {"source": "browser", "model": "coco-ssd"},
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": _mode,
        "has_last_scene": _last_scene is not None,
        "has_last_decision": _last_decision is not None,
        "runs": len(_metrics.runs),
    }


@app.get("/scorecard")
def scorecard() -> dict:
    """Live session scorecard (success rate, p50/p95 latency, tokens, cost)."""
    return build_scorecard(_metrics, session_label=f"bridge-{_mode}")


@app.get("/last")
def last() -> dict:
    if not _last_decision or not _last_result:
        raise HTTPException(status_code=404, detail="No decision yet. POST /scene first.")
    return {
        "scene": _last_scene.model_dump() if _last_scene else None,
        "decision": _last_decision.model_dump(),
        "result": _last_result,
    }


@app.post("/scene", response_model=DecisionResponse)
def process_scene(payload: IncomingScene) -> DecisionResponse:
    global _last_scene, _last_decision, _last_result

    if _reasoner is None:
        raise HTTPException(status_code=503, detail="Reasoner not initialized")

    t0 = time.perf_counter()
    scene = _to_scene(payload)
    _last_scene = scene

    console.print(
        f"[cyan][VISION][/] {len(scene.detections)} objects from dashboard — "
        + ", ".join(f"{d.label}({d.confidence:.2f})" for d in scene.detections[:6])
    )

    decision = _reasoner.decide(scene)
    _last_decision = decision
    cost = estimate_cost_usd(decision.input_tokens, decision.output_tokens)

    cmd = RobotCommand(
        name=decision.command,
        params=decision.params,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
    )
    result = _robot.execute(cmd)
    total = time.perf_counter() - t0

    _last_result = {
        "success": result.success,
        "message": result.message,
        "duration_s": result.duration_s,
        "final_state": result.final_state,
        "estimated_cost_usd": cost,
    }

    console.print(
        f"[magenta][REASON][/] {decision.command} {decision.params} "
        f"(conf={decision.confidence:.2f}, cost≈${cost:.6f}, {decision.latency_s:.2f}s)"
    )
    console.print(f"         {decision.reasoning}")
    color = "green" if result.success else "red"
    console.print(f"[{color}][ROBOT][/] {result.message}")

    _metrics.record(
        PipelineMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode=f"bridge-{_mode}",
            vision_latency_s=0.0,
            reasoning_latency_s=decision.latency_s,
            robot_latency_s=result.duration_s,
            total_latency_s=round(total, 4),
            command=decision.command,
            confidence=decision.confidence,
            success=result.success,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            estimated_cost_usd=cost,
            model_id=decision.model_id,
            notes=result.message,
        )
    )

    return DecisionResponse(
        command=decision.command,
        params=decision.params,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        latency_s=decision.latency_s,
        input_tokens=decision.input_tokens,
        output_tokens=decision.output_tokens,
        estimated_cost_usd=cost,
        model_id=decision.model_id,
        robot_success=result.success,
        robot_message=result.message,
        robot_state=result.final_state,
        total_latency_s=round(total, 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision bridge for robot-vision-dashboard")
    parser.add_argument("--mode", choices=["mock", "bedrock"], default=os.getenv("PIPELINE_MODE", "mock"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    os.environ["PIPELINE_MODE"] = args.mode

    console.print(
        f"[bold]Bridge listening on http://{args.host}:{args.port}[/]  mode={args.mode}"
    )
    console.print("POST /scene  |  GET /health  |  GET /last  |  GET /scorecard")

    import uvicorn

    uvicorn.run(
        "src.bridge_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
