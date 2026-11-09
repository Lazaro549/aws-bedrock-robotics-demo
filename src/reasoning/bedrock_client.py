"""Amazon Bedrock (Claude) reasoning client + mock for local testing."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from src.vision.adapter import SceneContext
from .prompts import SYSTEM_PROMPT, build_user_prompt


class RobotDecision(BaseModel):
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    # telemetry filled by the client
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""


class Reasoner(ABC):
    @abstractmethod
    def decide(self, scene: SceneContext) -> RobotDecision:
        ...


class MockReasoner(Reasoner):
    """Deterministic local reasoner — no AWS required."""

    def decide(self, scene: SceneContext) -> RobotDecision:
        start = time.perf_counter()

        labels = [d.label for d in scene.detections]
        confidences = {d.label: d.confidence for d in scene.detections}

        if "cup" in labels and confidences.get("cup", 0) > 0.8:
            decision = RobotDecision(
                command="APPROACH",
                params={"target_label": "cup"},
                confidence=0.85,
                reasoning="High-confidence cup detected. Approaching to prepare a potential pick.",
            )
        elif "person" in labels and confidences.get("person", 0) > 0.7:
            decision = RobotDecision(
                command="WAIT",
                params={"seconds": 2},
                confidence=0.78,
                reasoning="Person nearby. Waiting to avoid collision risk.",
            )
        elif not labels:
            decision = RobotDecision(
                command="STOP",
                params={},
                confidence=0.9,
                reasoning="No objects detected. Stopping for safety.",
            )
        else:
            target = max(scene.detections, key=lambda d: d.confidence)
            decision = RobotDecision(
                command="INSPECT",
                params={"target_label": target.label},
                confidence=0.7,
                reasoning=f"Inspecting highest-confidence object ({target.label}).",
            )

        decision.latency_s = round(time.perf_counter() - start, 4)
        decision.model_id = "mock-reasoner-v1"
        return decision


class BedrockReasoner(Reasoner):
    """Calls Claude on Amazon Bedrock."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        import boto3

        self.model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", max_tokens))
        self.temperature = float(os.getenv("BEDROCK_TEMPERATURE", temperature))

        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def decide(self, scene: SceneContext) -> RobotDecision:
        user_content = build_user_prompt(scene.to_prompt_block())

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_content},
            ],
        }

        start = time.perf_counter()
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        latency = time.perf_counter() - start

        raw = json.loads(response["body"].read())
        text = raw["content"][0]["text"]

        # Robust JSON extraction
        decision_dict = self._parse_json(text)

        usage = raw.get("usage", {})
        return RobotDecision(
            command=decision_dict.get("command", "STOP"),
            params=decision_dict.get("params", {}),
            confidence=float(decision_dict.get("confidence", 0.5)),
            reasoning=decision_dict.get("reasoning", ""),
            latency_s=round(latency, 4),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model_id=self.model_id,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: find first { ... }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {
                "command": "STOP",
                "params": {},
                "confidence": 0.3,
                "reasoning": f"Failed to parse model output: {text[:200]}",
            }
