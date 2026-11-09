# Architecture

## High-level data flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Vision Adapter  │     │    Reasoner      │     │  Robot Adapter   │
│  (sim or real)   │────▶│  (Mock/Bedrock)  │────▶│  (sim or ROS)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
   SceneContext              RobotDecision            ExecutionResult
         │                        │                        │
         └────────────────────────䅋────────────────────────┘
                                  │
                                  ▼
                         MetricsCollector
```

## Layer responsibilities

| Layer | Responsibility | Swap points |
|-------|----------------|-------------|
| **Vision** | Produce a structured `SceneContext` (detections + robot/env state) | Replace `SimulatedVision` with a real detector or ROS subscriber |
| **Reasoning** | Turn scene into a validated robot command + confidence + trace | `MockReasoner` ↔ `BedrockReasoner` (Claude) |
| **Robot** | Execute (or simulate) the command and return success + new state | Replace `RobotSimulator` with a ROS 2 publisher or serial driver |
| **Metrics** | Capture latencies, tokens, success for every run | Write to JSONL, CloudWatch, or your evaluation platform |

## Why this design

- **Testable without AWS** — mock mode exercises the full loop.
- **Measurable by default** — every decision carries latency and token counts.
- **Hardware-agnostic** — the same decision JSON can drive a simulator today and a real arm tomorrow.
- **Aligned with evaluation practice** — the output of this pipeline can be fed into an AI evaluation harness.

## Future AWS deployment sketch

```
API Gateway  →  Lambda (pipeline)  →  Bedrock (Claude)
                      │
                      └→  optional: IoT Core / ROS bridge for real robots
```

The local Python pipeline already mirrors the same stages, so moving to Lambda is mostly packaging and IAM.
