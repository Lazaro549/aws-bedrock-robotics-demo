# aws-bedrock-robotics-demo

**Vision → Reasoning → Action**  
End-to-end pipeline that connects computer vision, Amazon Bedrock (Claude), and a simulated robot.

> *I build AI and robotics systems, then measure how they behave.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

---

## What this does

1. **Vision** — Receives (or simulates) object detections from a camera stream.
2. **Reasoning** — Sends structured scene context to Claude on Amazon Bedrock.
3. **Action** — Generates a robot command (move, pick, stop, inspect…).
4. **Measure** — Logs latency, token usage, decision confidence and execution outcome.

The whole loop is designed to be **measurable** and easy to extend to real hardware or a real CV model.

```
[ Camera / Simulator ]
         ↓
   Vision Adapter  →  Scene Context (JSON)
         ↓
   Bedrock Claude  →  Robot Command + Reasoning Trace
         ↓
   Robot Simulator →  Execution Result + Metrics
```

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/Lazaro549/aws-bedrock-robotics-demo.git
cd aws-bedrock-robotics-demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your AWS credentials / region / model ID
```

You need:
- AWS credentials with `bedrock:InvokeModel` permission
- Access to a Claude model on Bedrock (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0` or newer)

### 3. Run the pipeline (local simulation)

```bash
# Fully simulated (no AWS call) — useful for testing the loop
python -m src.pipeline --mode mock

# Real Bedrock call + simulated robot
python -m src.pipeline --mode bedrock

# Continuous loop (every N seconds)
python -m src.pipeline --mode bedrock --loop --interval 5
```

Example output:

```
[VISION] Detected: person (0.92), cup (0.87), table (0.95)
[REASON] Claude decided: approach_and_inspect(target="cup")
         Confidence: 0.81 | Tokens: 312 in / 148 out | Latency: 1.42s
[ROBOT]  Executing: MOVE_TO (x=0.45, y=0.30) → PICK
[METRIC] Pipeline total: 1.67s | Success: true
```

---

## Project structure

```
aws-bedrock-robotics-demo/
├── src/
│   ├── vision/
│   │   ├── adapter.py          # Interface for real or simulated detections
│   │   └── simulator.py        # Generates realistic scene contexts
│   ├── reasoning/
│   │   ├── bedrock_client.py   # Amazon Bedrock (Claude) client
│   │   └── prompts.py          # System + user prompt templates
│   ├── robot/
│   │   ├── commands.py         # Command schema & validation
│   │   └── simulator.py        # Executes commands and returns outcomes
│   ├── metrics/
│   │   └── collector.py        # Latency, tokens, success rate
│   └── pipeline.py             # Orchestrator
├── tests/
├── docs/
│   └── architecture.md
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design principles

| Principle | Implementation |
|-----------|----------------|
| **Measurable** | Every run emits structured metrics (latency, tokens, confidence, success) |
| **Swappable** | Vision and robot layers are adapters — replace with real camera / ROS 2 / hardware |
| **Safe by default** | Mock mode works without AWS credentials; commands are validated before execution |
| **Observable** | Full reasoning trace is kept so you can inspect *why* the model chose an action |

---

## Extending the demo

### Connect a real vision model
Replace `src/vision/simulator.py` with a call to your existing `robot-vision-dashboard` detections or a YOLO / COCO-SSD endpoint.

### Connect real hardware or ROS 2
Implement a new adapter in `src/robot/` that publishes commands to a ROS topic or serial interface.

### Deploy on AWS
A minimal serverless sketch (API Gateway + Lambda + Bedrock) can be added under `infra/`. The current local pipeline already mirrors the same data flow.

---

## Metrics you can track

- End-to-end pipeline latency
- Bedrock input / output tokens and cost estimate
- Decision confidence (from the model response)
- Command success rate
- Scene complexity vs. decision time

These numbers are printed after every run and can be exported to JSON/CSV for later analysis.

---

## Related projects

- [robot-vision-dashboard](https://github.com/Lazaro549/robot-vision-dashboard) — Real-time browser computer vision
- [ai-evaluation-platform](https://github.com/Lazaro549/ai-evaluation-platform) — LLM evaluation harness
- [mcp-advanced-starter](https://github.com/Lazaro549/mcp-advanced-starter) — Advanced MCP server patterns
- [AWS-AI-Generative](https://github.com/Lazaro549/AWS-AI-Generative) — Generative AI patterns on AWS

---

## License

MIT © Lázaro Gómez Vitolo

---

**Built with a bias toward practical, measurable systems.**  
Mar del Plata, Argentina · [Portfolio](https://lazaro549.github.io/Portfolio/)
