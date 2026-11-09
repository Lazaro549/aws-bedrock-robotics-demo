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

```
[ Camera / Dashboard / Simulator ]
              ↓
        Vision Adapter  →  Scene Context (JSON)
              ↓
        Bedrock Claude  →  Robot Command + Reasoning Trace
              ↓
        Robot Simulator →  Execution Result + Metrics
```

---

## Two ways to run

### A) Local CLI (simulated vision)

```bash
git clone https://github.com/Lazaro549/aws-bedrock-robotics-demo.git
cd aws-bedrock-robotics-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline --mode mock
python -m src.pipeline --mode bedrock
```

### B) Live bridge with robot-vision-dashboard (real COCO-SSD)

```bash
# Terminal 1 — bridge
python -m src.bridge_server --mode mock --port 8787
# or: python -m src.bridge_server --mode bedrock --port 8787

# Terminal 2 — dashboard
cd ../robot-vision-dashboard
python -m http.server 8000
# open http://127.0.0.1:8000 → enable AI Bridge → Start camera
```

Full integration guide: [docs/integration-dashboard.md](docs/integration-dashboard.md)

---

## Quick start (CLI)

```bash
cp .env.example .env   # only needed for --mode bedrock
python -m src.pipeline --mode mock
python -m src.pipeline --mode bedrock --loop --interval 5
```

Example output:

```
[VISION] Detected: person (0.92), cup (0.87), table (0.95)
[REASON] Claude decided: APPROACH(target_label="cup")
         Confidence: 0.81 | Tokens: 312 in / 148 out | Latency: 1.42s
[ROBOT]  Executing: Approaching target 'cup'.
[METRIC] Pipeline total: 1.67s | Success: true
```

---

## Project structure

```
aws-bedrock-robotics-demo/
├── src/
│   ├── vision/           # SceneContext + simulated vision
│   ├── reasoning/        # Mock + Bedrock Claude clients
│   ├── robot/            # Command schema + simulator
│   ├── metrics/          # Latency / tokens / success
│   ├── pipeline.py       # CLI orchestrator
│   └── bridge_server.py  # FastAPI bridge for the dashboard
├── docs/
│   ├── architecture.md
│   └── integration-dashboard.md
├── tests/
├── .env.example
└── requirements.txt
```

---

## Design principles

| Principle | Implementation |
|-----------|----------------|
| **Measurable** | Every run emits structured metrics (latency, tokens, confidence, success) |
| **Swappable** | Vision and robot layers are adapters |
| **Safe by default** | Mock mode works without AWS credentials |
| **Observable** | Full reasoning trace kept for every decision |
| **Integrable** | Same models used by CLI and HTTP bridge |

---

## Related projects

- [robot-vision-dashboard](https://github.com/Lazaro549/robot-vision-dashboard) — Real-time browser computer vision (COCO-SSD)
- [ai-evaluation-platform](https://github.com/Lazaro549/ai-evaluation-platform) — LLM evaluation harness
- [mcp-advanced-starter](https://github.com/Lazaro549/mcp-advanced-starter) — Advanced MCP server patterns
- [AWS-AI-Generative](https://github.com/Lazaro549/AWS-AI-Generative) — Generative AI patterns on AWS

---

## License

MIT © Lázaro Gómez Vitolo

**Built with a bias toward practical, measurable systems.**  
Mar del Plata, Argentina · [Portfolio](https://lazaro549.github.io/Portfolio/)
