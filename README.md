# aws-bedrock-robotics-demo

**Vision → Reasoning → Action**  
End-to-end pipeline that connects computer vision, Amazon Bedrock (Claude), and a simulated robot.

> *I build AI and robotics systems, then measure how they behave.*

[![CI](https://github.com/Lazaro549/aws-bedrock-robotics-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/Lazaro549/aws-bedrock-robotics-demo/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

---

## What this does

1. **Vision** — Receives (or simulates) object detections from a camera stream.
2. **Reasoning** — Sends structured scene context to Claude on Amazon Bedrock.
3. **Action** — Generates a robot command (move, pick, stop, inspect…).
4. **Measure** — Logs latency, token usage, **estimated cost (USD)**, confidence and execution outcome.

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

## Install

```bash
git clone https://github.com/Lazaro549/aws-bedrock-robotics-demo.git
cd aws-bedrock-robotics-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Entry points after install:

```bash
bedrock-robotics-pipeline --mode mock
bedrock-robotics-bridge --mode mock --port 8787
```

(You can still use `python -m src.pipeline` / `python -m src.bridge_server`.)

---

## Two ways to run

### A) Local CLI (simulated vision)

```bash
bedrock-robotics-pipeline --mode mock
bedrock-robotics-pipeline --mode bedrock --loop --interval 5
```

### B) Live bridge with robot-vision-dashboard (real COCO-SSD)

```bash
# Terminal 1 — bridge
bedrock-robotics-bridge --mode mock --port 8787

# Terminal 2 — dashboard
cd ../robot-vision-dashboard
python -m http.server 8000
# open http://127.0.0.1:8000 → enable AI Bridge → Start camera
```

Full integration guide: [docs/integration-dashboard.md](docs/integration-dashboard.md)

### C) Bridge dry-run with curl (no dashboard)

With the bridge running on port 8787:

```bash
# Health
curl -s http://127.0.0.1:8787/health | python -m json.tool

# Send a synthetic scene
curl -s -X POST http://127.0.0.1:8787/scene \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": "curl-demo-1",
    "frame_width": 1280,
    "frame_height": 720,
    "detections": [
      {"label": "cup", "confidence": 0.93, "bbox": [120, 80, 260, 220], "track_id": 1},
      {"label": "person", "confidence": 0.71, "bbox": [400, 100, 700, 600], "track_id": 2}
    ],
    "robot_state": {"battery": 88, "source": "curl"},
    "environment": {"source": "cli", "model": "synthetic"}
  }' | python -m json.tool
```

Expected fields in the response: `command`, `params`, `confidence`, `reasoning`, `estimated_cost_usd`, `robot_success`, `total_latency_s`.

---

## Cost estimates

Each Bedrock call records an **approximate** USD cost from token counts:

- Defaults: $3 / 1M input tokens, $15 / 1M output tokens (Claude-class on Bedrock)
- Override with `BEDROCK_INPUT_PRICE_PER_1M` and `BEDROCK_OUTPUT_PRICE_PER_1M` in `.env`
- Mock mode reports `$0` (no model tokens)

These numbers are for engineering visibility in demos — not a substitute for AWS Cost Explorer.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on Python 3.11 and 3.12.

---

## Project structure

```
aws-bedrock-robotics-demo/
├── src/
│   ├── vision/
│   ├── reasoning/
│   ├── robot/
│   ├── metrics/          # collector + cost estimates
│   ├── pipeline.py
│   └── bridge_server.py  # FastAPI + lifespan
├── tests/
├── docs/
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Design principles

| Principle | Implementation |
|-----------|----------------|
| **Measurable** | Latency, tokens, estimated cost, confidence, success |
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
