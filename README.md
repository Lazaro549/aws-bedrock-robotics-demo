# aws-bedrock-robotics-demo

**Vision → Reasoning → Policy → Action → Measure**

End-to-end robotics-style pipeline: structured scenes in, robot commands out, with safety policy, cost estimates, scorecards, offline replay, and Mock vs Bedrock evaluation.

> *I build AI and robotics systems, then measure how they behave.*

[![CI](https://github.com/Lazaro549/aws-bedrock-robotics-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/Lazaro549/aws-bedrock-robotics-demo/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

---

## Features

| Layer | What you get |
|-------|----------------|
| **Vision** | Simulator, browser dashboard bridge, or JSON/JSONL replay |
| **Reasoning** | Mock (offline) or Claude on Amazon Bedrock |
| **Policy** | Battery, confidence, workspace clamp, person-near-target |
| **Robot** | Lightweight simulator with validated commands |
| **Measure** | Latency p50/p95, tokens, estimated USD cost, scorecard JSON+CSV |
| **Evaluate** | Same scenes on two reasoners (Mock vs Bedrock) |

---

## Install

```bash
git clone https://github.com/Lazaro549/aws-bedrock-robotics-demo.git
cd aws-bedrock-robotics-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Entry points:

```bash
bedrock-robotics-pipeline --mode mock
bedrock-robotics-bridge --mode mock --port 8787
bedrock-robotics-evaluate --scenes fixtures/scenes/sample.jsonl --left mock --right mock
```

Or: `python -m src.pipeline` / `src.bridge_server` / `src.evaluate`.

---

## Quick start

```bash
# Single simulated cycle
python -m src.pipeline --mode mock

# Offline replay (fixtures)
python -m src.pipeline --mode mock --scenes fixtures/scenes/sample.jsonl
python -m src.pipeline --mode mock --scenes fixtures/scenes/policy.jsonl

# Disable safety policy (A/B)
python -m src.pipeline --mode mock --scenes fixtures/scenes/policy.jsonl --no-policy

# Evaluate reasoners (no AWS needed for mock vs mock)
python -m src.evaluate --scenes fixtures/scenes/sample.jsonl --left mock --right mock

# Bridge + curl dry-run
python -m src.bridge_server --mode mock --port 8787
curl -s -X POST http://127.0.0.1:8787/scene \
  -H "Content-Type: application/json" \
  -d '{"detections":[{"label":"cup","confidence":0.93}]}' | python -m json.tool
```

Outputs land under `logs/` (JSONL metrics, scorecards, CSV, recorded scenes).

---

## Docker

```bash
docker compose up --build
# bridge on http://127.0.0.1:8787
```

---

## Safety policy (default on)

- Battery &lt; 15% → `STOP`
- Low decision confidence → `WAIT`
- Weak PICK/APPROACH target → `INSPECT` / `STOP`
- Person near target → `WAIT`
- `MOVE_TO` outside [0,1] → clamp

---

## Project layout

```
src/
  vision/          # adapter, simulator, file/JSONL replay
  reasoning/       # mock + Bedrock clients
  robot/           # commands, simulator, safety policy
  metrics/         # collector, cost, scorecard, CSV export
  pipeline.py
  bridge_server.py
  evaluate.py
fixtures/scenes/   # sample + policy stress scenes
infra/             # minimal SAM skeleton
```

---

## Related

- [robot-vision-dashboard](https://github.com/Lazaro549/robot-vision-dashboard)
- [Portfolio](https://lazaro549.github.io/Portfolio/)

## License

MIT © Lázaro Gómez Vitolo
