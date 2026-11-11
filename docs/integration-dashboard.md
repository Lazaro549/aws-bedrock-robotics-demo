# Integration: robot-vision-dashboard ↔ aws-bedrock-robotics-demo

This bridge closes the loop between **real browser computer vision** and **LLM robot reasoning**.

```
[ Webcam / Video ]
        ↓
 robot-vision-dashboard   (COCO-SSD + tracking in the browser)
        ↓  POST /scene  (JSON detections + telemetry)
 bridge_server.py         (FastAPI on :8787)
        ↓
 MockReasoner | Bedrock Claude
        ↓
 RobotSimulator
        ↓  JSON decision (+ estimated_cost_usd)
 dashboard UI             (command + reasoning shown live)
```

## 1. Start the bridge

```bash
cd aws-bedrock-robotics-demo
pip install -e ".[dev]"

bedrock-robotics-bridge --mode mock --port 8787
# or: bedrock-robotics-bridge --mode bedrock --port 8787
```

Health check:

```bash
curl -s http://127.0.0.1:8787/health | python -m json.tool
```

## 2. Dry-run without the dashboard

```bash
curl -s -X POST http://127.0.0.1:8787/scene \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": "curl-demo-1",
    "detections": [
      {"label": "cup", "confidence": 0.93, "bbox": [0.1, 0.1, 0.3, 0.4], "track_id": 1}
    ],
    "robot_state": {"battery": 90}
  }' | python -m json.tool
```

## 3. Run the dashboard with bridge enabled

```bash
cd robot-vision-dashboard
python -m http.server 8000
```

In the dashboard:

1. Enable **AI Bridge**
2. Endpoint `http://127.0.0.1:8787/scene` (or the Codespaces public URL + `/scene`)
3. Start camera or load a video
4. Watch decisions appear in the decision card / log

## 4. Payload contract

Dashboard → bridge (`POST /scene`): detections, optional bboxes, robot_state, frame size.

Bridge → dashboard: `command`, `params`, `confidence`, `reasoning`, `estimated_cost_usd`, robot outcome, latencies.

## Design notes

- CORS is open for local portfolio demos.
- Bboxes may be pixels or normalized; the bridge normalizes when `frame_width/height` are present.
- Throttling lives on the dashboard side (~1.5s) so Bedrock is not flooded.
- Cost fields are approximate (token × configured $/1M); override via `.env`.
