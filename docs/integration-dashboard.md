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
        ↓  JSON decision
 dashboard UI             (command + reasoning shown live)
```

## 1. Start the bridge

```bash
cd aws-bedrock-robotics-demo
source .venv/bin/activate
pip install -r requirements.txt

# Mock (no AWS)
python -m src.bridge_server --mode mock --port 8787

# Real Bedrock
python -m src.bridge_server --mode bedrock --port 8787
```

Health check:

```bash
curl http://127.0.0.1:8787/health
```

## 2. Run the dashboard with bridge enabled

```bash
cd robot-vision-dashboard
python -m http.server 8000
# open http://127.0.0.1:8000
```

In the dashboard:

1. Enable **AI Bridge**
2. Confirm endpoint `http://127.0.0.1:8787/scene`
3. Start camera or load a video
4. Watch detections stream to the bridge and decisions appear in the log / decision card

## 3. Payload contract

Dashboard → bridge (`POST /scene`):

```json
{
  "timestamp": "2026-11-09T04:00:00Z",
  "frame_id": "dash-1731",
  "frame_width": 1280,
  "frame_height": 720,
  "detections": [
    {
      "label": "cup",
      "confidence": 0.91,
      "bbox": [120, 80, 220, 200],
      "track_id": 3
    }
  ],
  "robot_state": {
    "battery": 90,
    "odometry_m": 12.6,
    "network": "Connected"
  },
  "environment": {
    "source": "browser",
    "model": "coco-ssd"
  }
}
```

Bridge → dashboard:

```json
{
  "command": "APPROACH",
  "params": { "target_label": "cup" },
  "confidence": 0.85,
  "reasoning": "High-confidence cup detected...",
  "latency_s": 1.42,
  "robot_success": true,
  "robot_message": "Approaching target 'cup'.",
  "total_latency_s": 1.45
}
```

## Design notes

- CORS is open for local portfolio demos.
- Bboxes may be pixels or normalized; the bridge normalizes when `frame_width/height` are present.
- Throttling lives on the dashboard side (default ~1.5s) so Bedrock is not flooded.
- The same `SceneContext` / `RobotDecision` models used by the CLI pipeline are reused here.
