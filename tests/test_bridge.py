"""Bridge HTTP tests (no AWS credentials required)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Force mock mode before importing the app module state used at startup
os.environ["PIPELINE_MODE"] = "mock"

from src.bridge_server import (  # noqa: E402
    IncomingDetection,
    IncomingScene,
    _normalize_bbox,
    _to_scene,
    app,
)
from src.reasoning.bedrock_client import MockReasoner


@pytest.fixture()
def client(monkeypatch):
    """TestClient with MockReasoner already initialized."""
    import src.bridge_server as bridge

    monkeypatch.setattr(bridge, "_reasoner", MockReasoner())
    monkeypatch.setattr(bridge, "_mode", "mock")
    monkeypatch.setattr(bridge, "_last_scene", None)
    monkeypatch.setattr(bridge, "_last_decision", None)
    monkeypatch.setattr(bridge, "_last_result", None)

    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "mock"
    assert body["has_last_scene"] is False
    assert body["has_last_decision"] is False


def test_last_before_scene_is_404(client):
    response = client.get("/last")
    assert response.status_code == 404


def test_scene_returns_decision(client):
    payload = {
        "frame_id": "test-001",
        "frame_width": 1280,
        "frame_height": 720,
        "detections": [
            {
                "label": "cup",
                "confidence": 0.92,
                "bbox": [100, 80, 220, 200],
                "track_id": 3,
            }
        ],
        "robot_state": {"battery": 90, "source": "test"},
        "environment": {"source": "pytest", "model": "coco-ssd"},
    }
    response = client.post("/scene", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "command" in body
    assert body["command"] in {
        "MOVE_TO",
        "APPROACH",
        "PICK",
        "PLACE",
        "INSPECT",
        "STOP",
        "WAIT",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["reasoning"], str)
    assert "robot_success" in body
    assert "total_latency_s" in body


def test_scene_then_last(client):
    payload = {
        "detections": [{"label": "person", "confidence": 0.88}],
    }
    post = client.post("/scene", json=payload)
    assert post.status_code == 200

    last = client.get("/last")
    assert last.status_code == 200
    body = last.json()
    assert body["decision"] is not None
    assert body["result"] is not None


def test_scene_empty_detections(client):
    response = client.post("/scene", json={"detections": []})
    assert response.status_code == 200
    body = response.json()
    assert body["command"] in {
        "MOVE_TO",
        "APPROACH",
        "PICK",
        "PLACE",
        "INSPECT",
        "STOP",
        "WAIT",
    }


def test_scene_invalid_confidence_rejected(client):
    payload = {
        "detections": [{"label": "cup", "confidence": 1.5}],
    }
    response = client.post("/scene", json=payload)
    assert response.status_code == 422


def test_normalize_bbox_pixels():
    result = _normalize_bbox([100, 50, 300, 250], 1000, 500)
    assert result is not None
    x1, y1, x2, y2 = result
    assert abs(x1 - 0.1) < 1e-3
    assert abs(y1 - 0.1) < 1e-3
    assert abs(x2 - 0.3) < 1e-3
    assert abs(y2 - 0.5) < 1e-3


def test_normalize_bbox_already_normalized():
    result = _normalize_bbox([0.1, 0.2, 0.4, 0.5], 1280, 720)
    assert result == (0.1, 0.2, 0.4, 0.5)


def test_normalize_bbox_none():
    assert _normalize_bbox(None, 100, 100) is None
    assert _normalize_bbox([1, 2], 100, 100) is None


def test_to_scene_preserves_track_id():
    payload = IncomingScene(
        frame_id="f1",
        detections=[
            IncomingDetection(label="bottle", confidence=0.8, track_id=7, bbox=[0.1, 0.1, 0.2, 0.3])
        ],
    )
    scene = _to_scene(payload)
    assert scene.frame_id == "f1"
    assert len(scene.detections) == 1
    assert scene.detections[0].attributes.get("track_id") == 7
