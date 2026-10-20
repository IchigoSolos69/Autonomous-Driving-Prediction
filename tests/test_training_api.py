"""Training status is read-only and never accepts a filesystem root over HTTP."""
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from adp.api import create_app


def test_unconfigured_training_screen_is_actionable(tmp_path):
    client = TestClient(create_app(tmp_path, device="cpu", frontend_dir=tmp_path))
    result = client.get("/api/training")
    assert result.status_code == 200
    assert result.json()["studies"] == []
    assert "--study-root" in result.json()["message"]


def test_progress_uses_only_configured_provider_and_rejects_paths_and_writes(tmp_path, monkeypatch):
    snapshot = {"version": 1, "generated_at": "2026-10-19T12:00:00Z", "poll_seconds": 5,
                "studies": [{"id": "original", "status": "running"}]}
    provider = SimpleNamespace(snapshot=Mock(return_value=snapshot))
    client = TestClient(create_app(tmp_path, device="cpu", frontend_dir=tmp_path, training_status=provider))
    monkeypatch.setattr("adp.api.model_inputs", lambda *args: (_ for _ in ()).throw(AssertionError("No training-screen inference")))
    assert client.get("/api/training").json() == snapshot
    provider.snapshot.assert_called_once_with()
    assert client.get("/api/training?root=C:/private").status_code == 422
    assert client.get("/api/training?study=original").status_code == 422
    assert client.post("/api/training", json={"root": "C:/private"}).status_code == 405
    assert client.delete("/api/training").status_code == 405
    provider.snapshot.assert_called_once()


def test_temporary_training_file_error_is_retryable(tmp_path):
    provider = SimpleNamespace(snapshot=Mock(side_effect=PermissionError("checkpoint rename")))
    client = TestClient(create_app(tmp_path, device="cpu", frontend_dir=tmp_path, training_status=provider))
    result = client.get("/api/training")
    assert result.status_code == 503
    assert "Retry" in result.json()["detail"]
