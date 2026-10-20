from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from adp import cloud_monitor


@pytest.fixture(autouse=True)
def no_real_cloud_access(monkeypatch, tmp_path):
    from adp import drive_evidence
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cloud_monitor, "_kaggle_cli", lambda: None)
    monkeypatch.setattr(cloud_monitor.subprocess, "run", lambda *_a, **_k: pytest.fail("Unexpected cloud call"))
    monkeypatch.setattr(drive_evidence, "configured_evidence", lambda: {})


def test_unconfigured_connections_are_concrete_and_do_not_claim_live_progress():
    result = cloud_monitor.check_cloud_connections()
    assert set(result) == {"colab", "kaggle"}
    assert all(row["status"] == "unconfigured" and row["phase"] == "connection_required" for row in result.values())
    assert "free-runtime API" in result["colab"]["message"]
    assert "Browser sign-in" in result["kaggle"]["message"]


def test_credential_probe_uses_only_presence_and_never_reads_secret_bytes(monkeypatch, tmp_path):
    directory = tmp_path / ".kaggle"
    directory.mkdir()
    (directory / "access_token").write_bytes(b"DO-NOT-READ")
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("Credential bytes must not be inspected"))
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("Credential bytes must not be inspected"))
    assert cloud_monitor._kaggle_credentials_present()


def test_official_oauth_credentials_presence_is_supported_without_reading_it(tmp_path):
    directory = tmp_path / ".kaggle"
    directory.mkdir()
    (directory / "credentials.json").write_bytes(b"DO-NOT-READ-OAUTH")
    assert cloud_monitor._kaggle_credentials_present()


@pytest.mark.parametrize("saved", ["queued", "running", "complete", "error", "canceled"])
def test_official_kaggle_status_is_read_only_and_labels_saved_run_scope(monkeypatch, saved):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "SECRET-NOT-IN-ARGV")
    monkeypatch.setattr(cloud_monitor, "_kaggle_cli", lambda: "official-kaggle")
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=f'{cloud_monitor.KAGGLE_KERNEL} has status "{saved}"', stderr="")
    monkeypatch.setattr(cloud_monitor.subprocess, "run", run)
    row = cloud_monitor.check_cloud_connections()["kaggle"]
    assert calls[0][0] == ["official-kaggle", "kernels", "status", cloud_monitor.KAGGLE_KERNEL]
    assert calls[0][1]["timeout"] == 30
    assert row["status"] == "checked" and row["saved_run_status"] == saved
    assert "interactive editor" in row["message"] and row["scope"] == "latest_saved_run_only"
    assert "SECRET" not in str(row)


@pytest.mark.parametrize("failure", ["exit", "timeout", "unrecognized", "oversized"])
def test_failed_kaggle_probes_never_expose_stderr_or_token(monkeypatch, failure):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "SECRET-TOKEN")
    monkeypatch.setattr(cloud_monitor, "_kaggle_cli", lambda: "official-kaggle")
    def run(*_args, **_options):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("SECRET-ARGV", 30)
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
            stdout="x" * (cloud_monitor.MAX_RESPONSE_BYTES + 1) if failure == "oversized" else "unrecognized",
            stderr="SECRET-TOKEN")
    monkeypatch.setattr(cloud_monitor.subprocess, "run", run)
    row = cloud_monitor.check_cloud_connections()["kaggle"]
    assert row["status"] == "unavailable" and "SECRET" not in str(row)


def test_configured_drive_evidence_is_separate_saved_source_not_live_runtime(monkeypatch):
    from adp import drive_evidence
    saved = {"status":"checked", "phase":"saved_notebook_status", "scope":"saved_notebook_heartbeat",
        "source_observed_at":"2026-10-19T00:00:00+00:00", "completed_jobs":12, "completed_epochs":240,
        "message":"Saved evidence, not live GPU proof."}
    monkeypatch.setattr(drive_evidence, "configured_evidence", lambda: {"colab":saved})
    result = cloud_monitor.check_cloud_connections()
    assert result["colab"] == saved and result["kaggle"]["status"] == "unconfigured"


def test_denied_status_falls_back_to_exact_owned_saved_notebook_metadata(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "SECRET-TOKEN")
    monkeypatch.setattr(cloud_monitor, "_kaggle_cli", lambda: "official-kaggle")
    calls = []
    def run(command, **options):
        calls.append(command)
        if command[2] == "status":
            return SimpleNamespace(returncode=1, stdout="", stderr="403 Forbidden SECRET-TOKEN")
        return SimpleNamespace(returncode=0, stdout='[{"ref":"adimaitre/interaction-lab-gpu-qualification","title":"Interaction Lab — GPU qualification","author":"Adi Maitre","lastRunTime":"2026-10-20T12:10:49.510000","totalVotes":0}]', stderr="")
    monkeypatch.setattr(cloud_monitor.subprocess, "run", run)
    row = cloud_monitor.check_cloud_connections()["kaggle"]
    assert calls == [["official-kaggle", "kernels", "status", cloud_monitor.KAGGLE_KERNEL],
        ["official-kaggle", "kernels", "list", "--mine", "--search", "interaction-lab-gpu-qualification", "--page-size", "5", "--format", "json"]]
    assert row["status"] == "checked" and row["phase"] == "saved_notebook_visible"
    assert row["scope"] == "saved_notebook_metadata" and "saved_run_status" not in row
    assert "Execution status is unavailable" in row["message"] and "SECRET" not in str(row)


@pytest.mark.parametrize("response", ["[]", '[{"ref":"someone/other"}]', '{"ref":"adimaitre/interaction-lab-gpu-qualification"}',
    '[{"ref":"adimaitre/interaction-lab-gpu-qualification"},{"ref":"adimaitre/interaction-lab-gpu-qualification"}]', "not-json"])
def test_metadata_fallback_refuses_wrong_missing_duplicate_or_invalid_records(monkeypatch, response):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "SECRET")
    monkeypatch.setattr(cloud_monitor, "_kaggle_cli", lambda: "official-kaggle")
    monkeypatch.setattr(cloud_monitor.subprocess, "run", lambda command, **options:
        SimpleNamespace(returncode=1, stdout="", stderr="403 Forbidden") if command[2] == "status" else
        SimpleNamespace(returncode=0, stdout=response, stderr=""))
    assert cloud_monitor.check_cloud_connections()["kaggle"]["status"] == "unavailable"
