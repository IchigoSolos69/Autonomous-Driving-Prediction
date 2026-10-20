"""The independent monitor transfers without training and stops only after its cycle."""
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from filelock import FileLock
import pytest

import adp.compute_monitor as monitor
from adp.data import read_json, write_json


@pytest.fixture(autouse=True)
def isolate_cloud_and_disk_checks(monkeypatch):
    from adp import cloud_monitor
    monkeypatch.setattr(cloud_monitor, "check_cloud_connections", lambda: {})
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))


@pytest.fixture
def configured(tmp_path):
    config = tmp_path / "private-server.json"
    value = {"host": "aadi@192.168.29.71", "identityfile": str(tmp_path / "not-opened-key"),
        "remote_python": "/home/aadi/interaction-lab/runtime/bin/python",
        "remote_source": "/home/aadi/interaction-lab/control/src",
        "remote_job": "/home/aadi/interaction-lab/jobs/server-pilot-v1",
        "local_root": str(tmp_path / "results"), "local_staging": str(tmp_path / "incoming")}
    write_json(config, value)
    hosts = tmp_path / "hosts.json"
    stamp = monitor._now()
    rows = [{"id": "local", "gpu": "Local GPU", "cleanup": {"status": "protected"}},
        {"id": "private", "status": "completed", "updated_at": stamp, "observed_at": stamp,
         "training": {"completed_epochs": 1, "planned_epochs": 1, "committed_examples": 1000, "total_examples": 1000},
         "transfer": {"status": "verified", "verified_at": stamp, "local_path": "D:/saved/receipt.json"},
         "cleanup": {"status": "completed", "message": "Owned pilot copies retired"}},
        {"id": "colab", "status": "completed", "message": "Archived cloud evidence"},
        {"id": "kaggle", "status": "completed", "message": "GPUs released"}]
    write_json(hosts, {"version": 1, "hosts": rows})
    return config, hosts, tmp_path / "monitor", rows


def test_cycle_is_ordered_transfer_only_and_cleanup_has_no_forced_timeout(configured, monkeypatch):
    config, hosts, _, rows = configured
    calls = []
    def child(command, *, timeout):
        calls.append((command, timeout))
        value = {"state": "blocked", "reason": "Original study is still running"} if "adp.study_cleanup" in command else {}
        return {"ok": True, "result": value, "returncode": 0}
    monkeypatch.setattr(monitor, "_command", child)
    result = monitor.run_cycle(config, hosts)
    assert len(calls) == 3
    assert calls[0][0][1].endswith("sync-private-server.py") and "--no-resume" in calls[0][0]
    assert calls[1][0][1].endswith("update-compute-hosts.py")
    assert calls[2][0][1:] == ["-m", "adp.study_cleanup", "--archive-completed", "--execute"]
    assert calls[0][1] == monitor.SYNC_TIMEOUT and calls[2][1] is None
    assert result["local_cleanup"]["state"] == "blocked"
    updated = read_json(hosts)["hosts"]
    assert updated[0]["cleanup"] == {"status": "protected", "message": "Original study is still running"}
    assert updated[1:] == rows[1:]


def test_ssh_failure_preserves_verified_history_cloud_rows_and_still_checks_cleanup(configured, monkeypatch):
    config, hosts, _, rows = configured
    calls = []
    def child(command, *, timeout):
        calls.append(command)
        if "adp.study_cleanup" in command:
            return {"ok": True, "result": {"state": "blocked", "reason": "Direction follow-up remains active"}, "returncode": 0}
        return {"ok": False, "error": "SSH connection timed out", "returncode": 255}
    monkeypatch.setattr(monitor, "_command", child)
    result = monitor.run_cycle(config, hosts)
    assert len(calls) == 2 and "private_error" in result
    after = read_json(hosts)["hosts"]
    assert after[1]["status"] == "unavailable"
    assert after[1]["training"] == rows[1]["training"] and after[1]["updated_at"] == rows[1]["updated_at"]
    assert after[1]["transfer"] == rows[1]["transfer"] and after[1]["cleanup"] == rows[1]["cleanup"]
    assert after[2:] == rows[2:]
    assert after[0]["cleanup"]["status"] == "protected"


def test_monitor_runs_immediately_once_and_writes_durable_bounded_status(configured, monkeypatch):
    config, hosts, state, _ = configured
    cycles = []
    monkeypatch.setattr(monitor, "run_cycle", lambda *args, **kwargs: cycles.append(True) or {"ok": True})
    result = monitor.run_monitor(config, state, hosts, once=True)
    assert cycles == [True] and result["cycles_completed"] == 1 and result["state"] == "stopped"
    assert result["interval_seconds"] == 300 and result["storage_reserve_bytes"] == 10 * 1024**3
    assert read_json(state / "status.json")["state"] == "stopped"
    assert (state / "monitor.log").stat().st_size < 1024**2
    assert monitor.monitor_status(state)["running"] is False


def test_single_instance_and_stop_request_never_kill_any_worker(configured, monkeypatch):
    config, hosts, state, _ = configured
    state.mkdir()
    with FileLock(str(state / "monitor.lock"), timeout=0):
        monkeypatch.setattr(monitor, "run_cycle", lambda *_args, **_kwargs: pytest.fail("Duplicate monitor cannot run a cycle"))
        assert monitor.run_monitor(config, state, hosts, once=True)["state"] == "already_running"
        assert monitor.request_stop(state)["state"] == "stop_requested"
        assert read_json(state / "stop.json")["version"] == 1
    assert monitor.request_stop(state)["state"] == "not_running"


def test_stop_waits_for_whole_cycle_and_does_not_wait_for_interval(configured, monkeypatch):
    config, hosts, state, _ = configured
    stages = []
    def cycle(*args, **kwargs):
        stages.append("sync")
        assert monitor.request_stop(state)["state"] == "stop_requested"
        stages.extend(["update", "safe_cleanup_finished"])
        return {"state": "finished"}
    monkeypatch.setattr(monitor, "run_cycle", cycle)
    monkeypatch.setattr(monitor.time, "sleep", lambda _: pytest.fail("Stop must not wait for the interval"))
    result = monitor.run_monitor(config, state, hosts)
    assert stages == ["sync", "update", "safe_cleanup_finished"] and result["cycles_completed"] == 1
    assert result["state"] == "stopped"


def test_transient_cycle_error_does_not_end_the_monitor(configured, monkeypatch):
    config, hosts, state, _ = configured
    calls = []
    def cycle(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("temporary child error")
        monitor.request_stop(state)
        return {"state": "finished"}
    monkeypatch.setattr(monitor, "run_cycle", cycle)
    clock = [0.]
    monkeypatch.setattr(monitor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    result = monitor.run_monitor(config, state, hosts, interval_seconds=60)
    assert len(calls) == 2 and result["cycles_completed"] == 2
    assert "temporary child error" in (state / "monitor.log").read_text()


def test_stale_pid_is_not_reported_as_running(configured, monkeypatch):
    _, _, state, _ = configured
    write_json(state / "status.json", {"pid": 123, "state": "running"})
    monkeypatch.setattr(monitor, "_alive", lambda _: True)
    observed = monitor.monitor_status(state)
    assert observed["running"] is False and observed["pid_stale"] is True and observed["state"] == "stopped"


def test_pythonw_uses_sibling_qualified_python_and_does_not_require_stdio(tmp_path, configured, monkeypatch):
    python = tmp_path / "runtime" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"qualified sibling marker")
    monkeypatch.setattr(monitor.sys, "executable", str(python.with_name("pythonw.exe")))
    assert monitor._python() == str(python)
    config, hosts, state, _ = configured
    monkeypatch.setattr(monitor, "run_cycle", lambda *args, **kwargs: {"state": "finished"})
    with monkeypatch.context() as silent:
        silent.setattr(monitor.sys, "stdout", None)
        silent.setattr(monitor.sys, "stderr", None)
        assert monitor.main(["--config", str(config), "--hosts", str(hosts), "--state-dir", str(state), "--once"]) == 0


def test_child_diagnostics_omit_credential_argv_and_timeout_is_safe(monkeypatch):
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="SSH error\nCommand ['ssh', '-i', 'SECRET-KEY-PATH'] failed"))
    value = monitor._command(["not-run"], timeout=15)
    assert "SECRET-KEY-PATH" not in value["error"] and "SSH error" in value["error"]
    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(["SENSITIVE-ARGV"], 15)
    monkeypatch.setattr(monitor.subprocess, "run", timeout)
    value = monitor._command(["not-run"], timeout=15)
    assert "SENSITIVE-ARGV" not in value["error"] and "timed out" in value["error"]


def test_logs_rotate_with_total_bound_and_postcheck_is_opt_in(configured, monkeypatch):
    config, hosts, state, _ = configured
    state.mkdir()
    logger = monitor._log(state)
    try:
        for number in range(600):
            monitor._event(logger, "test", detail="x" * 4000, number=number)
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    assert sum(path.stat().st_size for path in state.glob("monitor.log*")) <= 2 * 1024**2
    monkeypatch.setattr(monitor, "_command", lambda *_a, **_k: {"ok": True, "result": {"state": "blocked", "reason": "Study active"}})
    checks = []
    result = monitor.run_cycle(config, hosts, postcheck=lambda: checks.append(True) or {"state": "waiting", "message": "No validated queue plan"})
    assert checks == [True] and result["postcheck"]["state"] == "waiting"


def test_malformed_hosts_are_preserved_and_cleanup_still_runs(configured, monkeypatch):
    config, hosts, _, _ = configured
    hosts.write_text('{"version":1,"hosts":[{"id":"foreign"}]}')
    original = hosts.read_bytes()
    calls = []
    monkeypatch.setattr(monitor, "_command", lambda command, **kwargs: calls.append(command) or {"ok": False, "error": "Offline"})
    result = monitor.run_cycle(config, hosts)
    assert len(calls) == 2 and "local_publish_error" in result
    assert hosts.read_bytes() == original


def test_dynamic_ten_gib_reserve_pauses_transfers_but_keeps_safe_cleanup(configured, monkeypatch):
    config, hosts, _, _ = configured
    free = [monitor.MINIMUM_FREE_BYTES + monitor.MAX_TRANSFER_STAGING_BYTES - 1]
    monkeypatch.setattr(monitor.shutil, "disk_usage", lambda _: SimpleNamespace(free=free[0]))
    commands = []
    monkeypatch.setattr(monitor, "_command", lambda command, **kwargs: commands.append(command) or
        {"ok": True, "result": {"state":"blocked", "reason":"Study active"}})
    result = monitor.run_cycle(config, hosts)
    assert "10 GiB" in result["private_error"]
    assert len(commands) == 1 and "adp.study_cleanup" in commands[0]
    free[0] = 100 * 1024**3
    assert "private_error" not in monitor.run_cycle(config, hosts)
    assert len(commands) == 4


def test_cloud_checks_preserve_last_training_status_time_and_verified_receipts(configured, monkeypatch):
    from adp import cloud_monitor
    config, hosts, _, _ = configured
    document = read_json(hosts)
    old_stamp = "2026-10-19T00:00:00+00:00"
    for row in document["hosts"][2:]:
        row.update(observed_at=old_stamp, updated_at=old_stamp)
    write_json(hosts, document)
    monkeypatch.setattr(cloud_monitor, "check_cloud_connections", lambda: {
        host: {"status":"unconfigured", "phase":"connection_required", "scope":"connection_capability",
               "message":"Connect the supported read-only provider."} for host in ("colab", "kaggle")})
    monkeypatch.setattr(monitor, "_command", lambda command, **kwargs:
        {"ok": True, "result": {"state":"blocked", "reason":"Study active"}})
    result = monitor.run_cycle(config, hosts)
    assert result["colab_connection"]["status"] == "unconfigured"
    rows = read_json(hosts)["hosts"]
    for previous, current in zip(document["hosts"][2:], rows[2:]):
        assert {key:current[key] for key in previous} == previous
        assert current["connection"]["checked_at"] != old_stamp
    from adp.compute_status import read_compute_hosts
    for row in read_compute_hosts(hosts)[2:]:
        assert row["stale"] is True and row["observed_at"].startswith("2026-10-19")
        assert row["connection"]["checked_at"] != row["observed_at"]
