"""Delivery ordering, immutable export pruning and private GPU job ownership."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import adp.remote_control as control
from adp.data import read_json, write_json
from adp.model import ForecastModel
from adp.remote_transfer import MINIMUM_FREE_BYTES, validate_snapshot


@pytest.fixture
def server(tmp_path, monkeypatch):
    root = tmp_path / "private" / "jobs" / "server-pilot-v1"
    run = root / "run"
    run.mkdir(parents=True)
    model_config = {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1}
    saved = {"format_version": 2, "config": {"model": model_config}, "manifest_hash": "a" * 64,
             "preprocessing_hash": "b" * 64, "model_state_dict": ForecastModel(model_config).state_dict(),
             "completed_epochs": 1, "global_step": 2, "partial_epoch": False}
    torch.save(saved, run / "last.pt")
    protocol = {"purpose": "development qualification", "maximum_compute_seconds": 1800.,
                "config_sha256": "c" * 64, "cache_files": {"manifest.json": "a" * 64,
                "preprocessing.json": "b" * 64}}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(root / "job.json", protocol)
    write_json(root / "status.json", {"state": "completed"})
    write_json(root / "compute-ledger.json", {"protocol_sha256": protocol["protocol_sha256"], "charged_seconds": 42.})
    monkeypatch.setattr(control, "PRIVATE_JOB_ROOT", root)
    monkeypatch.setattr(control, "_gpu_processes", lambda: [])
    monkeypatch.setattr(control, "_process", lambda _: False)
    monkeypatch.setattr(control.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    return root


@pytest.fixture
def connection(tmp_path):
    value = {"host": "aadi@192.168.29.71", "identityfile": str(tmp_path / "identity"),
        "remote_python": "/home/aadi/interaction-lab/qualification-v1/runtime-cu126/bin/python",
        "remote_source": "/home/aadi/interaction-lab/control-v1/src",
        "remote_job": "/home/aadi/interaction-lab/jobs/server-pilot-v1",
        "local_root": str(tmp_path / "local-\u03b4"), "local_staging": str(tmp_path / "incoming")}
    path = tmp_path / "connection.json"
    write_json(path, value)
    return path, value


def transport(monkeypatch, root, config, events, *, corrupt=False):
    def ssh(settings, action, **kwargs):
        assert settings == config
        events.append(action)
        if action == "prune":
            snapshot_id = kwargs["snapshot_id"]
            receipt = Path(config["local_root"]) / "receipts" / f"{snapshot_id}.json"
            assert receipt.is_file()
            assert read_json(receipt) == kwargs["receipt"]
            validate_snapshot(Path(config["local_root"]) / snapshot_id)
            return control.prune(root, snapshot_id, kwargs["receipt"])
        return {"status": control.status, "snapshot": control.snapshot, "resume": control.resume}[action](root)

    def download(settings, snapshot_id, sizes, stage):
        events.append("download")
        exports, _ = control._paths(root)
        for name in sizes:
            shutil.copyfile(exports / snapshot_id / name, stage / name)
        if corrupt:
            (stage / "last.pt").write_bytes(b"corrupt checkpoint")

    monkeypatch.setattr(control, "_ssh", ssh)
    monkeypatch.setattr(control, "_download", download)


def test_real_cpu_import_receipt_precedes_remote_prune_and_preserves_active_checkpoint(server, connection, monkeypatch):
    path, config = connection
    events = []
    original = (server / "run" / "last.pt").read_bytes()
    transport(monkeypatch, server, config, events)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("Synchronization must remain on CPU"))
    first = control.sync_private_server(path)
    assert events == ["status", "snapshot", "download", "prune", "resume"]
    assert first["resume"]["state"] == "not_resumed"
    snapshot_id = first["delivered"]["snapshot_id"]
    assert validate_snapshot(Path(config["local_root"]) / snapshot_id)
    exports, _ = control._paths(server)
    assert not (exports / snapshot_id).exists()
    assert not (Path(config["local_staging"]) / snapshot_id).exists()
    assert (server / "run" / "last.pt").read_bytes() == original
    events.clear()
    second = control.sync_private_server(path)
    assert events == ["status", "snapshot", "resume"]
    assert second["snapshot"]["state"] == "unchanged"


def test_corruption_blocks_receipt_prune_and_resume(server, connection, monkeypatch):
    path, config = connection
    events = []
    transport(monkeypatch, server, config, events, corrupt=True)
    with pytest.raises(ValueError):
        control.sync_private_server(path)
    assert events == ["status", "snapshot", "download"]
    exports, state = control._paths(server)
    snapshot_id = read_json(state)["pending_snapshot_id"]
    assert (exports / snapshot_id / "last.pt").is_file()
    assert not (Path(config["local_root"]) / "receipts" / f"{snapshot_id}.json").exists()


def test_transfer_only_sync_never_requests_resume_even_for_paused_job(server, connection, monkeypatch):
    path, config = connection
    write_json(server / "status.json", {"state": "paused"})
    events = []
    transport(monkeypatch, server, config, events)
    with patch.object(control.subprocess, "Popen") as launch:
        result = control.sync_private_server(path, resume_training=False)
    assert events == ["status", "snapshot", "download", "prune"]
    assert result["before"]["state"] == "paused" and result["resume"]["state"] == "not_resumed"
    assert "disabled" in result["resume"]["reason"]
    launch.assert_not_called()


def test_disk_reserve_and_invalid_remote_identifier_fail_before_download(server, connection, monkeypatch):
    path, config = connection
    events = []
    transport(monkeypatch, server, config, events)
    monkeypatch.setattr(control.shutil, "disk_usage", lambda _: SimpleNamespace(free=MINIMUM_FREE_BYTES + 1))
    monkeypatch.setattr(control, "snapshot", lambda _: {"state": "snapshot", "snapshot_id": "d" * 64,
        "files": {"last.pt": {"size_bytes": 20, "sha256": "e" * 64}}, "snapshot_json_bytes": 20})
    with pytest.raises(OSError, match="reserve"):
        control.sync_private_server(path)
    assert events == ["status", "snapshot"]
    monkeypatch.setattr(control, "snapshot", lambda _: {"state": "snapshot", "snapshot_id": "../../escape"})
    with pytest.raises(ValueError, match="identifier"):
        control.sync_private_server(path)


def test_pending_export_is_reused_and_only_receipted_pending_export_can_be_pruned(server):
    first = control.snapshot(server)
    assert control.snapshot(server)["snapshot_id"] == first["snapshot_id"]
    _, state_path = control._paths(server)
    write_json(state_path, {**read_json(state_path), "last_pruned_snapshot_id": "f" * 64})
    with pytest.raises(ValueError, match="pending"):
        control.prune(server, "f" * 64, {})
    with pytest.raises(ValueError, match="verified"):
        control.prune(server, first["snapshot_id"], {})
    assert control.snapshot(server)["snapshot_id"] == first["snapshot_id"]


def test_missing_checkpoint_does_not_create_export(server):
    (server / "run" / "last.pt").unlink()
    assert control.snapshot(server) == {"state": "no_checkpoint"}
    assert not control._paths(server)[0].exists()


def test_resume_uses_frozen_source_and_refuses_duplicate_or_stale_owner(server, monkeypatch):
    write_json(server / "status.json", {"state": "paused"})
    source = server / "source" / "adp"
    source.mkdir(parents=True)
    (source / "remote_jobs.py").write_text("# frozen source")
    with patch.object(control.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as launch:
        result = control.resume(server)
        assert result["state"] == "launched"
        command = launch.call_args.args[0]
        assert command[1:4] == ["-m", "adp.remote_jobs", "run"]
        assert launch.call_args.kwargs["env"]["PYTHONPATH"] == str(server / "source")
        assert launch.call_args.kwargs["start_new_session"] is True
        monkeypatch.setattr(control, "_process", lambda pid: pid == 123)
        assert control.resume(server)["state"] == "already_owned"
        assert launch.call_count == 1
    monkeypatch.setattr(control, "_process", lambda _: False)
    write_json(server / ".remote-job.lock", {"pid": 999})
    assert control.resume(server)["state"] == "already_owned"
    assert (server / ".remote-job.lock").exists()


@pytest.mark.parametrize("state,charged", [("completed", 42), ("failed", 42), ("paused", 1800), ("not_started", 0)])
def test_resume_is_limited_to_paused_jobs_with_remaining_budget(server, state, charged):
    write_json(server / "status.json", {"state": state})
    protocol = read_json(server / "job.json")
    write_json(server / "compute-ledger.json", {"protocol_sha256": protocol["protocol_sha256"], "charged_seconds": charged})
    with patch.object(control.subprocess, "Popen") as launch:
        assert control.resume(server)["state"] == "not_resumed"
        launch.assert_not_called()


def test_status_reports_training_owner_gpu_pid_not_controller_process(server, monkeypatch):
    write_json(server / "launch.json", {"pid": 321})
    write_json(server / ".remote-job.lock", {"pid": 123})
    monkeypatch.setattr(control, "_process", lambda _: True)
    monkeypatch.setattr(control, "_gpu_processes", lambda: [{"pid": 123, "memory_mib": 400.}])
    result = control.status(server)
    assert result["training_pid"] == 123 and result["training_on_gpu"] is True


def test_config_paths_command_quoting_and_bounded_file_reader(connection, tmp_path):
    path, config = connection
    assert control._config(path) == config
    quoted = {**config, "remote_source": "/tmp/lab $(touch NEVER)"}
    with patch.object(control.subprocess, "run", return_value=SimpleNamespace(stdout='{"state":"completed"}')) as run:
        control._ssh(quoted, "status")
        argv = run.call_args.args[0]
        assert argv[0] == "ssh" and "StrictHostKeyChecking=yes" in argv
        assert f"PYTHONPATH={quoted['remote_source']}" in shlex.split(argv[-1])
        assert run.call_args.kwargs.get("shell") is not True

    def delivered(argv, **kwargs):
        words = shlex.split(argv[-1])
        assert "from adp.remote_transfer import _read" in words[4]
        assert words[-1] == "4"
        kwargs["stdout"].write(b"data")

    with patch.object(control.subprocess, "run", side_effect=delivered):
        control._download(config, "a" * 64, {"last.pt": 4}, tmp_path)
    assert (tmp_path / "last.pt").read_bytes() == b"data"
    write_json(path, {**config, "remote_job": "/home/aadi/../someone"})
    with pytest.raises(ValueError, match="traversal"):
        control._config(path)


def test_unlisted_files_excessive_sizes_and_foreign_jobs_are_rejected(server):
    base = {"snapshot_id": "a" * 64, "files": {"last.pt": {"sha256": "b" * 64, "size_bytes": 5}},
            "snapshot_json_bytes": 5}
    for files in ({"../secret": {"sha256": "b" * 64, "size_bytes": 5}},
                  {"last.pt": {"sha256": "b" * 64, "size_bytes": control.MAX_FILE_BYTES + 1}}):
        with pytest.raises(ValueError):
            control._download_index({**base, "files": files})
    with pytest.raises(ValueError, match="restricted"):
        control.status(server.parent)
    changed = read_json(server / "job.json")
    write_json(server / "job.json", {**changed, "maximum_compute_seconds": 99999})
    with pytest.raises(ValueError, match="hash differs"):
        control.status(server)
