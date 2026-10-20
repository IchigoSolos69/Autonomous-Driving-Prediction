"""Completed-job cleanup never reaches shared data or unverified results."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import pytest
import torch

from adp.compute_cleanup import cleanup_completed_job, main, plan_cleanup
from adp.data import read_json, write_json
from adp.model import ForecastModel
from adp.remote_jobs import prepare_job
from adp.remote_transfer import create_snapshot, import_snapshot


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    root = tmp_path / "server-pilot-v1"
    splits = {"train": [str(uuid.UUID(int=1))], "validation": [str(uuid.UUID(int=2))]}
    write_json(root / "cache" / "manifest.json", {"splits": splits})
    write_json(root / "cache" / "preprocessing.json", {"version": 1})
    for split, ids in splits.items():
        (root / "cache" / split).mkdir()
        (root / "cache" / split / f"{ids[0]}.npz").write_bytes(b"independent development cache")
    write_json(root / "source-manifest.json", {"splits": {**splits, "calibration": [], "evaluation": []}})
    source = root / "source" / "adp"
    source.mkdir(parents=True)
    (source / "training.py").write_text("# frozen source")
    (source / "remote_jobs.py").write_text("# frozen source")
    config = {"model": {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1}}
    protocol = prepare_job(root, config, source_root=source)
    config = read_json(root / "config.json")
    run = root / "run"
    run.mkdir()
    for name in ("last.pt", "best.pt"):
        torch.save({"format_version": 2, "config": config, "completed_epochs": 1, "global_step": 1,
            "partial_epoch": False, "manifest_hash": protocol["cache_files"]["manifest.json"],
            "preprocessing_hash": protocol["cache_files"]["preprocessing.json"],
            "model_state_dict": ForecastModel(config["model"]).state_dict()}, run / name)
    (root / "training.log").write_text("Keep the complete audit trail")
    write_json(root / "status.json", {"state": "completed", "protocol_sha256": protocol["protocol_sha256"],
                                      "result": {"completed_epochs": 1}})
    payload = root / "inputs.zip"
    payload.write_bytes(b"staged independent ZIP payload")
    write_json(root / "cleanup-leases.json", {"version": 1, "job_protocol_sha256": protocol["protocol_sha256"],
        "cache_owned": True, "consumers": {"private-server": "released", "cloud": "released"},
        "payload_files": {"inputs.zip": hashlib.sha256(payload.read_bytes()).hexdigest()}})
    snapshot = create_snapshot(run, tmp_path / "exports", root.name,
        {"job_protocol_sha256": protocol["protocol_sha256"], "config_sha256": protocol["config_sha256"],
         "manifest_sha256": protocol["cache_files"]["manifest.json"],
         "preprocessing_sha256": protocol["cache_files"]["preprocessing.json"]})
    received = import_snapshot(snapshot["path"], tmp_path / "verified local models")
    receipt = {**received["receipt"], "export_manifest": snapshot["manifest"]}
    return root, receipt, Path(received["path"])


def _targets(root):
    return [root / "cache", root / "inputs.zip", root / "run" / "last.pt", root / "run" / "best.pt"]


def test_plan_then_cleanup_preserves_local_weights_audit_and_is_idempotent(completed):
    root, receipt, local = completed
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    plan = plan_cleanup(root, receipt, _targets(root))
    assert plan["bytes_to_free"] > 0
    assert all((root / name).read_bytes() == value for name, value in before.items())
    result = cleanup_completed_job(root, receipt, _targets(root))
    assert result["state"] == "completed"
    assert not (root / "cache").exists() and not (root / "inputs.zip").exists()
    assert not (root / "run" / "last.pt").exists() and not (root / "run" / "best.pt").exists()
    for name in ("job.json", "config.json", "status.json", "source-manifest.json", "cleanup-leases.json", "training.log", "source/adp/training.py"):
        assert (root / name).read_bytes() == before[name]
    assert (local / "last.pt").is_file() and (local / "best.pt").is_file()
    again = cleanup_completed_job(root, receipt, _targets(root))
    assert again["state"] == "completed" and again["bytes_to_free"] == 0
    assert not (root / ".cleanup.lock").exists()


@pytest.mark.parametrize("failure", ["running", "locked", "live-pid", "cloud-pending", "shared-cache", "receipt", "protocol", "checkpoint", "unexpected-cache"])
def test_all_unsafe_conditions_block_the_whole_plan_before_deletion(completed, failure):
    root, receipt, _ = completed
    if failure == "running":
        status = read_json(root / "status.json")
        write_json(root / "status.json", {**status, "state": "paused"})
    elif failure == "locked":
        (root / ".remote-job.lock").write_text('{"pid":123}')
    elif failure == "live-pid":
        write_json(root / "launch.json", {"pid": 123})
    elif failure in {"cloud-pending", "shared-cache"}:
        leases = read_json(root / "cleanup-leases.json")
        if failure == "cloud-pending":
            leases["consumers"]["cloud"] = "pending"
        else:
            leases["cache_owned"] = False
        write_json(root / "cleanup-leases.json", leases)
    elif failure == "receipt":
        receipt = {**receipt, "manifest_sha256": "f" * 64}
    elif failure == "protocol":
        job = read_json(root / "job.json")
        write_json(root / "job.json", {**job, "epochs": 2})
    elif failure == "checkpoint":
        (root / "run" / "best.pt").write_bytes(b"new unverified weights")
    else:
        (root / "cache" / "calibration.npz").write_bytes(b"never delete unknown data")
    with patch("adp.compute_cleanup._alive", return_value=failure == "live-pid"):
        with pytest.raises(ValueError):
            cleanup_completed_job(root, receipt, _targets(root))
    assert (root / "inputs.zip").is_file()
    assert (root / "cache" / "train").is_dir()
    assert (root / "run" / "last.pt").is_file()
    assert not (root / "cleanup-receipt.json").exists()


@pytest.mark.parametrize("target", ["outside", "source", "metadata", "relative", "traversal"])
def test_foreign_and_audit_targets_are_never_allowlisted(completed, target):
    root, receipt, _ = completed
    paths = {"outside": root.parent / "original-study-cache", "source": root / "source",
             "metadata": root / "job.json", "relative": Path("cache"),
             "traversal": root / "cache" / ".." / "source"}
    with pytest.raises(ValueError):
        plan_cleanup(root, receipt, [paths[target]])
    assert (root / "source" / "adp" / "training.py").is_file()


def test_preflight_rechecks_all_files_and_leaves_data_when_inputs_change(completed, monkeypatch):
    root, receipt, _ = completed
    import adp.compute_cleanup as cleanup
    write = cleanup._write
    def change_after_journal(path, value):
        write(path, value)
        (root / "inputs.zip").write_bytes(b"changed between preflight and execution")
    monkeypatch.setattr(cleanup, "_write", change_after_journal)
    with pytest.raises(ValueError, match="hash"):
        cleanup_completed_job(root, receipt, _targets(root))
    assert (root / "cache" / "train").is_dir()
    assert (root / "run" / "last.pt").is_file()


def test_partial_delete_retry_uses_matching_cleanup_receipt(completed, monkeypatch):
    root, receipt, _ = completed
    unlink = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path.name == "inputs.zip":
            raise OSError("simulated interruption")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", interrupted)
        with pytest.raises(OSError, match="interruption"):
            cleanup_completed_job(root, receipt, _targets(root))
    assert (root / "cleanup-receipt.json").is_file()
    assert cleanup_completed_job(root, receipt, _targets(root))["state"] == "completed"


def test_links_or_ancestor_links_are_rejected(completed):
    root, receipt, _ = completed
    alias = root.parent / "linked-job"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symbolic links requires permission on this Windows host")
    with pytest.raises(ValueError, match="links"):
        plan_cleanup(alias, receipt, [alias / "inputs.zip"])


def test_shared_hardlinks_are_refused_even_with_owned_cache_claim(completed, monkeypatch):
    root, receipt, _ = completed
    import os
    try:
        os.link(root / "inputs.zip", root.parent / "shared-input.zip")
    except OSError:
        original = Path.stat
        def linked_stat(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == root / "inputs.zip":
                return SimpleNamespace(st_mode=value.st_mode, st_nlink=2)
            return value
        monkeypatch.setattr(Path, "stat", linked_stat)
    with pytest.raises(ValueError, match="shared"):
        plan_cleanup(root, receipt, [root / "inputs.zip"])
    assert (root / "inputs.zip").is_file()


def test_cli_defaults_to_plan(completed, capsys):
    root, receipt, _ = completed
    receipt_file = root / "verified-result.json"
    write_json(receipt_file, receipt)
    main(["--job-root", str(root), "--receipt-file", str(receipt_file), "--leases-file",
          str(root / "cleanup-leases.json"), "--paths", str(root / "inputs.zip"), "--plan"])
    assert json.loads(capsys.readouterr().out)["bytes_to_free"] > 0
    assert (root / "inputs.zip").is_file()
