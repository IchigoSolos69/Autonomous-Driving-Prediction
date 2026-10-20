"""Cleanup is an authenticated byte/ownership check, never a checkpoint loader."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid
import zipfile

import pytest

from adp.compute_queue import ComputeQueue, job_spec, spec_hash
from adp import queue_cleanup as cleanup


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cleanup._bytes(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    root = tmp_path_factory.mktemp("queue-cleanup-source")
    queue = ComputeQueue(root / "controller")
    queue.initialize()
    queue.plan("private")
    record = queue.claim("private", "fixture")
    queue.start(record["id"], record["lease_token"], "pid-123", device_count=1)
    jid = record["id"]
    job = root / "jobs" / jid
    split_ids = {"train": [str(uuid.UUID(int=n)) for n in range(1, 1001)],
                 "validation": [str(uuid.UUID(int=n)) for n in range(1001, 1065)]}
    write(job / "cache/manifest.json", {"splits": split_ids})
    write(job / "cache/preprocessing.json", {"version": 1})
    write(job / "source-manifest.json", {"splits": split_ids})
    for split, ids in split_ids.items():
        (job / "cache" / split).mkdir()
        for sid in ids:
            (job / "cache" / split / (sid + ".npz")).write_bytes(b"immutable cache")
    source = job / "source/adp"
    source.mkdir(parents=True)
    for name in ("__init__.py", "training.py", "remote_jobs.py", "model.py", "data.py", "metrics.py", "research.py", "risk.py"):
        (source / name).write_bytes(b"# frozen approved source\n")
    config = {"seed": 42, "model": {"use_graph": True, "use_map": True, "shared_world_ce_weight": 0.},
        "training": {"batch_size": 1, "accumulation_steps": 32, "amp": False, "num_workers": 0,
                     "checkpoint_seconds": 300, "minimum_free_gib": 8}}
    write(job / "config.json", config)
    cache = {path.relative_to(job / "cache").as_posix(): sha(path) for path in (job / "cache").rglob("*") if path.is_file()}
    protocol = {"version": 1, "purpose": "development qualification; not frozen-study benchmark evidence",
        "epochs": 20, "maximum_compute_seconds": 1800., "config_sha256": cleanup._protocol_hash(config),
        "cache_files": cache, "source_manifest_sha256": sha(job / "source-manifest.json"),
        "source_files": {path.name: sha(path) for path in source.iterdir()},
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "split_sha256": {key: cleanup._protocol_hash(value) for key, value in split_ids.items()}}
    protocol["protocol_sha256"] = cleanup._protocol_hash(protocol)
    status = {"state": "completed", "protocol_sha256": protocol["protocol_sha256"], "result": {"completed_epochs": 20}}
    write(job / "job.json", protocol)
    write(job / "status.json", status)
    write(job / "run/job.json", protocol)
    write(job / "run/config.json", config)
    write(job / "run/result.json", status)
    (job / "run/last.pt").write_bytes(b"strict CPU verified final checkpoint")
    (job / "run/best.pt").write_bytes(b"strict CPU verified best checkpoint")
    (job / "stdout.log").write_bytes(b"audit log")
    (root / "inputs.zip").write_bytes(b"shared approved payload preserved")
    checkpoint_metadata = {name: {"format_version": 2, "config": config, "completed_epochs": epoch,
        "global_step": epoch * 1000, "partial_epoch": False, "manifest_hash": cache["manifest.json"],
        "preprocessing_hash": cache["preprocessing.json"]} for name, epoch in (("last.pt", 20), ("best.pt", 12))}
    plan_hash = spec_hash({"version": 1, "host_seconds": 21600,
        "jobs": [job_spec("private", cfg, seed) for cfg in ("full", "no_graph", "no_map", "world_ce") for seed in (42, 43, 44)]})
    provenance = {"queue_job_spec": record["spec"], "queue_job_id": jid, "payload_sha256": cleanup.PAYLOAD_SHA256,
        "queue_protocol_sha256": plan_hash, "allocated_device_count": 1,
        "job_protocol_sha256": protocol["protocol_sha256"], "config_sha256": protocol["config_sha256"],
        "manifest_sha256": cache["manifest.json"], "preprocessing_sha256": cache["preprocessing.json"]}
    files = {path.name: {"sha256": sha(path), "size_bytes": path.stat().st_size} for path in (job / "run").iterdir()}
    metadata = {"format_version": 1, "run_id": jid, "provenance": provenance,
                "files": files, "checkpoint_metadata": checkpoint_metadata, "captured_at": "2026-10-20T01:00:00+00:00"}
    sid = cleanup._hash(metadata)
    metadata["snapshot_id"] = sid
    snapshot = root / "snapshots" / sid
    snapshot.mkdir(parents=True)
    for name in files:
        shutil.copy2(job / "run" / name, snapshot / name)
    write(snapshot / "snapshot.json", metadata)
    archive = root / "exports" / (sid + ".zip")
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in snapshot.iterdir():
            zipped.write(path, path.name)
    shared = {"snapshot_id": sid, "manifest_sha256": sha(snapshot / "snapshot.json"), "files": files,
              "verified": True, "verified_at": "2026-10-20T01:00:01+00:00", "local_directory": "D:/local/results/" + sid}
    envelope = {"queue_receipt": {**shared, "version": 1, "job_id": jid, "job_spec": record["spec"],
        "archive_sha256": sha(archive), "payload_sha256": cleanup.PAYLOAD_SHA256,
        "queue_protocol_sha256": plan_hash, "allocated_device_count": 1,
        "checkpoint_validation": "strict_cpu_observed_only_completed"},
        "import_receipt": {**shared, "format_version": 1, "checkpoint_validation": "cpu_weights_only"},
        "export": {"job_id": jid, "snapshot_id": sid, "archive_name": archive.name,
                   "bytes": archive.stat().st_size, "sha256": sha(archive)}}
    queue.finish(jid, record["lease_token"], outcome="completed", elapsed_seconds=1000,
                 wall_seconds=1000, checkpoint_sha256=files["last.pt"]["sha256"])
    return root, jid, envelope


@pytest.fixture
def case(tmp_path, pristine):
    source, jid, envelope = pristine
    root = tmp_path / "私有 queue"
    shutil.copytree(source, root)
    return root, jid, deepcopy(envelope)


def test_plan_preserves_every_byte_and_has_exact_targets(case):
    root, jid, envelope = case
    before = {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}
    result = cleanup.cleanup_completed_queue_job(root, jid, envelope)
    assert result["state"] == "planned" and len(result["files"]) == 1066
    assert all(entry["path"].endswith(".npz") or entry["path"].endswith(("/last.pt", "/best.pt")) for entry in result["files"])
    assert not (root / "cleanup" / (jid + ".json")).exists()
    assert all(sha(root / name) == digest for name, digest in before.items())


def test_completed_cleanup_preserves_sources_metadata_exports_and_is_idempotent(case):
    root, jid, envelope = case
    job = root / "jobs" / jid
    preserved = [root / "inputs.zip", root / "controller/queue.json", job / "stdout.log",
        job / "cache/manifest.json", job / "cache/preprocessing.json", job / "job.json", job / "status.json",
        job / "run/job.json", job / "run/config.json", job / "run/result.json",
        *list((job / "source/adp").iterdir()), *list((root / "snapshots" / envelope["export"]["snapshot_id"]).iterdir()),
        root / "exports" / envelope["export"]["archive_name"]]
    hashes = {path: sha(path) for path in preserved}
    result = cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert result["state"] == "completed"
    assert not list((job / "cache").rglob("*.npz"))
    assert not (job / "run/last.pt").exists() and not (job / "run/best.pt").exists()
    assert all(sha(path) == digest for path, digest in hashes.items())
    assert cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True) == result


def test_interrupted_cleanup_has_durable_per_file_plan_and_resumes(case, monkeypatch):
    root, jid, envelope = case
    original = Path.unlink
    original_sync = cleanup._sync
    synced = []
    def sync(path):
        synced.append(path)
        original_sync(path)
    monkeypatch.setattr(cleanup, "_sync", sync)
    removed = []
    def interrupted(path, *args, **kwargs):
        if path.suffix == ".npz":
            journal = cleanup._read(root / "cleanup" / (jid + ".json"))
            assert journal["state"] == "in_progress" and len(journal["files"]) == 1066
            assert root in synced and root / "cleanup" in synced
            if len(removed) == 2:
                raise OSError("power interruption")
            removed.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError, match="power interruption"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert len(removed) == 2 and all(not path.exists() for path in removed)
    monkeypatch.setattr(Path, "unlink", original)
    result = cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert result["state"] == "completed" and sum(entry["already_absent"] for entry in result["files"]) == 2


@pytest.mark.parametrize("mutation", ["queue_state", "queue_checkpoint", "status", "protocol", "source", "cache", "config", "missing", "owner", "extra_cache"])
def test_changed_or_incomplete_job_is_preserved(case, mutation):
    root, jid, envelope = case
    job = root / "jobs" / jid
    if mutation.startswith("queue_"):
        path = root / "controller/queue.json"
        data = cleanup._read(path)
        data["jobs"][0]["state" if mutation == "queue_state" else "checkpoint_sha256"] = "running" if mutation == "queue_state" else "a" * 64
        write(path, data)
    elif mutation == "status":
        status = cleanup._read(job / "status.json")
        status["result"]["completed_epochs"] = 19
        write(job / "status.json", status)
    elif mutation == "protocol":
        protocol = cleanup._read(job / "job.json")
        protocol["epochs"] = 19
        write(job / "job.json", protocol)
    elif mutation == "source":
        (job / "source/adp/training.py").write_bytes(b"changed source")
    elif mutation == "cache":
        next((job / "cache").rglob("*.npz")).write_bytes(b"changed cache")
    elif mutation == "config":
        write(job / "config.json", {"seed": 43})
    elif mutation == "missing":
        next((job / "cache").rglob("*.npz")).unlink()
    elif mutation == "owner":
        (job / ".remote-job.lock").write_text("active owner")
    else:
        (job / "cache/foreign.npz").write_bytes(b"preserve")
    before = sorted(path.name for path in (job / "cache").rglob("*.npz"))
    with pytest.raises(ValueError):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert sorted(path.name for path in (job / "cache").rglob("*.npz")) == before
    assert (job / "run/last.pt").exists()


@pytest.mark.parametrize("section,key,value", [
    ("queue_receipt", "job_id", "a" * 64),
    ("queue_receipt", "payload_sha256", "a" * 64),
    ("queue_receipt", "manifest_sha256", "a" * 64),
    ("queue_receipt", "checkpoint_validation", "unchecked"),
    ("queue_receipt", "allocated_device_count", 2),
    ("queue_receipt", "verified", False),
    ("queue_receipt", "verified_at", "2026-10-20"),
    ("import_receipt", "checkpoint_validation", "unchecked"),
    ("import_receipt", "local_directory", "D:/different-copy"),
    ("export", "archive_name", "../inputs.zip"),
    ("export", "sha256", "b" * 64),
])
def test_receipt_tampering_never_deletes(case, section, key, value):
    root, jid, envelope = case
    envelope[section][key] = value
    with pytest.raises(ValueError):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert len(list((root / "jobs" / jid / "cache").rglob("*.npz"))) == 1064


def test_hardlinked_cache_is_never_deleted(case, monkeypatch):
    root, jid, envelope = case
    target = next((root / "jobs" / jid / "cache").rglob("*.npz"))
    alias = root.parent / "another-consumer.npz"
    try:
        alias.hardlink_to(target)
    except PermissionError:
        # The Windows sandbox denies creating hardlinks; exercise the same stat guard.
        alias.write_bytes(target.read_bytes())
        original = Path.lstat
        def shared(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == target:
                return SimpleNamespace(**{name: 2 if name == "st_nlink" else getattr(value, name)
                    for name in dir(value) if name.startswith("st_")})
            return value
        monkeypatch.setattr(Path, "lstat", shared)
    with pytest.raises(ValueError, match="unshared"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert target.exists() and alias.exists()


def test_linked_job_directory_is_rejected(case):
    root, jid, envelope = case
    actual = root / "jobs" / jid
    other = root.parent / "outside-job"
    actual.rename(other)
    try:
        actual.symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer permissions")
    with pytest.raises(ValueError, match="links"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert (other / "run/last.pt").exists()


def test_journal_write_failure_happens_before_deletion(case, monkeypatch):
    root, jid, envelope = case
    def fail(*_):
        raise OSError("fsync failed")
    monkeypatch.setattr(cleanup, "_write", fail)
    with pytest.raises(OSError, match="fsync failed"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert len(list((root / "jobs" / jid / "cache").rglob("*.npz"))) == 1064


def test_status_recheck_stops_cleanup_if_job_changes(case, monkeypatch):
    root, jid, envelope = case
    original = Path.unlink
    def changed(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.suffix == ".npz":
            status_path = root / "jobs" / jid / "status.json"
            status = cleanup._read(status_path)
            status["state"] = "running"
            write(status_path, status)
        return result
    monkeypatch.setattr(Path, "unlink", changed)
    with pytest.raises(ValueError, match="status changed"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert len(list((root / "jobs" / jid / "cache").rglob("*.npz"))) == 1063
    assert (root / "jobs" / jid / "run/last.pt").exists()


def test_completed_journal_rejects_recreated_files_or_different_receipts(case):
    root, jid, envelope = case
    result = cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    target = root / result["files"][0]["path"]
    target.write_bytes(b"immutable cache")
    with pytest.raises(ValueError, match="replaced|reappeared"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)
    assert target.exists()
    target.unlink()
    envelope["queue_receipt"]["verified_at"] = "2026-10-20T02:00:00+00:00"
    with pytest.raises(ValueError, match="different verified"):
        cleanup.cleanup_completed_queue_job(root, jid, envelope, execute=True)


def test_module_is_standalone_and_cli_restricts_root(case, monkeypatch, capsys):
    root, jid, envelope = case
    spec = importlib.util.spec_from_file_location("standalone_queue_cleanup", Path(cleanup.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="fixed private"):
        module.main(["--root", str(root), "--job-id", jid])
    monkeypatch.setattr(module, "PRIVATE_ROOT", root)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(cleanup._bytes(envelope))))
    module.main(["--root", str(root), "--job-id", jid])
    assert json.loads(capsys.readouterr().out)["state"] == "planned"
    assert (root / "jobs" / jid / "run/last.pt").exists()


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'x' * (cleanup.MAX_JSON + 1)],
                         ids=["duplicate-key", "nonfinite", "array", "oversized"])
def test_bounded_json_rejects_ambiguous_or_excessive_input(data):
    with pytest.raises(ValueError):
        cleanup._json_bytes(data)
