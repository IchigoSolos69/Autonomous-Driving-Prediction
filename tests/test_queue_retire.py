"""Terminal retirement preserves exact provenance and survives an interrupted unlink."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tool(tmp_path):
    path = PROJECT / "scripts/queue-retire-v1.py"
    spec = importlib.util.spec_from_file_location("terminal_retirement_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / path.name)
    shutil.copy2(PROJECT / "scripts/queue-cleanup-v2.py", tmp_path / ("queue-cleanup-v2-" + module.ADAPTER_SHA256 + ".py"))
    shutil.copy2(PROJECT / "src/adp/queue_cleanup.py", tmp_path / "queue_cleanup.py")
    return module


@pytest.fixture
def case(tool, tmp_path, monkeypatch):
    base = tool.load_cleanup()
    root = tmp_path / "private queue 私有"
    root.mkdir()
    (root / ".worker.lock").write_bytes(b"0")
    payload = b"original fixture payload"
    base.PAYLOAD_SHA256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(tool, "PAYLOAD_BYTES", len(payload))
    write = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(base._bytes(value)))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    ids = {"train": [str(uuid.UUID(int=n)) for n in range(1, 1001)],
           "validation": [str(uuid.UUID(int=n)) for n in range(1001, 1065)]}
    shared = root / "input"
    write(shared / "cache/manifest.json", {"splits": ids})
    write(shared / "cache/preprocessing.json", {"version": 1})
    write(shared / "source-manifest.json", {"splits": ids})
    write(shared / "initial-config.json", {"version": 1})
    for name in ("remote_jobs.py", "training.py"):
        path = shared / "source/adp" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# approved source\n")
    for split, scenarios in ids.items():
        (shared / "cache" / split).mkdir()
        for sid in scenarios:
            (shared / "cache" / split / (sid + ".npz")).write_bytes(b"approved cache")
    index = {path.relative_to(shared).as_posix(): {"sha256": sha(path), "size": path.stat().st_size}
             for path in shared.rglob("*") if path.is_file()}
    write(shared / "package-files.json", index)
    monkeypatch.setattr(tool, "INDEX_SHA256", sha(shared / "package-files.json"))
    cache = {name[6:]: value["sha256"] for name, value in index.items() if name.startswith("cache/")}
    deliveries, records = [], []
    host_protocol = base._hash({"version": 1, "host_seconds": 21600, "jobs": tool.specifications(base)})
    for spec in tool.specifications(base):
        jid = base._hash(spec)
        job = root / "jobs" / jid
        for name in ("source",):
            shutil.copytree(shared / name, job / name)
        for name in ("manifest.json", "preprocessing.json"):
            target = job / "cache" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(shared / "cache" / name, target)
        shutil.copy2(shared / "source-manifest.json", job / "source-manifest.json")
        cfg = spec["configuration"]
        config = {"seed": spec["seed"], "model": {"use_graph": cfg != "no_graph", "use_map": cfg != "no_map",
                  "shared_world_ce_weight": .1 if cfg == "world_ce" else 0.},
                  "training": {"batch_size": 1, "accumulation_steps": 32, "amp": False, "num_workers": 0,
                               "checkpoint_seconds": 300, "minimum_free_gib": 8}}
        protocol = {"version": 1, "purpose": "development qualification; not frozen-study benchmark evidence",
            "epochs": 20, "maximum_compute_seconds": 1800., "config_sha256": base._protocol_hash(config),
            "cache_files": cache, "source_manifest_sha256": sha(job / "source-manifest.json"),
            "source_files": {path.name: sha(path) for path in (job / "source/adp").iterdir()},
            "split_counts": {name: len(value) for name, value in ids.items()},
            "split_sha256": {name: base._protocol_hash(value) for name, value in ids.items()}}
        protocol["protocol_sha256"] = base._protocol_hash(protocol)
        status = {"state": "completed", "protocol_sha256": protocol["protocol_sha256"], "result": {"completed_epochs": 20}}
        for name, value in (("job.json", protocol), ("config.json", config), ("status.json", status),
                            ("run/job.json", protocol), ("run/config.json", config), ("run/result.json", status)):
            write(job / name, value)
        files = {path.name: {"sha256": sha(path), "size_bytes": path.stat().st_size} for path in (job / "run").iterdir()}
        model_bytes = {"last.pt": b"verified final model", "best.pt": b"verified best model"}
        files.update({name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)} for name, data in model_bytes.items()})
        provenance = {"queue_job_spec": spec, "queue_job_id": jid, "payload_sha256": base.PAYLOAD_SHA256,
            "queue_protocol_sha256": host_protocol, "allocated_device_count": 1,
            "job_protocol_sha256": protocol["protocol_sha256"], "config_sha256": protocol["config_sha256"],
            "manifest_sha256": cache["manifest.json"], "preprocessing_sha256": cache["preprocessing.json"]}
        checkpoint = {"config": config, "manifest_hash": cache["manifest.json"], "preprocessing_hash": cache["preprocessing.json"],
                      "partial_epoch": False, "completed_epochs": 20, "global_step": 640}
        metadata = {"format_version": 1, "run_id": jid, "provenance": provenance, "files": files,
                    "checkpoint_metadata": {name: checkpoint for name in model_bytes}, "captured_at": "2026-10-20T01:00:00+00:00"}
        sid = base._hash(metadata)
        metadata["snapshot_id"] = sid
        folder = root / "snapshots" / sid
        folder.mkdir(parents=True)
        for name in files:
            if name in model_bytes:
                (folder / name).write_bytes(model_bytes[name])
            else:
                shutil.copy2(job / "run" / name, folder / name)
        write(folder / "snapshot.json", metadata)
        archive = root / "exports" / (sid + ".zip")
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive, "w") as zipped:
            for path in folder.iterdir():
                zipped.write(path, path.name)
        common = {"snapshot_id": sid, "manifest_sha256": sha(folder / "snapshot.json"), "files": files,
                  "verified": True, "verified_at": "2026-10-20T01:00:01+00:00", "local_directory": "D:/canonical/" + sid}
        submitted = {"queue_receipt": {**common, "version": 1, "job_id": jid, "job_spec": spec, "archive_sha256": sha(archive),
                     "payload_sha256": base.PAYLOAD_SHA256, "queue_protocol_sha256": host_protocol,
                     "allocated_device_count": 1, "checkpoint_validation": "strict_cpu_observed_only_completed"},
                     "import_receipt": {**common, "format_version": 1, "checkpoint_validation": "cpu_weights_only"},
                     "export": {"job_id": jid, "snapshot_id": sid, "archive_name": archive.name,
                                "bytes": archive.stat().st_size, "sha256": sha(archive)}}
        old_files = {"jobs/" + jid + "/cache/" + name: digest for name, digest in cache.items() if name.endswith(".npz")}
        old_files.update({"jobs/" + jid + "/run/" + name: files[name]["sha256"] for name in model_bytes})
        old = {"version": 1, "state": "completed", "context": {"job_id": jid, "job_protocol_sha256": protocol["protocol_sha256"],
               "snapshot_id": sid, "envelope_sha256": base._hash(submitted), "status_sha256": base._hash(status)},
               "files": [{"path": name, "sha256": digest} for name, digest in old_files.items()]}
        old["receipt_sha256"] = base._hash(old)
        write(root / "cleanup" / (jid + ".json"), old)
        deliveries.append({**submitted, "cleanup_receipt_sha256": sha(root / "cleanup" / (jid + ".json"))})
        records.append({"id": jid, "spec": spec, "spec_sha256": jid, "state": "completed", "lease": None,
                        "checkpoint_sha256": files["last.pt"]["sha256"], "charged_seconds": 100,
                        "attempts": [{"outcome": "completed", "charged_seconds": 100, "reserved_seconds": 1800}]})
    write(root / "controller/queue.json", {"version": 1, "host_seconds": 21600, "jobs": records})
    for name in ("worker.py", "compute_queue.py", "queue_cleanup.py", "pilot-helper.py"):
        (root / name).write_bytes(b"# approved immutable resource\n")
    (root / "inputs.zip").write_bytes(payload)
    resources = {name: sha(root / name) for name in ("worker.py", "compute_queue.py", "queue_cleanup.py", "pilot-helper.py", "inputs.zip")}
    plans = {host: [{**spec, "host": host, "dependencies": ["original_study", "direction_study", "cpu_reports", "training_cache_cleanup"] if host == "local" else []}
                     for spec in tool.specifications(base)] for host in ("local", "private", "colab", "kaggle")}
    approved = base._hash({"version": 1, "host_seconds": 21600, "payload_sha256": base.PAYLOAD_SHA256, "files": resources, "host_plans": plans})
    envelope = {"version": 1, "approved_plan_sha256": approved, "payload_sha256": base.PAYLOAD_SHA256, "files": resources, "deliveries": deliveries}
    write(root / "resources.json", {"files": resources, "payload_sha256": base.PAYLOAD_SHA256,
          "plan_sha256": host_protocol, "approved_plan_sha256": approved})
    return tool, base, root, envelope


def test_plan_is_exact_and_preserves_all_copies(case):
    tool, base, root, envelope = case
    result = tool.retire(root, envelope, base=base)
    assert result["state"] == "planned" and result["target_count"] == 1101
    assert len(list((root / "input/cache").rglob("*.npz"))) == 1064
    assert not (root / tool.JOURNAL).exists()


def test_terminal_retirement_preserves_provenance_and_inspects_without_models(case):
    tool, base, root, envelope = case
    preserved = {name: base._sha(root / name, 8 * 1024**2) for name in tool.preserved_paths(base, root, envelope)}
    result = tool.retire(root, envelope, execute=True, base=base)
    assert result["state"] == "completed" and result["remaining_files"] == 0 and result["target_count"] == 1101
    assert not (root / "inputs.zip").exists() and not list((root / "exports").glob("*.zip"))
    assert not list((root / "input/cache").rglob("*.npz"))
    assert all((root / "snapshots" / item["export"]["snapshot_id"] / "snapshot.json").is_file() for item in envelope["deliveries"])
    assert all((root / "jobs" / item["export"]["job_id"] / "source-manifest.json").is_file() for item in envelope["deliveries"])
    assert tool.retire(root, inspect=True, base=base) == result
    assert tool.retire(root, envelope, execute=True, base=base) == result
    assert all(base._sha(root / name, 8 * 1024**2) == digest for name, digest in preserved.items())
    journal_path = root / tool.JOURNAL
    journal = base._read(journal_path)
    journal["files"][0]["path"] = "unregistered.bin"
    journal["receipt_sha256"] = base._hash({key: value for key, value in journal.items() if key != "receipt_sha256"})
    journal_path.write_bytes(base._bytes(journal))
    with pytest.raises(ValueError, match="unexpected deletion set"):
        tool.retire(root, inspect=True, base=base)


def test_interruption_has_durable_full_plan_and_resumes(case, monkeypatch):
    tool, base, root, envelope = case
    original = Path.unlink
    removed = []
    def interrupt(path, *args, **kwargs):
        if path.suffix == ".npz":
            assert len(base._read(root / tool.JOURNAL)["files"]) == 1101
            if len(removed) == 2:
                raise OSError("simulated interruption")
            removed.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(OSError, match="interruption"):
        tool.retire(root, envelope, execute=True, base=base)
    monkeypatch.setattr(Path, "unlink", original)
    assert tool.retire(root, inspect=True, base=base)["state"] == "in_progress"
    result = tool.retire(root, envelope, execute=True, base=base)
    assert result["state"] == "completed" and result["remaining_files"] == 0


@pytest.mark.parametrize("kind", ["active", "missing_receipt", "changed_input", "owner", "corrupt_cleanup"])
def test_incomplete_or_changed_evidence_never_deletes(case, kind):
    tool, base, root, envelope = case
    jid = envelope["deliveries"][0]["export"]["job_id"]
    if kind == "active":
        path = root / "controller/queue.json"
        value = base._read(path); value["jobs"][0]["state"] = "running"; path.write_bytes(base._bytes(value))
    elif kind == "missing_receipt":
        envelope["deliveries"].pop()
    elif kind == "changed_input":
        next((root / "input/cache").rglob("*.npz")).write_bytes(b"changed")
    elif kind == "owner":
        (root / "jobs" / jid / ".remote-job.lock").write_bytes(b"owner")
    else:
        (root / "cleanup" / (jid + ".json")).write_bytes(b"{}")
    with pytest.raises(ValueError):
        tool.retire(root, envelope, execute=True, base=base)
    assert (root / "inputs.zip").is_file() and len(list((root / "input/cache").rglob("*.npz"))) == 1064


def test_reappearance_after_completion_is_preserved(case):
    tool, base, root, envelope = case
    tool.retire(root, envelope, execute=True, base=base)
    (root / "inputs.zip").write_bytes(b"reappeared")
    with pytest.raises(ValueError, match="reappeared"):
        tool.retire(root, inspect=True, base=base)
    assert (root / "inputs.zip").read_bytes() == b"reappeared"


def test_inspection_before_retirement_and_fixed_cli_root(case):
    tool, base, root, envelope = case
    assert tool.retire(root, inspect=True, base=base)["state"] == "not_started"
    with pytest.raises(ValueError, match="fixed private"):
        tool.main(["--root", str(root), "--inspect"])


def test_active_worker_lock_blocks_retirement(case):
    tool, base, root, envelope = case
    with tool.worker_lock(base, root):
        with pytest.raises(OSError):
            tool.retire(root, envelope, execute=True, base=base)
    assert (root / "inputs.zip").is_file()
