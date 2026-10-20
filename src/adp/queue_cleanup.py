"""Retire only completed private-queue job copies after authenticated local receipts.

Standalone standard-library controller: it never imports training code or weights.
The caller delivers receipts over authenticated SSH; local paths are metadata only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import uuid

PRIVATE_ROOT = Path("/home/aadi/interaction-lab/queues/development-v1")
PAYLOAD_SHA256 = "f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa"
MAX_JSON = 2 * 1024**2
MAX_FILE = 256 * 1024**2
MAX_TOTAL = 512 * 1024**2
FILES = {"last.pt", "best.pt", "history.json", "config.json", "run.json", "job.json",
         "environment.json", "qualification.json", "result.json", "training.png", "metrics.json"}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _protocol_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected an opaque SHA-256 identifier")
    return value


def _safe(path, root=None):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Cleanup requires explicit, non-traversing absolute paths")
    for item in (path, *path.parents):
        try:
            attributes = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("Cleanup paths must not contain links or junctions")
    path = path.resolve()
    if root is not None and (path == root or not path.is_relative_to(root)):
        raise ValueError("Cleanup path escapes the registered queue root")
    return path


def _regular(path, limit=MAX_FILE):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or not 0 < value.st_size <= limit:
        raise ValueError("Expected a bounded, regular, unshared file")
    return value


def _fingerprint(path):
    value = _regular(path, MAX_TOTAL)
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _sha(path, limit=MAX_FILE):
    path = _safe(path)
    _regular(path, limit)
    before = _fingerprint(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if _fingerprint(path) != before:
        raise ValueError("Cleanup input changed while hashing")
    return digest


def _json_bytes(data):
    if not data or len(data) > MAX_JSON:
        raise ValueError("Cleanup metadata exceeds its bounded JSON size")
    try:
        def pairs(items):
            value = dict(items)
            if len(value) != len(items):
                raise ValueError("Duplicate cleanup metadata key")
            return value
        value = json.loads(data, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid cleanup JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Cleanup metadata must be an object")
    return value


def _read(path):
    path = _safe(path)
    _regular(path, MAX_JSON)
    with path.open("rb") as stream:
        return _json_bytes(stream.read(MAX_JSON + 1))


def _sync(directory):
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write(path, value):
    _safe(path)
    if path.exists():
        _regular(path, MAX_JSON)
    data = _bytes(value)
    if len(data) > MAX_JSON:
        raise ValueError("Cleanup journal exceeds its metadata bound")
    descriptor, name = tempfile.mkstemp(prefix=".queue-cleanup-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _mutex(root):
    path = _safe(root / ".queue-cleanup.lock", root)
    if path.exists():
        _regular(path, 1024)
    with path.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_nlink != 1:
            raise ValueError("Cleanup lock must be unshared")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _spec(configuration, seed):
    if configuration not in {"full", "no_graph", "no_map", "world_ce"} or type(seed) is not int or seed not in {42, 43, 44}:
        raise ValueError("Unapproved private queue recipe")
    return {"version": 1, "purpose": "development_only", "budget_unit": "allocated_gpu_seconds",
        "host": "private", "recipe": "pilot-seed", "configuration": configuration, "seed": seed,
        "epochs": 20, "maximum_seconds": 1800, "payload_sha256": PAYLOAD_SHA256,
        "train_count": 1000, "selection_count": 64, "dependencies": []}


def _queue_job(root, job_id):
    queue = _read(root / "controller/queue.json")
    jobs = queue.get("jobs")
    if (type(queue.get("version")) is not int or queue["version"] != 1 or queue.get("host_seconds") != 21600
            or not isinstance(jobs, list) or len(jobs) > 48):
        raise ValueError("Invalid queue metadata")
    matches = [job for job in jobs if isinstance(job, dict) and job.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError("Cleanup job is not uniquely registered")
    job = matches[0]
    specification = job.get("spec")
    if (not isinstance(specification, dict) or _hash(specification) != _hash(_spec(specification.get("configuration"), specification.get("seed")))
            or _hash(specification) != job_id or job.get("spec_sha256") != job_id
            or job.get("state") != "completed" or job.get("lease") is not None):
        raise ValueError("Cleanup requires a completed immutable private queue job")
    attempts = job.get("attempts")
    cost = job.get("charged_seconds")
    if (not isinstance(attempts, list) or not 1 <= len(attempts) <= 32 or not isinstance(attempts[-1], dict)
            or attempts[-1].get("outcome") != "completed" or type(cost) not in (int, float)
            or not math.isfinite(cost) or not 0 <= cost <= 1800):
        raise ValueError("Completed queue job lacks settled budget evidence")
    _identifier(job.get("checkpoint_sha256"))
    if sum(item.get("charged_seconds", float("inf")) for item in attempts) != cost:
        raise ValueError("Queue job charges differ from settled attempts")
    if attempts[-1].get("charged_seconds", float("inf")) > attempts[-1].get("reserved_seconds", -1):
        raise ValueError("Completed queue job exceeds its reservation")
    return job


def _inactive(job):
    path = job / ".remote-job.lock"
    if path.exists() or path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError("Cleanup refused while the job has an owner lock")


def _verify(root, job_id, envelope):
    if not isinstance(envelope, dict) or set(envelope) != {"queue_receipt", "import_receipt", "export"}:
        raise ValueError("Expected the authenticated verified-result receipt envelope")
    record = _queue_job(root, job_id)
    specification = record["spec"]
    job = _safe(root / "jobs" / job_id, root)
    _inactive(job)
    protocol, config, status = (_read(job / name) for name in ("job.json", "config.json", "status.json"))
    digest = _identifier(protocol.get("protocol_sha256"))
    if (type(protocol.get("version")) is not int or protocol["version"] != 1
            or protocol.get("purpose") != "development qualification; not frozen-study benchmark evidence"
            or _protocol_hash({key: value for key, value in protocol.items() if key != "protocol_sha256"}) != digest
            or protocol.get("epochs") != 20 or protocol.get("maximum_compute_seconds") != 1800
            or protocol.get("config_sha256") != _protocol_hash(config)
            or status.get("state") != "completed" or status.get("protocol_sha256") != digest
            or not isinstance(status.get("result"), dict) or type(status["result"].get("completed_epochs")) is not int
            or status["result"]["completed_epochs"] != 20):
        raise ValueError("Cleanup requires intact completed 20-epoch job provenance")
    if (config.get("seed") != specification["seed"] or config.get("model", {}).get("use_graph") is not (specification["configuration"] != "no_graph")
            or config.get("model", {}).get("use_map") is not (specification["configuration"] != "no_map")
            or config.get("model", {}).get("shared_world_ce_weight") != (.1 if specification["configuration"] == "world_ce" else 0.)):
        raise ValueError("Job configuration differs from the approved queue recipe")
    for key, value in {"batch_size": 1, "accumulation_steps": 32, "amp": False, "num_workers": 0,
                       "checkpoint_seconds": 300, "minimum_free_gib": 8}.items():
        if type(config.get("training", {}).get(key)) is not type(value) or config["training"][key] != value:
            raise ValueError("Job training configuration differs from qualification")
    sources = protocol.get("source_files")
    if not isinstance(sources, dict) or not 2 <= len(sources) <= 64 or not {"remote_jobs.py", "training.py"} <= set(sources):
        raise ValueError("Invalid immutable source index")
    for name, expected in sources.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.py", name):
            raise ValueError("Invalid source filename")
        if _sha(_safe(job / "source/adp" / name, root)) != _identifier(expected):
            raise ValueError("Frozen job source changed")
    if {path.name for path in (job / "source/adp").iterdir()} != set(sources):
        raise ValueError("Unexpected frozen source file")
    if _sha(job / "source-manifest.json", MAX_JSON) != _identifier(protocol.get("source_manifest_sha256")):
        raise ValueError("Job source manifest changed")
    cache = protocol.get("cache_files")
    manifest = _read(job / "cache/manifest.json")
    splits = manifest.get("splits")
    if not isinstance(cache, dict) or not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("Only registered training and selection caches may be retired")
    expected_names = {"manifest.json", "preprocessing.json"}
    all_ids = []
    original = _read(job / "source-manifest.json")
    for split, count in (("train", 1000), ("validation", 64)):
        ids = splits[split]
        if (not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count
                or any(not isinstance(sid, str) or str(uuid.UUID(sid)) != sid for sid in ids)
                or not set(ids) <= set(original.get("splits", {}).get(split, []))
                or protocol.get("split_counts", {}).get(split) != count
                or protocol.get("split_sha256", {}).get(split) != _protocol_hash(ids)):
            raise ValueError("Cache scenario membership differs from its fixed pilot protocol")
        expected_names.update(f"{split}/{sid}.npz" for sid in ids)
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids) or set(cache) != expected_names:
        raise ValueError("Unexpected, overlapping or held-out cache entries")
    for name, expected in cache.items():
        _identifier(expected)
        if name in {"manifest.json", "preprocessing.json"} and _sha(job / "cache" / name, MAX_JSON) != expected:
            raise ValueError("Cache provenance metadata changed")
    for path in (job / "cache").rglob("*"):
        path = _safe(path, root)
        relative = path.relative_to(job / "cache").as_posix()
        if (path.is_dir() and relative not in {"train", "validation"}) or (not path.is_dir() and relative not in cache):
            raise ValueError("Unexpected cache entries are not cleanup targets")
    queue_receipt, imported, exported = (envelope[name] for name in ("queue_receipt", "import_receipt", "export"))
    if not all(isinstance(value, dict) for value in (queue_receipt, imported, exported)):
        raise ValueError("Invalid verified-result receipt objects")
    snapshot_id = _identifier(exported.get("snapshot_id"))
    queue_protocol = _hash({"version": 1, "host_seconds": 21600,
        "jobs": [_spec(cfg, seed) for cfg in ("full", "no_graph", "no_map", "world_ce") for seed in (42, 43, 44)]})
    if (set(exported) != {"job_id", "snapshot_id", "archive_name", "bytes", "sha256"}
            or exported.get("job_id") != job_id or exported.get("archive_name") != snapshot_id + ".zip"
            or type(exported.get("bytes")) is not int or not 0 < exported["bytes"] <= MAX_TOTAL):
        raise ValueError("Export identity or bounded archive metadata differs")
    archive = _safe(root / "exports" / exported["archive_name"], root)
    if archive.stat().st_size != exported["bytes"] or _sha(archive, MAX_TOTAL) != _identifier(exported.get("sha256")):
        raise ValueError("Immutable export archive changed after local verification")
    snapshot = _safe(root / "snapshots" / snapshot_id, root)
    metadata = _read(snapshot / "snapshot.json")
    if (metadata.get("format_version") != 1 or metadata.get("snapshot_id") != snapshot_id
            or _hash({key: value for key, value in metadata.items() if key != "snapshot_id"}) != snapshot_id
            or metadata.get("run_id") != job_id):
        raise ValueError("Snapshot identity differs from this completed job")
    files, checkpoints = metadata.get("files"), metadata.get("checkpoint_metadata")
    required = {"last.pt", "best.pt", "job.json", "config.json", "result.json"}
    if (not isinstance(files, dict) or not required <= set(files) or set(files) - FILES
            or not isinstance(checkpoints, dict) or set(checkpoints) != {"last.pt", "best.pt"}
            or {path.name for path in snapshot.iterdir()} != set(files) | {"snapshot.json"}):
        raise ValueError("Incomplete or unexpected immutable snapshot files")
    total = 0
    for name, entry in files.items():
        if (not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}
                or type(entry.get("size_bytes")) is not int or not 0 < entry["size_bytes"] <= (MAX_JSON if name.endswith(".json") else MAX_FILE)):
            raise ValueError("Invalid immutable snapshot file index")
        path = snapshot / name
        if path.stat().st_size != entry["size_bytes"] or _sha(path) != _identifier(entry["sha256"]):
            raise ValueError("Immutable snapshot bytes differ from the local receipt")
        total += entry["size_bytes"]
    if total > MAX_TOTAL:
        raise ValueError("Excessive immutable snapshot size")
    expected_provenance = {"queue_job_spec": specification, "queue_job_id": job_id,
        "payload_sha256": PAYLOAD_SHA256, "queue_protocol_sha256": queue_protocol, "allocated_device_count": 1,
        "job_protocol_sha256": digest, "config_sha256": protocol["config_sha256"],
        "manifest_sha256": cache["manifest.json"], "preprocessing_sha256": cache["preprocessing.json"]}
    if metadata.get("provenance") != expected_provenance:
        raise ValueError("Verified snapshot belongs to another queue protocol")
    for name in ("job.json", "config.json", "result.json"):
        expected = {"job.json": protocol, "config.json": config, "result.json": status}[name]
        if _read(snapshot / name) != expected or _sha(job / "run" / name, MAX_JSON) != files[name]["sha256"]:
            raise ValueError("Preserved job and export metadata differ")
    for name, saved in checkpoints.items():
        if (not isinstance(saved, dict) or saved.get("config") != config or saved.get("manifest_hash") != cache["manifest.json"]
                or saved.get("preprocessing_hash") != cache["preprocessing.json"] or saved.get("partial_epoch") is not False
                or type(saved.get("completed_epochs")) is not int or not 1 <= saved["completed_epochs"] <= 20
                or type(saved.get("global_step")) is not int or saved["global_step"] <= 0):
            raise ValueError("Verified checkpoint metadata differs from completed training")
    if checkpoints["last.pt"]["completed_epochs"] != 20 or record["checkpoint_sha256"] != files["last.pt"]["sha256"]:
        raise ValueError("Queue completion checkpoint differs from the canonical local result")
    manifest_hash = _sha(snapshot / "snapshot.json", MAX_JSON)
    shared = {"snapshot_id": snapshot_id, "manifest_sha256": manifest_hash, "files": files, "verified": True}
    if (type(queue_receipt.get("version")) is not int or queue_receipt["version"] != 1 or queue_receipt.get("job_id") != job_id
            or queue_receipt.get("job_spec") != specification or queue_receipt.get("payload_sha256") != PAYLOAD_SHA256
            or queue_receipt.get("archive_sha256") != exported["sha256"]
            or queue_receipt.get("queue_protocol_sha256") != queue_protocol or queue_receipt.get("allocated_device_count") != 1
            or queue_receipt.get("checkpoint_validation") != "strict_cpu_observed_only_completed"
            or type(imported.get("format_version")) is not int or imported["format_version"] != 1
            or any(item.get("verified") is not True for item in (queue_receipt, imported))
            or imported.get("checkpoint_validation") != "cpu_weights_only"
            or any(receipt.get(key) != value for receipt in (queue_receipt, imported) for key, value in shared.items())):
        raise ValueError("Durable strict CPU verification receipts do not match these exact remote bytes")
    for receipt in (queue_receipt, imported):
        local = receipt.get("local_directory")
        timestamp = receipt.get("verified_at")
        if (not isinstance(local, str) or len(local) > 4096 or local.replace("\\", "/").split("/")[-1] != snapshot_id
                or not isinstance(timestamp, str) or len(timestamp) > 64 or datetime.fromisoformat(timestamp).tzinfo is None):
            raise ValueError("Receipt is missing durable local-copy provenance")
    if queue_receipt["local_directory"] != imported["local_directory"]:
        raise ValueError("Local verification receipts refer to different canonical copies")
    targets = {f"jobs/{job_id}/cache/{name}": value for name, value in cache.items() if name.endswith(".npz")}
    targets.update({f"jobs/{job_id}/run/{name}": files[name]["sha256"] for name in ("last.pt", "best.pt")})
    context = {"job_id": job_id, "job_protocol_sha256": digest, "snapshot_id": snapshot_id,
        "envelope_sha256": _hash(envelope), "status_sha256": _hash(status)}
    return job, context, targets


def cleanup_completed_queue_job(root, job_id, receipt, *, execute=False):
    """Plan by default; retire only exact owned files after matching verified receipts."""
    root = _safe(root)
    if not root.is_dir():
        raise ValueError("Expected the existing private queue directory")
    job_id = _identifier(job_id)
    # Round-trip the caller's object through the same bounded, finite JSON boundary.
    receipt = _json_bytes(_bytes(receipt))
    with _mutex(root):
        job, context, targets = _verify(root, job_id, receipt)
        directory = _safe(root / "cleanup", root)
        journal_path = _safe(directory / (job_id + ".json"), root)
        previous = _read(journal_path) if journal_path.exists() else None
        if previous is not None and (previous.get("context") != context or previous.get("state") not in {"in_progress", "completed"}
                or previous.get("receipt_sha256") != _hash({key: value for key, value in previous.items() if key != "receipt_sha256"})):
            raise ValueError("Existing cleanup journal is corrupt or belongs to different verified results")
        prior = {entry["path"]: entry for entry in previous.get("files", [])} if previous else {}
        if previous and (len(prior) != len(previous["files"]) or set(prior) != set(targets)):
            raise ValueError("Existing cleanup journal has a different deletion set")
        files = []
        for name, digest in sorted(targets.items()):
            path = _safe(root / name, root)
            old = prior.get(name)
            if not path.exists():
                if old is None or old.get("sha256") != digest:
                    raise ValueError("Cleanup input is missing without a matching durable plan")
                files.append({**old, "already_absent": True})
                continue
            fingerprint = _fingerprint(path)
            if _sha(path) != digest or _fingerprint(path) != fingerprint:
                raise ValueError("Registered cleanup file changed before preflight")
            if old is not None and (old.get("sha256") != digest or old.get("fingerprint") != fingerprint):
                raise ValueError("Cleanup input was replaced after the durable plan")
            files.append({"path": name, "sha256": digest, "size_bytes": fingerprint[2],
                          "fingerprint": fingerprint, "already_absent": False})
        if previous and previous["state"] == "completed":
            if not all(entry["already_absent"] for entry in files):
                raise ValueError("Files reappeared after completed cleanup; they were preserved")
            return previous
        plan = {"version": 1, "state": "planned", "context": context, "files": files,
            "bytes_to_free": sum(entry["size_bytes"] for entry in files if not entry["already_absent"]),
            "preserved": "shared payload, source, metadata, logs, ledgers, immutable snapshots and ZIP exports"}
        if not execute:
            return plan
        directory.mkdir(exist_ok=True)
        _sync(root)
        plan.update(state="in_progress", updated_at=datetime.now(timezone.utc).isoformat())
        plan["receipt_sha256"] = _hash(plan)
        _write(journal_path, plan)
        for entry in files:
            if entry["already_absent"]:
                continue
            _inactive(job)
            _queue_job(root, job_id)
            if _hash(_read(job / "status.json")) != context["status_sha256"]:
                raise ValueError("Job status changed before deletion")
            path = _safe(root / entry["path"], root)
            if (_fingerprint(path) != entry["fingerprint"] or _sha(path) != entry["sha256"]
                    or _fingerprint(path) != entry["fingerprint"]):
                raise ValueError("Cleanup file changed immediately before deletion")
            path.unlink()
            _sync(path.parent)
        plan.pop("receipt_sha256")
        plan.update(state="completed", updated_at=datetime.now(timezone.utc).isoformat())
        plan["receipt_sha256"] = _hash(plan)
        _write(journal_path, plan)
        return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.root != PRIVATE_ROOT:
        raise ValueError("CLI cleanup is restricted to the fixed private development queue")
    receipt = _json_bytes(sys.stdin.buffer.read(MAX_JSON + 1))
    result = cleanup_completed_queue_job(args.root, args.job_id, receipt, execute=args.execute)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
