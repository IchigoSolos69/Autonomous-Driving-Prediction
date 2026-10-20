"""Checksummed saved-notebook evidence and immutable exports; never starts training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import zipfile

PAYLOAD_SHA256 = "f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa"
CONFIGURATIONS = ("full", "no_graph", "no_map", "world_ce")
SEEDS = (42, 43, 44)
MAX_JSON_BYTES = 128 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024**2
RESULT_FILES = {"last.pt", "best.pt", "history.json", "config.json", "run.json", "job.json", "environment.json",
                "qualification.json", "result.json", "training.png", "metrics.json"}
DEFAULT_CONFIG = Path("D:/InteractionLab/remote-compute/cloud-folders.json")
CONNECTION_PROBE = {"nonce":"interaction-lab-drive-check-20261020-v1",
    "purpose":"Interaction Lab Drive connection check; no training"}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Expected an opaque SHA-256 identity")
    return value


def _path(value):
    value = Path(value)
    if not value.is_absolute() or ".." in value.parts or any(path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) for path in (value, *value.parents)):
        raise ValueError("Saved evidence paths cannot contain traversal, links or junctions")
    return value


def _json(path):
    with _path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Saved evidence JSON exceeds its size bound")
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite evidence")))
    if not isinstance(value, dict):
        raise ValueError("Saved evidence must be a JSON object")
    return value


def _sha(path):
    with _path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stamp(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("Saved evidence requires a bounded UTC observation time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.timestamp() > datetime.now(timezone.utc).timestamp() + 300:
        raise ValueError("Saved evidence observation time is invalid or in the future")
    return value


def expected_plan(host):
    if host not in ("colab", "kaggle"):
        raise ValueError("Only the approved private notebook hosts are supported")
    specs = [{"version":1, "purpose":"development_only", "budget_unit":"allocated_gpu_seconds", "host":host,
        "recipe":"pilot-seed", "configuration":configuration, "seed":seed, "epochs":20, "maximum_seconds":1800,
        "payload_sha256":PAYLOAD_SHA256, "train_count":1000, "selection_count":64, "dependencies":[]}
        for configuration in CONFIGURATIONS for seed in SEEDS]
    return {_digest(_bytes(spec)):spec for spec in specs}, _digest(_bytes({"version":1,"host_seconds":21600,"jobs":specs}))


def validate_evidence(value, host):
    expected, protocol = expected_plan(host)
    required = {"version", "kind", "host", "queue", "payload_sha256", "queue_protocol_sha256", "published_at",
        "worker_observed_at", "state", "charged_gpu_seconds", "completed_jobs", "planned_jobs", "completed_epochs",
        "planned_epochs", "jobs", "exports"}
    if (set(value) != required or type(value["version"]) is not int or value["version"] != 1
            or value["kind"] != "saved_notebook_queue_evidence" or value["host"] != host
            or value["queue"] != "development-v1" or value["payload_sha256"] != PAYLOAD_SHA256
            or value["queue_protocol_sha256"] != protocol or value["planned_jobs"] != 12 or value["planned_epochs"] != 240
            or value["state"] not in ("running", "waiting", "paused", "completed", "blocked", "failed")):
        raise ValueError("Saved evidence differs from the approved finite development protocol")
    _stamp(value["published_at"]); _stamp(value["worker_observed_at"])
    charge = value["charged_gpu_seconds"]
    if isinstance(charge, bool) or not isinstance(charge, (int, float)) or not math.isfinite(charge) or not 0 <= charge <= 21600:
        raise ValueError("Saved evidence GPU charge exceeds the approved ledger")
    jobs = value["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 12 or {item.get("id") for item in jobs if isinstance(item, dict)} != set(expected):
        raise ValueError("Saved evidence requires all twelve unique approved jobs")
    for job in jobs:
        if (set(job) != {"id", "state", "completed_epochs"} or job["state"] not in ("queued", "leased", "running", "manual_start_required", "completed", "failed", "blocked", "paused")
                or type(job["completed_epochs"]) is not int or not 0 <= job["completed_epochs"] <= 20
                or (job["state"] == "completed" and job["completed_epochs"] != 20)):
            raise ValueError("Saved job state or complete-future epoch count is inconsistent")
    if (type(value["completed_jobs"]) is not int or value["completed_jobs"] != sum(job["state"] == "completed" for job in jobs)
            or type(value["completed_epochs"]) is not int or value["completed_epochs"] != sum(job["completed_epochs"] for job in jobs)
            or (value["state"] == "completed" and value["completed_jobs"] != 12)):
        raise ValueError("Saved aggregate progress does not match its jobs")
    exports = value["exports"]
    if not isinstance(exports, list) or len(exports) > 12:
        raise ValueError("Saved export list exceeds the finite queue")
    seen = set()
    for item in exports:
        if (not isinstance(item, dict) or set(item) != {"job_id", "snapshot_id", "archive_name", "bytes", "sha256", "checkpoints"}
                or item["job_id"] not in expected or item["job_id"] in seen or item["archive_name"] != _identity(item["snapshot_id"]) + ".zip"
                or type(item["bytes"]) is not int or not 0 < item["bytes"] <= MAX_ARCHIVE_BYTES
                or item["checkpoints"] != ["best.pt", "last.pt"]):
            raise ValueError("Saved export identity, size or checkpoint inventory is invalid")
        _identity(item["sha256"]); seen.add(item["job_id"])
    completed = {job["id"] for job in jobs if job["state"] == "completed"}
    if not completed.issubset(seen):
        raise ValueError("A completed saved job is missing its reproducible result archive")
    return value


def _atomic(path, raw):
    descriptor, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_saved_queue(root, destination, host="colab"):
    """Called in a foreground notebook after its owner explicitly mounts Drive."""
    root, destination = _path(root), _path(destination)
    expected, protocol = expected_plan(host)
    status, ledger = _json(root / "worker-status.json"), _json(root / "controller/queue.json")
    source_jobs = [job for job in ledger["jobs"] if job["spec"]["host"] == host]
    if len(source_jobs) != 12 or any(expected.get(job["id"]) != job["spec"] for job in source_jobs):
        raise ValueError("Source queue differs from the approved twelve development recipes")
    jobs = []
    for job in source_jobs:
        history = root / "jobs" / _identity(job["id"]) / "run/history.json"
        count = len(_json(history)["epochs"]) if history.is_file() else 0
        jobs.append({"id":job["id"], "state":job["state"], "completed_epochs":count})
    exports = []
    for record in status.get("exports", []):
        if set(record) != {"job_id", "snapshot_id", "archive_name", "bytes", "sha256"} or record["job_id"] not in expected:
            raise ValueError("Source export is not an approved opaque job result")
        identifier = _identity(record["snapshot_id"])
        if record["archive_name"] != identifier + ".zip" or type(record["bytes"]) is not int or not 0 < record["bytes"] <= MAX_ARCHIVE_BYTES:
            raise ValueError("Source export name or size is invalid")
        source = _path(root / "exports" / record["archive_name"])
        if source.stat().st_size != record["bytes"] or _sha(source) != _identity(record["sha256"]):
            raise ValueError("Source result archive differs from its authenticated notebook record")
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if (len(names) > 16 or len(names) != len(set(names)) or not set(names).issubset(RESULT_FILES | {"snapshot.json"})
                    or any(Path(name).name != name or "/" in name or "\\" in name for name in names)
                    or sum(entry.file_size for entry in archive.infolist()) > MAX_ARCHIVE_BYTES):
                raise ValueError("Source archive contains excessive, repeated or unbounded paths")
            entry = archive.getinfo("snapshot.json")
            if entry.file_size > MAX_JSON_BYTES:
                raise ValueError("Source snapshot metadata exceeds its bound")
            snapshot = json.loads(archive.read(entry))
            provenance = snapshot["provenance"]
            if (snapshot["snapshot_id"] != identifier or _digest(_bytes({key:value for key, value in snapshot.items() if key != "snapshot_id"})) != identifier
                    or set(names) != set(snapshot["files"]) | {"snapshot.json"} or provenance["queue_job_id"] != record["job_id"]
                    or provenance["queue_job_spec"] != expected[record["job_id"]]
                    or provenance["payload_sha256"] != PAYLOAD_SHA256 or provenance["queue_protocol_sha256"] != protocol
                    or not {"best.pt", "last.pt"}.issubset(archive.namelist())):
                raise ValueError("Source snapshot provenance/checkpoint inventory differs")
        exports.append({**record, "checkpoints":["best.pt", "last.pt"]})
    evidence = validate_evidence({"version":1, "kind":"saved_notebook_queue_evidence", "host":host, "queue":"development-v1",
        "payload_sha256":PAYLOAD_SHA256, "queue_protocol_sha256":protocol, "published_at":datetime.now(timezone.utc).isoformat(),
        "worker_observed_at":status["updated_at"], "state":status["state"], "charged_gpu_seconds":ledger["hosts"][host]["charged_seconds"],
        "completed_jobs":sum(job["state"] == "completed" for job in jobs), "planned_jobs":12,
        "completed_epochs":sum(job["completed_epochs"] for job in jobs), "planned_epochs":240, "jobs":jobs, "exports":exports}, host)
    destination.mkdir(parents=True, exist_ok=True)
    archives = _path(destination / "exports")
    archives.mkdir(exist_ok=True)
    for record in exports:
        source = root / "exports" / record["archive_name"]
        version = _path(archives / record["snapshot_id"])
        target = _path(version / record["archive_name"])
        if version.exists():
            if target.stat().st_size != record["bytes"] or _sha(target) != record["sha256"]:
                raise ValueError("Immutable Drive archive collision; existing bytes were preserved")
            continue
        stage = Path(tempfile.mkdtemp(prefix=".copy-", dir=archives))
        temporary = stage / record["archive_name"]
        try:
            with temporary.open("xb") as stream, source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, stream)
                stream.flush(); os.fsync(stream.fileno())
            if _sha(temporary) != record["sha256"]:
                raise ValueError("Drive archive copy checksum differs; no status was published")
            # Never replace a prior versioned archive, even if another exporter raced us.
            # A nonempty immutable directory cannot be replaced by directory rename.
            stage.rename(version)
        finally:
            if stage.exists():
                temporary.unlink(missing_ok=True)
                stage.rmdir()
    raw = _bytes(evidence)
    _atomic(destination / "status.json", _bytes({"sha256":_digest(raw), "payload":evidence}))
    return evidence


def read_saved_queue(directory, host="colab"):
    root = _path(directory)
    envelope = _json(root / "status.json")
    if set(envelope) != {"sha256", "payload"} or _digest(_bytes(envelope["payload"])) != _identity(envelope["sha256"]):
        raise ValueError("Saved notebook status checksum differs")
    evidence = validate_evidence(envelope["payload"], host)
    available = sum((_path(root / "exports" / item["snapshot_id"] / item["archive_name"])).is_file()
        and (root / "exports" / item["snapshot_id"] / item["archive_name"]).stat().st_size == item["bytes"] for item in evidence["exports"])
    return {"status":"checked", "phase":"saved_notebook_status", "scope":"saved_notebook_heartbeat",
        "source_observed_at":evidence["worker_observed_at"], "published_at":evidence["published_at"],
        "completed_jobs":evidence["completed_jobs"], "completed_epochs":evidence["completed_epochs"],
        "available_archives":available, "expected_archives":len(evidence["exports"]),
        "message":f"Saved {host.title()} evidence: {evidence['completed_jobs']}/12 jobs, {evidence['completed_epochs']}/240 epochs; {available}/{len(evidence['exports'])} result archives available. Saved notebook evidence is not live GPU proof. Hash/strict CPU verification is still required before cleanup or model use."}


def configure_folder(directory, host="colab", config=DEFAULT_CONFIG):
    expected_plan(host)
    directory, config = _path(directory), _path(config)
    if not directory.is_dir():
        raise ValueError("Choose an existing locally synced queue folder after Drive sign-in and export")
    value = _json(config) if config.exists() else {"version":1, "folders":{}}
    if set(value) != {"version", "folders"} or value["version"] != 1 or not isinstance(value["folders"], dict) or not set(value["folders"]).issubset({"colab", "kaggle"}):
        raise ValueError("Existing cloud-folder configuration is invalid and was preserved")
    value["folders"][host] = str(directory)
    config.parent.mkdir(parents=True, exist_ok=True)
    _atomic(config, _bytes(value))
    return {"state":"configured", "host":host, "scope":"saved_notebook_heartbeat", "directory":str(directory)}


def configured_evidence(config=DEFAULT_CONFIG):
    config = _path(config)
    if not config.exists():
        return {}
    value = _json(config)
    if set(value) != {"version", "folders"} or value["version"] != 1 or not isinstance(value["folders"], dict) or not set(value["folders"]).issubset({"colab", "kaggle"}):
        raise ValueError("Cloud-folder configuration is invalid; existing evidence was preserved")
    results = {}
    for host, directory in value["folders"].items():
        try:
            root = _path(directory)
            status = _path(root / "status.json")
            if not status.exists():
                if not root.is_dir() or _json(root / "connection-check.json") != CONNECTION_PROBE:
                    raise ValueError("Expected Drive connection probe is missing or invalid")
                results[host] = {"status":"checked", "phase":"connected_awaiting_evidence", "scope":"drive_connection_probe",
                    "message":"The expected Drive connection probe is readable. Awaiting saved queue evidence; folder availability does not establish a live notebook, GPU state or training progress."}
            else:
                results[host] = read_saved_queue(root, host)
        except (OSError, ValueError, KeyError, TypeError):
            results[host] = {"status":"unavailable", "phase":"saved_source_unavailable", "scope":"saved_notebook_heartbeat",
                "message":"The configured saved-notebook folder is unavailable or invalid. Check Drive sign-in, offline availability and status provenance. Earlier training observations, budgets and verified models remain unchanged."}
    return results


def watch_foreground(process, root, destination, host="colab"):
    """Optional wrapper for an already started, authorized foreground process."""
    try:
        while process.poll() is None:
            try:
                saved = publish_saved_queue(root, destination, host)
                print(json.dumps({"saved_jobs":saved["completed_jobs"], "saved_epochs":saved["completed_epochs"]}), flush=True)
            except Exception as error:
                print("Drive evidence not yet settled; source/leases retained: " + str(error)[:200], flush=True)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("The owned worker has not exited. Preserve its runtime and lease; do not start another owner.", flush=True)
        raise
    result = publish_saved_queue(root, destination, host)
    if process.returncode:
        raise RuntimeError("Foreground training exited unsuccessfully; saved partial evidence was retained")
    return result
