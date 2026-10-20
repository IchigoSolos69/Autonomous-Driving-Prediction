"""Private-server result delivery. Only immutable exports may be removed."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import sys

from adp.data import read_json, write_json
from adp.remote_transfer import (FILES, MAX_FILE_BYTES, MAX_JSON_BYTES, MAX_TOTAL_BYTES,
    MINIMUM_FREE_BYTES, create_snapshot, import_snapshot, prune_snapshot, validate_snapshot)

PRIVATE_JOB_ROOT = Path.home() / "interaction-lab/jobs/server-pilot-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected an opaque SHA-256 snapshot identifier")
    return value


def _json(path):
    path = Path(path)
    if path.is_symlink() or path.is_junction() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Expected bounded regular JSON metadata")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object metadata")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _job(path):
    path = Path(path)
    if path.is_symlink() or path.is_junction() or path.resolve() != PRIVATE_JOB_ROOT.resolve():
        raise ValueError("Controller is restricted to its private server-pilot-v1 job")
    protocol = _json(path / "job.json")
    digest = _id(protocol.get("protocol_sha256"))
    core = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if hashlib.sha256(json.dumps(core, sort_keys=True, allow_nan=False).encode()).hexdigest() != digest:
        raise ValueError("Private job protocol hash differs")
    return path.resolve(), protocol


@contextmanager
def _lock(path):
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("Another controller owns this job; verify its process before removing its lock") from error
    try:
        with stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_at": _now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def _process(pid):
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "posix":
        raise RuntimeError("Remote process inspection is supported on Linux only")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _gpu_processes():
    try:
        response = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
        rows = []
        for line in response.stdout.splitlines():
            pid, memory = (part.strip() for part in line.split(",", 1))
            if pid.isdecimal():
                rows.append({"pid": int(pid), "memory_mib": float(memory) if memory.replace(".", "", 1).isdigit() else None})
        return rows
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def status(job_root):
    root, protocol = _job(job_root)
    saved = _json(root / "status.json") if (root / "status.json").exists() else {"state": "not_started"}
    ledger = _json(root / "compute-ledger.json") if (root / "compute-ledger.json").exists() else {"charged_seconds": 0.}
    if ledger.get("protocol_sha256", protocol["protocol_sha256"]) != protocol["protocol_sha256"]:
        raise ValueError("Private job ledger provenance differs")
    owner_path = root / ".remote-job.lock"
    owner = _json(owner_path) if owner_path.exists() else None
    launch = _json(root / "launch.json") if (root / "launch.json").exists() else {}
    pid = owner.get("pid") if owner else launch.get("pid")
    active = _process(pid)
    gpu = _gpu_processes()
    return {"state": saved.get("state", "unknown"), "protocol_sha256": protocol["protocol_sha256"],
        "purpose": protocol["purpose"], "training": saved,
        "remaining_seconds": max(0., protocol["maximum_compute_seconds"] - ledger["charged_seconds"]),
        "charged_seconds": ledger["charged_seconds"], "training_pid": pid, "process_alive": active,
        "owner_lock_present": owner is not None, "training_on_gpu": active and any(row["pid"] == pid for row in gpu),
        "gpu_processes": gpu, "updated_at": _now()}


def _paths(root):
    return root.parent.parent / "exports" / root.name, root / "controller-sync.json"


def _export_result(exported):
    metadata = exported["manifest"]
    path = Path(exported["path"])
    return {"state": "snapshot", "snapshot_id": metadata["snapshot_id"], "files": metadata["files"],
            "snapshot_json_bytes": (path / "snapshot.json").stat().st_size,
            "manifest_sha256": exported["manifest_sha256"]}


def snapshot(job_root):
    root, protocol = _job(job_root)
    with _lock(root / ".controller.lock"):
        exports, state_path = _paths(root)
        state = _json(state_path) if state_path.exists() else {}
        pending = state.get("pending_snapshot_id")
        if pending:
            pending = _id(pending)
            loaded = validate_snapshot(exports / pending)
            return _export_result({"path": str(exports / pending), **loaded})
        run = root / "run"
        if not (run / "last.pt").is_file():
            return {"state": "no_checkpoint"}
        hashes = {name: _sha(run / name) for name in ("last.pt", "best.pt") if (run / name).is_file()}
        if hashes == state.get("last_verified_checkpoint_hashes"):
            return {"state": "unchanged", "snapshot_id": state.get("last_pruned_snapshot_id")}
        exported = create_snapshot(run, exports, root.name, {
            "job_protocol_sha256": protocol["protocol_sha256"], "config_sha256": protocol["config_sha256"],
            "manifest_sha256": protocol["cache_files"]["manifest.json"],
            "preprocessing_sha256": protocol["cache_files"]["preprocessing.json"], "purpose": protocol["purpose"]})
        state.update(pending_snapshot_id=exported["snapshot_id"], updated_at=_now())
        write_json(state_path, state)
        return _export_result(exported)


def prune(job_root, snapshot_id, receipt):
    root, _ = _job(job_root)
    snapshot_id = _id(snapshot_id)
    with _lock(root / ".controller.lock"):
        exports, state_path = _paths(root)
        state = _json(state_path) if state_path.exists() else {}
        pending = state.get("pending_snapshot_id")
        if snapshot_id != (pending if pending else state.get("last_pruned_snapshot_id")):
            raise ValueError("Receipt does not belong to the pending private-job export")
        result = prune_snapshot(exports, snapshot_id, receipt)
        state.update(pending_snapshot_id=None, last_pruned_snapshot_id=snapshot_id,
            last_verified_checkpoint_hashes={name: entry["sha256"] for name, entry in receipt["files"].items()
                                            if name in ("last.pt", "best.pt")}, updated_at=_now())
        write_json(state_path, state)
        return result


def resume(job_root):
    root, _ = _job(job_root)
    with _lock(root / ".controller.lock"):
        current = status(root)
        if current["owner_lock_present"] or current["process_alive"]:
            return {**current, "state": "already_owned"}
        if current["state"] != "paused" or current["remaining_seconds"] <= 0:
            return {"state": "not_resumed", "reason": current["state"], "remaining_seconds": current["remaining_seconds"]}
        source = root / "source"
        if not (source / "adp" / "remote_jobs.py").is_file():
            raise ValueError("Frozen private-job source is missing")
        environment = {**os.environ, "PYTHONPATH": str(source), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        command = [sys.executable, "-m", "adp.remote_jobs", "run", "--root", str(root),
                   "--device", "cuda", "--slice-seconds", str(min(1800., current["remaining_seconds"]))]
        with (root / "controller-training.stdout.log").open("ab") as stdout, \
                (root / "controller-training.stderr.log").open("ab") as stderr:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
        launch = {"pid": process.pid, "started_at": _now(), "source": str(source), "command": command}
        write_json(root / "launch.json", launch)
        return {"state": "launched", **launch}


def _config(path):
    value = _json(path)
    required = {"host", "identityfile", "remote_python", "remote_source", "remote_job", "local_root", "local_staging"}
    if set(value) != required or value["host"] != "aadi@192.168.29.71":
        raise ValueError("Expected the configured private-server connection only")
    for name in required:
        if not isinstance(value[name], str) or not value[name] or any(ord(char) < 32 for char in value[name]):
            raise ValueError("Connection settings must be nonempty strings without control characters")
    for name in ("remote_python", "remote_source", "remote_job"):
        path = PurePosixPath(value[name])
        if not path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError("Remote paths must be absolute POSIX paths without traversal")
    if value["remote_job"] != "/home/aadi/interaction-lab/jobs/server-pilot-v1":
        raise ValueError("Only the explicitly configured development job is supported")
    for name in ("identityfile", "local_root", "local_staging"):
        if not Path(value[name]).is_absolute():
            raise ValueError("Local paths must be absolute")
    local, staging = Path(value["local_root"]).resolve(), Path(value["local_staging"]).resolve()
    if local == staging or local.is_relative_to(staging) or staging.is_relative_to(local):
        raise ValueError("Result and incoming directories must be separate")
    return value


def _ssh(config, action, *, snapshot_id=None, receipt=None):
    words = ["env", f"PYTHONPATH={config['remote_source']}", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2",
             config["remote_python"], "-m", "adp.remote_control", action, "--job-root", config["remote_job"]]
    if snapshot_id is not None:
        words.extend(["--snapshot-id", _id(snapshot_id)])
    response = subprocess.run(["ssh", "-i", config["identityfile"], "-o", "BatchMode=yes", "-o",
        "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", config["host"],
        " ".join(shlex.quote(word) for word in words)], input=json.dumps(receipt) if receipt else None,
        text=True, capture_output=True, check=True, timeout=300)
    if len(response.stdout.encode()) > MAX_JSON_BYTES:
        raise ValueError("Remote controller response exceeds metadata limits")
    value = json.loads(response.stdout)
    if not isinstance(value, dict):
        raise ValueError("Remote controller must return a JSON object")
    return value


def _download_index(value):
    snapshot_id = _id(value.get("snapshot_id"))
    files = value.get("files")
    if not isinstance(files, dict) or "last.pt" not in files or set(files) - FILES:
        raise ValueError("Remote snapshot has invalid file names")
    sizes = {}
    for name, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}:
            raise ValueError("Invalid remote snapshot file metadata")
        _id(entry["sha256"])
        size = entry["size_bytes"]
        bound = MAX_JSON_BYTES if name.endswith(".json") else MAX_FILE_BYTES
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= bound:
            raise ValueError("Remote snapshot exceeds file size limits")
        sizes[name] = size
    metadata_size = value.get("snapshot_json_bytes")
    if isinstance(metadata_size, bool) or not isinstance(metadata_size, int) or not 0 < metadata_size <= MAX_JSON_BYTES:
        raise ValueError("Remote snapshot manifest size is invalid")
    if sum(sizes.values()) > MAX_TOTAL_BYTES:
        raise ValueError("Remote snapshot exceeds total size limits")
    sizes["snapshot.json"] = metadata_size
    return snapshot_id, sizes


def _local_directory(path):
    path = Path(path)
    if path.is_symlink() or path.is_junction():
        raise ValueError("Linked local transfer directories are not supported")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _download(config, snapshot_id, sizes, staging):
    remote = str(PurePosixPath(config["remote_job"]).parent.parent / "exports" / "server-pilot-v1" / _id(snapshot_id))
    reader = "import sys; from pathlib import Path; from adp.remote_transfer import _read; " \
             "sys.stdout.buffer.write(_read(Path(sys.argv[1]), int(sys.argv[2])))"
    for name, size in sizes.items():
        destination = staging / name
        if destination.is_symlink() or destination.is_junction():
            raise ValueError("Linked transfer files are not supported")
        command = " ".join(shlex.quote(word) for word in
            ("env", f"PYTHONPATH={config['remote_source']}", config["remote_python"], "-c", reader, f"{remote}/{name}", str(size)))
        with destination.open("wb") as output:
            subprocess.run(["ssh", "-i", config["identityfile"], "-o", "BatchMode=yes", "-o",
                "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", config["host"], command],
                stdout=output, stderr=subprocess.PIPE, check=True, timeout=900)
            output.flush()
            os.fsync(output.fileno())
        if destination.stat().st_size != size:
            raise ValueError("Downloaded snapshot file size differs from its index")


def sync_private_server(config_path, *, resume_training=True):
    """One periodic cycle: deliver, verify, acknowledge, prune, then safely resume."""
    config = _config(config_path)
    local = _local_directory(config["local_root"])
    incoming = _local_directory(config["local_staging"])
    with _lock(incoming / ".private-server-sync.lock"):
        before = _ssh(config, "status")
        exported = _ssh(config, "snapshot")
        delivered = None
        if exported.get("state") == "snapshot":
            snapshot_id, sizes = _download_index(exported)
            projected = sum(sizes.values())
            # Both staging and atomic publication can coexist on the same volume.
            for directory in (incoming, local):
                if shutil.disk_usage(directory).free - 2 * projected < MINIMUM_FREE_BYTES:
                    raise OSError("Transfer paused: preserve the 8 GiB reserve through staging and publication")
            stage = _local_directory(incoming / snapshot_id)
            if set(entry.name for entry in stage.iterdir()) - set(sizes):
                raise ValueError("Unexpected files in pending local transfer")
            _download(config, snapshot_id, sizes, stage)
            delivered = import_snapshot(stage, local)
            if delivered["snapshot_id"] != snapshot_id:
                raise ValueError("Imported snapshot differs from the remote transfer identity")
            _ssh(config, "prune", snapshot_id=snapshot_id, receipt=delivered["receipt"])
            for name in sizes:
                (stage / name).unlink()
            stage.rmdir()
        elif exported.get("state") not in ("no_checkpoint", "unchanged"):
            raise ValueError("Unexpected private-server snapshot response")
        after = _ssh(config, "resume") if resume_training else {
            "state": "not_resumed", "reason": "transfer_only; training resume disabled"}
        result = {"updated_at": _now(), "before": before, "snapshot": exported,
                  "delivered": delivered, "resume": after}
        write_json(local / "last-sync.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "snapshot", "prune", "resume"))
    parser.add_argument("--job-root", type=Path, default=PRIVATE_JOB_ROOT)
    parser.add_argument("--snapshot-id")
    args = parser.parse_args(argv)
    if args.action == "prune":
        raw = sys.stdin.read(MAX_JSON_BYTES + 1)
        if len(raw.encode()) > MAX_JSON_BYTES:
            raise ValueError("Receipt exceeds metadata limits")
        result = prune(args.job_root, args.snapshot_id, json.loads(raw))
    else:
        result = {"status": status, "snapshot": snapshot, "resume": resume}[args.action](args.job_root)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
