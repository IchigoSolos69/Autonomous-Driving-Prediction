"""Plan and retire only verified, completed independent development-job inputs.

The authenticated controller supplies a durable local-copy receipt with its exact
export_manifest. Shared studies, source code and audit metadata are never targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from adp.remote_transfer import MAX_JSON_BYTES, _checkpoint, _identifier, _json_bytes, _sync_directory


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _links(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _safe(path, root=None):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Cleanup paths must be explicit absolute paths without traversal")
    for parent in (path, *path.parents):
        if _links(parent):
            raise ValueError("Cleanup paths and their ancestors must not be links or junctions")
    resolved = path.resolve()
    if root is not None and (resolved == root or not resolved.is_relative_to(root)):
        raise ValueError("Cleanup target is outside the explicitly owned job directory")
    return resolved


def _json(path):
    path = _safe(path)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Expected bounded regular cleanup metadata")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Cleanup metadata must contain an object")
    return value


def _sha(path):
    if not path.is_file() or _links(path):
        raise ValueError("Cleanup checks require regular unlinked files")
    if path.stat().st_nlink != 1:
        raise ValueError("Hard-linked or shared cleanup files are not supported")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _alive(pid):
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.windll.kernel32.GetLastError() == 5
        try:
            code = ctypes.c_ulong()
            return not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _inactive(root):
    for name in (".remote-job.lock", ".controller.lock"):
        if (root / name).exists() or _links(root / name):
            raise ValueError("Cleanup refused while a job or transfer owner lock exists")
    if (root / "launch.json").exists() and _alive(_json(root / "launch.json").get("pid")):
        raise ValueError("Cleanup refused while the job's training process is alive")


def _write(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".cleanup-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _receipt(receipt, protocol):
    if (not isinstance(receipt, dict) or receipt.get("format_version") != 1
            or receipt.get("verified") is not True or receipt.get("checkpoint_validation") != "cpu_weights_only"):
        raise ValueError("A durable verified local-result receipt is required")
    manifest = receipt.get("export_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Receipt must include its exact verified export_manifest")
    snapshot_id = _identifier(receipt.get("snapshot_id"))
    core = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    if (manifest.get("snapshot_id") != snapshot_id
            or hashlib.sha256(_json_bytes(core)).hexdigest() != snapshot_id
            or hashlib.sha256(_json_bytes(manifest)).hexdigest() != receipt.get("manifest_sha256")
            or manifest.get("files") != receipt.get("files")):
        raise ValueError("Receipt and exported manifest hashes differ")
    provenance = manifest.get("provenance", {})
    if (not isinstance(provenance, dict) or provenance.get("job_protocol_sha256") != protocol["protocol_sha256"]
            or provenance.get("config_sha256") != protocol["config_sha256"]
            or provenance.get("manifest_sha256") != protocol["cache_files"]["manifest.json"]
            or provenance.get("preprocessing_sha256") != protocol["cache_files"]["preprocessing.json"]):
        raise ValueError("Verified result belongs to a different job protocol")
    files, checkpoints = manifest.get("files", {}), manifest.get("checkpoint_metadata", {})
    if (not isinstance(files, dict) or not isinstance(checkpoints, dict)
            or not {"last.pt", "best.pt"} <= set(files) or not {"last.pt", "best.pt"} <= set(checkpoints)):
        raise ValueError("Verified local last.pt and best.pt are required before input cleanup")
    for name in ("last.pt", "best.pt"):
        if not isinstance(files[name], dict) or not isinstance(checkpoints[name], dict):
            raise ValueError("Invalid verified checkpoint index")
        _identifier(files[name].get("sha256"))
        if (_hash(checkpoints[name].get("config")) != protocol["config_sha256"]
                or checkpoints[name].get("manifest_hash") != protocol["cache_files"]["manifest.json"]
                or checkpoints[name].get("preprocessing_hash") != protocol["cache_files"]["preprocessing.json"]):
            raise ValueError("Verified checkpoint configuration or data identity differs")
    if checkpoints["last.pt"].get("completed_epochs") != protocol["epochs"]:
        raise ValueError("Verified local result is not the completed final checkpoint")
    return manifest


def plan_cleanup(jobroot, receipt, paths, *, leases_file=None):
    """Preflight the entire deletion set without changing any files."""
    root = _safe(jobroot)
    if root.name != "server-pilot-v1" or not root.is_dir():
        raise ValueError("Cleanup is restricted to the independently owned server-pilot-v1 job")
    _inactive(root)
    protocol, status = _json(root / "job.json"), _json(root / "status.json")
    digest = _identifier(protocol.get("protocol_sha256"))
    if (protocol.get("version") != 1 or _hash({key: value for key, value in protocol.items() if key != "protocol_sha256"}) != digest
            or protocol.get("purpose") != "development qualification; not frozen-study benchmark evidence"):
        raise ValueError("Cleanup requires an intact independent development-job protocol")
    if (not isinstance(protocol.get("cache_files"), dict)
            or not {"manifest.json", "preprocessing.json"} <= set(protocol["cache_files"])
            or isinstance(protocol.get("epochs"), bool) or not isinstance(protocol.get("epochs"), int)
            or protocol["epochs"] < 1):
        raise ValueError("Invalid completed-job data or epoch protocol")
    _identifier(protocol.get("config_sha256"))
    _identifier(protocol.get("source_manifest_sha256"))
    if (status.get("state") != "completed" or status.get("protocol_sha256") != digest
            or not isinstance(status.get("result"), dict)
            or status["result"].get("completed_epochs") != protocol.get("epochs")):
        raise ValueError("Cleanup requires a completed job with the matching final epoch")
    if (_hash(_json(root / "config.json")) != protocol["config_sha256"]
            or _sha(_safe(root / "source-manifest.json")) != protocol["source_manifest_sha256"]):
        raise ValueError("Job audit provenance changed")
    lease_path = _safe(leases_file or root / "cleanup-leases.json")
    if lease_path != root / "cleanup-leases.json":
        raise ValueError("The explicit cleanup lease ledger must live in the owned job root")
    leases = _json(lease_path)
    if (set(leases) != {"version", "job_protocol_sha256", "cache_owned", "consumers", "payload_files"}
            or leases.get("version") != 1 or leases.get("job_protocol_sha256") != digest
            or leases.get("cache_owned") is not True or not isinstance(leases.get("consumers"), dict)
            or not isinstance(leases.get("payload_files"), dict)):
        raise ValueError("Cleanup requires an explicit exclusive-cache lease ledger")
    if any(value != "released" for value in leases["consumers"].values()):
        raise ValueError("Cleanup refused while any configured cloud or other consumer still needs these inputs")
    manifest = _receipt(receipt, protocol)
    requested = sorted({_safe(path, root).relative_to(root).as_posix() for path in paths})
    if not requested:
        raise ValueError("Specify at least one explicit cleanup target")
    if any(left != right and right.startswith(left + "/") for left in requested for right in requested):
        raise ValueError("Overlapping cleanup targets are not supported")
    context = {"job_protocol_sha256": digest, "receipt_sha256": _hash(receipt),
               "status_sha256": _hash(status), "leases_sha256": _hash(leases), "requested_paths": requested}
    journal_path = root / "cleanup-receipt.json"
    journal = _json(journal_path) if journal_path.exists() else None
    if journal is not None and (journal.get("context") != context or journal.get("receipt_sha256") != _hash(
            {key: value for key, value in journal.items() if key != "receipt_sha256"})):
        raise ValueError("Existing cleanup receipt belongs to another plan or is corrupt")
    previous = {entry["path"]: entry for entry in journal.get("files", [])} if journal else {}
    expected, directories = {}, set()
    cache_files = protocol.get("cache_files", {})
    for name, digest_value in cache_files.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or not (name in {"manifest.json", "preprocessing.json"} or
                        len(relative.parts) == 2 and relative.parts[0] in {"train", "validation"} and relative.suffix == ".npz")):
            raise ValueError("Protocol contains foreign or held-out cache paths")
        _identifier(digest_value)
    for target in requested:
        if target in {"cache", "cache/train", "cache/validation"}:
            matching = {f"cache/{name}": value for name, value in cache_files.items()
                        if target == "cache" or name.startswith(target.removeprefix("cache/") + "/")}
            expected.update(matching)
            directories.add(target)
            directory = root / target
            if directory.exists():
                if not directory.is_dir():
                    raise ValueError("Expected a job-owned cache directory")
                for path in directory.rglob("*"):
                    path = _safe(path, root)
                    relative = path.relative_to(root).as_posix()
                    if path.is_dir():
                        if relative not in {"cache/train", "cache/validation"}:
                            raise ValueError("Unexpected cache subdirectory")
                        directories.add(relative)
                    elif relative not in matching:
                        raise ValueError("Unexpected cache file; shared data cannot be cleaned")
        elif target in {"run/last.pt", "run/best.pt"}:
            expected[target] = manifest["files"][Path(target).name]["sha256"]
        elif "/" not in target and target.endswith(".zip") and target in leases["payload_files"]:
            expected[target] = _identifier(leases["payload_files"][target])
        else:
            raise ValueError("Cleanup target is not an allowlisted owned cache, captured checkpoint or payload ZIP")
    for name in ("last.pt", "best.pt"):
        path, key = _safe(root / "run" / name, root), f"run/{name}"
        if path.exists():
            if _sha(path) != manifest["files"][name]["sha256"]:
                raise ValueError("Checkpoint changed after the verified local result")
            data = path.read_bytes()
            if _checkpoint(data) != manifest["checkpoint_metadata"][name]:
                raise ValueError("Checkpoint metadata differs from the verified local result")
        elif key not in previous:
            raise ValueError("A verified source checkpoint is missing without a cleanup receipt")
    files = []
    for name, expected_hash in sorted(expected.items()):
        path = _safe(root / name, root)
        if not path.exists():
            if name not in previous or previous[name]["sha256"] != expected_hash:
                raise ValueError("Cleanup input is missing without an existing matching cleanup receipt")
            files.append({**previous[name], "already_absent": True})
            continue
        if not path.is_file() or _sha(path) != expected_hash:
            raise ValueError("Cleanup file hash differs from its immutable ownership index")
        files.append({"path": name, "sha256": expected_hash, "size_bytes": path.stat().st_size,
                      "fingerprint": _fingerprint(path), "already_absent": False})
    return {"version": 1, "job_root": str(root), "context": context, "files": files,
            "directories": sorted(directories, key=lambda value: (-value.count("/"), value)),
            "bytes_to_free": sum(entry["size_bytes"] for entry in files if not entry["already_absent"])}


def cleanup_completed_job(jobroot, receipt, paths, *, leases_file=None):
    """Execute a fully checked plan, preserving its durable, idempotent audit receipt."""
    root = _safe(jobroot)
    if root.name != "server-pilot-v1" or not root.is_dir():
        raise ValueError("Cleanup is restricted to the independently owned server-pilot-v1 job")
    paths = list(paths)
    lock = root / ".cleanup.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        plan = plan_cleanup(root, receipt, paths, leases_file=leases_file)
        journal = {**plan, "state": "in_progress", "updated_at": datetime.now(timezone.utc).isoformat()}
        journal["receipt_sha256"] = _hash(journal)
        journal_path = root / "cleanup-receipt.json"
        _write(journal_path, journal)
        # Re-check the complete set immediately before the first filesystem mutation.
        check = plan_cleanup(root, receipt, paths, leases_file=leases_file)
        if check != plan:
            raise ValueError("Cleanup inputs changed during preflight")
        for entry in plan["files"]:
            if entry["already_absent"]:
                continue
            _inactive(root)
            if (_hash(_json(root / "status.json")) != plan["context"]["status_sha256"]
                    or _hash(_json(root / "cleanup-leases.json")) != plan["context"]["leases_sha256"]):
                raise ValueError("Job status or dependency leases changed before deletion")
            path = _safe(root / entry["path"], root)
            if (_fingerprint(path) != entry["fingerprint"] or _sha(path) != entry["sha256"]
                    or _fingerprint(path) != entry["fingerprint"]):
                raise ValueError("Cleanup input changed before deletion")
            path.unlink()
        for name in plan["directories"]:
            path = _safe(root / name, root)
            if path.is_dir():
                path.rmdir()
        journal.update(state="completed", updated_at=datetime.now(timezone.utc).isoformat())
        journal.pop("receipt_sha256")
        journal["receipt_sha256"] = _hash(journal)
        _write(journal_path, journal)
        return journal
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--receipt-file", type=Path, required=True)
    parser.add_argument("--leases-file", type=Path)
    parser.add_argument("--paths", nargs="+", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    function = cleanup_completed_job if args.execute else plan_cleanup
    result = function(args.job_root, _json(args.receipt_file), args.paths, leases_file=args.leases_file)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
