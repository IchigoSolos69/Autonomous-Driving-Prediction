"""Bounded, resumable development jobs on a separately qualified compute host.

These jobs do not participate in either frozen benchmark study. Their cache is
an explicitly exported subset of training and selection scenarios only.
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
import random
import socket
import time
import uuid

from adp.data import read_json, write_json


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def select_ids(ids, count, seed=42):
    """Deterministic membership independent of the source manifest's ordering."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Subset count must be a positive integer")
    if len(ids) != len(set(ids)):
        raise ValueError("Source split contains duplicate scenarios")
    return sorted(random.Random(seed).sample(sorted(ids), min(count, len(ids))))


def _source_files(source_root=None):
    directory = Path(source_root) if source_root else Path(__file__).parent
    files = {path.name: _file_hash(path) for path in sorted(directory.glob("*.py"))}
    if "training.py" not in files or "remote_jobs.py" not in files:
        raise ValueError("The source snapshot must contain the adp Python package")
    return files


def _cache_files(root):
    cache = root / "cache"
    if cache.is_symlink() or cache.is_junction():
        raise ValueError("Remote caches must be self-contained, without symbolic links")
    manifest = read_json(cache / "manifest.json")
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "validation"}:
        raise ValueError("Remote cache must contain train and selection validation only")
    expected = {"manifest.json", "preprocessing.json"}
    all_ids = []
    for split in ("train", "validation"):
        ids = splits[split]
        if not isinstance(ids, list) or not ids:
            raise ValueError("Both development splits must contain scenarios")
        for sid in ids:
            if not isinstance(sid, str) or str(uuid.UUID(sid)) != sid:
                raise ValueError("Invalid AV2 scenario identifier")
            expected.add(f"{split}/{sid}.npz")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Development scenarios must be unique and disjoint")
    actual = set()
    for path in cache.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Remote caches must be self-contained, without symbolic links")
        if path.is_file():
            actual.add(path.relative_to(cache).as_posix())
        elif path.is_dir() and path.relative_to(cache).as_posix() not in splits:
            raise ValueError("Unexpected remote cache directory")
    if actual != expected:
        raise ValueError("Remote cache files differ from the exported train/selection manifest")
    return manifest, {name: _file_hash(cache / name) for name in sorted(expected)}


def prepare_job(job_root, config, *, epochs=1, max_runtime_seconds=1800, source_root=None):
    """Freeze an already staged cache and source snapshot; never copy data here.

    The caller stages cache/{manifest,preprocessing}.json, selected NPZ files and
    source-manifest.json. The latter is provenance metadata, not held-out labels.
    """
    root = Path(job_root).resolve()
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epoch count must be a positive integer")
    if not math.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0:
        raise ValueError("Compute budget must be finite and positive")
    manifest, cache_files = _cache_files(root)
    original = read_json(root / "source-manifest.json")
    for split in ("train", "validation"):
        if not set(manifest["splits"][split]).issubset(original["splits"][split]):
            raise ValueError("Development scenarios must originate in the matching official train/selection split")
    frozen = json.loads(json.dumps(config, allow_nan=False))
    frozen.setdefault("training", {}).update(batch_size=1, accumulation_steps=32, amp=False,
        num_workers=0, checkpoint_seconds=300, minimum_free_gib=8)
    value = {"version": 1, "purpose": "development qualification; not frozen-study benchmark evidence",
        "epochs": epochs, "maximum_compute_seconds": float(max_runtime_seconds),
        "config_sha256": _hash(frozen), "cache_files": cache_files,
        "source_manifest_sha256": _file_hash(root / "source-manifest.json"),
        "source_files": _source_files(source_root),
        "split_counts": {name: len(ids) for name, ids in manifest["splits"].items()},
        "split_sha256": {name: _hash(ids) for name, ids in manifest["splits"].items()},
        "limits": "One optimizer group or validation batch plus startup/checkpoint I/O can overrun a slice."
    }
    value["protocol_sha256"] = _hash(value)
    if (root / "job.json").exists():
        if read_json(root / "job.json") != value or read_json(root / "config.json") != frozen:
            raise ValueError("Remote job protocol is immutable; use a new job directory")
    else:
        write_json(root / "config.json", frozen)
        write_json(root / "job.json", value)
    return value


@contextmanager
def _owner_lock(root):
    path = root / ".remote-job.lock"
    owner = {"pid": os.getpid(), "hostname": socket.gethostname(), "started_at": _now()}
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("Remote job already has an owner; verify that process before removing its lock") from error
    try:
        with stream:
            json.dump(owner, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def run_job(job_root, *, device="cuda", slice_seconds=1800):
    """Resume this job's committed checkpoint within its separate compute ledger."""
    from adp.research import ensure_space
    from adp.training import train

    root = Path(job_root).resolve()
    if device not in {"cpu", "cuda"}:
        raise ValueError("Select cpu or cuda explicitly; automatic fallback changes qualification")
    if not math.isfinite(slice_seconds) or slice_seconds <= 0:
        raise ValueError("Slice duration must be finite and positive")
    with _owner_lock(root):
        protocol = read_json(root / "job.json")
        config = read_json(root / "config.json")
        expected = protocol.get("protocol_sha256")
        if protocol.get("version") != 1 or expected != _hash({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
            raise ValueError("Remote job protocol hash differs")
        if protocol["config_sha256"] != _hash(config) or protocol["source_files"] != _source_files():
            raise ValueError("Remote job configuration or source differs from its frozen snapshot")
        _, actual_cache = _cache_files(root)
        if actual_cache != protocol["cache_files"] or _file_hash(root / "source-manifest.json") != protocol["source_manifest_sha256"]:
            raise ValueError("Remote job data provenance differs from its frozen snapshot")
        ledger_path, status_path = root / "compute-ledger.json", root / "status.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else {
            "version": 1, "protocol_sha256": expected, "charged_seconds": 0., "slices": []}
        if ledger.get("protocol_sha256") != expected:
            raise ValueError("Compute ledger belongs to another protocol")
        if status_path.exists():
            previous = read_json(status_path)
            if previous.get("state") == "completed":
                if (previous.get("protocol_sha256") != expected
                        or previous.get("result", {}).get("completed_epochs", 0) != protocol["epochs"]
                        or not (root / "run" / "last.pt").is_file()):
                    raise ValueError("Completed job receipt or checkpoint is missing or inconsistent")
                return previous
        available = protocol["maximum_compute_seconds"] - ledger["charged_seconds"]
        if available <= 0:
            status = {"state": "budget_exhausted", "updated_at": _now(), "protocol_sha256": expected,
                      "charged_seconds": ledger["charged_seconds"], "remaining_seconds": 0.}
            write_json(status_path, status)
            return status
        allowance = min(float(slice_seconds), available)
        ensure_space(root, 8, 128 * 1024**2)
        entry = {"started_at": _now(), "device": device, "reserved_seconds": allowance,
                 "state": "reserved", "charged_seconds": allowance}
        ledger["slices"].append(entry)
        ledger["charged_seconds"] += allowance
        write_json(ledger_path, ledger)
        status = {"state": "running", "updated_at": _now(), "protocol_sha256": expected,
                  "device": device, "reserved_seconds": allowance, "purpose": protocol["purpose"]}
        write_json(status_path, status)
        started = time.monotonic()
        try:
            checkpoint = root / "run" / "last.pt"
            result = train(config, root / "cache", root / "run", epochs=protocol["epochs"], device=device,
                resume=checkpoint if checkpoint.exists() else None, max_runtime_seconds=allowance)
            status.update(state="completed" if result["completed_epochs"] >= protocol["epochs"] else "paused",
                          result=result)
            entry["state"] = status["state"]
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            entry["state"] = "failed"
            raise
        finally:
            elapsed = time.monotonic() - started
            ledger["charged_seconds"] += elapsed - allowance
            entry.update(charged_seconds=elapsed, finished_at=_now())
            write_json(ledger_path, ledger)
            status.update(updated_at=_now(), charged_seconds=ledger["charged_seconds"],
                          remaining_seconds=max(0., protocol["maximum_compute_seconds"] - ledger["charged_seconds"]))
            write_json(status_path, status)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze a staged train/selection cache and source")
    prepare.add_argument("--root", required=True, type=Path)
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--epochs", type=int, default=1)
    prepare.add_argument("--max-runtime-seconds", type=float, default=1800)
    run = commands.add_parser("run", help="Resume the bounded job")
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--slice-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_job(args.root, read_json(args.config), epochs=args.epochs,
                             max_runtime_seconds=args.max_runtime_seconds)
    else:
        result = run_job(args.root, device=args.device, slice_seconds=args.slice_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
