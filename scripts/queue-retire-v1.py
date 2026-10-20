"""Retire exact shared copies only after the private twelve-job queue is verified."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType

ROOT = Path("/home/aadi/interaction-lab/queues/development-v1")
ADAPTER_SHA256 = "7a4413533f728f8a86d4be481767b65ffb8b9f33e28be47ee439d99346a193c7"
INDEX_SHA256 = "3770e3f2c6df73ce4fb652a6c52a2841772f25feb70a3480bb035e788f107282"
PAYLOAD_BYTES = 231629839
JOURNAL = "cleanup/terminal-v1.json"


def load_cleanup():
    path = Path(__file__).absolute().parent / ("queue-cleanup-v2-" + ADAPTER_SHA256 + ".py")
    for item in (path, *path.parents):
        info = item.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError("Retirement tool paths must not contain links or junctions")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size < 128 * 1024:
        raise ValueError("Expected a bounded regular unshared cleanup adapter")
    with path.open("rb") as stream:
        data = stream.read(128 * 1024)
    if hashlib.sha256(data).hexdigest() != ADAPTER_SHA256:
        raise ValueError("Cleanup adapter differs from the separately reviewed source")
    module = ModuleType("interaction_terminal_cleanup_adapter")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), vars(module))
    return module.load_base()


@contextmanager
def worker_lock(base, root):
    path = base._safe(root / ".worker.lock", root)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 <= info.st_size <= 1024:
        raise ValueError("Expected the existing unshared worker lock")
    with path.open("r+b") as stream:
        if os.name == "nt":
            import msvcrt
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


def specifications(base):
    return [base._spec(cfg, seed) for cfg in ("full", "no_graph", "no_map", "world_ce")
            for seed in (42, 43, 44)]


def terminal_queue(base, root):
    queue = base._read(root / "controller/queue.json")
    expected = {base._hash(spec) for spec in specifications(base)}
    jobs = queue.get("jobs")
    if (not isinstance(jobs, list) or len(jobs) != 12
            or {item.get("id") for item in jobs if isinstance(item, dict)} != expected):
        raise ValueError("Retirement requires exactly the twelve approved private jobs")
    for identifier in sorted(expected):
        base._queue_job(root, identifier)
        base._inactive(root / "jobs" / identifier)
    return base._sha(root / "controller/queue.json", base.MAX_JSON)


def validate_envelope(base, root, envelope):
    if (not isinstance(envelope, dict) or set(envelope) != {"version", "approved_plan_sha256", "payload_sha256", "files", "deliveries"}
            or type(envelope["version"]) is not int or envelope["version"] != 1
            or envelope["payload_sha256"] != base.PAYLOAD_SHA256):
        raise ValueError("Expected the exact authenticated terminal receipt envelope")
    files = envelope["files"]
    if (not isinstance(files, dict) or set(files) != {"worker.py", "compute_queue.py", "queue_cleanup.py", "pilot-helper.py", "inputs.zip"}
            or files["inputs.zip"] != base.PAYLOAD_SHA256):
        raise ValueError("Retirement resource index differs from the approved queue")
    for digest in files.values():
        base._identifier(digest)
    plans = {}
    for host in ("local", "private", "colab", "kaggle"):
        plans[host] = [{**spec, "host": host, "dependencies": ["original_study", "direction_study", "cpu_reports", "training_cache_cleanup"] if host == "local" else []}
                       for spec in specifications(base)]
    approved = {"version": 1, "host_seconds": 21600, "payload_sha256": base.PAYLOAD_SHA256,
                "files": files, "host_plans": plans}
    if base._hash(approved) != envelope["approved_plan_sha256"]:
        raise ValueError("Approved queue plan hash differs from the exact resources and recipes")
    resources = base._read(root / "resources.json")
    if resources != {"files": files, "payload_sha256": base.PAYLOAD_SHA256,
                     "plan_sha256": base._hash({"version": 1, "host_seconds": 21600, "jobs": plans["private"]}),
                     "approved_plan_sha256": envelope["approved_plan_sha256"]}:
        raise ValueError("Remote resource provenance differs from the approved envelope")
    deliveries = envelope["deliveries"]
    if not isinstance(deliveries, list) or len(deliveries) != 12:
        raise ValueError("Twelve verified local deliveries are required")
    for spec, delivery in zip(specifications(base), deliveries):
        if not isinstance(delivery, dict) or set(delivery) != {"queue_receipt", "import_receipt", "export", "cleanup_receipt_sha256"}:
            raise ValueError("Invalid completed job delivery")
        base._identifier(delivery["cleanup_receipt_sha256"])
        queue_receipt, imported, exported = (delivery[name] for name in ("queue_receipt", "import_receipt", "export"))
        if (not all(isinstance(value, dict) for value in (queue_receipt, imported, exported))
                or queue_receipt.get("job_spec") != spec or queue_receipt.get("job_id") != base._hash(spec)
                or queue_receipt.get("verified") is not True or imported.get("verified") is not True
                or queue_receipt.get("checkpoint_validation") != "strict_cpu_observed_only_completed"
                or imported.get("checkpoint_validation") != "cpu_weights_only"
                or queue_receipt.get("payload_sha256") != base.PAYLOAD_SHA256
                or queue_receipt.get("allocated_device_count") != 1
                or queue_receipt.get("queue_protocol_sha256") != resources["plan_sha256"]
                or queue_receipt.get("archive_sha256") != exported.get("sha256")
                or exported.get("job_id") != base._hash(spec)):
            raise ValueError("Retirement needs matching strict CPU-verified local model receipts")
        for key in ("snapshot_id", "manifest_sha256", "files", "local_directory"):
            if queue_receipt.get(key) != imported.get(key):
                raise ValueError("Local verification receipts refer to different canonical copies")
        identifier = base._identifier(exported.get("snapshot_id"))
        if queue_receipt.get("snapshot_id") != identifier or exported.get("archive_name") != identifier + ".zip":
            raise ValueError("Retirement export is not the verified opaque snapshot")


def indexed_targets(base, root, envelope):
    """Reconstruct the deletion set even after an interrupted removal of exports."""
    index_path = root / "input/package-files.json"
    if base._sha(index_path, base.MAX_JSON) != INDEX_SHA256:
        raise ValueError("Shared inputs do not match the pinned original package index")
    index = base._read(index_path)
    first = base._read(root / "jobs" / envelope["deliveries"][0]["export"]["job_id"] / "job.json")
    cache = first.get("cache_files")
    if not isinstance(cache, dict) or len(cache) != 1066:
        raise ValueError("Unexpected shared pilot cache membership")
    targets = {"inputs.zip": {"sha256": base.PAYLOAD_SHA256, "size_bytes": PAYLOAD_BYTES}}
    for name, digest in cache.items():
        if not name.endswith(".npz"):
            continue
        parts = Path(name).parts
        if len(parts) != 2 or parts[0] not in {"train", "validation"}:
            raise ValueError("Unregistered shared cache path")
        entry = index.get("cache/" + name)
        if not isinstance(entry, dict) or entry.get("sha256") != digest or type(entry.get("size")) is not int or not 0 < entry["size"] <= 8 * 1024**2:
            raise ValueError("Shared cache bytes differ from the pinned package index")
        targets["input/cache/" + name] = {"sha256": digest, "size_bytes": entry["size"]}
    if len(targets) != 1065:
        raise ValueError("Expected exactly the approved 1,064 shared scenario caches")
    for delivery in envelope["deliveries"]:
        exported = delivery["export"]
        sid = base._identifier(exported["snapshot_id"])
        if type(exported.get("bytes")) is not int or not 0 < exported["bytes"] <= base.MAX_TOTAL:
            raise ValueError("Invalid bounded export size")
        targets["exports/" + sid + ".zip"] = {"sha256": base._identifier(exported["sha256"]), "size_bytes": exported["bytes"]}
        for name in ("last.pt", "best.pt"):
            entry = delivery["queue_receipt"]["files"].get(name)
            if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"} or type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= base.MAX_FILE:
                raise ValueError("Missing bounded canonical checkpoint identity")
            targets["snapshots/" + sid + "/" + name] = {"sha256": base._identifier(entry["sha256"]), "size_bytes": entry["size_bytes"]}
    if len(targets) != 1101:
        raise ValueError("Duplicate or incomplete terminal targets")
    return targets


def preserved_paths(base, root, envelope):
    preserved = {"resources.json", "controller/queue.json", "input/package-files.json"}
    preserved.update(name for name in envelope["files"] if name != "inputs.zip")
    for name, entry in base._read(root / "input/package-files.json").items():
        if not name.endswith(".npz"):
            preserved.add("input/" + name)
    for delivery in envelope["deliveries"]:
        jid, sid = delivery["export"]["job_id"], delivery["export"]["snapshot_id"]
        prefix = "jobs/" + jid + "/"
        preserved.update(prefix + name for name in ("job.json", "config.json", "status.json", "source-manifest.json", "cache/manifest.json", "cache/preprocessing.json"))
        preserved.update({"cleanup/" + jid + ".json", "snapshots/" + sid + "/snapshot.json"})
        protocol = base._read(root / prefix / "job.json")
        preserved.update(prefix + "source/adp/" + name for name in protocol["source_files"])
        for name in delivery["queue_receipt"]["files"]:
            if name not in {"last.pt", "best.pt"}:
                preserved.update({prefix + "run/" + name, "snapshots/" + sid + "/" + name})
    return preserved


def prepare(base, root, envelope):
    validate_envelope(base, root, envelope)
    terminal_queue(base, root)
    for name, entry in base._read(root / "input/package-files.json").items():
        path = base._safe(root / "input" / name, root)
        if base._sha(path, 8 * 1024**2) != entry["sha256"]:
            raise ValueError("Shared approved inputs changed before retirement")
    for name, digest in envelope["files"].items():
        if base._sha(root / name, base.MAX_TOTAL) != digest:
            raise ValueError("Original immutable queue resources changed")
    for delivery in envelope["deliveries"]:
        jid, sid = delivery["export"]["job_id"], delivery["export"]["snapshot_id"]
        submitted = {key: delivery[key] for key in ("queue_receipt", "import_receipt", "export")}
        job, context, old_targets = base._verify(root, jid, submitted)
        journal_path = root / "cleanup" / (jid + ".json")
        old = base._read(journal_path)
        if (base._sha(journal_path, base.MAX_JSON) != delivery["cleanup_receipt_sha256"]
                or old.get("state") != "completed" or old.get("context") != context
                or old.get("receipt_sha256") != base._hash({key: value for key, value in old.items() if key != "receipt_sha256"})
                or not isinstance(old.get("files"), list) or len(old["files"]) != len(old_targets)
                or {entry.get("path"): entry.get("sha256") for entry in old["files"]} != old_targets
                or any(base._safe(root / name, root).exists() for name in old_targets)):
            raise ValueError("Per-job cleanup lacks its exact completed durable absence proof")
    return {name: base._sha(root / name, 8 * 1024**2) for name in sorted(preserved_paths(base, root, envelope))}


def journal_check(base, root, value, envelope=None):
    if (not isinstance(value, dict) or set(value) != {"version", "tool", "state", "envelope", "envelope_sha256", "preserved", "files", "updated_at", "receipt_sha256"}
            or value["version"] != 1 or value["tool"] != "queue-retire-v1"
            or value["state"] not in {"in_progress", "completed"}
            or value["receipt_sha256"] != base._hash({key: item for key, item in value.items() if key != "receipt_sha256"})
            or value["envelope_sha256"] != base._hash(value["envelope"])
            or envelope is not None and value["envelope"] != envelope):
        raise ValueError("Terminal journal is corrupt or belongs to another authenticated envelope")
    validate_envelope(base, root, value["envelope"])
    terminal_queue(base, root)
    targets = indexed_targets(base, root, value["envelope"])
    entries = value["files"]
    if (not isinstance(entries, list) or len(entries) != len(targets)
            or {entry.get("path") for entry in entries if isinstance(entry, dict)} != set(targets)):
        raise ValueError("Terminal journal contains an unexpected deletion set")
    if not isinstance(value["preserved"], dict) or set(value["preserved"]) != preserved_paths(base, root, value["envelope"]):
        raise ValueError("Terminal journal is missing preserved provenance")
    for name, digest in value["preserved"].items():
        if base._sha(base._safe(root / name, root), 8 * 1024**2) != base._identifier(digest):
            raise ValueError("Preserved terminal provenance changed")
    remaining = 0
    for entry in entries:
        if (set(entry) != {"path", "sha256", "size_bytes", "fingerprint"}
                or {key: entry[key] for key in ("sha256", "size_bytes")} != targets[entry["path"]]
                or not isinstance(entry["fingerprint"], list) or len(entry["fingerprint"]) != 5
                or any(type(item) is not int for item in entry["fingerprint"])
                or entry["fingerprint"][2] != entry["size_bytes"]):
            raise ValueError("Terminal target differs from its exact approved byte index")
        path = base._safe(root / entry["path"], root)
        if path.exists():
            if value["state"] == "completed":
                raise ValueError("A target reappeared after completed retirement; it was preserved")
            if base._fingerprint(path) != entry["fingerprint"] or base._sha(path, base.MAX_TOTAL) != entry["sha256"]:
                raise ValueError("A terminal target changed after the durable plan")
            remaining += 1
    return remaining


def summary(base, root, value, remaining):
    return {"version": 1, "tool": "queue-retire-v1", "state": value["state"],
            "receipt_sha256": value["receipt_sha256"], "journal_file_sha256": base._sha(root / JOURNAL, base.MAX_JSON),
            "envelope_sha256": value["envelope_sha256"], "target_count": len(value["files"]),
            "remaining_files": remaining, "bytes_retired": sum(entry["size_bytes"] for entry in value["files"] if not (root / entry["path"]).exists()),
            "journal": JOURNAL}


def retire(root, envelope=None, *, execute=False, inspect=False, base=None):
    base = base or load_cleanup()
    root = base._safe(root)
    with worker_lock(base, root):
        path = base._safe(root / JOURNAL, root)
        if inspect:
            if not path.exists():
                return {"version": 1, "tool": "queue-retire-v1", "state": "not_started", "receipt_sha256": None,
                        "journal_file_sha256": None, "envelope_sha256": None, "target_count": 0, "remaining_files": 0,
                        "bytes_retired": 0, "journal": JOURNAL}
            value = base._read(path)
            return summary(base, root, value, journal_check(base, root, value))
        envelope = base._json_bytes(base._bytes(envelope))
        with base._mutex(root):
            if path.exists():
                value = base._read(path)
                remaining = journal_check(base, root, value, envelope)
                if value["state"] == "completed" or not execute:
                    return summary(base, root, value, remaining)
            else:
                preserved = prepare(base, root, envelope)
                targets = indexed_targets(base, root, envelope)
                entries = []
                for name, expected in sorted(targets.items()):
                    target = base._safe(root / name, root)
                    fingerprint = base._fingerprint(target)
                    if fingerprint[2] != expected["size_bytes"] or base._sha(target, base.MAX_TOTAL) != expected["sha256"] or base._fingerprint(target) != fingerprint:
                        raise ValueError("Terminal target changed before preflight")
                    entries.append({"path": name, **expected, "fingerprint": fingerprint})
                value = {"version": 1, "tool": "queue-retire-v1", "state": "in_progress", "envelope": envelope,
                         "envelope_sha256": base._hash(envelope), "preserved": preserved, "files": entries,
                         "updated_at": datetime.now(timezone.utc).isoformat()}
                value["receipt_sha256"] = base._hash(value)
                if not execute:
                    return {"version": 1, "tool": "queue-retire-v1", "state": "planned", "target_count": len(entries),
                            "bytes_to_free": sum(entry["size_bytes"] for entry in entries), "envelope_sha256": value["envelope_sha256"]}
                base._write(path, value)
            queue_path = root / "controller/queue.json"
            queue_fingerprint = base._fingerprint(queue_path)
            for entry in value["files"]:
                # Both controller locks are held; detect a replaced ledger or new job owner.
                if base._fingerprint(queue_path) != queue_fingerprint:
                    raise ValueError("Completed queue ledger changed during retirement")
                for delivery in envelope["deliveries"]:
                    base._inactive(root / "jobs" / delivery["export"]["job_id"])
                target = base._safe(root / entry["path"], root)
                if not target.exists():
                    continue
                if base._fingerprint(target) != entry["fingerprint"] or base._sha(target, base.MAX_TOTAL) != entry["sha256"] or base._fingerprint(target) != entry["fingerprint"]:
                    raise ValueError("Terminal target changed immediately before deletion")
                target.unlink()
                base._sync(target.parent)
            value.pop("receipt_sha256")
            value.update(state="completed", updated_at=datetime.now(timezone.utc).isoformat())
            value["receipt_sha256"] = base._hash(value)
            base._write(path, value)
            return summary(base, root, value, journal_check(base, root, value, envelope))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--inspect", action="store_true")
    action.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.root != ROOT:
        raise ValueError("Retirement CLI is restricted to the fixed private development queue")
    base = load_cleanup()
    envelope = None if args.inspect else base._json_bytes(sys.stdin.buffer.read(base.MAX_JSON + 1))
    print(json.dumps(retire(args.root, envelope, execute=args.execute, inspect=args.inspect, base=base), allow_nan=False))


if __name__ == "__main__":
    main()
