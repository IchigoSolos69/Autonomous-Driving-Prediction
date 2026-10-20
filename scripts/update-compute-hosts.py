"""Publish private-host evidence from a completed local sync, without contacting a host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adp.compute_status import LABELS, MAX_HOSTS_BYTES, _host
from adp.remote_transfer import validate_snapshot


def _read(path):
    path = Path(path)
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError("Host evidence must use regular unlinked files")
    with path.open("rb") as stream:
        raw = stream.read(MAX_HOSTS_BYTES + 1)
    if not raw or len(raw) > MAX_HOSTS_BYTES:
        raise ValueError("Host evidence exceeds its metadata bound")
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Host evidence must contain a JSON object")
    return value, raw


def _delivery(sync, result_root, previous):
    snapshot = sync.get("snapshot", {})
    if not isinstance(snapshot, dict):
        raise ValueError("Malformed sync snapshot evidence")
    archived = snapshot.get("state") == "no_checkpoint"
    identifier = snapshot.get("snapshot_id")
    if archived and previous.get("phase") == "development queue":
        matches = []
        for count, path in enumerate((result_root / "receipts").glob("*.json"), 1):
            if count > 64 or not re.fullmatch(r"[0-9a-f]{64}", path.stem):
                raise ValueError("Historical pilot receipt index is invalid or excessive")
            metadata, _ = _read(result_root / path.stem / "snapshot.json")
            provenance = metadata.get("provenance", {})
            if isinstance(provenance, dict) and provenance.get("job_protocol_sha256") == sync["before"].get("protocol_sha256"):
                matches.append(path.stem)
        if len(matches) != 1:
            raise ValueError("Historical pilot receipt must be identified independently of the development queue")
        identifier = matches[0]
    elif archived:
        transfer = previous.get("transfer", {})
        if not isinstance(transfer, dict) or transfer.get("status") != "verified":
            return None
        local = Path(transfer.get("local_path", "")).resolve()
        if local.parent == result_root.resolve():
            identifier = local.name
        elif local.parent == (result_root / "receipts").resolve() and local.suffix == ".json":
            identifier = local.stem
        else:
            raise ValueError("Previous verified receipt must remain inside this result archive")
    if (not archived and snapshot.get("state") not in {"snapshot", "unchanged"}) or not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("Expected a completed sync with an identified result snapshot")
    directory, receipt_path = result_root / identifier, result_root / "receipts" / (identifier + ".json")
    receipt, _ = _read(receipt_path)
    delivered = sync.get("delivered")
    if delivered is not None and (not isinstance(delivered, dict) or delivered.get("snapshot_id") != identifier
            or Path(delivered.get("path", "")).resolve() != directory.resolve()
            or Path(delivered.get("receipt_path", "")).resolve() != receipt_path.resolve()
            or delivered.get("receipt") != receipt):
        raise ValueError("Sync delivery differs from its persisted local receipt")
    verified = validate_snapshot(directory)
    metadata = verified["manifest"]
    if (receipt.get("format_version") != 1 or receipt.get("verified") is not True
            or receipt.get("checkpoint_validation") != "cpu_weights_only"
            or receipt.get("snapshot_id") != identifier or metadata["snapshot_id"] != identifier
            or receipt.get("manifest_sha256") != verified["manifest_sha256"]
            or receipt.get("files") != metadata["files"]
            or Path(receipt.get("local_directory", "")).resolve() != directory.resolve()
            or metadata["provenance"].get("job_protocol_sha256") != sync["before"].get("protocol_sha256")):
        raise ValueError("Verified local snapshot provenance differs from the sync receipt")
    job = _read(directory / "job.json")[0] if "job.json" in metadata["files"] else {}
    saved = _read(directory / "result.json")[0] if "result.json" in metadata["files"] else {}
    return {"receipt": receipt, "receipt_path": receipt_path, "job": job, "saved": saved,
            "checkpoint": verified["checkpoint_metadata"]["last.pt"], "archived": archived}


def _private(sync, result_root, previous, gpu, now):
    before, resumed = sync.get("before"), sync.get("resume", {})
    if not isinstance(before, dict) or not isinstance(resumed, dict) or not isinstance(sync.get("updated_at"), str):
        raise ValueError("Expected the controller's completed last-sync schema")
    delivered = _delivery(sync, result_root, previous)
    current = resumed if resumed.get("state") == "already_owned" else before
    saved = delivered["saved"] if delivered else current.get("training", {})
    if not isinstance(saved, dict):
        raise ValueError("Malformed saved training status")
    result = saved.get("result", {})
    if not isinstance(result, dict):
        raise ValueError("Malformed saved training result")
    state = current.get("state")
    status = state if state in {"completed", "failed", "paused", "not_started", "budget_exhausted"} else "unknown"
    message = "Standalone development qualification; this checkpoint is not frozen-study or deployment evidence."
    if state in {"running", "already_owned"} and current.get("process_alive") is True:
        status = "running"
        message += " Worker process was alive when checked."
        if current.get("training_on_gpu") is not True:
            message += " GPU execution was not observed."
    if resumed.get("state") == "launched":
        status, message = "waiting", "A resume process was launched; a subsequent host check must confirm execution."
    checkpoint = delivered["checkpoint"] if delivered else {}
    job = delivered["job"] if delivered else {}
    completed, planned = checkpoint.get("completed_epochs", result.get("completed_epochs")), job.get("epochs")
    counts = job.get("split_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("Malformed job scenario counts")
    total = counts.get("train")
    if delivered and delivered["archived"] and planned and completed is not None and completed >= planned:
        status = "completed"
        message = "Verified completed results remain in the local archive; no remote checkpoint is currently present."
    transfer = {"status": "waiting", "verified_at": None, "local_path": None}
    cleanup = previous.get("cleanup") or {"status": "unknown", "message": None}
    if not isinstance(cleanup, dict):
        raise ValueError("Malformed existing private-host cleanup evidence")
    if delivered:
        transfer = {"status": "verified", "verified_at": delivered["receipt"].get("verified_at"),
                    "local_path": str(delivered["receipt_path"])}
        if cleanup.get("status") in (None, "unknown"):
            cleanup = {"status": "completed", "message": "Controller completed verified export pruning; active checkpoints follow the separate cleanup policy."}
    value = {"id": "private", "label": LABELS["private"], "status": status, "phase": "development_pilot",
             "message": message, "gpu": previous.get("gpu") or gpu, "observed_at": sync["updated_at"],
             "updated_at": saved.get("updated_at") or current.get("updated_at") or sync["updated_at"],
             "training": {"completed_epochs": completed, "planned_epochs": planned,
                          "committed_examples": total if total is not None and planned is not None and completed is not None and completed >= planned else None,
                          "total_examples": total, "examples_per_second": result.get("training_examples_per_second")},
             "transfer": transfer, "cleanup": cleanup}
    _host(value, now)  # Validate using the reader's exact contract before publishing.
    return value


def update_hosts(last_sync, hosts_path, *, gpu="NVIDIA GeForce 940MX (4 GiB)", now=None):
    last_sync, hosts_path = Path(last_sync), Path(hosts_path)
    sync, _ = _read(last_sync)
    document, original = _read(hosts_path) if hosts_path.exists() else ({"version": 1, "hosts": []}, None)
    rows = document.get("hosts")
    if type(document.get("version")) is not int or document["version"] != 1 or not isinstance(rows, list) or len(rows) > 4:
        raise ValueError("Existing host file has an unsupported schema; it was preserved")
    identifiers = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(identifiers) != len(rows) or any(not isinstance(identifier, str) or identifier not in LABELS for identifier in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError("Existing host identities are invalid; the file was preserved")
    previous = next((row for row in rows if row["id"] == "private"), {})
    value = _private(sync, last_sync.resolve().parent, previous, gpu, time.time() if now is None else now)
    if previous.get("phase") == "development queue":
        _host(previous, time.time() if now is None else now)
        return previous
    document["hosts"] = [value if row["id"] == "private" else row for row in rows]
    if "private" not in identifiers:
        document["hosts"].append(value)
    content = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(content) > MAX_HOSTS_BYTES:
        raise ValueError("Combined host file exceeds its metadata bound")
    hosts_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".hosts-update-", dir=hosts_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        current = _read(hosts_path)[1] if hosts_path.exists() else None
        if current != original:
            raise RuntimeError("Host evidence changed during refresh; rerun to preserve the other records")
        os.replace(temporary, hosts_path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-sync", type=Path, default=Path("D:/InteractionLab/remote-compute/results/last-sync.json"))
    parser.add_argument("--hosts", type=Path, default=Path("D:/InteractionLab/remote-compute/hosts.json"))
    parser.add_argument("--gpu", default="NVIDIA GeForce 940MX (4 GiB)")
    args = parser.parse_args()
    try:
        print(json.dumps(update_hosts(args.last_sync, args.hosts, gpu=args.gpu), indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Host refresh failed; existing evidence was preserved: {error}\n")
