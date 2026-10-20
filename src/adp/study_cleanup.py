"""Retire derived training NPZs only after both local studies and reports finish.

Raw sources, held-out caches, models, manifests and reports remain untouched.
Missing evidence blocks cleanup; it never requests deletion approval or guesses.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import uuid

from filelock import FileLock, Timeout

from adp.compute_cleanup import _fingerprint, _hash, _links, _safe, _sha, _write
from adp.remote_transfer import (MINIMUM_FREE_BYTES, _checkpoint, _identifier, _json_bytes,
                                 _space, _sync_directory, _write_new)


STUDY_ROOT = Path("D:/InteractionLab/study-v1")
DIRECTION_ROOT = Path("D:/InteractionLab/direction-study-v1")
MODEL_ARCHIVE_ROOT = Path("D:/InteractionLab/model-archive/study-v1")
PLAN_NAME = "processed-train-cleanup-plan.json"
JOURNAL_NAME = "processed-train-cleanup-progress.jsonl"
MAX_METADATA_BYTES = 32 * 1024**2
MAX_JOURNAL_BYTES = 4 * 1024**2
GROUP_SIZE = 64
PROOF_REQUIREMENT = ("Provide a version-1 verified local-model receipt with protocol_sha256, "
    "dataset_manifest_sha256, preprocessing_sha256, frozen_selection_sha256, "
    "checkpoint_sha256, config_sha256 and an independent absolute checkpoint_copy path.")


def _json(path):
    path = _safe(path)
    if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Expected bounded, exclusive regular study metadata")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Study metadata must contain an object")
    return value


def _roots(root, direction_root):
    root, direction = _safe(root), _safe(direction_root)
    if root != STUDY_ROOT.resolve() or direction != DIRECTION_ROOT.resolve():
        raise ValueError("Only the fixed study-v1 and direction-study-v1 roots are eligible")
    if not root.is_dir() or not direction.is_dir():
        raise ValueError("Both configured study roots must exist")
    return root, direction


def _finished(root, direction):
    original = _json(root / "pipeline_status.json")
    followup = _json(direction / "followup_status.json")
    if original.get("status") != "completed" or original.get("phase") != "finished":
        raise ValueError("Original study is not completed; training cache is still needed for resumption")
    if followup.get("status") != "completed" or followup.get("phase") != "finished":
        raise ValueError("Direction follow-up and its final CPU reports are not completed; shared training cache is still needed")
    for directory in (root, direction):
        for path in directory.glob(".*.lock"):
            if path.name != ".processed-train-cleanup.lock" and (path.exists() or _links(path)):
                raise ValueError(f"Unknown study owner lock remains present: {path.name}")
    return original, followup


@contextmanager
def _study_locks(root, direction):
    _finished(root, direction)
    try:
        with ExitStack() as owners:
            for directory, name in ((root, "pipeline.lock"), (root, "runner.lock"),
                                    (direction, "followup.lock"), (direction, "direction.lock")):
                path = _safe(directory / name, directory)
                owners.enter_context(FileLock(str(path), timeout=0))
            _finished(root, direction)
            yield
    except Timeout as error:
        raise ValueError("A pipeline, training runner or direction workflow still owns its OS study lock") from error


def _report(folder, identity, manifest_hash, expected_kind):
    folder = _safe(folder)
    if not isinstance(identity, dict) or _safe(identity.get("output_dir", "")) != folder:
        raise ValueError("Completed failure-report directory is missing or differs from the recorded workflow")
    metadata = _json(folder / "analysis.json")
    provenance = metadata.get("provenance", {})
    if (metadata.get("id") != identity.get("id") or provenance.get("kind") != expected_kind
            or provenance.get("manifest_sha256") != manifest_hash
            or provenance.get("inference_device") != "cpu" or provenance.get("labels_used") != "selection_only"):
        raise ValueError("Completed CPU failure-report provenance differs from the study")
    files = metadata.get("files")
    if (not isinstance(files, dict) or not {"records.json.gz", "summary.json", "summary.csv", "failures.png"} <= set(files)
            or len(files) > 68):
        raise ValueError("Completed failure-report file index is missing")
    actual = set()
    total = 0
    for path in folder.rglob("*"):
        path = _safe(path, folder)
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    if actual != set(files) | {"analysis.json"}:
        raise ValueError("Completed failure-report files differ from their recorded index")
    for name, entry in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Failure-report path is unsafe")
        path = _safe(folder.joinpath(*relative.parts), folder)
        if not isinstance(entry, dict) or path.stat().st_size != entry.get("bytes") or _sha(path) != entry.get("sha256"):
            raise ValueError("Completed failure-report bytes changed")
        total += path.stat().st_size
    if total > 64 * 1024**2:
        raise ValueError("Completed failure report exceeds its portable size limit")
    return _sha(folder / "analysis.json")


def _evidence(root, direction, proof, *, _before_archive=False):
    original, followup = _finished(root, direction)
    if not isinstance(proof, dict) or proof.get("version") != 1 or proof.get("verified") is not True:
        raise ValueError(PROOF_REQUIREMENT)
    manifest_path = root / "cache" / "manifest.json"
    protocol, manifest = _json(root / "protocol.json"), _json(manifest_path)
    manifest_hash, protocol_hash = _sha(manifest_path), _sha(root / "protocol.json")
    preprocessing_hash = _sha(root / "cache" / "preprocessing.json")
    frozen = _json(root / "frozen_selection.json")
    frozen_hash = _sha(root / "frozen_selection.json")
    for key, value in {"protocol_sha256": protocol_hash, "dataset_manifest_sha256": manifest_hash,
                       "preprocessing_sha256": preprocessing_hash, "frozen_selection_sha256": frozen_hash}.items():
        if proof.get(key) != value:
            raise ValueError(f"Local-model receipt does not match immutable {key}")
    if protocol.get("dataset_manifest_sha256") != manifest_hash or _sha(root / "raw" / "manifest.json") != manifest_hash:
        raise ValueError("Protocol, raw manifest and processed manifest identities differ")
    training = manifest["splits"]["train"]
    if (not isinstance(training, list) or not training or len(training) != len(set(training))
            or any(not isinstance(sid, str) or str(uuid.UUID(sid)) != sid for sid in training)
            or hashlib.sha256("\n".join(training).encode()).hexdigest() != protocol["split_hashes"]["train"]):
        raise ValueError("Training membership differs from its immutable protocol")
    if set(training).intersection(sid for split, ids in manifest["splits"].items() if split != "train" for sid in ids):
        raise ValueError("Training scenarios overlap a preserved benchmark split")
    listing_path = root / "listings" / "train.json"
    listing = _json(listing_path)
    if (manifest["listing_sha256"]["train"] != _sha(listing_path) or listing.get("complete") is not True
            or listing.get("source") != "https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/train/"
            or not set(training) <= set(listing["ids"])):
        raise ValueError("Training IDs are not verified against the complete official training listing")
    selected = _safe(frozen["checkpoint"], root)
    if selected != root / "runs" / frozen["deployment_run"] / "best.pt":
        raise ValueError("Selected checkpoint path differs from the frozen deployment run")
    checkpoint_hash = _identifier(frozen["checkpoint_sha256"])
    copy = _safe(proof.get("checkpoint_copy", ""))
    if copy.is_relative_to(root) or copy.is_relative_to(direction):
        raise ValueError("The verified model copy must be independent of both study roots")
    if (proof.get("checkpoint_sha256") != checkpoint_hash or _sha(selected) != checkpoint_hash
            or not _before_archive and _sha(copy) != checkpoint_hash):
        raise ValueError("Independent local model copy does not match the frozen selected checkpoint")
    checked_model = selected if _before_archive else copy
    if checked_model.stat().st_size > 256 * 1024**2:
        raise ValueError("Model copy exceeds checkpoint validation limits")
    saved = _checkpoint(checked_model.read_bytes())
    if (_hash(saved["config"]) != proof.get("config_sha256") or saved["manifest_hash"] != manifest_hash
            or saved["preprocessing_hash"] != preprocessing_hash):
        raise ValueError("Strict CPU model validation differs from the receipt or preprocessing")
    promotion = _json(root / "promotion.json")
    if promotion.get("deployment_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Final original-study results do not identify the selected checkpoint")
    direction_protocol, experiment = _json(direction / "protocol.json"), _json(direction / "experiment.json")
    direction_hash = _sha(direction / "protocol.json")
    if (direction_protocol.get("reference_root") != str(root) or _safe(direction_protocol["cache_dir"]) != root / "cache"
            or direction_protocol.get("manifest_sha256") != manifest_hash
            or direction_protocol.get("preprocessing_sha256") != preprocessing_hash
            or direction_protocol.get("train_ids") != training
            or experiment.get("status") != "completed" or experiment.get("protocol_sha256") != direction_hash
            or followup.get("direction_result", {}).get("status") != "completed"):
        raise ValueError("Direction study is incomplete or references different training inputs")
    comparison = _json(direction / "comparison.json")
    if comparison.get("protocol_sha256") != direction_hash or not (direction / "REPORT.md").is_file():
        raise ValueError("Final direction comparison and report are missing or inconsistent")
    reports = {"reference_analysis_sha256": _report(direction / "reference-analysis", followup.get("reference_analysis"), manifest_hash, "reference"),
               "direction_analysis_sha256": _report(direction / "direction-analysis", followup.get("direction_analysis"), manifest_hash, "direction-study")}
    return sorted(training), {"protocol_sha256": protocol_hash, "manifest_sha256": manifest_hash,
        "preprocessing_sha256": preprocessing_hash, "frozen_selection_sha256": frozen_hash,
        "checkpoint_sha256": checkpoint_hash, "proof_sha256": _hash(proof),
        "original_status_sha256": _hash(original), "followup_status_sha256": _hash(followup),
        "direction_protocol_sha256": direction_hash, "experiment_sha256": _sha(direction / "experiment.json"),
        "comparison_sha256": _sha(direction / "comparison.json"), **reports}


def _journal(root, plan):
    path = root / JOURNAL_NAME
    if not path.exists():
        return {"cursor": 0, "pending_stop": None, "completed": False, "valid_bytes": 0}
    _safe(path, root)
    if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > MAX_JOURNAL_BYTES:
        raise ValueError("Cleanup progress journal is not a bounded exclusive regular file")
    lines = path.read_bytes().splitlines(keepends=True)
    cursor, pending, completed, valid_bytes = 0, None, False, 0
    for number, raw in enumerate(lines):
        if not raw.endswith(b"\n") and number == len(lines) - 1:
            break
        entry = json.loads(raw)
        if not isinstance(entry, dict) or entry.get("plan_sha256") != plan["plan_sha256"] or completed:
            raise ValueError("Cleanup journal belongs to a different or completed plan")
        kind = entry.get("kind")
        if number == 0:
            if kind != "header" or entry.get("version") != 1:
                raise ValueError("Cleanup journal header is invalid")
        elif kind == "intent" and pending is None and entry.get("start") == cursor:
            stop = entry.get("stop")
            if isinstance(stop, bool) or not isinstance(stop, int) or not cursor < stop <= min(len(plan["files"]), cursor + GROUP_SIZE):
                raise ValueError("Cleanup intent range is invalid")
            pending = stop
        elif kind == "done" and pending is not None and entry.get("stop") == pending:
            cursor, pending = pending, None
        elif kind == "completed" and pending is None and cursor == len(plan["files"]):
            completed = True
        else:
            raise ValueError("Cleanup journal ordering is invalid")
        valid_bytes += len(raw)
    if lines and not valid_bytes:
        raise ValueError("Cleanup journal has no durable header")
    return {"cursor": cursor, "pending_stop": pending, "completed": completed, "valid_bytes": valid_bytes}


def _plan(root, direction, proof):
    training, context = _evidence(root, direction, proof)
    target = _safe(root / "cache" / "train", root)
    path = root / PLAN_NAME
    if path.exists():
        plan = _json(path)
        if (plan.get("plan_sha256") != _hash({key: value for key, value in plan.items() if key != "plan_sha256"})
                or plan.get("context") != context or plan.get("root") != str(root) or plan.get("target") != str(target)
                or [entry["name"] for entry in plan.get("files", [])] != [f"{sid}.npz" for sid in training]):
            raise ValueError("Immutable cleanup plan is corrupt or belongs to other study inputs")
        journal = _journal(root, plan)
    else:
        if (root / JOURNAL_NAME).exists():
            raise ValueError("Cleanup journal exists without its immutable plan")
        plan = {"version": 1, "root": str(root), "target": str(target), "context": context, "files": []}
        journal = {"cursor": 0, "pending_stop": None, "completed": False, "valid_bytes": 0}
    allowed_absent = journal["pending_stop"] or journal["cursor"]
    expected = {f"{sid}.npz" for sid in training}
    actual = set()
    if target.exists():
        if not target.is_dir():
            raise ValueError("Expected the fixed processed training directory")
        for child in target.iterdir():
            _safe(child, target)
            if not child.is_file() or child.name not in expected or child.stat().st_nlink != 1:
                raise ValueError("Training directory contains unregistered, shared or linked entries")
            actual.add(child.name)
    elif allowed_absent != len(training):
        raise ValueError("Training directory is missing without a complete write-ahead cleanup journal")
    for index, sid in enumerate(training):
        name, item = f"{sid}.npz", target / f"{sid}.npz"
        if name not in actual:
            if index >= allowed_absent:
                raise ValueError("Training cache is incomplete without an authorized cleanup intent")
            continue
        digest, fingerprint = _sha(item), _fingerprint(item)
        if path.exists():
            entry = plan["files"][index]
            if entry.get("sha256") != digest or entry.get("fingerprint") != fingerprint:
                raise ValueError("Training cache bytes changed after cleanup planning")
        else:
            plan["files"].append({"name": name, "sha256": digest, "size_bytes": item.stat().st_size, "fingerprint": fingerprint})
    if not path.exists():
        plan["plan_sha256"] = _hash(plan)
    return plan, journal, sum(entry["size_bytes"] for entry in plan["files"] if entry["name"] in actual)


def _ready_result(root, direction, proof):
    plan, journal, remaining = _plan(root, direction, proof)
    return {"state": "completed" if journal["completed"] else "ready", "plan": plan,
            "progress": journal, "files_to_remove": len(plan["files"]) - journal["cursor"], "bytes_to_free": remaining}


def plan_study_cleanup(root=STUDY_ROOT, direction_root=DIRECTION_ROOT, proof=None):
    """Verify a data-read-only plan while holding the actual study OS locks."""
    try:
        root, direction = _roots(root, direction_root)
        with _study_locks(root, direction):
            return _ready_result(root, direction, proof)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        return {"state": "blocked", "reason": str(error), "required_proof": PROOF_REQUIREMENT,
                "preserved": ["raw sources and maps", "validation/calibration/evaluation", "models", "reports", "manifests"]}


def _append(path, value):
    with path.open("ab") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def execute_study_cleanup(root=STUDY_ROOT, direction_root=DIRECTION_ROOT, proof=None):
    """Remove only fixed train/*.npz; use bounded write-ahead groups for interruption."""
    try:
        root, direction = _roots(root, direction_root)
        with _study_locks(root, direction):
            return _execute_owned(root, direction, proof)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return {"state": "blocked", "reason": str(error), "required_proof": PROOF_REQUIREMENT}


def _execute_owned(root, direction, proof):
    result = _ready_result(root, direction, proof)
    if result["state"] == "completed":
        return result
    lock = root / ".processed-train-cleanup.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        return {"state": "blocked", "reason": "Another training-cache cleanup owns this root"}
    try:
        with stream:
            stream.write(str(os.getpid()))
        current = _ready_result(root, direction, proof)
        if current["state"] != "ready" or current["plan"] != result["plan"]:
            return {"state": "blocked", "reason": "Study inputs changed while acquiring cleanup ownership"}
        plan, progress = current["plan"], current["progress"]
        plan_path, journal_path = root / PLAN_NAME, root / JOURNAL_NAME
        if not plan_path.exists():
            projected = len(_json_bytes(plan))
            if projected > MAX_METADATA_BYTES or shutil.disk_usage(root).free - 2 * projected - MAX_JOURNAL_BYTES < MINIMUM_FREE_BYTES:
                return {"state": "blocked", "reason": "Cannot durably store cleanup audit while preserving the 8 GiB reserve"}
            _write(plan_path, plan)
        if not journal_path.exists():
            _append(journal_path, {"kind": "header", "version": 1, "plan_sha256": plan["plan_sha256"]})
            _sync_directory(root)
        else:
            with journal_path.open("r+b") as journal_stream:
                journal_stream.truncate(progress["valid_bytes"])
                journal_stream.flush()
                os.fsync(journal_stream.fileno())
            if progress["valid_bytes"] == 0:
                _append(journal_path, {"kind": "header", "version": 1, "plan_sha256": plan["plan_sha256"]})
        cursor, pending = progress["cursor"], progress["pending_stop"]
        while cursor < len(plan["files"]):
            _finished(root, direction)
            stop = pending or min(cursor + GROUP_SIZE, len(plan["files"]))
            if pending is None:
                _append(journal_path, {"kind": "intent", "start": cursor, "stop": stop, "plan_sha256": plan["plan_sha256"]})
            for entry in plan["files"][cursor:stop]:
                item = _safe(Path(plan["target"]) / entry["name"], root / "cache" / "train")
                if not item.exists():
                    if pending is None:
                        raise ValueError("Training input vanished before this cleanup group")
                    continue
                if (_fingerprint(item) != entry["fingerprint"] or _sha(item) != entry["sha256"]
                        or _fingerprint(item) != entry["fingerprint"]):
                    raise ValueError("Training input changed immediately before deletion")
                item.unlink()
            if Path(plan["target"]).is_dir():
                _sync_directory(Path(plan["target"]))
            _append(journal_path, {"kind": "done", "stop": stop, "plan_sha256": plan["plan_sha256"]})
            cursor, pending = stop, None
        target = _safe(root / "cache" / "train", root)
        if target.is_dir():
            target.rmdir()
            _sync_directory(target.parent)
        _append(journal_path, {"kind": "completed", "plan_sha256": plan["plan_sha256"]})
        return {"state": "completed", "removed_files": len(plan["files"]), "plan_sha256": plan["plan_sha256"],
                "plan_path": str(plan_path), "journal_path": str(journal_path), "preserved_models": True}
    finally:
        lock.unlink(missing_ok=True)


def _archive_selected(root, direction):
    archive_root = _safe(MODEL_ARCHIVE_ROOT)
    if archive_root.is_relative_to(root) or archive_root.is_relative_to(direction):
        raise ValueError("The fixed model archive must remain independent of both studies")
    frozen = _json(root / "frozen_selection.json")
    selected = _safe(frozen["checkpoint"], root)
    if selected != root / "runs" / frozen["deployment_run"] / "best.pt":
        raise ValueError("Selected checkpoint path differs from the frozen deployment run")
    digest = _identifier(frozen["checkpoint_sha256"])
    if not selected.is_file() or not 0 < selected.stat().st_size <= 256 * 1024**2:
        raise ValueError("Selected checkpoint exceeds the model archive size limit")
    content = selected.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Frozen selected checkpoint bytes changed before archival")
    saved = _checkpoint(content)
    destination = _safe(archive_root / digest, archive_root)
    copy, proof_path = destination / "selected.pt", destination / "cleanup-proof.json"
    proof = {"version": 1, "verified": True, "checkpoint_validation": "strict_cpu_observed_only",
        "protocol_sha256": _sha(root / "protocol.json"),
        "dataset_manifest_sha256": _sha(root / "cache" / "manifest.json"),
        "preprocessing_sha256": _sha(root / "cache" / "preprocessing.json"),
        "frozen_selection_sha256": _sha(root / "frozen_selection.json"),
        "checkpoint_sha256": digest, "config_sha256": _hash(saved["config"]), "checkpoint_copy": str(copy)}
    # Verify every report/protocol guard before creating any archive; source bytes stand in for its not-yet-published copy.
    _evidence(root, direction, proof, _before_archive=True)
    if destination.exists():
        if (not destination.is_dir() or {path.name for path in destination.iterdir()} != {"selected.pt", "cleanup-proof.json"}
                or _sha(_safe(copy, destination)) != digest or _json(proof_path) != proof):
            raise ValueError("Existing model archive is corrupt or belongs to different proof; it will not be overwritten")
    else:
        existing_parent = archive_root
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        _space(existing_parent, 2 * len(content) + len(_json_bytes(proof)) + MAX_JOURNAL_BYTES)
        archive_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".model-archive-", dir=archive_root))
        try:
            _write_new(stage / "selected.pt", content)
            if _sha(stage / "selected.pt") != digest or _checkpoint((stage / "selected.pt").read_bytes()) != saved:
                raise ValueError("Independent model copy failed strict CPU/readback verification")
            _write_new(stage / "cleanup-proof.json", _json_bytes(proof))
            _sync_directory(stage)
            if destination.exists() or _links(destination):
                raise ValueError("Model archive appeared during publication; refusing to overwrite it")
            stage.rename(destination)
            _sync_directory(archive_root)
        finally:
            if stage.exists() and not _links(stage) and stage.resolve().parent == archive_root and stage.name.startswith(".model-archive-"):
                shutil.rmtree(stage)
    _evidence(root, direction, proof)
    return proof, proof_path


def archive_and_cleanup(root=STUDY_ROOT, direction_root=DIRECTION_ROOT):
    """Archive and prove the selected model automatically, then retire eligible training caches."""
    try:
        root, direction = _roots(root, direction_root)
        with _study_locks(root, direction):
            proof, proof_path = _archive_selected(root, direction)
            result = _execute_owned(root, direction, proof)
            return {**result, "proof_path": str(proof_path), "archived_checkpoint": proof["checkpoint_copy"]}
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        return {"state": "blocked", "reason": str(error), "preserved_models": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--direction-root", type=Path, default=DIRECTION_ROOT)
    parser.add_argument("--proof-file", type=Path)
    parser.add_argument("--archive-completed", action="store_true",
                        help="Automatically archive/prove the selected model after both studies finish; requires --execute")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.archive_completed and (not args.execute or args.proof_file):
        parser.error("--archive-completed requires --execute and supplies its own proof; omit --proof-file")
    proof = _json(args.proof_file) if args.proof_file else None
    function = execute_study_cleanup if args.execute else plan_study_cleanup
    result = archive_and_cleanup(args.root, args.direction_root) if args.archive_completed else function(args.root, args.direction_root, proof)
    if "plan" in result:
        result = {key: value for key, value in result.items() if key != "plan"} | {
            "plan_sha256": result["plan"]["plan_sha256"], "target": result["plan"]["target"]}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
