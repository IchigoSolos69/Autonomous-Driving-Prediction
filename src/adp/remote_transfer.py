"""Verified immutable result snapshots; active training directories are never pruned.

The SSH controller transports directories and returns receipts over its authenticated
connection. A receipt is evidence from that controller, not a digital signature.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


FILES = {"last.pt", "best.pt", "history.json", "config.json", "run.json", "job.json",
         "environment.json", "qualification.json", "result.json", "training.png", "metrics.json"}
MAX_FILE_BYTES = 256 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MAX_JSON_BYTES = 8 * 1024**2
MINIMUM_FREE_BYTES = 8 * 1024**3
MARKER = ".remote-transfer-exports.json"
_MARKER_VALUE = {"format_version": 1, "purpose": "immutable_training_result_exports"}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected an opaque SHA-256 snapshot identifier")
    return value


def _linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _directory(path, create=False):
    path = Path(path).absolute()
    if _linked(path):
        raise ValueError("Linked directories are not supported")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("Expected an existing directory")
    return path.resolve()


def _read(path, limit):
    if _linked(path) or not path.is_file():
        raise ValueError(f"Expected a regular unlinked file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError(f"Empty or excessive file: {path.name}")
    return data


def _json(data):
    try:
        value = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid snapshot JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Snapshot JSON must contain an object")
    return value


def _space(root, incoming):
    if shutil.disk_usage(root).free - incoming < MINIMUM_FREE_BYTES:
        raise OSError("Transfer paused: incoming snapshot would breach the 8 GiB free-space reserve")


def _checkpoint(data):
    import torch
    from adp.model import ForecastModel

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 20000 or sum(entry.file_size for entry in entries) > 1024**3:
                raise ValueError("Expanded checkpoint exceeds transfer limits")
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Checkpoint failed CPU weights-only validation") from error
    if (not isinstance(saved, dict) or saved.get("format_version") not in (1, 2)
            or not isinstance(saved.get("config"), dict)
            or not isinstance(saved.get("model_state_dict"), dict)
            or not saved["model_state_dict"]):
        raise ValueError("Unsupported training checkpoint metadata")
    for value in saved["model_state_dict"].values():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains invalid model tensors")
    config = saved["config"].get("model", saved["config"])
    if not isinstance(config, dict):
        raise ValueError("Checkpoint requires a model configuration object")
    limits = {"width": (128, 16, 512), "heads": (4, 1, 16), "temporal_layers": (2, 1, 8),
              "graph_layers": (2, 0, 8), "modes": (6, 6, 6)}
    for key, (default, lower, upper) in limits.items():
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"Unsupported checkpoint model configuration: {key}")
    if config.get("width", 128) % config.get("heads", 4):
        raise ValueError("Checkpoint model width must be divisible by its attention heads")
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = ForecastModel(config).cpu().eval()
        if sum(value.numel() for value in model.parameters()) > 25_000_000:
            raise ValueError("Checkpoint model exceeds the supported parameter budget")
        expected = model.state_dict()
        for name, value in saved["model_state_dict"].items():
            if name in expected and (value.shape != expected[name].shape or value.dtype != expected[name].dtype):
                raise ValueError(f"Checkpoint model tensor shape or dtype differs: {name}")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        inputs = {"history": torch.zeros(1, 2, 50, 8, device="cpu"),
                  "history_mask": torch.ones(1, 2, 50, dtype=torch.bool, device="cpu"),
                  "actor_mask": torch.ones(1, 2, dtype=torch.bool, device="cpu"),
                  "actor_types": torch.zeros(1, 2, dtype=torch.long, device="cpu"),
                  "dynamic_mask": torch.ones(1, 2, dtype=torch.bool, device="cpu"),
                  "map_points": torch.zeros(1, 1, 20, 2, device="cpu"),
                  "map_features": torch.zeros(1, 1, 4, device="cpu"),
                  "map_mask": torch.ones(1, 1, dtype=torch.bool, device="cpu")}
        with torch.inference_mode():
            output = model(**inputs)
        if output["means"].shape != (1, 6, 2, 60, 2) or any(
                not torch.isfinite(output[key]).all() for key in ("means", "scales", "logits", "risk_logits")):
            raise ValueError("Checkpoint model produced invalid CPU predictions")
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        raise ValueError("Checkpoint failed strict CPU model loading/inference") from error
    manifest_hash = _identifier(saved.get("manifest_hash"))
    preprocessing_hash = saved.get("preprocessing_hash")
    if preprocessing_hash is not None:
        _identifier(preprocessing_hash)
    return {"format_version": saved["format_version"], "config": saved["config"],
            "manifest_hash": manifest_hash, "preprocessing_hash": preprocessing_hash,
            "completed_epochs": saved.get("completed_epochs"), "global_step": saved.get("global_step"),
            "partial_epoch": saved.get("partial_epoch", False)}


def _metadata(root):
    raw = _read(root / "snapshot.json", MAX_JSON_BYTES)
    metadata = _json(raw)
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported result snapshot version")
    snapshot_id = _identifier(metadata.get("snapshot_id"))
    core = {key: value for key, value in metadata.items() if key != "snapshot_id"}
    if _sha(_json_bytes(core)) != snapshot_id:
        raise ValueError("Snapshot identity does not match its manifest")
    if (not isinstance(metadata.get("run_id"), str) or not metadata["run_id"]
            or len(metadata["run_id"]) > 200 or not isinstance(metadata.get("provenance"), dict)):
        raise ValueError("Invalid snapshot run provenance")
    files = metadata.get("files")
    if not isinstance(files, dict) or "last.pt" not in files or set(files) - FILES:
        raise ValueError("Missing or unallowlisted snapshot files")
    total = 0
    for name, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}:
            raise ValueError("Invalid snapshot file index")
        _identifier(entry["sha256"])
        size = entry["size_bytes"]
        limit = MAX_JSON_BYTES if name.endswith(".json") else MAX_FILE_BYTES
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= limit:
            raise ValueError("Invalid snapshot file size")
        total += size
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Result snapshot exceeds the total size limit")
    if {entry.name for entry in root.iterdir()} != set(files) | {"snapshot.json"}:
        raise ValueError("Unexpected or missing entries in result snapshot")
    return metadata, _sha(raw), total + len(raw)


def validate_snapshot(path):
    """Verify every byte and CPU-load checkpoints without initializing CUDA."""
    root = _directory(path)
    metadata, manifest_hash, _ = _metadata(root)
    checkpoints = {}
    for name, entry in metadata["files"].items():
        data = _read(root / name, MAX_JSON_BYTES if name.endswith(".json") else MAX_FILE_BYTES)
        if len(data) != entry["size_bytes"] or _sha(data) != entry["sha256"]:
            raise ValueError(f"Snapshot SHA-256 or size mismatch: {name}")
        if name.endswith(".pt"):
            checkpoints[name] = _checkpoint(data)
        elif name.endswith(".json"):
            _json(data)
    last = checkpoints["last.pt"]
    for saved in checkpoints.values():
        if any(saved[key] != last[key] for key in ("config", "manifest_hash", "preprocessing_hash")):
            raise ValueError("Snapshot checkpoints have inconsistent run provenance")
    if checkpoints != metadata.get("checkpoint_metadata"):
        raise ValueError("Snapshot checkpoint metadata does not match its tensors")
    for name, key in (("manifest_sha256", "manifest_hash"), ("preprocessing_sha256", "preprocessing_hash")):
        if name in metadata["provenance"] and metadata["provenance"][name] != last[key]:
            raise ValueError("Snapshot provenance does not match its checkpoint")
    return {"manifest": metadata, "manifest_sha256": manifest_hash, "checkpoint_metadata": checkpoints}


def _cleanup_stage(stage, root):
    if stage.exists() and not _linked(stage) and stage.resolve().parent == root and stage.name.startswith(".transfer-"):
        shutil.rmtree(stage)


def create_snapshot(run_dir, export_root, run_id, provenance):
    """Copy allowlisted artifacts into a new immutable export; retain live checkpoints."""
    run = _directory(run_dir)
    root = _directory(export_root, create=True)
    if run == root or run.is_relative_to(root) or root.is_relative_to(run):
        raise ValueError("Export and active training directories must be separate")
    marker = root / MARKER
    if marker.exists() or _linked(marker):
        if _json(_read(marker, MAX_JSON_BYTES)) != _MARKER_VALUE:
            raise ValueError("Invalid immutable export root marker")
    else:
        _write_new(marker, _json_bytes(_MARKER_VALUE))
    selected = [name for name in sorted(FILES) if (run / name).exists() or _linked(run / name)]
    if "last.pt" not in selected:
        raise ValueError("A resumable last.pt is required before exporting")
    projected = sum((run / name).stat().st_size for name in selected)
    if projected > MAX_TOTAL_BYTES:
        raise ValueError("Result snapshot exceeds the total size limit")
    _space(root, projected + MAX_JSON_BYTES)
    stage = Path(tempfile.mkdtemp(prefix=".transfer-", dir=root))
    try:
        files, checkpoints = {}, {}
        for name in selected:
            limit = MAX_JSON_BYTES if name.endswith(".json") else MAX_FILE_BYTES
            data = _read(run / name, limit)
            _space(root, len(data) + MAX_JSON_BYTES)
            if name.endswith(".pt"):
                checkpoints[name] = _checkpoint(data)
            elif name.endswith(".json"):
                _json(data)
            _write_new(stage / name, data)
            files[name] = {"sha256": _sha(data), "size_bytes": len(data)}
        core = {"format_version": 1, "run_id": run_id, "provenance": provenance,
                "captured_at": datetime.now(timezone.utc).isoformat(), "files": files,
                "checkpoint_metadata": checkpoints,
                "consistency": "Files captured independently; checkpoint metadata is authoritative."}
        snapshot_id = _sha(_json_bytes(core))
        _write_new(stage / "snapshot.json", _json_bytes({**core, "snapshot_id": snapshot_id}))
        validated = validate_snapshot(stage)
        destination = root / snapshot_id
        _sync_directory(stage)
        stage.rename(destination)
        _sync_directory(root)
        return {"snapshot_id": snapshot_id, "path": str(destination), **validated}
    finally:
        _cleanup_stage(stage, root)


def import_snapshot(source, local_root):
    """Verify and atomically publish a received directory, then persist its receipt."""
    source = _directory(source)
    root = _directory(local_root, create=True)
    metadata, manifest_hash, total = _metadata(source)
    snapshot_id = metadata["snapshot_id"]
    destination = root / snapshot_id
    if destination.exists() or _linked(destination):
        loaded = validate_snapshot(destination)
        if loaded["manifest_sha256"] != manifest_hash:
            raise ValueError("Existing local snapshot has a different manifest")
    else:
        _space(root, total + MAX_JSON_BYTES)
        stage = Path(tempfile.mkdtemp(prefix=".transfer-", dir=root))
        try:
            for name in ["snapshot.json", *metadata["files"]]:
                data = _read(source / name, MAX_JSON_BYTES if name.endswith(".json") else MAX_FILE_BYTES)
                _space(root, len(data) + MAX_JSON_BYTES)
                _write_new(stage / name, data)
            loaded = validate_snapshot(stage)
            if loaded["manifest_sha256"] != manifest_hash:
                raise ValueError("Snapshot manifest changed during transfer")
            _sync_directory(stage)
            stage.rename(destination)
            _sync_directory(root)
        finally:
            _cleanup_stage(stage, root)
    receipts = _directory(root / "receipts", create=True)
    receipt_path = receipts / f"{snapshot_id}.json"
    receipt = {"format_version": 1, "snapshot_id": snapshot_id, "manifest_sha256": manifest_hash,
               "files": metadata["files"], "verified": True, "checkpoint_validation": "cpu_weights_only",
               "verified_at": datetime.now(timezone.utc).isoformat(), "local_directory": str(destination)}
    if _linked(receipt_path):
        raise ValueError("Receipt paths must not be links")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".transfer-receipt-", dir=receipts)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt_path)
        _sync_directory(receipts)
    finally:
        temporary.unlink(missing_ok=True)
    return {"snapshot_id": snapshot_id, "path": str(destination), "receipt": receipt,
            "receipt_path": str(receipt_path)}


def prune_snapshot(export_root, snapshot_id, receipt):
    """Remove only a verified immutable export after receiving a local-copy receipt.

    The controller must deliver the receipt over authenticated SSH. Live run
    last.pt/best.pt are outside this marked export root and remain resumable.
    """
    snapshot_id = _identifier(snapshot_id)
    root = _directory(export_root)
    if _json(_read(root / MARKER, MAX_JSON_BYTES)) != _MARKER_VALUE:
        raise ValueError("Refusing deletion outside a marked immutable export root")
    if (not isinstance(receipt, dict) or receipt.get("format_version") != 1
            or receipt.get("verified") is not True or receipt.get("snapshot_id") != snapshot_id
            or receipt.get("checkpoint_validation") != "cpu_weights_only"):
        raise ValueError("A verified local import receipt is required before remote deletion")
    path = root / snapshot_id
    if not path.exists() and not _linked(path):
        return {"snapshot_id": snapshot_id, "deleted": False, "already_absent": True}
    if _linked(path) or path.resolve().parent != root:
        raise ValueError("Refusing linked or escaping deletion target")
    loaded = validate_snapshot(path)
    if (loaded["manifest"]["snapshot_id"] != snapshot_id
            or receipt.get("manifest_sha256") != loaded["manifest_sha256"]
            or receipt.get("files") != loaded["manifest"]["files"]):
        raise ValueError("Receipt does not verify these exact remote snapshot bytes")
    shutil.rmtree(path)
    _sync_directory(root)
    return {"snapshot_id": snapshot_id, "deleted": True, "already_absent": False}
