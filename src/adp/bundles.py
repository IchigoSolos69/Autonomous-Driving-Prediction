"""Bounded, versioned inference artifacts; training checkpoints stay separate."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import torch

from adp.data import TYPES, read_json


MAX_FILE_BYTES = 256 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 16 * 1024**2
FILES = {"weights.pt", "calibration.json", "model-card.md", "evaluation.json", "evaluation.png", "training.png"}


def input_schema(cache_dir):
    settings = read_json(Path(cache_dir) / "preprocessing.json")
    return _schema({"schema_version": 1, "preprocessing_version": settings.get("version", 1),
                    "history_steps": 50, "future_steps": 60, "sample_hz": 10,
                    "max_actors": settings.get("max_actors", 32),
                    "max_polylines": settings.get("max_polylines", 128),
                    "map_radius_m": settings.get("map_radius", 100),
                    "actor_types": TYPES, "history_features": 8, "polyline_points": 20,
                    "map_features": 4, "coordinate_frame": "focal_local_metres"})


def _schema(value):
    result = dict(value)
    fixed = {"schema_version": 1, "preprocessing_version": 1, "history_steps": 50,
             "future_steps": 60, "sample_hz": 10, "history_features": 8,
             "polyline_points": 20, "map_features": 4,
             "coordinate_frame": "focal_local_metres", "actor_types": TYPES}
    if any(result.get(key) != item for key, item in fixed.items()):
        raise ValueError("Unsupported inference input schema")
    for key, lower, upper in (("max_actors", 2, 128), ("max_polylines", 1, 1024)):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or not lower <= value <= upper:
            raise ValueError(f"Invalid input schema {key}")
        result[key] = int(value)
    radius = result.get("map_radius_m")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 < radius <= 1000:
        raise ValueError("Invalid input schema map radius")
    result["map_radius_m"] = float(radius)
    if set(result) != set(fixed) | {"max_actors", "max_polylines", "map_radius_m"}:
        raise ValueError("Unknown inference input schema fields")
    return result


def schemas_compatible(expected, actual):
    try:
        return _schema(expected) == _schema(actual)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _bounded_json(path):
    if Path(path).stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Bundle metadata is too large")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Bundle JSON metadata must contain an object")
    return value


def _weights(path):
    from adp.model import ForecastModel
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Inference weights exceed the supported file size")
    if not zipfile.is_zipfile(path):
        raise ValueError("Only modern weights-only PyTorch archives are supported")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 1000 or sum(info.file_size for info in archive.infolist()) > MAX_TOTAL_BYTES:
            raise ValueError("Expanded inference weights are too large")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") != 1:
        raise ValueError("Unsupported inference checkpoint format")
    raw_config = saved.get("config")
    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("model"), dict):
        raise ValueError("Inference checkpoint requires a model configuration object")
    config = raw_config["model"]
    limits = {"width": (128, 16, 512), "heads": (4, 1, 16),
              "temporal_layers": (2, 1, 8), "graph_layers": (2, 0, 8), "modes": (6, 6, 6)}
    for key, (default, low, high) in limits.items():
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"Unsupported model configuration: {key}")
    if config.get("width", 128) % config.get("heads", 4):
        raise ValueError("Model width must divide evenly into attention heads")
    for key in ("use_graph", "use_map"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"Invalid model configuration: {key}")
    _schema(saved.get("input_schema", {}))
    with torch.device("meta"):
        model = ForecastModel(config)
    state = saved.get("model_state_dict", {})
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Model tensor names do not match the architecture")
    if sum(value.numel() for value in expected.values()) > 25_000_000:
        raise ValueError("Model exceeds the supported parameter budget")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError(f"Invalid inference tensor: {name}")
    return saved


def export_model(checkpoint, cache_dir, output_dir, calibration=None, model_card=None, evaluation=None):
    """Export to a new directory without optimizer, scheduler, scaler, or RNG state."""
    from adp.training import load_checkpoint
    checkpoint, cache_dir, output_dir = Path(checkpoint), Path(cache_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new inference bundle directory")
    saved = load_checkpoint(checkpoint)
    manifest = read_json(cache_dir / "manifest.json")
    if saved["manifest_hash"] != _sha(cache_dir / "manifest.json"):
        raise ValueError("Export cache does not match the checkpoint provenance")
    if saved.get("preprocessing_hash") and saved["preprocessing_hash"] != _sha(cache_dir / "preprocessing.json"):
        raise ValueError("Export preprocessing does not match the checkpoint")
    schema = input_schema(cache_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".bundle-export-", dir=output_dir.parent))
    try:
        inference = {key: saved[key] for key in ("config", "manifest_hash", "preprocessing_hash", "risk_supervision_observations", "data_audit") if key in saved}
        inference["format_version"] = 1
        inference["config"] = {"model": saved["config"].get("model", saved["config"])}
        inference["input_schema"] = schema
        inference["model_state_dict"] = {key: value.detach().cpu().contiguous() for key, value in saved["model_state_dict"].items()}
        torch.save(inference, stage / "weights.pt")
        weight_hash = _sha(stage / "weights.pt")
        original_hash = _sha(checkpoint)
        if calibration:
            calibrated = _bounded_json(calibration)
            if calibrated.get("checkpoint_sha256") != original_hash or calibrated.get("manifest_hash") != saved["manifest_hash"]:
                raise ValueError("Calibration does not belong to the source checkpoint and manifest")
            if calibrated.get("preprocessing_hash") not in (None, saved.get("preprocessing_hash")):
                raise ValueError("Calibration preprocessing does not match")
            calibrated["source_checkpoint_sha256"] = original_hash
            calibrated["checkpoint_sha256"] = weight_hash
            _json(stage / "calibration.json", calibrated)
        if model_card:
            if Path(model_card).stat().st_size > MAX_METADATA_BYTES:
                raise ValueError("Model card is too large")
            shutil.copyfile(model_card, stage / "model-card.md")
        else:
            (stage / "model-card.md").write_text("# Research forecasting model\n\nNoncommercial research beta. Proximity and footprint scores are experimental proxies, not driving-safety guarantees.\n", encoding="utf-8")
        evidence = None
        evaluation = Path(evaluation) if evaluation else checkpoint.parent / "evaluation" / "metrics.json"
        if evaluation.is_file():
            evidence = _bounded_json(evaluation)
            if evidence.get("checkpoint_sha256") not in (None, original_hash):
                raise ValueError("Evaluation belongs to a different source checkpoint")
            _json(stage / "evaluation.json", evidence)
        for name, image in (("evaluation", evaluation.parent / "evaluation.png"), ("training", checkpoint.parent / "training.png")):
            if image.is_file():
                if image.stat().st_size > MAX_METADATA_BYTES:
                    raise ValueError("Report image is too large")
                shutil.copyfile(image, stage / f"{name}.png")
        provenance = {"source_checkpoint_sha256": original_hash, "training_manifest_sha256": saved["manifest_hash"],
                      "preprocessing_sha256": saved.get("preprocessing_hash"),
                      "training_scenario_ids": manifest["splits"].get("train", []),
                      "selection_scenario_ids": manifest["splits"].get("validation", []),
                      "calibration_scenario_ids": manifest["splits"].get("calibration", []) if calibration else []}
        metadata = {"format_version": 1, "bundle_id": weight_hash, "input_schema": schema,
                    "model": {"architecture": "temporal_graph_scene_mixture_v1", "config": inference["config"]["model"]},
                    "provenance": provenance, "evaluation": evidence,
                    "evaluation_provenance_verified": bool(evidence and evidence.get("checkpoint_sha256") == original_hash),
                    "files": {path.name: {"sha256": _sha(path), "size_bytes": path.stat().st_size} for path in stage.iterdir()}}
        _json(stage / "bundle.json", metadata)
        load_bundle(stage)
        stage.rename(output_dir)
        return metadata
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_bundle(path):
    """Validate an installed directory; use install_bundle for untrusted ZIPs."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("Expected a bundle directory; install ZIP bundles before loading")
    if (root / "bundle.json").is_symlink():
        raise ValueError("Bundle metadata must not be a link")
    metadata = _bounded_json(root / "bundle.json")
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported model bundle version")
    schema = _schema(metadata.get("input_schema", {}))
    files = metadata.get("files", {})
    if not isinstance(files, dict) or not {"weights.pt", "model-card.md"} <= set(files) or set(files) - FILES:
        raise ValueError("Unexpected or missing model bundle files")
    if {entry.name for entry in root.iterdir()} != set(files) | {"bundle.json"}:
        raise ValueError("Unlisted files in model bundle")
    total = 0
    for name, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError("Invalid bundle file index")
        target = root / name
        if target.is_symlink() or not target.is_file() or target.resolve().parent != root:
            raise ValueError("Bundle links and directories are not supported")
        size = target.stat().st_size
        if not 0 < size <= MAX_FILE_BYTES or size != entry.get("size_bytes") or _sha(target) != entry.get("sha256"):
            raise ValueError(f"Bundle file size or SHA-256 mismatch: {name}")
        total += size
    if total > MAX_TOTAL_BYTES or metadata.get("bundle_id") != files["weights.pt"]["sha256"]:
        raise ValueError("Invalid bundle size or identifier")
    saved = _weights(root / "weights.pt")
    if not schemas_compatible(schema, saved["input_schema"]):
        raise ValueError("Bundle and checkpoint input schemas differ")
    provenance = metadata.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Bundle provenance must contain an object")
    if provenance.get("training_manifest_sha256") != saved.get("manifest_hash"):
        raise ValueError("Bundle training provenance differs from its checkpoint")
    if "evaluation.json" in files and _bounded_json(root / "evaluation.json") != metadata.get("evaluation"):
        raise ValueError("Bundle evaluation evidence differs from its indexed summary")
    if metadata.get("evaluation_provenance_verified") and (metadata.get("evaluation") or {}).get("checkpoint_sha256") != provenance.get("source_checkpoint_sha256"):
        raise ValueError("Bundle evaluation checkpoint identity is not verified")
    if "model" in metadata and (not isinstance(metadata["model"], dict) or metadata["model"].get("config") != saved["config"]["model"]):
        raise ValueError("Bundle model configuration differs from its checkpoint")
    if "calibration.json" in files:
        from adp.calibration import apply_calibration
        calibration = _bounded_json(root / "calibration.json")
        if calibration.get("checkpoint_sha256") != metadata["bundle_id"] or calibration.get("manifest_hash") != saved.get("manifest_hash"):
            raise ValueError("Bundle calibration does not match the weights and provenance")
        if calibration.get("preprocessing_hash") not in (None, saved.get("preprocessing_hash")):
            raise ValueError("Bundle calibration preprocessing differs")
        apply_calibration({"logits": torch.zeros(1), "risk_logits": torch.zeros(1)}, calibration)
    return {"metadata": metadata, "input_schema": schema,
            "paths": {"checkpoint": root / "weights.pt", "model_card": root / "model-card.md",
                      "calibration": root / "calibration.json" if "calibration.json" in files else None,
                      "evaluation": root / "evaluation.json" if "evaluation.json" in files else None,
                      "reports_dir": root}}


def install_bundle(source, models_dir):
    """Validate before atomic promotion; preserve the previous active version."""
    source, models_dir = Path(source).resolve(), Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".bundle-import-", dir=models_dir))
    try:
        if source.is_dir():
            total = 0
            for path in source.iterdir():
                if path.is_symlink() or not path.is_file() or path.name not in FILES | {"bundle.json"}:
                    raise ValueError("Unexpected model bundle entry")
                if path.stat().st_size > (MAX_METADATA_BYTES if path.name == "bundle.json" else MAX_FILE_BYTES):
                    raise ValueError("Model bundle file is too large")
                total += path.stat().st_size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("Model bundle is too large")
                shutil.copyfile(path, stage / path.name)
        else:
            with zipfile.ZipFile(source) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(names) != len(set(name.casefold() for name in names)) or len(entries) > len(FILES) + 1:
                    raise ValueError("Duplicate or excessive bundle ZIP entries")
                if sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES:
                    raise ValueError("Expanded model bundle is too large")
                for entry in entries:
                    name = PurePosixPath(entry.filename)
                    if entry.filename not in FILES | {"bundle.json"} or name.name != entry.filename or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                        raise ValueError("Unsafe bundle ZIP path or link")
                    limit = MAX_METADATA_BYTES if entry.filename == "bundle.json" else MAX_FILE_BYTES
                    if entry.file_size > limit:
                        raise ValueError("Model bundle ZIP entry is too large")
                    with archive.open(entry) as incoming, (stage / name.name).open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        loaded = load_bundle(stage)
        bundle_id = loaded["metadata"]["bundle_id"]
        destination = models_dir / bundle_id
        if destination.exists():
            load_bundle(destination)
        else:
            stage.rename(destination)
        active_path = models_dir / "active.json"
        active = _bounded_json(active_path) if active_path.is_file() else {}
        for name in ("current", "previous"):
            value = active.get(name)
            if value is not None and (not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("Invalid installed model version index")
        previous = active.get("current") if active.get("current") != bundle_id else active.get("previous")
        temporary = models_dir / "active.json.tmp"
        _json(temporary, {"format_version": 1, "current": bundle_id, "previous": previous})
        os.replace(temporary, active_path)
        obsolete = active.get("previous")
        if obsolete and obsolete not in (bundle_id, previous):
            old_path = models_dir / obsolete
            if old_path.is_dir() and not old_path.is_symlink() and old_path.resolve().parent == models_dir:
                try:
                    load_bundle(old_path)
                    shutil.rmtree(old_path)
                except (ValueError, OSError):
                    logging.warning("Installed the new model, but retained an unreadable previous bundle: %s", old_path)
        return load_bundle(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
