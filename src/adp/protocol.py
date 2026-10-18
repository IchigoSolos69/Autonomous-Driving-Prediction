"""Fixed benchmark membership and explicit, independently verifiable provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path

from adp.data import manifest_hash, read_json, write_json


def ids_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def create_protocol(cache_dir, output, provenance_manifests=()):
    cache_dir, output = Path(cache_dir), Path(output)
    manifest = read_json(cache_dir / "manifest.json")
    snapshots = {}
    for path in [cache_dir / "manifest.json", *map(Path, provenance_manifests)]:
        value = read_json(path)
        if "splits" not in value:
            raise ValueError(f"Not a dataset manifest: {path}")
        snapshots[manifest_hash(path)] = value
    value = {"version": 1, "dataset_manifest_sha256": manifest_hash(cache_dir / "manifest.json"),
             "split_hashes": {split: ids_hash(ids) for split, ids in manifest["splits"].items()},
             "provenance_manifests": snapshots,
             "policy": "Selection only on validation; calibration only on calibration; evaluation locked until checkpoint freeze"}
    if output.exists():
        if read_json(output) != value:
            raise ValueError("Benchmark protocol is immutable; use a new output path")
    else:
        write_json(output, value)
    return value


def validate_benchmark(saved, cache_dir, protocol, split="evaluation", calibration=None):
    from adp.bundles import input_schema

    protocol = read_json(protocol) if isinstance(protocol, (str, Path)) else protocol
    manifest = read_json(Path(cache_dir) / "manifest.json")
    if protocol.get("version") != 1 or protocol["dataset_manifest_sha256"] != manifest_hash(Path(cache_dir) / "manifest.json"):
        raise ValueError("Benchmark dataset/protocol identifiers differ")
    ids = manifest["splits"][split]
    if ids_hash(ids) != protocol["split_hashes"].get(split) or len(ids) != len(set(ids)):
        raise ValueError("Benchmark split membership differs or contains duplicates")
    if saved is None:
        return {"split_sha256": ids_hash(ids), "baseline": True}
    provenance = protocol["provenance_manifests"].get(saved.get("manifest_hash"))
    if provenance is None:
        raise ValueError("Checkpoint training provenance is absent from benchmark protocol")
    exposed = set(provenance["splits"].get("train", []))
    if split == "evaluation":
        exposed.update(provenance["splits"].get("validation", []))
        exposed.update(provenance["splits"].get("calibration", []))
    if exposed.intersection(ids):
        raise ValueError("Benchmark leaks checkpoint training/selection/calibration scenarios")
    if calibration:
        fitted = protocol["provenance_manifests"].get(calibration.get("manifest_hash"))
        if fitted is None or (split == "evaluation" and set(fitted["splits"].get("calibration", [])).intersection(ids)):
            raise ValueError("Calibration provenance is missing or overlaps evaluation")
    schema = input_schema(cache_dir)
    if saved.get("input_schema") is not None and saved["input_schema"] != schema:
        raise ValueError("Checkpoint input schema differs from benchmark")
    # Legacy checkpoints only recorded the byte hash; require an exact match.
    if "input_schema" not in saved:
        from adp.data import preprocessing_hash
        if saved.get("preprocessing_hash") != preprocessing_hash(cache_dir):
            raise ValueError("Legacy checkpoint preprocessing differs from benchmark")
    return {"split_sha256": ids_hash(ids), "training_manifest_sha256": saved["manifest_hash"],
            "training_selection_calibration_disjoint": True}


def paired_comparison(candidate, reference, bootstrap=1000, seed=42):
    """Resample matched scenarios, never independent actors or unpaired models."""
    import numpy as np

    if bootstrap < 1:
        raise ValueError("Paired comparison requires at least one bootstrap replicate")
    def indexed(records):
        result = {}
        for item in records:
            if item["scenario_id"] in result:
                raise ValueError("Paired comparison requires one record per scenario")
            result[item["scenario_id"]] = item
        return result
    left, right = indexed(candidate), indexed(reference)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical nonempty scenario sets")
    keys = ("jointFDE", "jointADE", "focal_minFDE", "minFDE", "minADE", "miss_rate", "NLL")
    rng = np.random.default_rng(seed)
    ids = sorted(left)
    draws = rng.integers(len(ids), size=(bootstrap, len(ids)))
    comparisons = {}
    for key in keys:
        values = []
        for sid in ids:
            a, b = left[sid].get(key), right[sid].get(key)
            if a is None or b is None or (isinstance(a, list) and not a) or (isinstance(b, list) and not b):
                values.append(np.nan)
            else:
                values.append(float(np.mean(a)) - float(np.mean(b)))
        values = np.asarray(values)
        valid = np.isfinite(values)
        if not valid.any():
            comparisons[key] = {"delta": None, "interval95": None, "eligible_scenarios": 0}
            continue
        sampled = values[draws]
        counts = np.isfinite(sampled).sum(1)
        estimates = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
        comparisons[key] = {"delta": float(values[valid].mean()),
                            "interval95": np.percentile(estimates, [2.5, 97.5]).tolist(),
                            "eligible_scenarios": int(valid.sum())}
    primary = comparisons["jointFDE"]["interval95"]
    return {"scenario_count": len(ids), "bootstrap": bootstrap, "seed": seed, "unit": "scenario",
            "direction": "candidate minus reference; negative error differences favor candidate",
            "aggregation": "scene means; actor metrics are scene-normalized for paired differences",
            "metrics": comparisons, "jointFDE_gate_passed": bool(primary and primary[1] < 0)}
