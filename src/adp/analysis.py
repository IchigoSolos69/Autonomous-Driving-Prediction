"""Selection-only failure analysis and bounded, non-executable recorded replays."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading

import numpy as np
import torch

from adp.data import TYPES, collate_scenes, load_scene, manifest_hash, public_scene, read_json
from adp.metrics import evaluate_predictions
from adp.model import ForecastModel, constant_velocity, model_inputs
from adp.training import load_checkpoint
from adp.research import ensure_space


MAX_FOLDER_BYTES = 64 * 1024**2
MAX_RECORD_BYTES = 256 * 1024**2
MAX_CASE_BYTES = 24 * 1024**2
MAX_CASES = 64
FILTERS = ("city", "actor_type", "role", "speed_bin", "turn_bin", "density_bin", "history_bin")
SLICE_DEFINITIONS = {
    "speed_bin": "Observed cutoff speed [0,1), [1,5), [5,15), [15,infinity) m/s; missing cutoff is unknown",
    "turn_bin": "Signed heading change from observed step 39 to 49: left >15 degrees, right <-15 degrees; missing endpoints are unknown",
    "density_bin": "Other retained dynamic actors at the cutoff within 20 metres: 0, 1-3, 4+",
    "history_bin": "Observed history coverage across 50 steps: <80% or >=80%",
    "metrics": "Displacements use only retained scored dynamic actors with all 60 finite future states; scene joint errors retain shared-world alignment",
}
_OPAQUE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _json_bytes(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_json(path, value, compressed=False):
    data = _json_bytes(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(gzip.compress(data, mtime=0) if compressed else data)


def _bounded_json(path, limit, compressed=False):
    try:
        with (gzip.open(path, "rb") if compressed else Path(path).open("rb")) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Expanded analysis metadata exceeds its supported size")
        return json.loads(data, parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Non-finite JSON value")))
    except (OSError, UnicodeError, json.JSONDecodeError, EOFError, RecursionError) as error:
        raise ValueError(f"Cannot read analysis metadata: {Path(path).name}") from error


def actor_descriptors(scene):
    """All slice assignments depend exclusively on observed states."""
    history, mask = np.asarray(scene["history"]), np.asarray(scene["history_mask"], dtype=bool)
    dynamic = np.asarray(scene["dynamic_mask"], dtype=bool) & np.asarray(scene["actor_mask"], dtype=bool)
    result = []
    for index, actor_id in enumerate(scene["actor_ids"]):
        cutoff = bool(mask[index, 49])
        speed = float(np.linalg.norm(history[index, 49, 2:4])) if cutoff else None
        speed_bin = "unknown" if speed is None else ("<1" if speed < 1 else "1-<5" if speed < 5 else "5-<15" if speed < 15 else ">=15")
        turn = "unknown"
        if cutoff and mask[index, 39]:
            a = math.atan2(history[index, 39, 4], history[index, 39, 5])
            b = math.atan2(history[index, 49, 4], history[index, 49, 5])
            delta = math.atan2(math.sin(b-a), math.cos(b-a))
            turn = "left" if delta > math.radians(15) else "right" if delta < -math.radians(15) else "straight"
        density = "unknown"
        if cutoff:
            distance = np.linalg.norm(history[:, 49, :2] - history[index, 49, :2], axis=-1)
            neighbors = dynamic & mask[:, 49] & (distance <= 20)
            neighbors[index] = False
            count = int(neighbors.sum())
            density = "0" if count == 0 else "1-3" if count <= 3 else "4+"
        result.append({"actor_id": str(actor_id), "city": str(scene["metadata"]["city"]),
            "actor_type": TYPES[int(scene["actor_types"][index])],
            "role": "focal" if str(actor_id) == scene["metadata"]["focal_track_id"] else "other",
            "speed_bin": speed_bin, "turn_bin": turn, "density_bin": density,
            "history_bin": ">=80%" if mask[index].mean() >= .8 else "<80%"})
    return result


def actor_metrics(scene, outputs):
    future = np.asarray(scene["future"])
    complete = np.asarray(scene["future_mask"], dtype=bool).all(-1) & np.isfinite(future).all(axis=(1, 2))
    requested = np.asarray(scene["scored_mask"], dtype=bool) & np.asarray(scene["dynamic_mask"], dtype=bool) & np.asarray(scene["actor_mask"], dtype=bool)
    means = outputs["means"][0].detach().cpu().numpy()
    rows = []
    for index in range(len(scene["actor_ids"])):
        eligible = bool(requested[index] and complete[index])
        row = {"eligible": eligible, "requested": bool(requested[index]),
               "excluded_incomplete_future": bool(requested[index] and not complete[index]),
               "minADE": None, "minFDE": None, "miss_rate": None}
        if eligible:
            distance = np.linalg.norm(means[:, index] - future[index], axis=-1)
            row.update(minADE=float(distance.mean(-1).min()), minFDE=float(distance[:, -1].min()))
            row["miss_rate"] = float(row["minFDE"] > 2)
        rows.append(row)
    return rows


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _cohort(records, labels, filters=None):
    filters = {key: value for key, value in (filters or {}).items() if value is not None and value != ""}
    if set(filters) - set(FILTERS):
        raise ValueError("Unknown analysis filter")
    actors, selected = [], []
    for record in records:
        matching = [actor for actor in record["actors"] if all(actor.get(key) == value for key, value in filters.items())]
        if matching:
            selected.append(record)
            actors.extend(matching)
    metrics = {}
    for label in labels:
        eligible = [actor["metrics"][label] for actor in actors if actor["metrics"][label]["eligible"]]
        focal = [actor["metrics"][label] for actor in actors if actor["role"] == "focal" and actor["metrics"][label]["eligible"]]
        sources = [record.get("source_selection", {}) for record in selected]
        total = sum(source.get("total_scored", 0) for source in sources)
        retained = sum(source.get("retained_scored", 0) for source in sources)
        metrics[label] = {key: _mean([actor[key] for actor in eligible]) for key in ("minADE", "minFDE", "miss_rate")}
        metrics[label].update({"focal_minADE": _mean([actor["minADE"] for actor in focal]),
            "focal_minFDE": _mean([actor["minFDE"] for actor in focal]),
            "focal_miss_rate": _mean([actor["miss_rate"] for actor in focal]),
            "jointADE": _mean([record["metrics"][label].get("jointADE") for record in selected]),
            "jointFDE": _mean([record["metrics"][label].get("jointFDE") for record in selected]),
            "NLL": _mean([record["metrics"][label].get("NLL") for record in selected]),
            "footprint_collision_proxy": _mean([record["metrics"][label].get("footprint_collision_proxy") for record in selected]),
            "eligible_actors": len(eligible), "requested_actors": sum(actor["metrics"][label]["requested"] for actor in actors),
            "excluded_incomplete_future": sum(actor["metrics"][label]["excluded_incomplete_future"] for actor in actors),
            "retained_target_coverage": retained / total if total else None})
    return {"actors": len(actors), "scenarios": len(selected), "small_sample": len(selected) < 50}, metrics, selected, actors


def _detail(metadata, records, filters=None):
    labels = metadata["model_labels"]
    counts, metrics, selected, actors = _cohort(records, labels, filters)
    slices = []
    for dimension in FILTERS:
        for value in sorted({actor[dimension] for actor in actors}):
            values = dict(filters or {})
            values[dimension] = value
            c, m, _, _ = _cohort(selected, labels, values)
            slices.append({"dimension": dimension, "value": value, "actor_count": c["actors"],
                "scenario_count": c["scenarios"], "small_sample": c["small_sample"], "metrics": m})
    lookup = {record["scenario_id"]: record for record in selected}
    cases = []
    for preview in metadata["cases"]:
        record = lookup.get(preview["scenario_id"])
        if record is None:
            continue
        descriptor = [actor for actor in record["actors"] if all(actor.get(key) == value for key, value in (filters or {}).items() if value is not None and value != "")]
        cases.append({**preview, "actors": [{key: value for key, value in actor.items() if key != "metrics"} for actor in descriptor]})
    counts["cases"] = len(cases)
    return {key: metadata[key] for key in ("id", "title", "split", "development_only", "model_labels", "provenance", "slice_definitions")} | {
        "counts": counts, "metrics": metrics, "slices": slices, "cases": cases,
        "preview_limit": MAX_CASES, "preview_reduction": metadata.get("preview_reduction", 0),
        "joint_metric_scope": "Whole-scene errors for scenarios containing matching actors; worlds remain shared"}


def _prediction(scene, outputs, label, checkpoint_id, seed, provenance):
    variant = label.rsplit("-", 1)[0] if label.startswith(("control-", "directional-")) else label
    return {"scenario_id": scene["id"], "actor_ids": scene["actor_ids"],
        "trajectories": outputs["means"][0].detach().cpu().tolist(),
        "scales": outputs["scales"][0].detach().cpu().tolist(),
        "probabilities": outputs["logits"][0].softmax(-1).detach().cpu().tolist(),
        "risk": {"raw": outputs["risk_logits"][0].sigmoid().detach().cpu().tolist(), "calibrated": None,
                 "status": "experimental_recorded_uncalibrated", "applicability": "development_only"},
        "model": {"kind": "constant_velocity" if checkpoint_id is None else "trained", "id": checkpoint_id or "constant_velocity_v1"},
        "calibration_id": None, "coordinate_frame": scene["metadata"]["coordinate_frame"],
        "scenario": public_scene(scene), "stress": None, "elapsed_ms": 0., "source": "recorded_analysis",
        "calibration_applicability": "unknown", "analysis_label": variant, "analysis_seed": seed,
        "recorded": True, "development_only": True, "analysis_provenance": provenance}


def _models(study_root):
    root = Path(study_root).resolve()
    protocol = read_json(root / "protocol.json") if (root / "protocol.json").is_file() else {}
    if protocol.get("kind") == "direction-study":
        experiment = read_json(root / "experiment.json")
        if experiment.get("status") != "completed":
            raise ValueError("The six-run direction experiment must complete before failure analysis")
        models = []
        for variant in ("control", "directional"):
            for seed in (42, 43, 44):
                name = f"{variant}-{seed}"
                run = experiment["runs"][name]
                checkpoint = Path(run["checkpoint"])
                if not run.get("completed") or manifest_hash(checkpoint) != run["last_sha256"]:
                    raise ValueError("An experiment final checkpoint changed or is incomplete")
                models.append((name, checkpoint, seed))
        cache = Path(protocol["cache_dir"])
        expected = protocol["selection_ids"]
        provenance = {"kind": "direction-study", "protocol_sha256": manifest_hash(root / "protocol.json"),
                      "experiment_sha256": manifest_hash(root / "experiment.json"), "seeds": [42, 43, 44]}
    else:
        state = read_json(root / "pipeline_status.json")
        if state.get("status") != "completed":
            raise ValueError("The current frozen study must complete before failure analysis")
        frozen = read_json(root / "frozen_selection.json")
        checkpoint = Path(frozen["checkpoint"])
        if manifest_hash(checkpoint) != frozen["checkpoint_sha256"]:
            raise ValueError("Frozen reference checkpoint changed")
        models = [("reference", checkpoint, int(frozen["deployment_run"].rsplit("-", 1)[-1]))]
        cache = root / "cache"
        expected = read_json(cache / "manifest.json")["splits"]["validation"]
        provenance = {"kind": "reference", "frozen_selection_sha256": manifest_hash(root / "frozen_selection.json")}
    manifest = read_json(cache / "manifest.json")
    if manifest["splits"].get("validation") != expected or not expected or len(expected) != len(set(expected)):
        raise ValueError("Analysis selection membership differs from the frozen protocol")
    prohibited = set(manifest["splits"].get("calibration", [])) | set(manifest["splits"].get("evaluation", []))
    if set(expected) & prohibited:
        raise ValueError("Selection IDs overlap calibration or locked evaluation")
    provenance.update({"manifest_sha256": manifest_hash(cache / "manifest.json"),
        "preprocessing_sha256": manifest_hash(cache / "preprocessing.json"), "selection_ids": expected,
        "checkpoint_sha256": {label: manifest_hash(path) for label, path, _ in models},
        "inference_device": "cpu", "calibration_applied": False, "labels_used": "selection_only",
        "analysis_source_sha256": manifest_hash(__file__),
        "slice_definitions_sha256": hashlib.sha256(_json_bytes(SLICE_DEFINITIONS)).hexdigest()})
    return cache, models + [("constant_velocity", None, None)], provenance


def _load_models(specifications, cache):
    models = []
    for label, path, seed in specifications:
        network = None
        if path is not None:
            checkpoint = load_checkpoint(path)
            if checkpoint.get("preprocessing_hash") != manifest_hash(cache / "preprocessing.json"):
                raise ValueError("Analysis model and cache preprocessing differ")
            if checkpoint.get("manifest_hash") != manifest_hash(cache / "manifest.json"):
                raise ValueError("Analysis checkpoint does not belong to the study manifest")
            network = ForecastModel(checkpoint["config"].get("model", checkpoint["config"]))
            network.load_state_dict(checkpoint["model_state_dict"])
            network.eval()
        models.append((label, network, manifest_hash(path) if path else None, seed))
    return models


def _preview_order(records, kind, labels):
    def value(record, key, selected):
        return _mean([record["metrics"][label].get(key) for label in selected])
    def descending(number):
        return -number if number is not None else float("inf")
    if kind == "direction-study":
        control = [label for label in labels if label.startswith("control-")]
        directional = [label for label in labels if label.startswith("directional-")]
        def regression(record):
            a, b = value(record, "jointFDE", directional), value(record, "jointFDE", control)
            return a-b if a is not None and b is not None else None
        first = sorted(records, key=lambda record: (descending(regression(record)), record["scenario_id"]))
        second = sorted(records, key=lambda record: (descending(value(record, "focal_minFDE", directional)), record["scenario_id"]))
        chosen = first[:32] + second[:32] + first + second
    else:
        chosen = sorted(records, key=lambda record: (descending(record["metrics"][labels[0]].get("focal_minFDE")), record["scenario_id"]))
    unique = {}
    for record in chosen:
        unique.setdefault(record["scenario_id"], record)
    return list(unique.values())[:MAX_CASES]


def _plot(stage, detail):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    axes = figure.subplots(1, 2)
    labels = detail["model_labels"]
    for axis, key in zip(axes, ("focal_minFDE", "jointFDE")):
        values = [detail["metrics"][label].get(key) for label in labels]
        axis.bar(np.arange(len(labels)), [value or 0 for value in values], color="#327b96")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel("Metres (lower is better)")
        axis.set_title(key)
    figure.suptitle("Selection-set development evidence · recorded forecasts · no safety claim")
    figure.savefig(stage / "failures.png", dpi=120)


def analyze_failures(study_root, output_dir):
    """Build a new portable report; validation labels only, CPU inference only."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a new analysis output directory; recorded reports are immutable")
    ensure_space(output.parent, additional_bytes=MAX_FOLDER_BYTES + 2*1024**2)
    torch.set_num_threads(2)
    cache, specifications, provenance = _models(study_root)
    networks = _load_models(specifications, cache)
    labels = [item[0] for item in networks]
    records = []
    with torch.inference_mode():
        for sid in provenance["selection_ids"]:
            if not _OPAQUE.fullmatch(sid):
                raise ValueError("Invalid scenario identifier in selection manifest")
            scene = load_scene(cache / "validation" / f"{sid}.npz")
            if scene["id"] != sid:
                raise ValueError("Cached selection scenario ID does not match its filename")
            batch = collate_scenes([scene])
            inputs = model_inputs(batch)
            actors = actor_descriptors(scene)
            for actor in actors:
                actor["metrics"] = {}
            row = {"scenario_id": sid, "city": scene["metadata"]["city"], "actors": actors,
                "source_selection": scene["metadata"].get("selection", {}), "metrics": {}}
            for label, network, _, _ in networks:
                outputs = network(**inputs) if network is not None else constant_velocity(inputs)
                metric = evaluate_predictions(outputs, batch)[0]
                row["metrics"][label] = {key: value for key, value in metric.items() if key not in (
                    "risk_labels", "risk_raw_probabilities", "risk_calibrated_probabilities", "minADE", "minFDE", "miss_rate", "brierFDE", "ADE_at_FDE_best_mode")}
                for actor, values in zip(actors, actor_metrics(scene, outputs)):
                    actor["metrics"][label] = values
            records.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".analysis-", dir=output.parent))
    try:
        metadata = {"format_version": 1, "title": "Map-direction failure analysis" if provenance["kind"] == "direction-study" else "Reference model failure analysis",
            "split": "validation", "development_only": True, "model_labels": labels,
            "provenance": provenance, "slice_definitions": SLICE_DEFINITIONS, "cases": []}
        metadata["id"] = hashlib.sha256(_json_bytes({"provenance": provenance, "format_version": 1})).hexdigest()[:24]
        _write_json(stage / "records.json.gz", records, True)
        ordered = _preview_order(records, provenance["kind"], labels)
        for row in ordered:
            sid = row["scenario_id"]
            scene = load_scene(cache / "validation" / f"{sid}.npz")
            batch = collate_scenes([scene])
            predictions = []
            with torch.inference_mode():
                for label, network, checkpoint_id, seed in networks:
                    outputs = network(**model_inputs(batch)) if network is not None else constant_velocity(model_inputs(batch))
                    predictions.append(_prediction(scene, outputs, label, checkpoint_id, seed, provenance))
            payload = {"analysis_id": metadata["id"], "scenario": public_scene(scene), "predictions": predictions}
            if len(_json_bytes(payload)) > MAX_CASE_BYTES:
                continue
            case_path = stage / "cases" / f"{sid}.json.gz"
            _write_json(case_path, payload, True)
            if sum(path.stat().st_size for path in stage.rglob("*") if path.is_file()) > MAX_FOLDER_BYTES - 2*1024**2:
                case_path.unlink()
                continue
            directional = [label for label in labels if label.startswith("directional-")]
            control = [label for label in labels if label.startswith("control-")]
            primary = directional or [labels[0]]
            focal = next((actor for actor in row["actors"] if actor["role"] == "focal"), row["actors"][0])
            joint = _mean([row["metrics"][label]["jointFDE"] for label in primary])
            control_joint = _mean([row["metrics"][label]["jointFDE"] for label in control])
            metadata["cases"].append({"scenario_id": sid, "city": row["city"], "focal_type": focal["actor_type"],
                "actor_count": len(row["actors"]), "focal_minFDE": _mean([row["metrics"][label]["focal_minFDE"] for label in primary]),
                "jointFDE": joint, "delta_jointFDE": joint-control_joint if joint is not None and control_joint is not None else None})
        metadata["preview_reduction"] = len(ordered)-len(metadata["cases"])
        detail = _detail(metadata, records)
        _write_json(stage / "summary.json", detail)
        with (stage / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dimension", "value", "actors", "scenarios", "small_sample", "model", "minADE", "minFDE", "miss_rate", "jointFDE", "focal_minFDE", "excluded_incomplete_future"])
            all_entry = {"dimension": "all", "value": "all", "actor_count": detail["counts"]["actors"], "scenario_count": detail["counts"]["scenarios"], "small_sample": detail["counts"]["small_sample"], "metrics": detail["metrics"]}
            for entry in [all_entry, *detail["slices"]]:
                for label, metric in entry["metrics"].items():
                    cells = [entry[key] for key in ("dimension", "value", "actor_count", "scenario_count", "small_sample")]+[label]+[metric.get(key) for key in ("minADE", "minFDE", "miss_rate", "jointFDE", "focal_minFDE", "excluded_incomplete_future")]
                    writer.writerow(["'"+cell if isinstance(cell, str) and cell.lstrip().startswith(("=", "+", "-", "@")) else cell for cell in cells])
        _plot(stage, detail)
        metadata["files"] = {path.relative_to(stage).as_posix(): {"sha256": manifest_hash(path), "bytes": path.stat().st_size}
            for path in stage.rglob("*") if path.is_file()}
        _write_json(stage / "analysis.json", metadata)
        validate_analysis(stage)
        os.replace(stage, output)
        return {"id": metadata["id"], "output_dir": str(output), "counts": detail["counts"], "development_only": True,
                "model_labels": labels, "preview_reduction": metadata["preview_reduction"]}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _finite_json(value, depth=0):
    if depth > 32:
        raise ValueError("Analysis metadata is nested too deeply")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Analysis contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 200:
                raise ValueError("Invalid analysis metadata key")
            _finite_json(item, depth+1)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item, depth+1)


def _valid_point(value):
    return (isinstance(value, list) and len(value) == 2
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


def _validate_case(case, metadata, sid, record):
    try:
        if case["analysis_id"] != metadata["id"] or case["scenario"]["id"] != sid:
            raise ValueError("Recorded case identifier differs from its manifest")
        scene = case["scenario"]
        actors = scene["actors"]
        count = len(actors)
        if not 1 <= count <= 32 or scene["observed_steps"] != 50 or scene["future_steps"] != 60 or scene["dt"] != .1:
            raise ValueError("Recorded case has an unsupported scene shape")
        if not isinstance(scene["city"], str) or not 0 < len(scene["city"]) <= 160 or scene["city"] != record["city"]:
            raise ValueError("Recorded city differs from its analysis descriptors")
        ids = [actor["id"] for actor in actors]
        if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not 0 < len(item) <= 160 for item in ids):
            raise ValueError("Invalid recorded actor identifiers")
        if ids != [actor["actor_id"] for actor in record["actors"]]:
            raise ValueError("Recorded actor order differs from its analysis descriptors")
        focal, ego = scene["focal_track_id"], scene["ego_track_id"]
        if focal not in ids or (ego is not None and ego not in ids):
            raise ValueError("Recorded focal or ego actor is missing")
        for actor, descriptor in zip(actors, record["actors"]):
            if (actor["type"] not in TYPES or type(actor["focal"]) is not bool or type(actor["ego"]) is not bool
                    or len(actor["history"]) != 50 or len(actor["headings"]) != 50 or len(actor["reference_future"]) != 60):
                raise ValueError("Recorded actor history/future shape differs")
            for point in actor["history"] + actor["reference_future"]:
                if point is not None and not _valid_point(point):
                    raise ValueError("Invalid recorded coordinate")
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)) for value in actor["headings"]):
                raise ValueError("Invalid recorded heading")
            valid = [point is not None for point in actor["history"]]
            if valid != [heading is not None for heading in actor["headings"]]:
                raise ValueError("Recorded position/heading observation masks differ")
            if (actor["focal"] != (actor["id"] == focal) or actor["ego"] != (actor["id"] == ego)
                    or actor["type"] != descriptor["actor_type"] or descriptor["city"] != scene["city"]
                    or descriptor["role"] != ("focal" if actor["focal"] else "other")
                    or descriptor["history_bin"] != (">=80%" if sum(valid) >= 40 else "<80%")):
                raise ValueError("Recorded actor metadata differs from its analysis descriptors")
            complete = all(point is not None for point in actor["reference_future"])
            if any(values["eligible"] != (values["requested"] and complete) for values in descriptor["metrics"].values()):
                raise ValueError("Recorded future masks differ from analysis eligibility")
        selection = scene["selection"]
        dynamic = sum(actor["type"] in TYPES[:5] for actor in actors)
        if (selection != record["source_selection"] or any(type(selection[key]) is not int or selection[key] < 0
                for key in ("retained_dynamic", "total_dynamic", "retained_scored", "total_scored"))
                or selection["retained_dynamic"] != dynamic or selection["total_dynamic"] < dynamic
                or not selection["retained_scored"] <= min(count, selection["total_scored"])):
            raise ValueError("Recorded retained-target selection counts differ")
        frame = scene["coordinate_frame"]
        if (frame["name"] != "focal_local" or frame["units"] != "m" or not _valid_point(frame["origin"])
                or type(frame["rotation"]) not in (float, int) or not math.isfinite(frame["rotation"])):
            raise ValueError("Invalid recorded coordinate frame")
        for lane in scene["map"]["lanes"]:
            for key in ("points", "left_boundary", "right_boundary"):
                points = lane[key]
                if not isinstance(points, list) or not points or not all(_valid_point(point) for point in points):
                    raise ValueError("Invalid recorded lane geometry")
        for polygons in (scene["map"]["crosswalks"], scene["map"]["drivable_areas"]):
            for polygon in polygons:
                if not isinstance(polygon, list) or not polygon or not all(_valid_point(point) for point in polygon):
                    raise ValueError("Invalid recorded map polygon")
        if len(case["predictions"]) != len(metadata["model_labels"]):
            raise ValueError("Recorded case model count differs")
        for label, prediction in zip(metadata["model_labels"], case["predictions"]):
            variant = label.rsplit("-", 1)[0] if label.startswith(("control-", "directional-")) else label
            if (prediction["scenario_id"] != sid or prediction["actor_ids"] != ids or prediction["analysis_label"] != variant
                    or prediction["recorded"] is not True or prediction["development_only"] is not True
                    or prediction["scenario"] != scene or prediction["coordinate_frame"] != frame
                    or prediction["calibration_id"] is not None or prediction["stress"] is not None
                    or prediction["analysis_provenance"] != metadata["provenance"]):
                raise ValueError("Recorded prediction actor/model alignment differs")
            if prediction.get("analysis_seed") != (int(label.rsplit("-", 1)[1]) if variant in ("control", "directional") else prediction.get("analysis_seed")):
                raise ValueError("Recorded prediction seed alignment differs")
            seed = prediction["analysis_seed"]
            if (label == "constant_velocity" and seed is not None) or (label != "constant_velocity" and (type(seed) is not int or seed not in (42, 43, 44))):
                raise ValueError("Recorded prediction has an invalid training seed")
            if type(prediction["elapsed_ms"]) not in (int, float) or not math.isfinite(prediction["elapsed_ms"]) or prediction["elapsed_ms"] < 0:
                raise ValueError("Recorded inference duration is invalid")
            expected_id = metadata["provenance"]["checkpoint_sha256"].get(label, "constant_velocity_v1")
            if (prediction["model"]["id"] != expected_id
                    or prediction["model"]["kind"] != ("constant_velocity" if label == "constant_velocity" else "trained")
                    or prediction["calibration_applicability"] != "unknown"
                    or prediction["risk"]["status"] != "experimental_recorded_uncalibrated"
                    or prediction["risk"]["applicability"] != "development_only"):
                raise ValueError("Recorded model/calibration provenance differs")
            for key in ("trajectories", "scales"):
                values = np.asarray(prediction[key])
                if values.dtype.kind not in "fi" or values.shape != (6, count, 60, 2) or not np.isfinite(values).all():
                    raise ValueError("Recorded prediction must contain six aligned actor worlds")
                if key == "scales" and ((values <= 0).any() or (values > 20).any()):
                    raise ValueError("Recorded Gaussian scales exceed model bounds")
            probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
            if (probabilities.shape != (6,) or any(type(value) not in (int, float) for value in prediction["probabilities"])
                    or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
                raise ValueError("Recorded world probabilities are invalid")
            if prediction["risk"]["calibrated"] is not None or len(prediction["risk"]["raw"]) != count:
                raise ValueError("Analysis records must not claim calibrated risk")
            risk = np.asarray(prediction["risk"]["raw"], dtype=np.float64)
            if (risk.shape != (count,) or any(type(value) not in (int, float) for value in prediction["risk"]["raw"])
                    or not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any()):
                raise ValueError("Recorded raw risk must be one scalar between zero and one per actor")
        _finite_json(case)
    except (KeyError, TypeError, OverflowError, AttributeError, IndexError) as error:
        raise ValueError("Recorded case schema is incomplete") from error


def validate_analysis(directory):
    """Validate a directory before copying; no pickle, weights or external paths."""
    root = Path(directory)
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()) or not root.is_dir():
        raise ValueError("Choose a real analysis directory, not a link")
    actual, total = set(), 0
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(folder) / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("Analysis folders cannot contain filesystem links")
        for name in files:
            path = Path(folder) / name
            actual.add(path.relative_to(root).as_posix())
            total += path.stat().st_size
            if total > MAX_FOLDER_BYTES or len(actual) > MAX_CASES + 8:
                raise ValueError("Analysis folder exceeds the 64 MiB or file-count limit")
    metadata = _bounded_json(root / "analysis.json", 2*1024**2)
    try:
        if metadata["format_version"] != 1 or metadata["split"] != "validation" or metadata["development_only"] is not True or not re.fullmatch(r"[0-9a-f]{24}", metadata["id"]):
            raise ValueError("Unsupported analysis format or evidence scope")
        if not isinstance(metadata["title"], str) or not 1 <= len(metadata["title"]) <= 200:
            raise ValueError("Invalid analysis title")
        labels = metadata["model_labels"]
        expected_labels = ([f"{variant}-{seed}" for variant in ("control", "directional") for seed in (42, 43, 44)] + ["constant_velocity"]
            if metadata["provenance"].get("kind") == "direction-study" else ["reference", "constant_velocity"])
        if labels != expected_labels:
            raise ValueError("Invalid analysis model labels")
        if metadata["id"] != hashlib.sha256(_json_bytes({"provenance": metadata["provenance"], "format_version": 1})).hexdigest()[:24]:
            raise ValueError("Analysis identifier differs from its provenance")
        if len(metadata["cases"]) > MAX_CASES or len({case["scenario_id"] for case in metadata["cases"]}) != len(metadata["cases"]):
            raise ValueError("Invalid or repeated analysis replay cases")
        files = metadata["files"]
        expected = {"records.json.gz", "summary.json", "summary.csv", "failures.png"} | {f"cases/{case['scenario_id']}.json.gz" for case in metadata["cases"]}
        if set(files) != expected or actual != expected | {"analysis.json"}:
            raise ValueError("Analysis files differ from the manifest; executable/model files are forbidden")
        for name, info in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Analysis path escapes its folder")
            path = root.joinpath(*relative.parts)
            if info["bytes"] != path.stat().st_size or info["sha256"] != manifest_hash(path):
                raise ValueError("Analysis file hash/size mismatch")
        records = _bounded_json(root / "records.json.gz", MAX_RECORD_BYTES, True)
        ids = metadata["provenance"]["selection_ids"]
        if not 1 <= len(records) <= 2000 or len(ids) != len(set(ids)) or [record["scenario_id"] for record in records] != ids:
            raise ValueError("Analysis records differ from complete selection membership")
        if metadata["slice_definitions"] != SLICE_DEFINITIONS:
            raise ValueError("Analysis slice definitions differ from the frozen protocol")
        if (metadata["provenance"].get("labels_used") != "selection_only"
                or metadata["provenance"].get("calibration_applied") is not False
                or metadata["provenance"].get("inference_device") != "cpu"):
            raise ValueError("Analysis provenance must describe uncalibrated CPU selection-set predictions")
        checkpoints = metadata["provenance"]["checkpoint_sha256"]
        if (not isinstance(checkpoints, dict) or set(checkpoints) != set(labels) - {"constant_velocity"}
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in checkpoints.values())):
            raise ValueError("Analysis checkpoint provenance is incomplete")
        for record in records:
            if not _OPAQUE.fullmatch(record["scenario_id"]) or set(record["metrics"]) != set(labels) or not 1 <= len(record["actors"]) <= 32:
                raise ValueError("Invalid scenario or model alignment in analysis statistics")
            if not isinstance(record["city"], str) or not 0 < len(record["city"]) <= 160:
                raise ValueError("Invalid city in analysis statistics")
            actor_ids = [actor["actor_id"] for actor in record["actors"]]
            if len(actor_ids) != len(set(actor_ids)):
                raise ValueError("Repeated actor in analysis statistics")
            source = record["source_selection"]
            if (not isinstance(source, dict) or any(type(source[key]) is not int or source[key] < 0
                    for key in ("retained_dynamic", "total_dynamic", "retained_scored", "total_scored"))
                    or source["retained_dynamic"] != sum(actor["actor_type"] in TYPES[:5] for actor in record["actors"])
                    or source["total_dynamic"] < source["retained_dynamic"]
                    or not source["retained_scored"] <= min(len(record["actors"]), source["total_scored"])):
                raise ValueError("Invalid retained-target coverage counts")
            for metric in record["metrics"].values():
                if not isinstance(metric, dict):
                    raise ValueError("Invalid scene metric record")
                for key in ("jointADE", "jointFDE", "focal_minADE", "focal_minFDE", "focal_miss_rate", "NLL", "footprint_collision_proxy"):
                    value = metric[key]
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                        raise ValueError("Invalid scene metric value")
            for actor in record["actors"]:
                if (any(not isinstance(actor[key], str) or not 0 < len(actor[key]) <= 160 for key in (*FILTERS, "actor_id"))
                        or actor["city"] != record["city"]
                        or set(actor["metrics"]) != set(labels) or actor["actor_type"] not in TYPES or actor["role"] not in ("focal", "other")
                        or actor["speed_bin"] not in ("unknown", "<1", "1-<5", "5-<15", ">=15")
                        or actor["turn_bin"] not in ("unknown", "left", "right", "straight")
                        or actor["density_bin"] not in ("unknown", "0", "1-3", "4+") or actor["history_bin"] not in ("<80%", ">=80%")):
                    raise ValueError("Invalid analysis actor descriptors")
                for values in actor["metrics"].values():
                    if any(type(values[key]) is not bool for key in ("eligible", "requested", "excluded_incomplete_future")):
                        raise ValueError("Invalid analysis actor eligibility")
                    for key in ("minADE", "minFDE", "miss_rate"):
                        value = values[key]
                        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                            raise ValueError("Invalid analysis displacement metric")
                    if values["eligible"] != all(values[key] is not None for key in ("minADE", "minFDE", "miss_rate")):
                        raise ValueError("Incomplete futures cannot have primary displacement metrics")
                    if (values["eligible"] and not values["requested"]) or values["miss_rate"] not in (None, 0, 1):
                        raise ValueError("Invalid scored-actor eligibility or miss event")
                    if values["excluded_incomplete_future"] != (values["requested"] and not values["eligible"]):
                        raise ValueError("Inconsistent incomplete-future exclusions")
        _finite_json(records)
        _finite_json(metadata)
        lookup = {record["scenario_id"]: record for record in records}
        for item in metadata["cases"]:
            sid = item["scenario_id"]
            if sid not in ids or not _OPAQUE.fullmatch(sid):
                raise ValueError("Recorded case is outside selection membership")
            if (type(item["actor_count"]) is not int or item["actor_count"] != len(lookup[sid]["actors"])
                    or item["focal_type"] not in TYPES or item["city"] != lookup[sid]["city"]):
                raise ValueError("Recorded preview descriptors differ from full statistics")
            for key in ("focal_minFDE", "jointFDE", "delta_jointFDE"):
                value = item[key]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
                    raise ValueError("Invalid recorded preview metric")
            _validate_case(_bounded_json(root / "cases" / f"{sid}.json.gz", MAX_CASE_BYTES, True), metadata, sid, lookup[sid])
    except (KeyError, TypeError, OverflowError, AttributeError, IndexError) as error:
        raise ValueError("Analysis manifest/statistics schema is incomplete") from error
    return metadata, records


class AnalysisStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cached = None

    @staticmethod
    def _fingerprint(path):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Analysis folders cannot contain filesystem links")
        entries = []
        for folder, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                item = Path(folder) / name
                if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
                    raise ValueError("Analysis folders cannot contain filesystem links")
            for name in files:
                item = Path(folder) / name
                info = item.stat()
                entries.append((item.relative_to(path).as_posix(), info.st_size, info.st_mtime_ns))
                if len(entries) > MAX_CASES + 8:
                    raise ValueError("Analysis contains too many files")
        return tuple(sorted(entries))

    def _load(self, path):
        fingerprint = self._fingerprint(path)
        if self.cached is not None and self.cached[:2] == (path, fingerprint):
            return self.cached[2:]
        metadata, records = validate_analysis(path)
        # Retain only the active report: repeated filter changes should not inflate
        # and parse every recorded replay, nor retain several full studies in RAM.
        self.cached = (path, fingerprint, metadata, records)
        return metadata, records

    def _path(self, analysis_id):
        if not isinstance(analysis_id, str) or not re.fullmatch(r"[0-9a-f]{24}", analysis_id):
            raise KeyError("Analysis not found")
        path = self.root / analysis_id
        if not path.is_dir() or path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise KeyError("Analysis not found")
        return path

    @staticmethod
    def _summary(metadata, records):
        counts, _, _, _ = _cohort(records, metadata["model_labels"])
        counts["cases"] = len(metadata["cases"])
        return {"id": metadata["id"], "title": metadata["title"], "split": "validation", "development_only": True,
            "counts": counts, "model_labels": metadata["model_labels"]}

    def list(self):
        summaries = []
        with self.lock:
            for path in sorted(self.root.iterdir()):
                if not re.fullmatch(r"[0-9a-f]{24}", path.name):
                    continue
                try:
                    metadata, records = self._load(path)
                    summaries.append(self._summary(metadata, records))
                except (ValueError, OSError):
                    continue
        return summaries

    def install(self, source):
        with self.lock:
            metadata, records = validate_analysis(source)
            source = Path(source).resolve()
            target = self.root / metadata["id"]
            if target.exists():
                installed, installed_records = validate_analysis(target)
                if manifest_hash(target / "analysis.json") != manifest_hash(source / "analysis.json"):
                    raise ValueError("An analysis with this provenance already exists with different contents")
                return self._summary(installed, installed_records)
            total = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
            if shutil.disk_usage(self.root).free - total < 8 * 1024**3:
                raise ValueError("Analysis import paused: preserve at least 8 GiB free on the destination drive")
            stage = Path(tempfile.mkdtemp(prefix=".analysis-import-", dir=self.root))
            try:
                for name in ["analysis.json", *metadata["files"]]:
                    src = source.joinpath(*PurePosixPath(name).parts)
                    if src.is_symlink() or not src.is_file():
                        raise ValueError("Analysis source changed during import")
                    dst = stage.joinpath(*PurePosixPath(name).parts)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst, follow_symlinks=False)
                copied, _ = validate_analysis(stage)
                if copied != metadata:
                    raise ValueError("Analysis source changed during import")
                os.replace(stage, target)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            return self._summary(metadata, records)

    def detail(self, analysis_id, filters=None):
        with self.lock:
            metadata, records = self._load(self._path(analysis_id))
            return _detail(metadata, records, filters)

    def case(self, analysis_id, scenario_id):
        if not isinstance(scenario_id, str) or not _OPAQUE.fullmatch(scenario_id):
            raise KeyError("Recorded case not found")
        with self.lock:
            path = self._path(analysis_id)
            metadata, _ = self._load(path)
            if scenario_id not in {item["scenario_id"] for item in metadata["cases"]}:
                raise KeyError("Recorded case not found")
            return _bounded_json(path / "cases" / f"{scenario_id}.json.gz", MAX_CASE_BYTES, True)
