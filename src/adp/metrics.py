"""Offline forecast metrics and calibrated *proximity* risk evaluation.

Risk labels concern retained dynamic actors and 3 m centroid proximity. Footprint
overlap is a separate class-size proxy; neither quantity is a crash probability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch


FOOTPRINT_DIMENSIONS = {
    0: (4.5, 1.8), 1: (0.6, 0.6), 2: (2.0, 0.8),
    3: (1.8, 0.6), 4: (12.0, 2.5),
}
FOOTPRINT_NAMES = {0: "vehicle", 1: "pedestrian", 2: "motorcyclist", 3: "cyclist", 4: "bus"}


def risk_labels(future, future_mask, dynamic_mask):
    """Return float labels [B,A] and conservative eligible mask [B,A].

    A label is positive when two valid retained dynamic centroids are strictly
    less than 3 m apart at a shared future timestamp. Eligibility additionally
    requires complete futures for *all* retained dynamic actors and at least two
    such actors. Padding does not affect eligibility. Incomplete positives are
    retained as labels but must not enter training/calibration without the mask.
    """
    if future.ndim != 4 or future.shape[-1] != 2:
        raise ValueError("future must have shape [B,A,T,2]")
    valid = future_mask.bool() & torch.isfinite(future).all(dim=-1)
    dynamic = dynamic_mask.bool()
    safe = torch.where(valid[..., None], future, torch.zeros_like(future))
    distances = (safe[:, :, None] - safe[:, None, :]).square().sum(dim=-1)
    pairs = dynamic[:, :, None, None] & dynamic[:, None, :, None]
    pairs = pairs & valid[:, :, None] & valid[:, None, :]
    self_pair = torch.eye(future.shape[1], device=future.device, dtype=torch.bool)
    pairs = pairs & ~self_pair[None, :, :, None]
    labels = ((distances < 9.0) & pairs).any(dim=-1).any(dim=-1).to(future.dtype)
    complete = (valid.all(dim=-1) | ~dynamic).all(dim=-1)
    eligible = dynamic & complete[:, None] & (dynamic.sum(dim=-1) >= 2)[:, None]
    return labels, eligible


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    exp = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -logits))


def reliability(probabilities, labels, bins=10):
    """Equal-width reliability bins; include p=1 in the final bin."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probabilities.size != labels.size or bins < 1:
        raise ValueError("probabilities and labels must match, and bins must be positive")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    result, ece = [], 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        confidence = float(probabilities[selected].mean()) if count else None
        frequency = float(labels[selected].mean()) if count else None
        if count:
            ece += count * abs(confidence - frequency)
        result.append({"lower": index / bins, "upper": (index + 1) / bins,
                       "count": count, "mean_probability": confidence,
                       "event_frequency": frequency})
    return {"ece": ece / probabilities.size if probabilities.size else None,
            "count": int(probabilities.size), "bins": result}


def average_precision(probabilities, labels):
    """Step-integrated precision-recall area with tied scores grouped together."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = labels.sum()
    if not labels.size or not positives:
        return None
    order = np.argsort(-probabilities, kind="stable")
    scores, truth = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), truth.size - 1]
    cumulative = np.cumsum(truth)[ends]
    precision = cumulative / (ends + 1)
    increments = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * increments))


def binary_scores(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not labels.size:
        return {"brier": None, "nll": None, "auprc": None,
                "reliability": reliability([], [])}
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return {"brier": float(np.square(probabilities - labels).mean()),
            "nll": float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()),
            "auprc": average_precision(probabilities, labels),
            "reliability": reliability(probabilities, labels)}


def oriented_boxes_overlap(center_a, heading_a, dimensions_a,
                           center_b, heading_b, dimensions_b):
    """Separating-axis rectangle intersection, broadcasting leading dimensions.

    Touching edges count as overlap. Dimensions are (length, width) in metres.
    """
    a = np.stack((np.cos(heading_a), np.sin(heading_a)), axis=-1)
    b = np.stack((np.cos(heading_b), np.sin(heading_b)), axis=-1)
    av = np.stack((-a[..., 1], a[..., 0]), axis=-1)
    bv = np.stack((-b[..., 1], b[..., 0]), axis=-1)
    delta = np.asarray(center_b) - np.asarray(center_a)
    overlap = np.ones(np.broadcast_shapes(delta.shape[:-1], a.shape[:-1], b.shape[:-1]), dtype=bool)
    for axis in (a, av, b, bv):
        separation = np.abs(np.sum(delta * axis, axis=-1))
        radius_a = (dimensions_a[0] * np.abs(np.sum(a * axis, axis=-1))
                    + dimensions_a[1] * np.abs(np.sum(av * axis, axis=-1))) / 2
        radius_b = (dimensions_b[0] * np.abs(np.sum(b * axis, axis=-1))
                    + dimensions_b[1] * np.abs(np.sum(bv * axis, axis=-1))) / 2
        overlap &= separation <= radius_a + radius_b + 1e-9
    return overlap


def footprint_collision_proxy(trajectories, actor_types, dynamic_mask=None,
                              initial_positions=None, initial_headings=None):
    """Return per-mode actor involvement and any-overlap proxies [K].

    Headings follow the path tangent, preserving the previous heading while
    stationary. Checks sample timestamps (10 Hz), not swept volumes.
    """
    positions = _numpy(trajectories).astype(np.float64)
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("trajectories must have shape [K,A,T,2]")
    k, actors, steps, _ = positions.shape
    types = _numpy(actor_types).astype(int)
    active = np.isin(types, list(FOOTPRINT_DIMENSIONS))
    if dynamic_mask is not None:
        active &= _numpy(dynamic_mask).astype(bool)
    if initial_positions is None:
        initial_positions = positions[0, :, 0]
    previous = np.broadcast_to(_numpy(initial_positions), (k, actors, 2)).copy()
    heading = np.broadcast_to(np.zeros(actors) if initial_headings is None
                              else _numpy(initial_headings), (k, actors)).copy()
    headings = np.zeros((k, actors, steps))
    for t in range(steps):
        change = positions[:, :, t] - previous
        moving = np.linalg.norm(change, axis=-1) > 1e-4
        heading = np.where(moving, np.arctan2(change[..., 1], change[..., 0]), heading)
        headings[:, :, t] = heading
        previous = positions[:, :, t]
    collisions = np.zeros((k, actors), dtype=bool)
    indices = np.flatnonzero(active)
    for offset, actor_a in enumerate(indices):
        for actor_b in indices[offset + 1:]:
            hit = oriented_boxes_overlap(
                positions[:, actor_a], headings[:, actor_a], FOOTPRINT_DIMENSIONS[int(types[actor_a])],
                positions[:, actor_b], headings[:, actor_b], FOOTPRINT_DIMENSIONS[int(types[actor_b])],
            ).any(axis=-1)
            collisions[:, actor_a] |= hit
            collisions[:, actor_b] |= hit
    denominator = int(active.sum())
    return {"actor_rate": collisions.sum(axis=-1) / denominator if denominator else np.zeros(k),
            "any_overlap": collisions.any(axis=-1)}


def _calibrated_logits(logits, calibration, kind):
    if calibration is None:
        return logits
    method = getattr(calibration, f"calibrate_{kind}_logits", None)
    if callable(method):
        return _numpy(method(torch.as_tensor(logits))).astype(np.float64)
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must be a mapping or expose calibrate_*_logits")
    if kind == "risk":
        nested = calibration.get("risk", calibration)
        slope = nested.get("risk_slope", nested.get("slope", 1.0))
        intercept = nested.get("risk_intercept", nested.get("intercept", nested.get("risk_bias", 0.0)))
        temperature = nested.get("risk_temperature", 1.0)
        if temperature <= 0:
            raise ValueError("risk_temperature must be positive")
        return slope * logits / temperature + intercept
    temperature = calibration.get("mode_temperature", 1.0)
    if temperature <= 0:
        raise ValueError("mode_temperature must be positive")
    return logits / temperature


def evaluate_predictions(outputs, batch, calibration=None):
    """Build one JSON-safe record per scene; no predictions are paired across modes.

    Displacement metrics use complete scored dynamic futures. NLL is joint
    mixture NLL in nats per valid coordinate; mode ECE uses nearest *joint FDE*
    as an oracle label, which is distinct from event-probability calibration.
    """
    means, scales = _numpy(outputs["means"]), _numpy(outputs["scales"])
    logits = _numpy(outputs["logits"])
    risk_logits = _numpy(outputs["risk_logits"])
    future, mask = _numpy(batch["future"]), _numpy(batch["future_mask"]).astype(bool)
    dynamic = _numpy(batch["dynamic_mask"]).astype(bool)
    actor_mask = _numpy(batch.get("actor_mask", dynamic)).astype(bool)
    dynamic &= actor_mask
    valid = mask & np.isfinite(future).all(axis=-1)
    complete = valid.all(axis=-1)
    requested_scored = _numpy(batch.get("scored_mask", dynamic)).astype(bool) & dynamic
    scored = requested_scored & complete
    labels, eligible = risk_labels(torch.as_tensor(future), torch.as_tensor(mask), torch.as_tensor(dynamic))
    labels, eligible = _numpy(labels), _numpy(eligible)
    probabilities = _softmax(_calibrated_logits(logits, calibration, "mode"))
    raw_probabilities = _softmax(logits)
    raw_risk = _sigmoid(risk_logits)
    calibrated_risk = _sigmoid(_calibrated_logits(risk_logits, calibration, "risk"))
    types = _numpy(batch.get("actor_types", np.zeros(dynamic.shape, dtype=int)))
    histories = _numpy(batch["history"]) if "history" in batch else None
    history_mask = _numpy(batch["history_mask"]).astype(bool) if "history_mask" in batch else None
    ids = batch.get("scenario_ids", [str(index) for index in range(len(future))])
    selections = batch.get("selections", [{} for _ in range(len(future))])
    records = []
    for index in range(len(future)):
        target = scored[index]
        record = {"scenario_id": str(ids[index]), "minADE": [], "minFDE": [],
                  "miss_rate": [], "brierFDE": [], "jointADE": None, "jointFDE": None,
                  "focal_minADE": None, "focal_minFDE": None, "focal_miss_rate": None,
                  "NLL": None, "mode_confidence": None, "mode_correct": None,
                  "mode_confidence_raw": None, "mode_correct_raw": None,
                  "risk_labels": labels[index, eligible[index]].astype(int).tolist(),
                  "risk_raw_probabilities": raw_risk[index, eligible[index]].tolist(),
                  "risk_calibrated_probabilities": calibrated_risk[index, eligible[index]].tolist()}
        dynamic_count = int(dynamic[index].sum())
        exclusion = ("fewer_than_two_dynamic_actors" if dynamic_count < 2 else
                     "incomplete_dynamic_future" if not eligible[index].any() else None)
        record["coverage"] = {
            "retained_actors": int(actor_mask[index].sum()), "retained_dynamic": dynamic_count,
            "retained_scored_dynamic": int(requested_scored[index].sum()),
            "evaluated_scored_dynamic": int(target.sum()),
            "scored_excluded_incomplete_future": int((requested_scored[index] & ~complete[index]).sum()),
            "risk_eligible_dynamic": int(eligible[index].sum()),
            "risk_excluded_dynamic": int((dynamic[index] & ~eligible[index]).sum()),
            "focal_eligible": int(bool(target[0])) if target.size else 0,
            "risk_scene_exclusion": exclusion,
        }
        if selections[index]:
            record["source_selection"] = {key: int(selections[index][key]) for key in
                                          ("retained_dynamic", "total_dynamic", "retained_scored", "total_scored")
                                          if key in selections[index]}
        if target.any():
            predicted = means[index][:, target]
            truth = future[index, target]
            distance = np.linalg.norm(predicted - truth[None], axis=-1)
            ade, fde = distance.mean(axis=-1), distance[:, :, -1]
            best = fde.argmin(axis=0)
            best_fde = fde[best, np.arange(best.size)]
            record.update({"minADE": ade.min(axis=0).tolist(), "minFDE": best_fde.tolist(),
                           "miss_rate": (best_fde > 2).astype(float).tolist(),
                           "brierFDE": (best_fde + (1 - probabilities[index, best]) ** 2).tolist(),
                           "jointADE": float(ade.mean(axis=-1).min()),
                           "jointFDE": float(fde.mean(axis=-1).min())})
            if target[0]:
                focal_fde = float(fde[:, 0].min())
                record.update({"focal_minADE": float(ade[:, 0].min()), "focal_minFDE": focal_fde,
                               "focal_miss_rate": float(focal_fde > 2)})
            chosen = int(fde.mean(axis=-1).argmin())
            record["mode_confidence"] = float(probabilities[index].max())
            record["mode_correct"] = int(probabilities[index].argmax() == chosen)
            record["mode_confidence_raw"] = float(raw_probabilities[index].max())
            record["mode_correct_raw"] = int(raw_probabilities[index].argmax() == chosen)
            sigma = np.clip(scales[index][:, target], 1e-3, None)
            log_density = -0.5 * (((truth[None] - predicted) / sigma) ** 2
                                  + 2 * np.log(sigma) + math.log(2 * math.pi))
            component = log_density.sum(axis=(1, 2, 3)) + np.log(np.clip(probabilities[index], 1e-300, 1))
            peak = component.max()
            record["NLL"] = float(-(peak + np.log(np.exp(component - peak).sum())) / truth.size)
        initial_positions = initial_headings = None
        if histories is not None:
            indices = np.full(histories.shape[1], histories.shape[2] - 1)
            if history_mask is not None:
                indices = np.maximum(np.where(history_mask[index], np.arange(histories.shape[2]), -1).max(axis=-1), 0)
            last = histories[index, np.arange(histories.shape[1]), indices]
            initial_positions, initial_headings = last[:, :2], np.arctan2(last[:, 4], last[:, 5])
        overlaps = footprint_collision_proxy(means[index], types[index], dynamic[index], initial_positions, initial_headings)
        record["footprint_collision_proxy"] = float(np.dot(probabilities[index], overlaps["any_overlap"]))
        record["footprint_actor_collision_proxy"] = float(np.dot(probabilities[index], overlaps["actor_rate"]))
        records.append(record)
    return records


def _aggregate(records):
    result = {"scenario_count": len(records)}
    keys = ("minADE", "minFDE", "miss_rate", "brierFDE", "jointADE", "jointFDE", "NLL",
            "focal_minADE", "focal_minFDE", "focal_miss_rate",
            "footprint_collision_proxy", "footprint_actor_collision_proxy")
    for key in keys:
        values = []
        for record in records:
            value = record.get(key)
            values.extend(value if isinstance(value, list) else [] if value is None else [value])
        result[key] = float(np.mean(values)) if values else None
    truth = [value for record in records for value in record.get("risk_labels", [])]
    raw = [value for record in records for value in record.get("risk_raw_probabilities", [])]
    calibrated = [value for record in records for value in record.get("risk_calibrated_probabilities", [])]
    positives = int(sum(truth))
    result["risk"] = {"raw": binary_scores(raw, truth), "calibrated": binary_scores(calibrated, truth),
                      "eligible_count": len(truth), "positive_count": positives,
                      "prevalence": positives / len(truth) if truth else None}
    mode = [(record["mode_confidence"], record["mode_correct"]) for record in records
            if record.get("mode_confidence") is not None]
    result["mode_reliability"] = reliability([pair[0] for pair in mode], [pair[1] for pair in mode])
    result["modeECE"] = result["mode_reliability"]["ece"]
    raw_mode = [(record["mode_confidence_raw"], record["mode_correct_raw"]) for record in records
                if record.get("mode_confidence_raw") is not None]
    result["mode_reliability_raw"] = reliability([pair[0] for pair in raw_mode], [pair[1] for pair in raw_mode])
    result["modeECE_raw"] = result["mode_reliability_raw"]["ece"]
    counts = ("retained_actors", "retained_dynamic", "retained_scored_dynamic", "evaluated_scored_dynamic",
              "scored_excluded_incomplete_future", "risk_eligible_dynamic", "risk_excluded_dynamic", "focal_eligible")
    coverage = {key: sum(record.get("coverage", {}).get(key, 0) for record in records) for key in counts}
    coverage["scored_evaluation_fraction"] = (coverage["evaluated_scored_dynamic"] / coverage["retained_scored_dynamic"]
                                               if coverage["retained_scored_dynamic"] else None)
    coverage["risk_eligibility_fraction"] = (coverage["risk_eligible_dynamic"] / coverage["retained_dynamic"]
                                             if coverage["retained_dynamic"] else None)
    coverage["focal_evaluation_fraction"] = coverage["focal_eligible"] / len(records) if records else None
    coverage["risk_exclusion_scenarios"] = {
        reason: sum(record.get("coverage", {}).get("risk_scene_exclusion") == reason for record in records)
        for reason in ("fewer_than_two_dynamic_actors", "incomplete_dynamic_future")}
    sources = [record["source_selection"] for record in records if "source_selection" in record]
    coverage["source_metadata_scenarios"] = len(sources)
    for population in ("dynamic", "scored"):
        total = sum(source.get(f"total_{population}", 0) for source in sources)
        retained = sum(source.get(f"retained_{population}", 0) for source in sources)
        coverage[f"source_total_{population}"] = total if sources else None
        coverage[f"source_retained_{population}"] = retained if sources else None
        coverage[f"{population}_retention_fraction"] = retained / total if total else None
    result["coverage"] = coverage
    return result


def summarize_records(records, bootstrap=200, seed=42, calibration=None):
    """Aggregate actors, resampling whole scenarios for percentile 95% intervals."""
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    # Multiple records for one scenario stay in the same bootstrap cluster.
    records = list(records)
    summary = _aggregate(records)
    risk = summary["risk"]
    risk_fit = calibration.get("risk", {}) if calibration is not None else {}
    calibration_positives = risk_fit.get("positives")
    evaluation_sufficient = risk["positive_count"] >= 50
    calibration_sufficient = calibration_positives is not None and calibration_positives >= 50
    risk["evaluation_status"] = "sufficient" if evaluation_sufficient else "insufficient_positive_events"
    risk["calibration"] = {
        "applied": calibration is not None, "fitted": bool(risk_fit.get("fitted", False)),
        "eligible_count": risk_fit.get("count"), "positive_count": calibration_positives,
        "status": ("not_applied" if calibration is None else
                   "insufficient_positive_events" if not calibration_sufficient else
                   "fitted" if risk_fit.get("fitted") else "not_fitted"),
    }
    risk["status"] = ("insufficient_positive_events" if not evaluation_sufficient else
                      "uncalibrated" if calibration is None else
                      "insufficient_positive_events" if not calibration_sufficient else
                      "sufficient" if risk_fit.get("fitted") else "not_fitted")
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record.get("scenario_id", str(index)), []).append(record)
    clusters = list(groups.values())
    keys = [key for key, value in summary.items() if isinstance(value, (float, int)) and key != "scenario_count"]
    estimates = {key: [] for key in keys}
    for phase in ("raw", "calibrated"):
        for metric in ("brier", "nll", "auprc"):
            estimates[f"risk.{phase}.{metric}"] = []
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap if clusters else 0):
        sampled = [record for index in rng.integers(len(clusters), size=len(clusters)) for record in clusters[index]]
        result = _aggregate(sampled)
        for key, samples in estimates.items():
            value = result
            for part in key.split("."):
                value = value[part]
            if value is not None and np.isfinite(value):
                samples.append(value)
    summary["confidence_intervals"] = {
        key: np.percentile(values, [2.5, 97.5]).tolist() if values else None
        for key, values in estimates.items()
    }
    summary["bootstrap"] = {"replicates": bootstrap, "seed": seed, "unit": "scenario", "clusters": len(clusters)}
    summary["definitions"] = {
        "forecast_population": "retained scored dynamic actors with complete finite futures",
        "focal_actor": "actor index 0; scored only with complete finite future",
        "displacement_units": "metres", "miss_threshold_m": 2.0,
        "risk_event": "centroid distance strictly below 3 m to any other retained dynamic actor within 6 s",
        "risk_eligibility": "at least two retained dynamic actors; all retained dynamic futures complete and finite",
        "risk_minimum_positive_labels": 50,
        "mode_reliability_target": "nearest joint-FDE mode; this is not collision probability calibration",
        "NLL": "joint Gaussian-mixture negative log density, normalized per valid coordinate (nats)",
        "footprint_collision_proxy": {
            "dimensions_length_width_m": {FOOTPRINT_NAMES[key]: list(value) for key, value in FOOTPRINT_DIMENSIONS.items()},
            "dimension_source": "fixed class assumptions; not measured actor dimensions",
            "orientation": "path tangent; previous heading retained while stationary",
            "sampling_hz": 10, "geometry": "oriented rectangle intersection at sampled timestamps",
            "aggregation": "scene-mode-probability weighted proxy; uncalibrated; retained dynamic actors only",
        },
    }
    return summary
