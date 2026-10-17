"""Seeded observation corruptions and explicitly nonreactive counterfactuals."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


class StressEligibilityError(ValueError):
    """The selected actor/map cannot support the requested counterfactual."""


def _features(history, mask):
    """Recompute motion from observed positions without looking at future data."""
    for actor in range(len(history)):
        indices = np.flatnonzero(mask[actor])
        for offset, time in enumerate(indices):
            if offset:
                previous = indices[offset - 1]
                history[actor, time, 2:4] = (history[actor, time, :2] - history[actor, previous, :2]) / ((time - previous) * 0.1)
            velocity = history[actor, time, 2:4]
            if np.linalg.norm(velocity) > 1e-4:
                angle = np.arctan2(velocity[1], velocity[0])
                history[actor, time, 4:6] = (np.sin(angle), np.cos(angle))
        history[actor, ~mask[actor], :7] = 0
        history[actor, :, 6] = mask[actor]


def _active(scene):
    types = np.asarray(scene["actor_types"])
    active = np.asarray(scene.get("dynamic_mask", types < 5), dtype=bool).copy()
    active &= np.asarray(scene.get("actor_mask", np.ones(len(types))), dtype=bool)
    return active


def _counterfactual_actor(scene, actor_id):
    history, mask = scene["history"], scene["history_mask"]
    ids = list(scene["actor_ids"])
    if actor_id is not None and actor_id not in ids:
        raise StressEligibilityError(f"Actor {actor_id!r} is not retained in this scenario; choose one of {ids}.")
    candidates = [ids.index(actor_id)] if actor_id is not None else list(range(1, len(ids))) + [0]
    types = np.asarray(scene["actor_types"])
    active = _active(scene)
    for actor in candidates:
        if (active[actor] and types[actor] in (0, 2, 3, 4)
                and mask[actor, -11:].all()
                and np.linalg.norm(history[actor, -11, 2:4]) > 0.5):
            return actor
    raise StressEligibilityError("Choose a moving vehicle/cyclist with 11 consecutive final history states (speed > 0.5 m/s).")


def _polyline(lane):
    points = np.asarray(lane.get("centerline", lane.get("centerline_xy", [])), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise StressEligibilityError("Cut-in needs adjacent lane centerlines with at least two points.")
    points = points[:, :2]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    points = points[np.r_[True, lengths > 1e-6]]
    if len(points) < 2 or not np.isfinite(points).all():
        raise StressEligibilityError("Cut-in lane centerline is degenerate or non-finite.")
    return points


def _project(points, position):
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=-1)
    progress = np.clip(((position - points[:-1]) * segments).sum(axis=-1) / lengths ** 2, 0, 1)
    nearest = points[:-1] + progress[:, None] * segments
    index = np.linalg.norm(nearest - position, axis=-1).argmin()
    along = np.r_[0, np.cumsum(lengths)]
    return float(np.linalg.norm(nearest[index] - position)), float(along[index] + progress[index] * lengths[index]), segments[index] / lengths[index]


def _sample(points, distances):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    along = np.r_[0, np.cumsum(lengths)]
    if np.max(distances) > along[-1] + 1e-6:
        raise StressEligibilityError("Cut-in needs longer connected lane centerlines for the 7 s intervention; choose a slower actor or a scenario with stitched adjacent lanes.")
    return np.stack([np.interp(distances, along, points[:, coordinate]) for coordinate in range(2)], axis=-1)


def _extend_lane(lane, lanes, required_distance):
    """Join coincident, unambiguous forward segments; never extrapolate a road."""
    points = _polyline(lane)
    visited = {str(lane.get("id"))}
    for _ in range(64):
        if np.linalg.norm(np.diff(points, axis=0), axis=-1).sum() >= required_distance:
            return points
        direction = points[-1]-points[-2]
        direction /= np.linalg.norm(direction)
        candidates = []
        for candidate in lanes:
            if str(candidate.get("id")) in visited:
                continue
            following = _polyline(candidate)
            gap = np.linalg.norm(following[0]-points[-1])
            tangent = following[1]-following[0]
            alignment = np.dot(direction, tangent/np.linalg.norm(tangent))
            if gap <= .5 and alignment > .5:
                candidates.append((alignment, candidate, following))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .05):
            break
        _, selected, following = candidates[0]
        visited.add(str(selected.get("id")))
        points = np.concatenate([points, following[1:]])
    return points


def _cut_in(scene, actor, severity, times):
    lanes = scene.get("map_lanes", [])
    if isinstance(lanes, dict):
        lanes = list(lanes.values())
    usable = [lane for lane in lanes if lane.get("is_drivable", True)]
    if not usable:
        raise StressEligibilityError("Cut-in requires map_lanes with drivable centerlines and declared left/right neighbor IDs.")
    start = scene["history"][actor, -11]
    direction = start[2:4] / np.linalg.norm(start[2:4])
    candidates = []
    for lane in usable:
        points = _polyline(lane)
        distance, along, tangent = _project(points, start[:2])
        if np.dot(tangent, direction) > 0.5:
            candidates.append((distance, lane, points, along))
    if not candidates:
        raise StressEligibilityError("No lane aligned with this actor's observed direction supports a cut-in.")
    distance, source, source_points, source_s = min(candidates, key=lambda item: item[0])
    if distance > 4:
        raise StressEligibilityError("Actor is more than 4 m from a supported lane centerline; select an actor on a lane.")
    by_id = {str(lane.get("id", lane.get("lane_id"))): lane for lane in usable}
    adjacent = []
    for field in ("left_neighbor_id", "right_neighbor_id"):
        neighbor_id = source.get(field)
        neighbor = by_id.get(str(neighbor_id)) if neighbor_id is not None else None
        if neighbor is None:
            continue
        points = _polyline(neighbor)
        separation, along, tangent = _project(points, start[:2])
        if 1 <= separation <= 8 and np.dot(tangent, direction) > 0.5:
            adjacent.append((separation, str(neighbor_id), points, along))
    if not adjacent:
        raise StressEligibilityError("This actor has no declared nearby same-direction drivable neighbor; choose another actor/scenario.")
    _, neighbor_id, target_points, target_s = min(adjacent, key=lambda item: (item[0], item[1]))
    travel = np.linalg.norm(start[2:4]) * times
    source_points = _extend_lane(source, usable, source_s+travel[-1])
    target_points = _extend_lane(by_id[neighbor_id], usable, target_s+travel[-1])
    source_path = _sample(source_points, source_s + travel)
    source_path += start[:2] - source_path[0]
    target_path = _sample(target_points, target_s + travel)
    duration = 4 - 2 * severity
    progress = np.clip(times / duration, 0, 1)
    blend = 10 * progress ** 3 - 15 * progress ** 4 + 6 * progress ** 5
    return source_path * (1 - blend[:, None]) + target_path * blend[:, None], {
        "source_lane_id": str(source.get("id", source.get("lane_id"))),
        "target_lane_id": neighbor_id, "transition_seconds": duration}


def apply_stress(scene: dict, spec: dict) -> dict:
    """Return a deep copy; observation edits retain every logged future label.

    Severity is in [0,1]: noise sigma=severity metres, dropout probability=
    0.5*severity, braking deceleration=6*severity m/s². Counterfactuals replace
    the selected actor's final observed second and six-second future using
    only its initial observed state; all other agents replay their logged paths.
    A zero severity is always an exact trajectory/mask no-op.
    """
    kind = spec.get("kind")
    if kind not in ("position_noise", "dropout", "braking", "cut_in"):
        raise ValueError("kind must be position_noise, dropout, braking, or cut_in")
    severity = float(spec.get("severity", 0.35))
    if not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("severity must be a finite number in [0,1]")
    seed = int(spec.get("seed", 42))
    result = deepcopy(scene)
    history = np.asarray(result["history"])
    mask = np.asarray(result["history_mask"], dtype=bool).copy()
    if history.ndim != 3 or history.shape[1:] != (50, 8) or mask.shape != history.shape[:2]:
        raise ValueError("single-scene history must have shape [A,50,8] and history_mask [A,50]")
    if not np.issubdtype(history.dtype, np.floating):
        history = history.astype(np.float32)
    result["history"], result["history_mask"] = history, mask
    metadata = {"kind": kind, "severity": severity, "seed": seed,
                "label": "observation_corruption" if kind in ("position_noise", "dropout") else "counterfactual_nonreactive_replay",
                "other_agents_reactive": False, "actor_id": spec.get("actor_id")}
    result["stress"] = metadata
    if severity == 0:
        return result
    rng = np.random.default_rng(seed)
    if kind in ("position_noise", "dropout"):
        selected = _active(result)
        if spec.get("actor_id") is not None:
            if spec["actor_id"] not in result["actor_ids"]:
                raise StressEligibilityError("Requested observation-stress actor is not retained in this scenario.")
            selected[:] = False
            selected[result["actor_ids"].index(spec["actor_id"])] = True
        selected_states = mask & selected[:, None]
        if kind == "position_noise":
            history[:, :, :2] += rng.normal(0, severity, size=history[:, :, :2].shape) * selected_states[..., None]
            metadata["position_sigma_m"] = severity
        else:
            removed = (rng.random(mask.shape) < severity * 0.5) & selected_states
            # Keep each actor's last measured state so it remains observable.
            for actor in np.flatnonzero(selected):
                valid_indices = np.flatnonzero(mask[actor])
                if valid_indices.size:
                    removed[actor, valid_indices[-1]] = False
            mask[removed] = False
            metadata.update({"dropout_probability": severity * 0.5, "dropped_states": int(removed.sum())})
        # Recompute only affected actors; an explicitly selected stress leaves all
        # other inputs byte-for-byte unchanged.
        changed_history, changed_mask = history[selected].copy(), mask[selected].copy()
        _features(changed_history, changed_mask)
        history[selected] = changed_history
        return result
    actor = _counterfactual_actor(result, spec.get("actor_id"))
    metadata["actor_id"] = result["actor_ids"][actor]
    times = np.arange(71, dtype=np.float64) * 0.1
    start = history[actor, -11].copy()
    speed = np.linalg.norm(start[2:4])
    if kind == "braking":
        deceleration = 6 * severity
        moving_time = np.minimum(times, speed / deceleration)
        travel = speed * moving_time - 0.5 * deceleration * moving_time ** 2
        trajectory = start[:2] + travel[:, None] * (start[2:4] / speed)
        metadata.update({"deceleration_mps2": deceleration, "stop_seconds": float(speed / deceleration)})
    else:
        trajectory, details = _cut_in(result, actor, severity, times)
        metadata.update(details)
    history[actor, -11:, :2] = trajectory[:11]
    selected_history, selected_mask = history[actor:actor + 1].copy(), mask[actor:actor + 1].copy()
    _features(selected_history, selected_mask)
    history[actor, -11:] = selected_history[0, -11:]
    future = np.asarray(result["future"])
    if future.shape != (len(history), 60, 2):
        raise ValueError("single-scene future must have shape [A,60,2]")
    if not np.issubdtype(future.dtype, np.floating):
        future = future.astype(np.float32)
        result["future"] = future
    future[actor] = trajectory[11:]
    if "future_heading" in result:
        headings = np.asarray(result["future_heading"]).copy()
        previous_heading = float(np.arctan2(history[actor, -1, 4], history[actor, -1, 5]))
        changes = np.diff(trajectory[10:], axis=0)
        for step, change in enumerate(changes):
            if np.linalg.norm(change) > 1e-4:
                previous_heading = float(np.arctan2(change[1], change[0]))
            headings[actor, step] = previous_heading
        result["future_heading"] = headings
    # Simulated futures are fully specified for the intervened actor, while
    # original masks for every replayed actor remain unchanged.
    result["future_mask"] = np.asarray(result["future_mask"], dtype=bool).copy()
    result["future_mask"][actor] = True
    metadata["intervention_start_seconds"] = -1.0
    return result
