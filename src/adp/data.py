"""AV2 parquet/JSON ingestion. All actor and map selection uses history only."""
from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import random
import time
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TYPES = ["vehicle", "pedestrian", "motorcyclist", "cyclist", "bus", "static", "background", "construction", "riderless_bicycle", "unknown"]
SPLITS = ("train", "validation", "calibration", "evaluation")
S3_ROOT = "datasets/av2/motion-forecasting/"
S3_URL = "https://argoverse.s3.amazonaws.com/"
_connections = threading.local()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocessing_hash(cache_dir):
    path = Path(cache_dir)/"preprocessing.json"
    return manifest_hash(path) if path.exists() else None


def _get(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "argoverse.s3.amazonaws.com":
        raise ValueError("Only the official AV2 HTTPS bucket is supported")
    for attempt in range(4):
        try:
            connection = getattr(_connections, "connection", None)
            if connection is None:
                connection = http.client.HTTPSConnection(parsed.hostname, timeout=60)
                _connections.connection = connection
            connection.request("GET", parsed.path+("?"+parsed.query if parsed.query else ""))
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise OSError(f"AV2 download returned HTTP {response.status}: {parsed.path}")
            return payload
        except Exception:
            if getattr(_connections, "connection", None) is not None:
                _connections.connection.close()
            _connections.connection = None
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def _list(prefix, delimiter=None):
    continuation = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if continuation:
            query["continuation-token"] = continuation
        document = ET.fromstring(_get(S3_URL + "?" + urllib.parse.urlencode(query)))
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        if delimiter:
            yield from (e.text for e in document.findall("s:CommonPrefixes/s:Prefix", ns))
        else:
            yield from (e.text for e in document.findall("s:Contents/s:Key", ns))
        continuation = document.findtext("s:NextContinuationToken", namespaces=ns)
        if not continuation:
            break


def _select_scenarios(split, count, seed):
    # UUID-prefix block sampling avoids listing 200,000 folders for a small experiment.
    prefixes = [f"{i:02x}" for i in range(256)]
    random.Random(seed).shuffle(prefixes)
    ids = []
    for prefix in prefixes:
        ids.extend(p.rstrip("/").rsplit("/", 1)[-1] for p in _list(S3_ROOT + split + "/" + prefix, "/"))
        if len(ids) >= count:
            return sorted(ids, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())[:count]
    raise ValueError(f"Only {len(ids)} scenarios available for requested {count} in {split}")


def download(root="data/raw", train_count=5000, val_count=2000, seed=42, workers=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if train_count < 1 or val_count < 3:
        raise ValueError("Need at least one training and three validation scenarios")
    if path.exists():
        manifest = read_json(path)
        expected = {"train_count": train_count, "val_count": val_count, "seed": seed}
        if manifest["selection"] != expected:
            raise ValueError("Manifest is immutable. Use a different --raw-dir for a different subset.")
    else:
        train_ids = _select_scenarios("train", train_count, seed)
        val_ids = _select_scenarios("val", val_count, seed)
        quarter = max(1, val_count // 4)
        manifest = {"version": 1, "source": "Argoverse 2 Motion Forecasting", "license": "CC-BY-NC-SA-4.0",
                    "sampling": "Seeded UUID-prefix blocks, SHA256 order within selected blocks; not a uniform full-dataset sample",
                    "selection": {"train_count": train_count, "val_count": val_count, "seed": seed},
                    "splits": {"train": train_ids, "validation": val_ids[:quarter],
                               "calibration": val_ids[quarter:2*quarter], "evaluation": val_ids[2*quarter:]}}
        write_json(path, manifest)

    def fetch(item):
        split, scenario_id = item
        source_split = "train" if split == "train" else "val"
        directory = root / split / scenario_id
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{S3_ROOT}{source_split}/{scenario_id}/"
        selected = [prefix+f"scenario_{scenario_id}.parquet", prefix+f"log_map_archive_{scenario_id}.json"]
        for key in selected:
            target = directory / key.rsplit("/", 1)[-1]
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(_get(S3_URL + urllib.parse.quote(key, safe="/")))
                temporary.replace(target)
        return scenario_id

    items = [(split, sid) for split, ids in manifest["splits"].items() for sid in ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, sid in enumerate(pool.map(fetch, items), 1):
            if i % (100 if len(items) > 100 else 10) == 0 or i == len(items):
                print(f"Downloaded {i}/{len(items)} scenarios ({sid})", flush=True)
    return manifest


def to_local(points, origin, heading):
    points = np.asarray(points, dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    return (points - origin) @ np.array([[c, -s], [s, c]], dtype=np.float32)


def to_world(points, origin, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(points) @ np.array([[c, s], [-s, c]]) + origin


def _points(value):
    if isinstance(value, dict):
        value = value.get("xyz", value.get("polyline", value.get("points", [])))
    if not value:
        return np.empty((0, 2), dtype=np.float32)
    if isinstance(value[0], dict):
        return np.array([[p["x"], p["y"]] for p in value], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)[:, :2]


def _values(value):
    return value.values() if isinstance(value, dict) else value


def _resample(points, count=20):
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    d = np.linspace(0, lengths[-1], count)
    return np.stack([np.interp(d, lengths, points[:, axis]) for axis in (0, 1)], axis=-1).astype(np.float32)


def load_av2(parquet_path, map_path, max_actors=32, max_polylines=128, map_radius=100.0):
    import pandas as pd
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    df = pd.read_parquet(parquet_path)
    df["track_id"] = df["track_id"].astype(str)
    if max_actors < 2 or max_polylines < 1 or map_radius <= 0:
        raise ValueError("Need max_actors >=2, max_polylines >=1 and positive map_radius")
    sid = str(df["scenario_id"].iloc[0]) if "scenario_id" in df else Path(parquet_path).stem.removeprefix("scenario_")
    focal = str(df["focal_track_id"].iloc[0])
    observed = df[df.timestep < 50]
    focal_last = observed[(observed.track_id == focal) & (observed.timestep == 49)]
    if len(focal_last) != 1:
        raise ValueError(f"Focal actor must have one state at cutoff: {sid}")
    row = focal_last.iloc[0]
    origin = np.array([row.position_x, row.position_y], dtype=np.float32)
    rotation = float(row.heading)
    latest = observed.sort_values("timestep").groupby("track_id", sort=False).tail(1)
    latest = latest[latest.timestep == 49].copy()
    latest["distance"] = np.hypot(latest.position_x-origin[0], latest.position_y-origin[1])
    ids = latest.sort_values(["distance", "track_id"]).track_id.astype(str).tolist()
    selected = [focal] + (["AV"] if "AV" in ids and focal != "AV" else [])
    selected += [track_id for track_id in ids if track_id not in selected]
    selected = selected[:max_actors]
    n = len(selected)
    history = np.zeros((n, 50, 8), dtype=np.float32)
    history_mask = np.zeros((n, 50), dtype=bool)
    future = np.zeros((n, 60, 2), dtype=np.float32)
    future_mask = np.zeros((n, 60), dtype=bool)
    future_heading = np.zeros((n, 60), dtype=np.float32)
    types = np.full(n, 9, dtype=np.int64)
    categories = np.zeros(n, dtype=np.int64)
    grouped = {str(track_id): track.sort_values("timestep") for track_id, track in df.groupby("track_id", sort=False)}
    for i, track_id in enumerate(selected):
        track = grouped[track_id]
        types[i] = TYPES.index(track.object_type.iloc[0]) if track.object_type.iloc[0] in TYPES else 9
        categories[i] = int(track.object_category.iloc[0])
        timesteps = track.timestep.to_numpy(dtype=np.int64)
        if len(np.unique(timesteps)) != len(timesteps):
            raise ValueError(f"Duplicate actor timestep: {sid}/{track_id}")
        states = track[["position_x", "position_y", "velocity_x", "velocity_y", "heading"]].to_numpy(dtype=np.float32)
        observed_rows = (timesteps >= 0) & (timesteps < 50)
        if not np.isfinite(states[observed_rows]).all():
            raise ValueError(f"Non-finite observed state for actor {track_id}, scenario {sid}")
        positions = to_local(states[:, :2], origin, rotation)
        velocities = to_local(states[:, 2:4], np.zeros(2), rotation)
        angles = states[:, 4]-rotation
        indices = timesteps[observed_rows]
        history[i, indices, :2] = positions[observed_rows]
        history[i, indices, 2:4] = velocities[observed_rows]
        history[i, indices, 4] = np.sin(angles[observed_rows])
        history[i, indices, 5] = np.cos(angles[observed_rows])
        history[i, indices, 6] = 1
        history[i, indices, 7] = indices/49
        history_mask[i, indices] = True
        future_rows = (timesteps >= 50) & (timesteps < 110) & np.isfinite(states[:, [0, 1, 4]]).all(-1)
        indices = timesteps[future_rows]-50
        future[i, indices] = positions[future_rows]
        future_heading[i, indices] = angles[future_rows]
        future_mask[i, indices] = True
    city = str(df.city.iloc[0])
    raw_map = read_json(map_path)
    lanes, crossings, areas, polylines = [], [], [], []
    for lane in _values(raw_map.get("lane_segments", {})):
        left = to_local(_points(lane["left_lane_boundary"]), origin, rotation)
        right = to_local(_points(lane["right_lane_boundary"]), origin, rotation)
        if not len(left) or not len(right):
            continue
        center = (_resample(left) + _resample(right))/2
        if np.min(np.linalg.norm(center, axis=1)) > map_radius:
            continue
        item = {"id": str(lane["id"]), "centerline": center.tolist(), "points": center.tolist(),
                "left_boundary": left.tolist(), "right_boundary": right.tolist(),
                "left_neighbor_id": str(lane["left_neighbor_id"]) if lane.get("left_neighbor_id") is not None else None,
                "right_neighbor_id": str(lane["right_neighbor_id"]) if lane.get("right_neighbor_id") is not None else None,
                "is_drivable": lane.get("lane_type", "VEHICLE") in ("VEHICLE", "BUS")}
        lanes.append(item)
        polylines.append((center, [1, 0, 0, float(lane.get("is_intersection", False))]))
    for crossing in _values(raw_map.get("pedestrian_crossings", {})):
        edge1 = to_local(_points(crossing["edge1"]), origin, rotation)
        edge2 = to_local(_points(crossing["edge2"]), origin, rotation)
        polygon = np.concatenate([edge1, edge2[::-1]])
        crossings.append(polygon.tolist())
        polylines.append((_resample(polygon), [0, 1, 0, 0]))
    for area in _values(raw_map.get("drivable_areas", {})):
        points = to_local(_points(area.get("area_boundary", [])), origin, rotation)
        if len(points) >= 3:
            areas.append(points.tolist())
    polylines.sort(key=lambda p: float(np.min(np.linalg.norm(p[0], axis=1))))
    polylines = polylines[:max_polylines]
    map_points = np.zeros((max(1, len(polylines)), 20, 2), dtype=np.float32)
    map_features = np.zeros((len(map_points), 4), dtype=np.float32)
    map_mask = np.zeros(len(map_points), dtype=bool)
    for i, (points, features) in enumerate(polylines):
        map_points[i], map_features[i], map_mask[i] = _resample(points), features, True
    sdf = np.zeros((1, 256, 256), dtype=np.float32)
    if areas:
        polygons = unary_union([Polygon(a).buffer(0) for a in areas])
        grid = np.arange(256, dtype=np.float32)-127.5
        x, y = np.meshgrid(grid, grid)
        inside = contains_xy(polygons, x, y)
        sdf[0] = distance_transform_edt(inside) - distance_transform_edt(~inside)
    total_dynamic = int(latest.object_type.isin(TYPES[:5]).sum())
    meta = {"id": sid, "city": city, "focal_track_id": focal, "ego_track_id": "AV" if "AV" in selected else None,
            "coordinate_frame": {"origin": origin.tolist(), "rotation": rotation, "units": "m", "name": "focal_local"},
            "map": {"lanes": lanes, "crosswalks": crossings, "drivable_areas": areas},
            "selection": {"retained_dynamic": int((types < 5).sum()), "total_dynamic": total_dynamic,
                          "retained_scored": int((categories >= 2).sum()),
                          "total_scored": int((latest.object_category >= 2).sum())}}
    return {"id": sid, "actor_ids": selected, "metadata": meta, "map_lanes": lanes,
            "history": history, "history_mask": history_mask, "future": future, "future_mask": future_mask,
            "future_heading": future_heading, "actor_types": types, "actor_mask": np.ones(n, dtype=bool),
            "dynamic_mask": types < 5, "scored_mask": categories >= 2,
            "map_points": map_points, "map_features": map_features, "map_mask": map_mask,
            "sdf": sdf, "has_drivable": np.array(bool(areas))}


def _cache_worker(task):
    raw_dir, cache_dir, split, scenario_id, options = task
    target = cache_dir/split/f"{scenario_id}.npz"
    written = not target.exists()
    if written:
        directory = raw_dir/split/scenario_id
        parquet = directory/f"scenario_{scenario_id}.parquet"
        map_path = directory/f"log_map_archive_{scenario_id}.json"
        if not parquet.exists() or not map_path.exists():
            raise FileNotFoundError(f"Missing raw files for {scenario_id}. Resume adp download before preprocessing.")
        scene = load_av2(parquet, map_path, **options)
        arrays = {k: v for k, v in scene.items() if isinstance(v, np.ndarray)}
        arrays["metadata_json"] = np.array(json.dumps({k: scene[k] for k in ("id", "actor_ids", "metadata", "map_lanes")}))
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(target)
    else:
        scene = load_scene(target)
    item = {"id": scenario_id, "city": scene["metadata"]["city"], "actor_count": len(scene["actor_ids"]),
            "focal_type": TYPES[int(scene["actor_types"][0])], "split": split}
    audit = {"risk_eligible_scenarios": 0, "risk_eligible_actors": 0, "risk_positive_actors": 0}
    _audit_risk(scene, audit)
    return item, audit, written


def preprocess(raw_dir="data/raw", cache_dir="data/cache", workers=4, **options):
    defaults = {"max_actors": 32, "max_polylines": 128, "map_radius": 100}
    options = {**defaults, **options}
    radius = float(options["map_radius"])
    options["map_radius"] = int(radius) if radius.is_integer() else radius
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    manifest = read_json(raw_dir / "manifest.json")
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = cache_dir / "manifest.json"
    if existing.exists() and manifest_hash(existing) != manifest_hash(raw_dir / "manifest.json"):
        raise ValueError("Cache belongs to a different manifest. Use a new cache directory.")
    existing.write_bytes((raw_dir / "manifest.json").read_bytes())
    config_path = cache_dir / "preprocessing.json"
    signature = {"version": 1, **options}
    if config_path.exists():
        if {**defaults, **read_json(config_path)} != signature:
            raise ValueError("Preprocessing options changed. Use a new cache directory.")
    else:
        write_json(config_path, signature)
    if workers < 1:
        raise ValueError("Preprocessing workers must be positive")
    count, catalog, audit, tasks = 0, [], {}, []
    for split, ids in manifest["splits"].items():
        audit[split] = {"scenarios": len(ids), "risk_eligible_scenarios": 0, "risk_eligible_actors": 0, "risk_positive_actors": 0}
        target_dir = cache_dir / split
        target_dir.mkdir(exist_ok=True)
        for scenario_id in ids:
            tasks.append((raw_dir, cache_dir, split, scenario_id, options))

    def collect(results):
        nonlocal count
        for done, (item, counters, written) in enumerate(results, 1):
            catalog.append(item)
            for key, value in counters.items():
                audit[item["split"]][key] += value
            count += int(written)
            if done % (100 if len(tasks) > 100 else 10) == 0 or done == len(tasks):
                print(f"Prepared {done}/{len(tasks)} scenarios ({count} newly cached)", flush=True)

    if workers == 1 or len(tasks) < 4:
        collect(map(_cache_worker, tasks))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            collect(pool.map(_cache_worker, tasks))
    write_json(cache_dir / "catalog.json", {"scenarios": catalog, "total": len(catalog)})
    write_json(cache_dir / "audit.json", audit)
    return {"written": count, "manifest_hash": manifest_hash(existing), "risk_audit": audit}


def _audit_risk(scene, totals):
    dynamic = scene["dynamic_mask"] & scene["actor_mask"]
    if dynamic.sum() >= 2 and scene["future_mask"][dynamic].all():
        totals["risk_eligible_scenarios"] += 1
        totals["risk_eligible_actors"] += int(dynamic.sum())
        future = scene["future"][dynamic]
        pairs = np.linalg.norm(future[:, None]-future[None, :], axis=-1) < 3
        pairs[np.arange(len(future)), np.arange(len(future))] = False
        totals["risk_positive_actors"] += int(pairs.any(axis=(1, 2)).sum())


def load_scene(path):
    with np.load(path, allow_pickle=False) as archive:
        scene = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        scene.update(json.loads(str(archive["metadata_json"])))
    return scene


class CachedDataset:
    def __init__(self, cache_dir, split):
        self.cache_dir, self.split = Path(cache_dir), split
        self.manifest_hash = manifest_hash(self.cache_dir / "manifest.json")
        self.ids = read_json(self.cache_dir / "manifest.json")["splits"][split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return load_scene(self.cache_dir / self.split / f"{self.ids[index]}.npz")


def collate_scenes(scenes):
    import torch
    actors = max(len(s["actor_ids"]) for s in scenes)
    maps = max(len(s["map_points"]) for s in scenes)
    result = {"scenario_ids": [s["id"] for s in scenes], "actor_ids": [s["actor_ids"] for s in scenes],
              "selections": [s["metadata"].get("selection", {}) for s in scenes]}
    actor_keys = {"history", "history_mask", "future", "future_mask", "future_heading", "actor_types", "actor_mask", "dynamic_mask", "scored_mask"}
    map_keys = {"map_points", "map_features", "map_mask"}
    for key, value in scenes[0].items():
        if not isinstance(value, np.ndarray):
            continue
        values = []
        for scene in scenes:
            array = scene[key]
            size = actors if key in actor_keys else maps if key in map_keys else None
            if size is not None:
                pad = [(0, size-len(array))] + [(0, 0)]*(array.ndim-1)
                array = np.pad(array, pad)
            values.append(array)
        result[key] = torch.from_numpy(np.stack(values))
    return result


def public_scene(scene):
    meta = scene["metadata"]
    actors = []
    for i, actor_id in enumerate(scene["actor_ids"]):
        actors.append({"id": actor_id, "type": TYPES[int(scene["actor_types"][i])],
                       "focal": actor_id == meta["focal_track_id"], "ego": actor_id == meta["ego_track_id"],
                       "history": [p.tolist() if valid else None for p, valid in zip(scene["history"][i, :, :2], scene["history_mask"][i])],
                       "headings": [float(np.arctan2(h[4], h[5])) if valid else None for h, valid in zip(scene["history"][i], scene["history_mask"][i])],
                       "reference_future": [p.tolist() if valid else None for p, valid in zip(scene["future"][i], scene["future_mask"][i])]})
    return {**meta, "actors": actors, "dt": 0.1, "observed_steps": 50, "future_steps": 60}
