"""User-selected AV2 imports. Source files are read; only completed caches are published."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import uuid

import numpy as np

from adp.data import TYPES, load_av2, load_scene, read_json, write_json


class ImportManager:
    def __init__(self, root, options=None, max_scenarios=10_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.options = {"max_actors": 32, "max_polylines": 128, "map_radius": 100, **(options or {})}
        self.max_scenarios = max_scenarios
        self.lock = threading.RLock()
        self.sources, self.jobs = {}, {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="av2-import")

    def grant_directory(self, directory):
        directory = Path(directory).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("Choose an AV2 scenario directory.")
        token = secrets.token_urlsafe(32)
        with self.lock:
            if len(self.sources) >= 32:
                self.sources.pop(next(iter(self.sources)))
            self.sources[token] = directory
        return {"source_token": token, "display_name": directory.name}

    def start(self, source_token):
        with self.lock:
            directory = self.sources.pop(source_token, None)
            if directory is None:
                raise ValueError("Choose the source directory again; its selection token has expired.")
            if any(job["status"] in ("queued", "scanning", "running") for job in self.jobs.values()):
                self.sources[source_token] = directory
                raise ValueError("An import is already running. Wait for it or cancel it first.")
            while len(self.jobs) >= 20:
                self.jobs.pop(next(iter(self.jobs)))
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {"id": job_id, "status": "queued", "display_name": directory.name,
                                 "processed": 0, "total": 0, "imported": 0, "errors": [],
                                 "cancel": threading.Event()}
        self.executor.submit(self._run, job_id, directory)
        return self.status(job_id)

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return {key: value.copy() if isinstance(value, list) else value
                    for key, value in self.jobs[job_id].items() if key != "cancel"}

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            self.jobs[job_id]["cancel"].set()
        return self.status(job_id)

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def _run(self, job_id, directory):
        staging = self.root / (".partial-" + job_id)
        job = self.jobs[job_id]
        try:
            self._update(job_id, status="scanning")
            candidates = []
            for folder, folders, files in os.walk(directory, followlinks=False):
                folders[:] = sorted(name for name in folders if not (Path(folder) / name).is_symlink())
                if job["cancel"].is_set():
                    self._update(job_id, status="cancelled")
                    return
                for name in sorted(files):
                    if name.startswith("scenario_") and name.endswith(".parquet"):
                        candidates.append(Path(folder) / name)
                        if len(candidates) > self.max_scenarios:
                            raise ValueError(f"This selection exceeds {self.max_scenarios:,} scenarios. Choose a smaller directory.")
            if not candidates:
                raise ValueError("No scenario_*.parquet files found. Select an extracted AV2 Motion Forecasting directory.")
            self._update(job_id, total=len(candidates), status="running")
            (staging / "imported").mkdir(parents=True)
            catalog, ids, seen = [], [], set()
            for position, parquet in enumerate(candidates, 1):
                if job["cancel"].is_set():
                    self._update(job_id, status="cancelled")
                    return
                sid = parquet.name[len("scenario_"):-len(".parquet")]
                try:
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sid):
                        raise ValueError("Unsupported scenario filename")
                    map_path = parquet.with_name(f"log_map_archive_{sid}.json")
                    for source in (parquet, map_path):
                        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(directory):
                            raise ValueError("Scenario files must remain inside the selected directory")
                    if parquet.stat().st_size > 128 * 1024**2 or map_path.stat().st_size > 32 * 1024**2:
                        raise ValueError("Scenario files exceed the import size limit (128 MB tracks / 32 MB map)")
                    from pyarrow.parquet import ParquetFile
                    if ParquetFile(parquet).metadata.num_rows > 1_000_000:
                        raise ValueError("Scenario exceeds the one-million-row import limit")
                    if sid in seen:
                        raise ValueError("Duplicate scenario ID in the selected directory")
                    scene = load_av2(parquet, map_path, **self.options)
                    if scene["id"] != sid:
                        raise ValueError("Scenario ID does not match its filename")
                    validate_scene(scene, self.options)
                    arrays = {key: value for key, value in scene.items() if isinstance(value, np.ndarray)}
                    arrays["metadata_json"] = np.array(json.dumps({key: scene[key] for key in ("id", "actor_ids", "metadata", "map_lanes")}))
                    target = staging / "imported" / f"{sid}.npz"
                    np.savez_compressed(target, **arrays)
                    validate_scene(load_scene(target), self.options)
                    seen.add(sid)
                    ids.append(sid)
                    catalog.append({"id": sid, "city": scene["metadata"]["city"], "actor_count": len(scene["actor_ids"]),
                                    "focal_type": TYPES[int(scene["actor_types"][0])], "split": "imported", "source": "imported"})
                except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
                    with self.lock:
                        if len(job["errors"]) < 100:
                            job["errors"].append({"scenario": sid, "reason": str(error)})
                self._update(job_id, processed=position, imported=len(ids))
            if job["cancel"].is_set():
                self._update(job_id, status="cancelled")
                return
            if not ids:
                raise ValueError("No valid AV2 scenarios were imported. See the per-scenario errors.")
            write_json(staging / "manifest.json", {"splits": {"imported": ids}, "source": "user_import", "job_id": job_id})
            write_json(staging / "preprocessing.json", {"version": 1, **self.options})
            write_json(staging / "catalog.json", {"scenarios": catalog, "total": len(catalog)})
            write_json(staging / "import.json", {"id": job_id, "display_name": directory.name, "errors": job["errors"]})
            # Publish a complete directory; benchmark manifests and raw files are never rewritten.
            with self.lock:
                if job["cancel"].is_set():
                    self._update(job_id, status="cancelled")
                    return
                staging.replace(self.root / job_id)
                self._update(job_id, status="completed")
        except Exception as error:
            self._update(job_id, status="failed", error=str(error))
        finally:
            if staging.exists():
                # staging is constructed from our root and a generated UUID, never from user input.
                if staging.is_symlink() or staging.resolve().parent != self.root:
                    raise RuntimeError("Refusing to remove an import staging directory outside the data root")
                shutil.rmtree(staging.resolve())

    def caches(self):
        return sorted(path for path in self.root.iterdir() if path.is_dir() and not path.name.startswith(".")
                      and (path / "manifest.json").is_file() and (path / "catalog.json").is_file())

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["cancel"].set()
        self.executor.shutdown(wait=True, cancel_futures=True)


def validate_scene(scene, options):
    actors = len(scene["actor_ids"])
    if not 1 <= actors <= options["max_actors"] or scene["history"].shape != (actors, 50, 8):
        raise ValueError("Invalid actor/history shape")
    if scene["future"].shape != (actors, 60, 2) or not scene["history_mask"][0, 49]:
        raise ValueError("A focal observation at the prediction cutoff is required")
    if not np.isfinite(scene["history"][scene["history_mask"]]).all():
        raise ValueError("Observed actor states contain non-finite values")
    if not np.isfinite(scene["map_points"]).all() or not np.isfinite(scene["map_features"]).all():
        raise ValueError("Map geometry contains non-finite values")
    for collection in (scene["metadata"]["map"]["drivable_areas"], scene["metadata"]["map"]["crosswalks"]):
        if any(not np.isfinite(np.asarray(points)).all() for points in collection):
            raise ValueError("Map geometry contains non-finite values")
