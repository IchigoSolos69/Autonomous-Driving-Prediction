"""Bounded real-data acquisition, pilot measurements and resumable study execution."""
from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
import urllib.parse

from adp.data import S3_ROOT, S3_URL, _get, _list, manifest_hash, read_json, write_json

GIB = 1024**3
COUNTS = {"train": 40000, "validation": 2000, "calibration": 5000, "evaluation": 5000}


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def metered_operation(root, device):
    """Charge inference/calibration to the same cap; a crash loses at most one reserved slice."""
    if str(device) == "cpu":
        yield lambda: None
        return
    ledger = Path(root) / "study_ledger.json"
    state = read_json(ledger) if ledger.exists() else {"elapsed_seconds": 0., "runs": {}}
    available = 40*3600 - state["elapsed_seconds"]
    if available <= 0:
        raise RuntimeError("Study reached its 40 GPU-hour ceiling")
    reserved = min(300., available)
    state["elapsed_seconds"] += reserved
    write_json(ledger, state)
    started = time.monotonic()
    def check():
        nonlocal reserved
        elapsed = time.monotonic() - started
        needed = min(available, 300 * (int(elapsed / 300) + 1))
        if needed > reserved:
            state["elapsed_seconds"] += needed - reserved
            reserved = needed
            write_json(ledger, state)
        if elapsed >= available:
            raise RuntimeError("Study reached its 40 GPU-hour ceiling; no partial evaluation is promoted")
    try:
        yield check
        check()
    finally:
        state["elapsed_seconds"] += (time.monotonic()-started) - reserved
        write_json(ledger, state)


def ensure_space(path, minimum_free_gib=8, additional_bytes=0):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free - additional_bytes < minimum_free_gib * GIB:
        raise RuntimeError(f"Storage paused: {free/GIB:.2f} GiB free; preserve {minimum_free_gib} GiB. Free space and resume.")
    return free


def remaining_storage(root, projected_bytes):
    root = Path(root)
    existing = sum(path.stat().st_size for name in ("raw", "cache")
                   for path in (root / name).rglob("*") if path.is_file())
    return max(0, projected_bytes - existing)


def _scenario_id(value):
    value = str(value)
    if str(uuid.UUID(value)) != value:
        raise ValueError(f"Invalid AV2 scenario identifier: {value}")
    return value


def _listing(split, directory):
    path = Path(directory) / f"{split}.json"
    if path.exists():
        result = read_json(path)
        if result.get("source") != S3_URL + S3_ROOT + split + "/" or not result.get("complete"):
            raise ValueError("Unrecognized or incomplete official scenario listing")
        return [_scenario_id(sid) for sid in result["ids"]]
    ids = sorted({_scenario_id(p.rstrip("/").rsplit("/", 1)[-1]) for p in _list(S3_ROOT + split + "/", "/")})
    write_json(path, {"source": S3_URL + S3_ROOT + split + "/", "complete": True, "ids": ids})
    return ids


def inspected_ids(project_root):
    """Include smoke/audit caches, even if a catalog/manifest was not preserved."""
    seen, manifests = set(), []
    for directory in (Path(project_root) / "data", Path(project_root) / "artifacts"):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "manifest.json":
                value = read_json(path)
                if "splits" in value:
                    manifests.append(path)
                    seen.update(sid for ids in value["splits"].values() for sid in ids)
            elif path.suffix in {".npz", ".parquet"}:
                candidate = path.stem.removeprefix("scenario_")
                try:
                    seen.add(_scenario_id(candidate))
                except ValueError:
                    pass
    return seen, manifests


def prepare_study(root, project_root, counts=None, seed=42):
    root = Path(root)
    counts = COUNTS if counts is None else counts
    if set(counts) != set(COUNTS) or any(not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("All four study split counts must be positive integers")
    path = root / "raw" / "manifest.json"
    if path.exists():
        value = read_json(path)
        if value.get("selection") != {"counts": counts, "seed": seed}:
            raise ValueError("Study manifest is immutable; select a new root")
        _, manifests = inspected_ids(project_root)
        _prepare_auxiliary(root, value, manifests, manifest_hash(path))
        return value
    ensure_space(root)
    seen, manifests = inspected_ids(project_root)
    train_ids = _listing("train", root / "listings")
    val_ids = _listing("val", root / "listings")
    order = lambda ids: sorted(ids, key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).digest())
    training = order(train_ids)
    available = order(set(val_ids) - seen)
    needed = sum(counts[key] for key in counts if key != "train")
    if len(training) < counts["train"] or len(available) < needed:
        raise ValueError("Not enough unseen official scenarios for the fixed study")
    offset, splits = 0, {"train": training[:counts["train"]]}
    for split in ("validation", "calibration", "evaluation"):
        splits[split] = available[offset:offset+counts[split]]
        offset += counts[split]
    if len({sid for ids in splits.values() for sid in ids}) != sum(counts.values()):
        raise ValueError("Official split selection contains duplicate/overlapping scenario IDs")
    value = {"version": 2, "source": "Argoverse 2 Motion Forecasting", "license": "CC-BY-NC-SA-4.0",
             "sampling": "SHA256(seed:id) order over complete official split listings; previously inspected validation excluded",
             "selection": {"counts": counts, "seed": seed}, "splits": splits,
             "excluded_validation_ids": sorted(seen.intersection(val_ids)),
             "listing_sha256": {split: manifest_hash(root / "listings" / f"{split}.json") for split in ("train", "val")}}
    # Match write_json's platform newline encoding without publishing early.
    parent_hash = hashlib.sha256(json.dumps(value, indent=2, allow_nan=False).replace("\n", os.linesep).encode()).hexdigest()
    _prepare_auxiliary(root, value, manifests, parent_hash)
    # Publish the main manifest last; retries also repair older partial preparations.
    write_json(path, value)
    return value


def _prepare_auxiliary(root, value, manifests, parent_hash):
    for source in manifests:
        destination = root / "provenance" / f"{manifest_hash(source)}.json"
        if not destination.exists() or manifest_hash(destination) != manifest_hash(source):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    # Pilot membership is training-only: holdout outcomes are never pilot inputs.
    pilot = {**value, "selection": {"parent_sha256": parent_hash, "pilot": True},
             "splits": {"train": value["splits"]["train"][:1000], "validation": [], "calibration": [], "evaluation": []}}
    pilot_path = root / "pilot" / "raw" / "manifest.json"
    if pilot_path.exists() and read_json(pilot_path) != pilot:
        raise ValueError("Pilot manifest is immutable")
    if not pilot_path.exists():
        write_json(pilot_path, pilot)


def acquire(raw_dir, workers=4, minimum_free_gib=8):
    raw_dir = Path(raw_dir)
    manifest = read_json(raw_dir / "manifest.json")
    if workers < 1 or workers > 16:
        raise ValueError("Download workers must be between 1 and 16")
    write_lock = threading.Lock()
    def fetch(item):
        split, sid = item
        _scenario_id(sid)
        if split not in COUNTS:
            raise ValueError("Unsupported split")
        folder = raw_dir / split / sid
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{S3_ROOT}{'train' if split == 'train' else 'val'}/{sid}/"
        for name in (f"scenario_{sid}.parquet", f"log_map_archive_{sid}.json"):
            target = folder / name
            if target.exists() and target.stat().st_size:
                continue
            ensure_space(raw_dir, minimum_free_gib, 16*1024**2)
            payload = _get(S3_URL + urllib.parse.quote(prefix + name, safe="/"))
            with write_lock:
                ensure_space(raw_dir, minimum_free_gib, len(payload) + 16*1024**2)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(target)
        return sid
    items = [(split, sid) for split, ids in manifest["splits"].items() for sid in ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit bounded batches so a failed disk guard cannot leave 52k queued jobs.
        for start in range(0, len(items), workers * 4):
            list(pool.map(fetch, items[start:start+workers*4]))
            print(f"Acquired {min(len(items), start+workers*4)}/{len(items)}", flush=True)
    return {"scenarios": len(items), "manifest_sha256": manifest_hash(raw_dir / "manifest.json")}


def profile_pilot(root, config, device="cuda", epochs=3):
    import torch
    from torch.utils.data import DataLoader
    from adp.data import CachedDataset, collate_scenes
    from adp.model import ForecastModel, forecast_loss, model_inputs
    from adp.training import gpu_smoke_test, resolve_device

    root = Path(root)
    if (root / "schedule.json").exists():
        raise ValueError("The study schedule is frozen; pilot measurements cannot be replaced")
    if epochs < 1:
        raise ValueError("Pilot epochs must be positive")
    settings = config.get("training", {})
    microbatch, accumulation = int(settings.get("batch_size", 4)), int(settings.get("accumulation_steps", 8))
    if microbatch < 1 or accumulation < 1:
        raise ValueError("Pilot microbatch and accumulation must be positive")
    selected = resolve_device(device)
    qualification = gpu_smoke_test(device, config) if selected.type == "cuda" else {"device": "cpu"}
    model = ForecastModel(config["model"]).to(selected).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = DataLoader(CachedDataset(root / "pilot" / "cache", "train"), batch_size=microbatch,
                        collate_fn=collate_scenes, num_workers=int(settings.get("num_workers", 0)),
                        pin_memory=selected.type == "cuda")
    use_amp = selected.type == "cuda" and bool(settings.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        torch.cuda.reset_peak_memory_stats(selected)
    started, visits, timings = time.monotonic(), 0, []
    for _ in range(epochs):
        epoch_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = {k: v.to(selected) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(selected.type, dtype=torch.float16, enabled=use_amp):
                loss = forecast_loss(model(**model_inputs(batch)), batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pilot loss")
            count = batch["history"].shape[0]
            group_start = (index // accumulation) * accumulation * microbatch
            group_examples = min(accumulation * microbatch, len(loader.dataset) - group_start)
            scaler.scale(loss * count / group_examples).backward()
            if (index+1) % accumulation == 0 or index+1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            visits += batch["history"].shape[0]
        if use_amp:
            torch.cuda.synchronize(selected)
        timings.append(time.monotonic() - epoch_start)
    seconds = time.monotonic() - started
    byte_count = sum(p.stat().st_size for directory in ("raw", "cache") for p in (root / "pilot" / directory).rglob("*") if p.is_file())
    pilot_count = len(loader.dataset)
    manifest = read_json(root / "raw" / "manifest.json")
    total = sum(map(len, manifest["splits"].values()))
    projected_bytes = int(byte_count / pilot_count * total)
    projected_hours20 = 6 * len(manifest["splits"]["train"]) * 20 / (visits / seconds) / 3600 * 1.35
    schedule = 20 if projected_hours20 <= 40 else 15
    previous_path = root / "pilot.json"
    if previous_path.exists():
        previous = read_json(previous_path)
        if previous.get("config_sha256") not in (None, config_fingerprint(config)):
            raise ValueError("Existing pilot used a different configuration; use a new study root")
        schedule = min(schedule, previous["epochs"])
        write_json(root / "pilot_previous.json", previous)
    report = {"version": 2, "qualification": qualification, "pilot_scenarios": pilot_count,
              "config_sha256": config_fingerprint(config), "config": config,
              "measurement_settings": {"microbatch": microbatch, "accumulation_steps": accumulation,
                                       "effective_batch": microbatch * accumulation, "amp": use_amp,
                                       "num_workers": int(settings.get("num_workers", 0))},
              "training_examples_per_second": visits / seconds, "epoch_seconds": timings,
              "raw_cache_bytes": byte_count, "projected_raw_cache_bytes": projected_bytes,
              "projected_study_gpu_hours20_including_35pct_overhead": projected_hours20,
              "epochs": schedule, "gpu_hour_ceiling": 40,
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(selected) if use_amp else None,
              "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(selected) if use_amp else None,
              "risk_audit": read_json(root / "pilot" / "cache" / "audit.json"),
              "model_weights_retained": False, "note": "Training-only pilot; projection is not promised throughput"}
    write_json(root / "pilot.json", report)
    if projected_hours20 * schedule / 20 > 40:
        raise RuntimeError("Even a 15-epoch study exceeds 40 projected GPU-hours; pilot recorded, full study paused")
    # Full data are separate from the pilot; both must fit simultaneously.
    ensure_space(root, additional_bytes=projected_bytes)
    return report


def study_configs(config):
    configs = []
    for name in ("full", "no_graph", "no_map", "world_ce"):
        candidate = json.loads(json.dumps(config))
        candidate["seed"] = 42
        candidate.setdefault("training", {})["checkpoint_selection"] = "joint_minFDE"
        candidate["training"]["minimum_free_gib"] = 8
        candidate["model"].update({"use_graph": name != "no_graph", "use_map": name != "no_map",
                                    "shared_world_ce_weight": .1 if name == "world_ce" else 0.})
        configs.append((name, candidate))
    return configs


def run_study(root, config, device="cuda"):
    from filelock import FileLock, Timeout
    try:
        with FileLock(str(Path(root) / "runner.lock"), timeout=0):
            return _run_study(root, config, device)
    except Timeout as error:
        raise RuntimeError("A study runner already owns this root") from error


def _run_study(root, config, device="cuda"):
    """Only train/selection. Calibration and locked evaluation remain explicit commands."""
    from adp.training import load_checkpoint, train
    from adp.reporting import plot_training

    root = Path(root)
    pilot = read_json(root / "pilot.json")
    if pilot.get("config_sha256") != config_fingerprint(config):
        raise ValueError("Pilot configuration differs from study configuration; run a matching pilot first")
    if pilot["projected_study_gpu_hours20_including_35pct_overhead"] * pilot["epochs"] / 20 > 40:
        raise ValueError("Pilot budget does not authorize this study schedule")
    schedule = root / "schedule.json"
    value = {"epochs": pilot["epochs"], "gpu_hour_ceiling": 40, "configs": study_configs(config),
             "pilot_sha256": manifest_hash(root / "pilot.json"),
             "manifest_sha256": manifest_hash(root / "cache" / "manifest.json")}
    if schedule.exists() and read_json(schedule) != json.loads(json.dumps(value)):
        raise ValueError("Study schedule/configuration is frozen")
    write_json(schedule, value)
    ledger = root / "study_ledger.json"
    state = read_json(ledger) if ledger.exists() else {"elapsed_seconds": 0, "runs": {}}
    def execute(name, candidate):
        output = root / "runs" / name
        if state["runs"].get(name, {}).get("completed"):
            return
        while not state["runs"].get(name, {}).get("completed"):
            ensure_space(root, additional_bytes=64*1024**2)
            remaining = min(900, 40*3600 - state["elapsed_seconds"])
            if remaining <= 0:
                raise RuntimeError("Study reached 40 GPU-hour ceiling")
            # A crash charges at most the reserved 15-minute slice, not downtime.
            started = time.monotonic()
            state["runs"][name] = {"completed": False, "reserved_seconds": remaining}
            state["elapsed_seconds"] += remaining
            write_json(ledger, state)
            result = train(candidate, root / "cache", output, pilot["epochs"], device,
                           output / "last.pt" if (output / "last.pt").exists() else None,
                           max_runtime_seconds=remaining)
            state["elapsed_seconds"] -= remaining - (time.monotonic()-started)
            state["runs"][name] = {"completed": result["completed_epochs"] >= pilot["epochs"],
                                    "result": result}
            write_json(ledger, state)
        plot_training(output)
    for name, candidate in study_configs(config):
        execute(f"{name}-42", candidate)
    def key(name):
        saved = load_checkpoint(root / "runs" / name / "best.pt")
        return tuple(saved["best_selection"]["key"])
    winner = min((f"{name}-42" for name, _ in study_configs(config)), key=key)
    selected_config = next(c for name, c in study_configs(config) if winner == f"{name}-42")
    for seed in (43, 44):
        candidate = json.loads(json.dumps(selected_config))
        candidate["seed"] = seed
        execute(f"{winner.rsplit('-', 1)[0]}-{seed}", candidate)
    candidates = [winner, *[f"{winner.rsplit('-', 1)[0]}-{seed}" for seed in (43, 44)]]
    deployment = min(candidates, key=key)
    checkpoint = root / "runs" / deployment / "best.pt"
    frozen = {"configuration": winner.rsplit("-", 1)[0], "deployment_run": deployment,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": manifest_hash(checkpoint),
              "all_seed_runs": candidates, "selection_key": key(deployment),
              "test_outcomes_used_for_selection": False}
    target = root / "frozen_selection.json"
    if target.exists() and read_json(target) != json.loads(json.dumps(frozen)):
        raise ValueError("Deployment checkpoint was already frozen")
    write_json(target, frozen)
    return frozen


def fit_cv_scale(cache_dir, budget_check=None):
    """Fit one bounded isotropic Gaussian scale, exclusively on calibration futures."""
    import numpy as np
    from adp.data import CachedDataset, collate_scenes
    from adp.model import constant_velocity, model_inputs
    squared, coordinates, scenarios = 0., 0, 0
    for scene in CachedDataset(cache_dir, "calibration"):
        if budget_check:
            budget_check()
        eligible = scene["dynamic_mask"] & scene["actor_mask"] & scene["scored_mask"]
        eligible &= (scene["future_mask"] & np.isfinite(scene["future"]).all(-1)).all(-1)
        if not eligible.any():
            continue
        prediction = constant_velocity(model_inputs(collate_scenes([scene])))["means"][0, 0].numpy()
        residual = prediction[eligible] - scene["future"][eligible]
        squared += float(np.square(residual.astype(np.float64)).sum())
        coordinates += residual.size
        scenarios += 1
    return {"scale_m": float(np.clip(np.sqrt(squared / coordinates), .15, 20)) if coordinates else 1.,
            "fitted": coordinates > 0, "coordinates": coordinates, "scenarios": scenarios,
            "split": "calibration", "manifest_sha256": manifest_hash(Path(cache_dir) / "manifest.json"),
            "bounds_m": [.15, 20], "population": "complete retained scored dynamic futures"}


def finalize_study(root, reference_checkpoint, device="cuda"):
    from filelock import FileLock, Timeout
    try:
        with FileLock(str(Path(root) / "runner.lock"), timeout=0):
            return _finalize_study(root, reference_checkpoint, device)
    except Timeout as error:
        raise RuntimeError("A study runner already owns this root") from error


def _finalize_study(root, reference_checkpoint, device="cuda"):
    """Frozen models only: calibrate, evaluate once, compare, then publish a gate result."""
    from types import SimpleNamespace
    from adp.cli import _evaluate
    from adp.protocol import paired_comparison
    from adp.training import calibrate
    root, reference_checkpoint = Path(root), Path(reference_checkpoint).resolve()
    frozen = read_json(root / "frozen_selection.json")
    if len(frozen["all_seed_runs"]) != 3 or len(set(frozen["all_seed_runs"])) != 3 or frozen["deployment_run"] not in frozen["all_seed_runs"]:
        raise ValueError("The frozen study must include three distinct seed runs and its selected deployment run")
    if manifest_hash(frozen["checkpoint"]) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen deployment checkpoint changed")
    checkpoints = {name: root / "runs" / name / "best.pt" for name in frozen["all_seed_runs"]}
    checkpoints["reference"] = reference_checkpoint
    plan = {"version": 1, "checkpoint_sha256": {name: manifest_hash(path) for name, path in checkpoints.items()},
            "protocol_sha256": manifest_hash(root / "protocol.json"), "deployment_run": frozen["deployment_run"],
            "bootstrap": 1000, "stress_scenario_limit": 512, "stress_seed": 42, "stress_severity": .35}
    target = root / "evaluation_plan.json"
    if target.exists() and read_json(target) != plan:
        raise ValueError("Evaluation is locked to the original frozen model set; create a new study")
    write_json(target, plan)
    evaluation_root = root / "evaluation"
    scale_path = root / "cv_scale.json"
    if scale_path.exists():
        scale = read_json(scale_path)
    else:
        with metered_operation(root, device) as check:
            scale = fit_cv_scale(root / "cache", check)
    if (scale.get("manifest_sha256") != manifest_hash(root / "cache" / "manifest.json")
            or scale.get("split") != "calibration" or not .15 <= scale.get("scale_m", 0) <= 20):
        raise ValueError("Constant-velocity scale lacks matching calibration provenance")
    write_json(scale_path, scale)
    def evaluate(name, checkpoint=None, calibration=None, cv_scale=1., stress_kind=None):
        directory = evaluation_root / name
        args = SimpleNamespace(checkpoint=str(checkpoint) if checkpoint else None,
            calibration=str(calibration) if calibration else None, cache_dir=root / "cache",
            output_dir=directory, split="evaluation", device=device, protocol=root / "protocol.json",
            bootstrap=1000, seed=42, limit=512 if stress_kind else None,
            cv_scale=cv_scale, kind=stress_kind, severity=.35)
        identity = {"checkpoint_sha256": manifest_hash(checkpoint) if checkpoint else None,
                    "manifest_hash": manifest_hash(root / "cache" / "manifest.json"),
                    "benchmark_protocol_sha256": manifest_hash(root / "protocol.json"),
                    "calibration_sha256": manifest_hash(calibration) if calibration else None,
                    "calibration_applied": bool(calibration and not stress_kind),
                    "split": "evaluation", "scenario_limit": args.limit,
                    "constant_velocity_scale_m": cv_scale if checkpoint is None else None,
                    "stress": {"kind": stress_kind, "severity": .35, "seed": 42, "other_agents_reactive": False} if stress_kind else None}
        request_path = directory / "evaluation_request.json"
        if request_path.exists() and read_json(request_path) != identity:
            raise ValueError("Evaluation output request changed; keep the frozen results and use a new study")
        if not request_path.exists():
            write_json(request_path, identity)
        if (directory / "metrics.json").exists() and (directory / "records.json").exists():
            report = read_json(directory / "metrics.json")
        else:
            with metered_operation(root, device) as check:
                args.budget_check = check
                report = _evaluate(args, stress=stress_kind is not None)
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError("Evaluation report does not match the exact frozen request")
        records = read_json(directory / "records.json")
        actual_ids = [record["scenario_id"] for record in records]
        skipped = [item["scenario_id"] for item in report.get("skipped", [])]
        membership = read_json(root / "cache" / "manifest.json")["splits"]["evaluation"]
        expected_ids = membership[:args.limit] if args.limit is not None else membership
        combined = actual_ids + skipped
        if (len(combined) != len(set(combined)) or set(combined) != set(expected_ids)
                or (not stress_kind and skipped) or report.get("scenario_count") != len(records)
                or report.get("bootstrap", {}).get("replicates") != 1000):
            raise ValueError("Evaluation is incomplete, duplicated, or uses a different bootstrap protocol")
        return report
    reports = {}
    for name, checkpoint in checkpoints.items():
        calibration_path = None
        if name != "reference":
            output = root / "calibration" / name
            calibration_path = output / "calibration.json"
            if not calibration_path.exists():
                with metered_operation(root, device) as check:
                    calibrate(checkpoint, root / "cache", output, device, budget_check=check)
        reports[name] = evaluate(name, checkpoint, calibration_path)
    reports["constant_velocity_fixed"] = evaluate("constant_velocity_fixed")
    reports["constant_velocity_fitted"] = evaluate("constant_velocity_fitted", cv_scale=scale["scale_m"])
    chosen = frozen["deployment_run"]
    candidate = read_json(evaluation_root / chosen / "records.json")
    comparisons = {name: paired_comparison(candidate, read_json(evaluation_root / name / "records.json"), 1000, 42)
                   for name in ("reference", "constant_velocity_fixed", "constant_velocity_fitted")}
    result = {"version": 1, "deployment_checkpoint_sha256": frozen["checkpoint_sha256"],
              "comparisons": comparisons,
              "promotion_gate_passed": all(value["jointFDE_gate_passed"] for value in comparisons.values()),
              "seed_metrics": {name: reports[name] for name in frozen["all_seed_runs"]},
              "risk_experimental": reports[chosen]["risk"]["status"] != "sufficient",
              "scope": "Research forecasting only; no driving-safety claim"}
    write_json(root / "promotion.json", result)
    for kind in ("position_noise", "dropout", "braking", "cut_in"):
        evaluate("stress-"+kind, checkpoints[chosen], stress_kind=kind)
    return result


def run_pipeline(root, project_root, config, reference_checkpoint, device="cuda", workers=4):
    from filelock import FileLock, Timeout
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / "pipeline.lock"), timeout=0):
            # Prevent idle system sleep only while this process is doing the study.
            # The display may turn off and the user's power settings stay unchanged.
            import sys
            keep_awake = None
            previous_state = 0
            if sys.platform == "win32":
                import ctypes
                keep_awake = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
                keep_awake.argtypes = [ctypes.c_uint32]
                keep_awake.restype = ctypes.c_uint32
                previous_state = keep_awake(0x80000001)
                if not previous_state:
                    raise ctypes.WinError(ctypes.get_last_error())
            try:
                return _run_pipeline(root, project_root, config, reference_checkpoint, device, workers)
            finally:
                if keep_awake is not None:
                    keep_awake(previous_state)
    except Timeout as error:
        raise RuntimeError("A study pipeline already owns this root") from error


def _run_pipeline(root, project_root, config, reference_checkpoint, device="cuda", workers=4):
    """Resume the approved sequence; never promote a model merely because training ended."""
    from adp.data import preprocess
    from adp.protocol import create_protocol
    root = Path(root)
    state = root / "pipeline_status.json"
    phase = "prepare"
    try:
        write_json(state, {"phase": phase, "status": "running"})
        prepare_study(root, project_root)
        phase = "pilot"
        write_json(state, {"phase": phase, "status": "running"})
        if not (root / "pilot.json").exists():
            acquire(root / "pilot" / "raw", workers)
            preprocess(root / "pilot" / "raw", root / "pilot" / "cache", workers=workers, minimum_free_gib=8)
            profile_pilot(root, config, device)
        pilot = read_json(root / "pilot.json")
        if pilot.get("config_sha256") != config_fingerprint(config):
            raise ValueError("Pipeline pilot configuration does not match; run a matching pilot")
        phase = "acquire"
        write_json(state, {"phase": phase, "status": "running"})
        ensure_space(root, additional_bytes=remaining_storage(root, pilot["projected_raw_cache_bytes"]))
        acquire(root / "raw", workers)
        preprocess(root / "raw", root / "cache", workers=workers, minimum_free_gib=8)
        create_protocol(root / "cache", root / "protocol.json", list((root / "provenance").glob("*.json")))
        phase = "train"
        write_json(state, {"phase": phase, "status": "running"})
        selected = run_study(root, config, device)
        phase = "finalize"
        write_json(state, {"phase": phase, "status": "running", "selected_checkpoint": selected["checkpoint_sha256"]})
        result = finalize_study(root, reference_checkpoint, device)
        if result["promotion_gate_passed"]:
            from adp.bundles import export_model, load_bundle
            bundle_path = root / "promoted-model-bundle"
            if bundle_path.exists():
                bundle = load_bundle(bundle_path)
                if bundle["metadata"]["provenance"]["source_checkpoint_sha256"] != selected["checkpoint_sha256"]:
                    raise ValueError("Promoted bundle belongs to a different checkpoint")
            else:
                measured = result["seed_metrics"][selected["deployment_run"]]
                card = root / "qualified-model-card.md"
                card.write_text("# Interaction Lab research forecasting model\n\n"
                    "Noncommercial AV2 research use. No driving-safety claim.\n\n"
                    "Data: 40,000 official training scenarios; fresh official validation partitions "
                    "of 2,000 selection, 5,000 calibration and 5,000 locked evaluation scenarios. "
                    "Previously inspected validation IDs excluded. Exact manifests and protocol accompany this study.\n\n"
                    "50 observed and 60 future steps at 10 Hz; up to 32 retained actors and 128 map polylines; "
                    "six shared scene modes, bounded diagonal Gaussian scales. "
                    "Selected by scene joint minFDE; paired 95% scenario-bootstrap improvement over both references. "
                    "Three seeds reported; deployment checkpoint chosen using selection outcomes only.\n\n"
                    "Risk is a <3 m centroid near-conflict proxy, including benign interactions. "
                    "Incomplete groups are excluded. Fixed class footprints are not measured dimensions. "
                    "Imported and stressed populations have unknown calibration applicability. "
                    "Counterfactuals replay nonreactive other actors.\n\n"
                    "Measured held-out results:\n\n```json\n" + json.dumps({key: measured.get(key) for key in
                    ("jointFDE", "minADE", "minFDE", "miss_rate", "NLL", "risk", "coverage", "matched_overlap")},
                    indent=2, allow_nan=False) + "\n```\n", encoding="utf-8")
                export_model(selected["checkpoint"], root / "cache", bundle_path,
                    root / "calibration" / selected["deployment_run"] / "calibration.json", card,
                    root / "evaluation" / selected["deployment_run"] / "metrics.json")
            result["inference_bundle"] = str(bundle_path.resolve())
            write_json(root / "promotion.json", result)
        write_json(state, {"phase": "finished", "status": "completed", "promotion_gate_passed": result["promotion_gate_passed"]})
        return result
    except Exception as error:
        write_json(state, {"phase": phase, "status": "stopped", "error": str(error), "resume": "Run the same study pipeline command"})
        raise
