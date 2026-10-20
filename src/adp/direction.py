"""Selection-only, paired lane-direction experiment with a separate compute ceiling."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

from adp.data import manifest_hash, read_json, write_json
from adp.research import config_fingerprint, ensure_space

SEEDS = (42, 43, 44)
VARIANTS = {"control": "xy_zero_tangent", "directional": "xy_tangent"}
GPU_SECONDS = 12 * 3600
QUALIFICATION_RESERVE = 1.5 * 3600


class DirectionBudgetExhausted(RuntimeError):
    pass


def _freeze(path, value):
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment changed: {path.name}; choose a new experiment root")
    else:
        write_json(path, value)
    return value


def _source_hash():
    directory = Path(__file__).parent
    return config_fingerprint({name: manifest_hash(directory / name) for name in
        ("direction.py", "model.py", "training.py", "data.py", "metrics.py")})


def direction_configs(base_config):
    result = []
    for seed in SEEDS:
        for name, encoding in VARIANTS.items():
            config = json.loads(json.dumps(base_config))
            config["seed"] = seed
            config.setdefault("model", {})["map_point_encoding"] = encoding
            config["model"]["use_map"] = True
            config.setdefault("training", {}).update(checkpoint_selection="joint_minFDE", minimum_free_gib=8)
            result.append({"name": f"{name}-{seed}", "config": config})
    return result


@contextmanager
def charged_operation(root, label, max_seconds=300):
    """Reserve before executing; a process crash retains its reservation in the ledger."""
    path = Path(root) / "direction_ledger.json"
    state = read_json(path) if path.exists() else {"version": 1, "elapsed_seconds": 0., "operations": []}
    if state.get("version") != 1 or not math.isfinite(state.get("elapsed_seconds", -1)) or state["elapsed_seconds"] < 0:
        raise ValueError("Invalid direction-study compute ledger")
    remaining = GPU_SECONDS - state["elapsed_seconds"]
    if remaining <= 0:
        raise DirectionBudgetExhausted("The separate 12 GPU-hour ceiling is exhausted")
    reserved = min(float(max_seconds), remaining)
    started = time.monotonic()
    record = {"label": label, "reserved_seconds": reserved, "status": "reserved"}
    state["operations"].append(record)
    state["elapsed_seconds"] += reserved
    write_json(path, state)
    def check():
        nonlocal reserved
        elapsed = time.monotonic() - started
        needed = min(remaining, 300 * (int(elapsed / 300) + 1))
        if needed > reserved:
            state["elapsed_seconds"] += needed - reserved
            reserved = needed
            record["reserved_seconds"] = reserved
            write_json(path, state)
        if elapsed >= remaining:
            raise DirectionBudgetExhausted("The separate 12 GPU-hour ceiling is exhausted; partial runs are not compared")
    try:
        yield remaining, check
        check()
    finally:
        elapsed = time.monotonic() - started
        state["elapsed_seconds"] += elapsed - reserved
        record.update(status="returned", elapsed_seconds=elapsed)
        write_json(path, state)


def _prepare(reference_root, root):
    from adp.training import load_checkpoint
    reference_root = Path(reference_root).resolve()
    cache = reference_root / "cache"
    manifest = read_json(cache / "manifest.json")
    training, selection = manifest["splits"]["train"], manifest["splits"]["validation"]
    if (len(training) != len(set(training)) or len(selection) != len(set(selection))
            or len(training) != 40000 or len(selection) != 2000 or set(training) & set(selection)):
        raise ValueError("Direction study requires the existing 40,000 train and 2,000 selection scenarios")
    pilot = read_json(reference_root / "pilot" / "cache" / "manifest.json")
    pilot_ids = pilot["splits"]["train"]
    if len(pilot_ids) != 1000 or len(set(pilot_ids)) != 1000 or not set(pilot_ids) <= set(training) or any(
            ids for split, ids in pilot["splits"].items() if split != "train"):
        raise ValueError("Pilot must contain exactly 1,000 training-only scenarios")
    schedule = read_json(reference_root / "schedule.json")
    candidates = []
    for name, config in schedule["configs"]:
        if not config["model"].get("use_map", True):
            continue
        run = reference_root / "runs" / f"{name}-42"
        last, best = load_checkpoint(run / "last.pt"), load_checkpoint(run / "best.pt")
        if config.get("seed") != 42 or last["config"] != config or best["config"] != config:
            raise ValueError("Reference screening checkpoint configuration differs from its frozen schedule")
        if last["completed_epochs"] != schedule["epochs"] or last.get("training_progress"):
            raise ValueError(f"Reference screening run is incomplete: {name}-42")
        if any(saved["manifest_hash"] != manifest_hash(cache / "manifest.json") or
               saved.get("preprocessing_hash") != manifest_hash(cache / "preprocessing.json") for saved in (last, best)):
            raise ValueError("Reference checkpoint/cache provenance differs")
        selection_key = best.get("best_selection", {})
        if (selection_key.get("metric") != "joint_minFDE" or len(selection_key.get("key", [])) != 2
                or not all(math.isfinite(value) for value in selection_key["key"])):
            raise ValueError("Reference screening lacks joint/focal selection evidence")
        key = (*selection_key["key"], selection_key["epoch"], name)
        candidates.append((key, name, config, manifest_hash(run / "best.pt")))
    if not candidates:
        raise ValueError("No completed map-enabled reference screening configuration")
    _, name, base, checkpoint_hash = min(candidates, key=lambda item: item[0])
    protocol = {"version": 1, "kind": "direction-study", "development_only": True,
        "reference_root": str(reference_root), "cache_dir": str(cache), "seeds": list(SEEDS),
        "train_ids": training, "selection_ids": selection, "pilot_ids": pilot_ids,
        "manifest_sha256": manifest_hash(cache / "manifest.json"),
        "preprocessing_sha256": manifest_hash(cache / "preprocessing.json"),
        "pilot_manifest_sha256": manifest_hash(reference_root / "pilot" / "cache" / "manifest.json"),
        "pilot_preprocessing_sha256": manifest_hash(reference_root / "pilot" / "cache" / "preprocessing.json"),
        "reference_schedule_sha256": manifest_hash(reference_root / "schedule.json"),
        "reference_completion_sha256": manifest_hash(reference_root / "pipeline_status.json"),
        "selected_screening_run": f"{name}-42", "selected_screening_checkpoint_sha256": checkpoint_hash,
        "base_config": base, "source_sha256": _source_hash(),
        "gpu_hour_ceiling": 12, "reserved_qualification_hours": 1.5,
        "comparison": "final fixed epoch; directional minus control; scenarios clustered across all three paired seeds",
        "heldout_labels_used": False, "calibration_fitted": False, "automatic_promotion": False,
        "slice_definitions": {"speed_boundaries_m_s": [1, 5, 15], "turning_degrees": 15,
            "turning_observed_steps": [39, 49], "density_radius_m": 20,
            "density_bins": [0, [1, 3], [4, None]], "history_coverage_boundary": .8,
            "small_sample_scenarios": 50}}
    return _freeze(root / "protocol.json", protocol)


def audit_lane_order(reference_root, pilot_ids):
    """Compare ordered boundary endpoints with official lane successor topology."""
    import numpy as np
    from adp.data import _points, _resample, _values
    counts = {"maps": 0, "successor_pairs": 0, "forward": 0, "reverse": 0, "ambiguous": 0}
    for sid in pilot_ids:
        path = Path(reference_root) / "pilot" / "raw" / "train" / sid / f"log_map_archive_{sid}.json"
        raw = read_json(path)
        lanes = {str(lane["id"]): lane for lane in _values(raw.get("lane_segments", {}))}
        centers = {}
        for lid, lane in lanes.items():
            left, right = _points(lane["left_lane_boundary"]), _points(lane["right_lane_boundary"])
            if len(left) and len(right):
                centers[lid] = (_resample(left) + _resample(right)) / 2
        counts["maps"] += 1
        for lid, lane in lanes.items():
            if lid not in centers:
                continue
            for successor in lane.get("successors", []):
                if str(successor) not in centers:
                    continue
                first, second = centers[lid], centers[str(successor)]
                forward = float(np.linalg.norm(first[-1] - second[0]))
                reverse = float(np.linalg.norm(first[0] - second[-1]))
                key = "ambiguous" if abs(forward - reverse) <= .5 else "forward" if forward < reverse else "reverse"
                counts[key] += 1
                counts["successor_pairs"] += 1
    qualified = counts["forward"] > 0 and counts["reverse"] == 0
    return {**counts, "qualified": qualified, "source": "https://github.com/argoverse/av2-api/blob/main/src/av2/map/lane_segment.py",
            "method": "training pilot only; center(end) to successor(start), compared with reversed endpoints; 0.5m ambiguity tolerance"}


def _profile(reference_root, config, device, check):
    import numpy as np
    import random
    import torch
    from torch.utils.data import DataLoader
    from adp.data import CachedDataset, collate_scenes
    from adp.model import ForecastModel, forecast_loss, model_inputs
    from adp.training import resolve_device, gpu_smoke_test
    selected = resolve_device(device)
    if selected.type == "cuda":
        gpu_smoke_test(str(selected), config)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = ForecastModel(config["model"]).to(selected).train()
    settings = config["training"]
    batch_size, accumulation = int(settings.get("batch_size", 4)), int(settings.get("accumulation_steps", 8))
    if batch_size < 1 or accumulation < 1:
        raise ValueError("Pilot microbatch and gradient accumulation must be positive")
    loader = DataLoader(CachedDataset(Path(reference_root) / "pilot" / "cache", "train"),
        batch_size=batch_size, num_workers=int(settings.get("num_workers", 0)), collate_fn=collate_scenes,
        generator=torch.Generator().manual_seed(44), pin_memory=selected.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("learning_rate", 3e-4)),
                                  weight_decay=float(settings.get("weight_decay", .01)))
    amp = selected.type == "cuda" and bool(settings.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
        torch.cuda.synchronize(selected)
    started, visits = time.monotonic(), 0
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            check()
            batch = {key: value.to(selected) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.autocast(selected.type, dtype=torch.float16, enabled=amp):
                loss = forecast_loss(model(**model_inputs(batch)), batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite directional pilot loss")
            size = batch["history"].shape[0]
            start = index // accumulation * accumulation * batch_size
            group = min(accumulation * batch_size, len(loader.dataset) - start)
            scaler.scale(loss * size / group).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 1.)), error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            visits += size
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
    return {"examples_per_second": visits / (time.monotonic() - started), "examples": visits,
        "config_sha256": config_fingerprint(config), "device": str(selected), "epochs": 3,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(selected) if selected.type == "cuda" else None,
        "weights_retained": False}


def freeze_epochs(profiles, already_charged_seconds, training_count=40000):
    if set(profiles) != set(VARIANTS):
        raise ValueError("Profile both controlled variants before freezing the schedule")
    slower = min(value["examples_per_second"] for value in profiles.values())
    if not math.isfinite(slower) or slower <= 0:
        raise ValueError("Both variants need finite positive pilot throughput")
    epoch_seconds = 6 * training_count / slower * 1.35
    usable = GPU_SECONDS - max(QUALIFICATION_RESERVE, already_charged_seconds)
    epochs = min(15, int(usable // epoch_seconds))
    if epochs < 1:
        raise DirectionBudgetExhausted("The measured pilot leaves insufficient budget for one uniform epoch")
    return {"epochs": epochs, "slower_examples_per_second": slower,
        "projected_six_run_epoch_seconds_including_35pct_overhead": epoch_seconds,
        "qualification_reserve_seconds": QUALIFICATION_RESERVE,
        "projected_training_seconds": epoch_seconds * epochs, "gpu_hour_ceiling": 12}


def compare_seed_records(records, bootstrap=1000, seed=42):
    """A scenario is resampled once with all three paired seeds kept together."""
    import numpy as np
    from adp.protocol import paired_comparison
    if bootstrap < 1:
        raise ValueError("Bootstrap replicates must be positive")
    expected = {f"{variant}-{number}" for variant in VARIANTS for number in SEEDS}
    if set(records) != expected:
        raise ValueError("Comparison requires all six completed paired runs")
    indexed = {}
    for name, rows in records.items():
        table = {row["scenario_id"]: row for row in rows}
        if len(table) != len(rows) or not table:
            raise ValueError("Comparison requires one record per scenario")
        indexed[name] = table
    ids = sorted(next(iter(indexed.values())))
    if any(set(table) != set(ids) for table in indexed.values()):
        raise ValueError("All six runs must contain identical scenario sets")
    draws = np.random.default_rng(seed).integers(len(ids), size=(bootstrap, len(ids)))
    metrics = {}
    for key in ("jointFDE", "jointADE", "focal_minFDE", "focal_minADE", "minFDE", "minADE", "miss_rate", "NLL", "footprint_collision_proxy"):
        deltas = []
        for sid in ids:
            values = []
            for number in SEEDS:
                left, right = indexed[f"directional-{number}"][sid].get(key), indexed[f"control-{number}"][sid].get(key)
                values.append(float(np.mean(left)) - float(np.mean(right)) if left is not None and right is not None
                    and np.size(left) and np.size(right) else np.nan)
            deltas.append(float(np.mean(values)) if np.isfinite(values).all() else np.nan)
        values = np.asarray(deltas)
        valid = np.isfinite(values)
        sampled = values[draws]
        counts = np.isfinite(sampled).sum(1)
        estimates = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
        metrics[key] = {"delta": float(values[valid].mean()) if valid.any() else None,
            "interval95": np.percentile(estimates, [2.5, 97.5]).tolist() if estimates.size else None,
            "eligible_scenarios": int(valid.sum())}
    return {"version": 1, "development_only": True, "scenario_count": len(ids), "seeds": list(SEEDS),
        "bootstrap": bootstrap, "bootstrap_seed": seed, "unit": "scenario, clustered across all three paired seeds",
        "direction": "directional minus control; negative error differences favor directional",
        "checkpoint_policy": "final fixed epoch, never per-run best checkpoint", "primary_metric": "jointFDE",
        "metrics": metrics, "individual_seeds": {str(number): paired_comparison(
            records[f"directional-{number}"], records[f"control-{number}"], bootstrap, seed) for number in SEEDS},
        "automatic_promotion": False, "interpretation": "Development-set evidence; no locked-holdout or calibration claim"}


def _write_experiment_report(root, comparison, ledger):
    primary = comparison["metrics"]["jointFDE"]
    interval = primary["interval95"]
    conclusion = ("No eligible complete-future scenes for the primary metric." if interval is None else
        "Directional features improved selection-set joint minFDE in this comparison." if interval[1] < 0 else
        "Directional features worsened selection-set joint minFDE in this comparison." if interval[0] > 0 else
        "The selection-set comparison is inconclusive: the 95% interval includes zero.")
    lines = ["# Map-direction experiment — development evidence", "", conclusion, "",
        f"Six paired runs; seeds 42, 43, 44; {comparison['epochs']} fixed epochs each. Final last.pt checkpoints were compared.",
        "All runs use the same eight-input map encoder. Control tangent channels are zero; directional channels contain signed unit lane tangents.",
        "", f"Primary directional-minus-control joint minFDE: {primary['delta']} m; 95% interval: {interval}.",
        f"Eligible scenarios: {primary['eligible_scenarios']}. The 1,000 bootstrap replicates resample whole scenarios with all three seed pairs together.",
        "", "| Seed | Joint minFDE difference (m) | 95% interval |", "| --- | ---: | --- |"]
    for seed, result in comparison["individual_seeds"].items():
        metric = result["metrics"]["jointFDE"]
        lines.append(f"| {seed} | {metric['delta']} | {metric['interval95']} |")
    lines.extend(["", f"Conservatively charged compute: {ledger['elapsed_seconds'] / 3600:.4f} hours of the separate 12-hour ceiling.",
        "Learning curves are saved in each runs/<variant>-<seed>/training.png. Full secondary errors, likelihood, overlap proxies and coverage are in comparison.json and scores/.",
        "", "This uses 40,000 training scenarios and the existing 2,000-scenario selection set. The training-only pilot verified lane order against successor topology.",
        "Calibration and locked evaluation labels were not read. No calibrators were fitted, no weights were promoted, and risk remains experimental.",
        "These intervals describe development-set evidence after configuration selection. They are not a fresh-holdout performance claim or a driving-safety claim.",
        "", f"Protocol SHA256: {comparison['protocol_sha256']}", f"Schedule SHA256: {comparison['schedule_sha256']}", ""])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _score(checkpoint, cache, directory, ids):
    import torch
    from adp.data import CachedDataset, collate_scenes
    from adp.metrics import evaluate_predictions, summarize_records
    from adp.model import ForecastModel, model_inputs
    from adp.training import load_checkpoint
    from adp.reporting import write_report
    torch.set_num_threads(2)
    ensure_space(directory, additional_bytes=32 * 1024**2)
    identity = {"checkpoint_sha256": manifest_hash(checkpoint), "manifest_sha256": manifest_hash(cache / "manifest.json"),
                "split": "validation", "device": "cpu", "calibration_applied": False}
    request = directory / "request.json"
    _freeze(request, identity)
    if (directory / "records.json").exists() and (directory / "metrics.json").exists():
        rows = read_json(directory / "records.json")
        report = read_json(directory / "metrics.json")
        if report.get("direction_study_identity") != identity or [row["scenario_id"] for row in rows] != ids:
            raise ValueError("Cached selection scores differ from the frozen complete request")
        return rows, report
    saved = load_checkpoint(checkpoint)
    model = ForecastModel(saved["config"]["model"]).eval()
    model.load_state_dict(saved["model_state_dict"])
    partial = directory / "records.partial.json"
    rows = read_json(partial) if partial.exists() else []
    if [row["scenario_id"] for row in rows] != ids[:len(rows)]:
        raise ValueError("Partial selection scores have invalid scenario membership")
    dataset = CachedDataset(cache, "validation")
    if dataset.ids != ids or len(rows) > len(ids):
        raise ValueError("Selection cache membership differs from the frozen protocol")
    with torch.inference_mode():
        for index in range(len(rows), len(dataset)):
            scene = dataset[index]
            batch = collate_scenes([scene])
            rows.extend(evaluate_predictions(model(**model_inputs(batch)), batch))
            if (index + 1) % 50 == 0:
                ensure_space(directory, additional_bytes=16 * 1024**2)
                write_json(partial, rows)
                print(f"Selection CPU score {directory.name}: {index + 1}/{len(dataset)}", flush=True)
    ensure_space(directory, additional_bytes=32 * 1024**2)
    report = summarize_records(rows, bootstrap=0, seed=42)
    report.update(direction_study_identity=identity, development_only=True, calibrated_risk_validated=False)
    write_report(directory, report, rows)
    partial.unlink(missing_ok=True)
    return rows, report


def run_direction_study(reference_root, root="D:/InteractionLab/direction-study-v1", device="cuda"):
    from filelock import FileLock, Timeout
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    if root == reference_root or root in reference_root.parents or reference_root in root.parents:
        raise ValueError("Use a separate experiment root outside the reference study")
    completion = reference_root / "pipeline_status.json"
    if not completion.exists() or read_json(completion).get("status") != "completed" or read_json(completion).get("phase") != "finished":
        return {"status": "waiting_for_reference", "development_only": True,
                "reason": "Reference pipeline must finish successfully before any follow-up GPU work"}
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / "direction.lock"), timeout=0):
            return _run(reference_root, root, device)
    except Timeout as error:
        raise RuntimeError("A direction-study runner already owns this root") from error


def _run(reference_root, root, device):
    from adp.training import load_checkpoint, train
    from adp.reporting import plot_training
    status = {"status": "incomplete_experiment", "development_only": True, "runs": {}}
    try:
        ensure_space(root, additional_bytes=128 * 1024**2)
        protocol = _prepare(reference_root, root)
        configurations = direction_configs(protocol["base_config"])
        pilot_path = root / "pilot.json"
        if (pilot_path.exists() or list(root.glob("pilot-*.json")) or list((root / "runs").glob("*/last.pt"))) and not (root / "direction_ledger.json").exists():
            raise ValueError("Compute ledger is missing; existing experiment work cannot receive a fresh budget")
        if not pilot_path.exists():
            audit = audit_lane_order(reference_root, protocol["pilot_ids"])
            write_json(root / "lane-order-audit.json", audit)
            if not audit["qualified"]:
                raise ValueError("Training-pilot successor ordering is unqualified; inspect lane-order-audit.json")
            profiles = {}
            for variant in VARIANTS:
                path = root / f"pilot-{variant}.json"
                config = next(item["config"] for item in configurations if item["name"] == f"{variant}-42")
                if path.exists():
                    profile = read_json(path)
                    if profile["config_sha256"] != config_fingerprint(config):
                        raise ValueError("Frozen pilot configuration changed")
                else:
                    with charged_operation(root, f"pilot-{variant}") as (_, check):
                        profile = _profile(reference_root, config, device, check)
                    write_json(path, profile)
                profiles[variant] = profile
            charged = read_json(root / "direction_ledger.json")["elapsed_seconds"]
            value = {"version": 1, "profiles": profiles, "lane_order": audit, "charged_seconds_at_freeze": charged,
                     "protocol_sha256": manifest_hash(root / "protocol.json"),
                     **freeze_epochs(profiles, charged, len(protocol["train_ids"]))}
            _freeze(pilot_path, value)
        pilot = read_json(pilot_path)
        expected_epochs = freeze_epochs(pilot["profiles"], pilot["charged_seconds_at_freeze"], len(protocol["train_ids"]))
        if (pilot.get("protocol_sha256") != manifest_hash(root / "protocol.json") or not pilot["lane_order"]["qualified"]
                or any(pilot[key] != value for key, value in expected_epochs.items())):
            raise ValueError("Pilot schedule or provenance differs from the frozen protocol")
        for variant, profile in pilot["profiles"].items():
            config = next(item["config"] for item in configurations if item["name"] == f"{variant}-42")
            if profile["config_sha256"] != config_fingerprint(config):
                raise ValueError("Pilot variant configuration differs from the frozen protocol")
        schedule = _freeze(root / "schedule.json", {"version": 1, "epochs": pilot["epochs"],
            "configs": configurations, "pilot_sha256": manifest_hash(pilot_path),
            "protocol_sha256": manifest_hash(root / "protocol.json")})
        status.update(epochs=schedule["epochs"], protocol_sha256=manifest_hash(root / "protocol.json"),
                      schedule_sha256=manifest_hash(root / "schedule.json"))
        write_json(root / "experiment.json", status)
        for item in configurations:
            name, config = item["name"], item["config"]
            directory = root / "runs" / name
            checkpoint = directory / "last.pt"
            while True:
                saved = load_checkpoint(checkpoint) if checkpoint.exists() else None
                if saved and saved["completed_epochs"] > schedule["epochs"]:
                    raise ValueError("Checkpoint exceeds the frozen fixed-epoch schedule")
                if saved and saved["completed_epochs"] == schedule["epochs"] and not saved.get("training_progress"):
                    if (saved["config"] != config or saved["manifest_hash"] != protocol["manifest_sha256"]
                            or saved.get("preprocessing_hash") != protocol["preprocessing_sha256"]):
                        raise ValueError("Completed checkpoint configuration or manifest changed")
                    break
                ensure_space(root, additional_bytes=64 * 1024**2)
                with charged_operation(root, f"train-{name}", 900) as (remaining, _):
                    train(config, protocol["cache_dir"], directory, schedule["epochs"], device,
                          checkpoint if checkpoint.exists() else None, max_runtime_seconds=min(900., remaining))
            status["runs"][name] = {"checkpoint": str(checkpoint), "last_sha256": manifest_hash(checkpoint), "completed": True}
            write_json(root / "experiment.json", status)
            plot_training(directory)
        all_records, reports = {}, {}
        for name, run in status["runs"].items():
            all_records[name], reports[name] = _score(Path(run["checkpoint"]), Path(protocol["cache_dir"]),
                root / "scores" / name, protocol["selection_ids"])
        comparison = compare_seed_records(all_records)
        comparison["seed_metrics"] = reports
        comparison["epochs"] = schedule["epochs"]
        comparison["protocol_sha256"] = manifest_hash(root / "protocol.json")
        comparison["schedule_sha256"] = manifest_hash(root / "schedule.json")
        write_json(root / "comparison.json", comparison)
        ledger = read_json(root / "direction_ledger.json")
        _write_experiment_report(root, comparison, ledger)
        status.update(status="completed", comparison="comparison.json", report="REPORT.md", gpu_ledger=ledger)
        write_json(root / "experiment.json", status)
        return status
    except DirectionBudgetExhausted as error:
        status.update(reason=str(error), budget_exhausted=True, resumable=False, checkpoints_preserved=True)
        write_json(root / "experiment.json", status)
        return status
    except BaseException as error:
        status.update(reason=str(error) or type(error).__name__, resumable=True)
        write_json(root / "experiment.json", status)
        raise
