"""Single-device training, portable checkpoints, and held-out calibration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adp.calibration import fit_platt, fit_temperature, mode_targets


def environment_report() -> dict:
    """Inspect the current interpreter and driver without changing the system."""
    packages = {}
    for name in ("torch", "numpy", "pandas", "pyarrow", "av2", "fastapi"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver = None
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            driver = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            driver = str(exc)
    return {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu_driver": driver,
    }


def resolve_device(device: str = "auto", require_cuda: bool = False) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda, or cuda:<index>.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a compatible CUDA PyTorch wheel or select --device cpu.")
    if require_cuda and resolved.type != "cuda":
        raise RuntimeError("The GPU smoke test requires a CUDA device; no GPU is available.")
    return resolved


def gpu_smoke_test(device: str = "auto", config: dict | None = None) -> dict:
    """Exercise actual CUDA kernels, optionally including model forward/backward."""
    selected = resolve_device(device, require_cuda=True)
    with torch.cuda.device(selected):
        values = torch.randn(64, 64, device=selected, requires_grad=True)
        (values @ values.T).square().mean().backward()
        if not torch.isfinite(values.grad).all():
            raise RuntimeError("CUDA smoke test produced non-finite gradients.")
        if config is not None:
            from adp.model import ForecastModel, forecast_loss, model_inputs

            batch = _smoke_batch(selected)
            model = ForecastModel(config.get("model", config)).to(selected)
            outputs = model(**model_inputs(batch))
            loss, _ = _loss_parts(forecast_loss(outputs, batch))
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("The model CUDA smoke test produced non-finite loss.")
        torch.cuda.synchronize(selected)
        properties = torch.cuda.get_device_properties(selected)
    return {
        "ok": True, "device": str(selected), "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(selected)),
        "total_memory_bytes": properties.total_memory,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "model_checked": config is not None,
    }


def _smoke_batch(device: torch.device) -> dict:
    # Synthetic inputs validate kernels and shapes only; they are never evaluation results.
    history = torch.zeros(1, 2, 50, 8, device=device)
    history[..., 5:7] = 1
    history[..., 7] = torch.linspace(-1, 0, 50, device=device)
    return {
        "history": history,
        "history_mask": torch.ones(1, 2, 50, dtype=torch.bool, device=device),
        "actor_mask": torch.ones(1, 2, dtype=torch.bool, device=device),
        "dynamic_mask": torch.ones(1, 2, dtype=torch.bool, device=device),
        "actor_types": torch.zeros(1, 2, dtype=torch.long, device=device),
        "future": torch.zeros(1, 2, 60, 2, device=device),
        "future_mask": torch.ones(1, 2, 60, dtype=torch.bool, device=device),
        "map_points": torch.zeros(1, 1, 20, 2, device=device),
        "map_features": torch.zeros(1, 1, 4, device=device),
        "map_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        "sdf": torch.ones(1, 1, 256, 256, device=device),
    }


def manifest_hash(cache_dir: str | Path) -> str:
    manifest = Path(cache_dir) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest}. Preprocess the selected dataset first.")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _loss_parts(result) -> tuple[torch.Tensor, dict]:
    if isinstance(result, torch.Tensor):
        return result, {"loss": result}
    if isinstance(result, tuple):
        loss, metrics = result
        return loss, {"loss": loss, **metrics}
    return result["loss"], result


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        # Different machines can have different GPU counts; restore only matching devices.
        for index, rng in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng.cpu(), index)


def _portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable(item) for item in value)
    return value


def _atomic_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(_portable(checkpoint), temporary)
        # Windows scanners can briefly hold the destination after a checkpoint read.
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError("Unsupported checkpoint format; expected format_version 1 or 2.")
    return checkpoint


def _objective_values(parts: dict) -> dict:
    values = {key: value for key, value in parts.items() if not key.endswith("_count")}
    return {key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in values.items()}


def _validation_displacement(outputs: dict, batch: dict) -> dict:
    totals = {"joint_sum": 0.0, "joint_count": 0, "focal_sum": 0.0, "focal_count": 0}
    if "means" not in outputs:
        return totals
    future = batch["future"].float()
    dynamic = batch["dynamic_mask"].bool() & batch["actor_mask"].bool()
    complete = (batch["future_mask"].bool() & torch.isfinite(future).all(-1)).all(-1)
    scored = batch.get("scored_mask", dynamic).bool() & dynamic & complete
    distances = torch.linalg.vector_norm(outputs["means"].float()[..., -1, :]-future[:, None, :, -1], dim=-1)
    joint = torch.where(scored[:, None], distances, 0).sum(-1)/scored.sum(-1).clamp_min(1)[:, None]
    eligible = scored.any(-1)
    focal_eligible = complete[:, 0] & dynamic[:, 0]
    totals.update(joint_sum=float(joint.amin(-1)[eligible].sum()), joint_count=int(eligible.sum()),
                  focal_sum=float(distances[:, :, 0].amin(-1)[focal_eligible].sum()),
                  focal_count=int(focal_eligible.sum()))
    return totals


def _selection_key(selection: str, validation_loss: float, joint_minfde: float | None,
                   focal_minfde: float | None, epoch: int) -> tuple:
    if selection == "validation_loss":
        return validation_loss, epoch
    if selection != "joint_minFDE":
        raise ValueError("training.checkpoint_selection must be validation_loss or joint_minFDE")
    if joint_minfde is None or focal_minfde is None:
        raise ValueError("joint_minFDE checkpoint selection requires complete scored dynamic validation actors and focal futures")
    return joint_minfde, focal_minfde, epoch


def train(
    config: dict, cache_dir: str | Path, output_dir: str | Path,
    epochs: int = 20, device: str = "auto", resume: str | Path | None = None,
    overfit_batches: int | None = None,
    max_runtime_seconds: float | None = None,
) -> dict:
    """Resume committed optimizer groups; budget checks bracket groups/validation batches.

    A slice can overrun by one optimizer group or validation batch, plus startup
    and checkpoint I/O. Uncommitted validation is restarted without training replay.
    """
    from adp.data import CachedDataset, collate_scenes
    from adp.model import ForecastModel, forecast_loss, model_inputs

    started = time.monotonic()
    if epochs < 1 or (overfit_batches is not None and overfit_batches < 1):
        raise ValueError("Epochs and overfit_batches must be positive.")
    if max_runtime_seconds is not None and (not math.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0):
        raise ValueError("max_runtime_seconds must be finite and positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("last.pt", "best.pt")) and (
        resume is None or Path(resume).resolve().parent != output_dir.resolve()
    ):
        raise ValueError("Output directory already contains checkpoints. Resume that run or choose another output directory.")
    config = json.loads(json.dumps(config))
    settings = config.get("training", {})
    selection = settings.get("checkpoint_selection", "validation_loss")
    if selection not in ("validation_loss", "joint_minFDE"):
        raise ValueError("training.checkpoint_selection must be validation_loss or joint_minFDE")
    seed = int(config.get("seed", settings.get("seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected = resolve_device(device)
    if selected.type == "cuda":
        gpu_smoke_test(str(selected), config)
        torch.cuda.reset_peak_memory_stats(selected)
    data_hash = manifest_hash(cache_dir)
    from adp.data import preprocessing_hash, read_json
    prep_hash = preprocessing_hash(cache_dir)
    audit_path = Path(cache_dir)/"audit.json"
    data_audit = read_json(audit_path) if audit_path.exists() else {}
    training_data = CachedDataset(cache_dir, "train")
    validation_data = CachedDataset(cache_dir, "validation")
    batch_size = int(settings.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("training.batch_size must be positive.")
    if overfit_batches is not None:
        training_data = Subset(training_data, range(min(len(training_data), overfit_batches * batch_size)))
    if not len(training_data) or not len(validation_data):
        raise ValueError("Both train and validation splits need at least one cached scenario.")
    loader_options = {
        "batch_size": batch_size, "collate_fn": collate_scenes,
        "num_workers": int(settings.get("num_workers", 0)), "pin_memory": selected.type == "cuda",
    }
    # Sampler order and worker seeds must not consume the model's dropout RNG.
    sampler_generator = torch.Generator().manual_seed(seed + 1)
    loader_generator = torch.Generator().manual_seed(seed + 2)
    validation_generator = torch.Generator().manual_seed(seed + 3)
    validation_loader = DataLoader(validation_data, shuffle=False, generator=validation_generator, **loader_options)
    model = ForecastModel(config.get("model", config)).to(selected)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings.get("learning_rate", 3e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-2)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = selected.type == "cuda" and bool(settings.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation = int(settings.get("accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("training.accumulation_steps must be positive.")
    checkpoint_seconds = float(settings.get("checkpoint_seconds", 300))
    if checkpoint_seconds <= 0:
        raise ValueError("training.checkpoint_seconds must be positive.")
    minimum_free_gib = float(settings.get("minimum_free_gib", 0))
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("training.minimum_free_gib must be finite and nonnegative")
    start_epoch, global_step, best_score, history = 0, 0, math.inf, []
    risk_supervision_observations = 0
    best_selection, progress = None, None
    training_seconds, examples_seen, objective_totals = 0.0, 0, {}
    if resume is not None:
        saved = load_checkpoint(resume)
        required = {"optimizer_state_dict", "scheduler_state_dict", "rng_state", "completed_epochs", "global_step"}
        if not required.issubset(saved):
            raise ValueError("Resume requires a training last.pt checkpoint, not an inference model bundle.")
        if saved["manifest_hash"] != data_hash:
            raise ValueError("Resume refused: the dataset manifest differs from the checkpoint.")
        if "preprocessing_hash" in saved and saved["preprocessing_hash"] != prep_hash:
            raise ValueError("Resume refused: preprocessing differs from the checkpoint.")
        if {key: value for key, value in saved["config"].items() if key != "data"} != {
            key: value for key, value in config.items() if key != "data"
        }:
            raise ValueError("Resume refused: configuration differs from the checkpoint. Use the saved configuration.")
        if saved.get("overfit_batches") != overfit_batches:
            raise ValueError("Resume refused: overfit_batches differs from the checkpoint.")
        if saved.get("partial_epoch") and saved.get("format_version") == 1:
            raise ValueError("Resume refused: legacy partial-epoch checkpoints have no committed sample cursor. Resume an epoch-boundary checkpoint.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        if scheduler.T_max != epochs:
            # Extending a run extends its cosine horizon from the restored epoch.
            scheduler.T_max = epochs
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * (1 + math.cos(math.pi * scheduler.last_epoch / epochs)) / 2
        if saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch = int(saved["completed_epochs"])
        global_step = int(saved["global_step"])
        if global_step:
            # The scheduler's first-step warning flag is not in optimizer state_dict.
            optimizer._opt_called = True
        best_score = float(saved["best_validation_loss"])
        history = saved.get("history", [])
        risk_supervision_observations = int(saved.get("risk_supervision_observations", 0))
        best_selection = saved.get("best_selection")
        if best_selection is None and math.isfinite(best_score):
            best_selection = {"metric": selection, "key": [best_score], "epoch": start_epoch}
        statistics = saved.get("statistics", {})
        examples_seen = int(statistics.get("examples_seen", 0))
        training_seconds = float(statistics.get("training_seconds", 0))
        objective_totals = statistics.get("objective_totals", {})
        if saved.get("format_version") == 2:
            sampler_generator.set_state(saved["sampler_generator_state"].cpu())
            loader_generator.set_state(saved["loader_generator_state"].cpu())
            validation_generator.set_state(saved["validation_generator_state"].cpu())
            progress = saved.get("training_progress")
            if saved.get("partial_epoch"):
                if progress is None or progress["epoch"] != start_epoch:
                    raise ValueError("Resume refused: partial checkpoint is missing its committed sample order/cursor.")
                order, cursor = progress["order"], progress["cursor"]
                if (sorted(order) != list(range(len(training_data))) or not 0 <= cursor <= len(order)
                        or (cursor != len(order) and cursor % (accumulation * batch_size))):
                    raise ValueError("Resume refused: partial checkpoint has an invalid committed sample order/cursor.")
        _restore_rng(saved["rng_state"])
        if saved.get("partial_epoch"):
            print(f"Resuming epoch {start_epoch + 1} after {progress['cursor']} committed examples.", flush=True)
    last_save = time.monotonic()

    def budget_expired():
        return max_runtime_seconds is not None and time.monotonic() - started >= max_runtime_seconds

    def save_checkpoint(path, payload):
        if minimum_free_gib:
            from adp.research import ensure_space

            def tensor_bytes(value):
                if isinstance(value, torch.Tensor):
                    return value.numel()*value.element_size()
                if isinstance(value, dict):
                    return sum(tensor_bytes(item) for item in value.values())
                if isinstance(value, (list, tuple)):
                    return sum(tensor_bytes(item) for item in value)
                return 0

            reserve = max(64*1024**2, 2*tensor_bytes(payload) + 1024**2)
            ensure_space(output_dir, minimum_free_gib, reserve)
        _atomic_checkpoint(path, payload)

    def checkpoint(completed_epochs: int, partial_epoch: bool = False) -> dict:
        return {
            "format_version": 2, "config": config, "manifest_hash": data_hash,
            "preprocessing_hash": prep_hash, "data_audit": data_audit,
            "risk_supervision_observations": risk_supervision_observations,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "completed_epochs": completed_epochs,
            "partial_epoch": partial_epoch, "global_step": global_step,
            "best_validation_loss": best_score, "history": history,
            "best_selection": best_selection, "training_progress": progress if partial_epoch else None,
            "sampler_generator_state": sampler_generator.get_state(),
            "loader_generator_state": loader_generator.get_state(),
            "validation_generator_state": validation_generator.get_state(),
            "statistics": {"examples_seen": examples_seen, "training_seconds": training_seconds,
                           "objective_totals": objective_totals},
            "rng_state": _rng_state(), "overfit_batches": overfit_batches,
            "torch_version": str(torch.__version__),
        }

    completed_epochs, budget_exhausted = start_epoch, False
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if progress is None:
            order = (torch.randperm(len(training_data), generator=sampler_generator).tolist()
                     if overfit_batches is None else list(range(len(training_data))))
            progress = {"epoch": epoch, "order": order, "cursor": 0, "loss_total": 0.0,
                        "example_count": 0, "component_totals": {},
                        "loader_epoch_state": loader_generator.get_state()}
        order, cursor = progress["order"], progress["cursor"]
        training_total, training_count = progress["loss_total"], progress["example_count"]
        component_totals = progress["component_totals"]
        if budget_expired():
            budget_exhausted = True
            save_checkpoint(output_dir / "last.pt", checkpoint(epoch, partial_epoch=True))
            break
        loader_generator.set_state(progress["loader_epoch_state"].cpu())
        training_loader = DataLoader(training_data, sampler=order[cursor:], generator=loader_generator, **loader_options)
        last_tick = time.monotonic()
        for local_batch_index, batch in enumerate(training_loader):
            batch_index = cursor // batch_size + local_batch_index
            if batch_index % accumulation == 0 and budget_expired():
                budget_exhausted = True
                save_checkpoint(output_dir / "last.pt", checkpoint(epoch, partial_epoch=True))
                break
            batch = _move_batch(batch, selected)
            count = int(batch["history"].shape[0])
            with torch.autocast(device_type=selected.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(**model_inputs(batch))
                loss, parts = _loss_parts(forecast_loss(outputs, batch))
            risk_supervision_observations += int(parts.get("risk_supervision_count", 0))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {batch_index + 1}.")
            group_start = (batch_index // accumulation) * accumulation
            group_examples = min(accumulation * batch_size, len(training_data) - group_start * batch_size)
            scaler.scale(loss * count / group_examples).backward()
            boundary = (batch_index + 1) % accumulation == 0 or training_count + count == len(training_data)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings.get("gradient_clip", 1.0)), error_if_nonfinite=not use_amp,
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += int(scaler.get_scale() >= previous_scale)
            training_total += float(loss.detach()) * count
            training_count += count
            examples_seen += count
            for name, value in _objective_values(parts).items():
                component_totals[name] = component_totals.get(name, 0.0) + value * count
                objective_totals[name] = objective_totals.get(name, 0.0) + value * count
            now = time.monotonic()
            training_seconds += now - last_tick
            last_tick = now
            if boundary:
                progress.update(cursor=training_count, loss_total=training_total, example_count=training_count)
            budget_exhausted = boundary and budget_expired()
            if boundary and (time.monotonic() - last_save >= checkpoint_seconds or budget_exhausted):
                save_checkpoint(output_dir / "last.pt", checkpoint(epoch, partial_epoch=True))
                last_save = time.monotonic()
            if budget_exhausted:
                break
        if budget_exhausted:
            break
        model.eval()
        if "validation_epoch_state" not in progress:
            progress["validation_epoch_state"] = validation_generator.get_state()
            progress["validation_rng_state"] = _rng_state()
        validation_generator.set_state(progress["validation_epoch_state"].cpu())
        _restore_rng(progress["validation_rng_state"])
        validation_total, validation_count = 0.0, 0
        validation_components = {}
        displacement = {"joint_sum": 0.0, "joint_count": 0, "focal_sum": 0.0, "focal_count": 0}
        with torch.inference_mode():
            for batch in validation_loader:
                if budget_expired():
                    budget_exhausted = True
                    break
                batch = _move_batch(batch, selected)
                outputs = model(**model_inputs(batch))
                loss, parts = _loss_parts(forecast_loss(outputs, batch))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite validation loss at epoch {epoch + 1}.")
                count = int(batch["history"].shape[0])
                validation_total += float(loss) * count
                validation_count += count
                for name, value in _objective_values(parts).items():
                    validation_components[name] = validation_components.get(name, 0.0) + value * count
                for name, value in _validation_displacement(outputs, batch).items():
                    displacement[name] += value
                if budget_expired():
                    budget_exhausted = True
                    break
        if budget_exhausted:
            validation_generator.set_state(progress["validation_epoch_state"].cpu())
            _restore_rng(progress["validation_rng_state"])
            save_checkpoint(output_dir / "last.pt", checkpoint(epoch, partial_epoch=True))
            break
        score = validation_total / validation_count
        joint_minfde = (displacement["joint_sum"] / displacement["joint_count"]
                        if displacement["joint_count"] else None)
        focal_minfde = (displacement["focal_sum"] / displacement["focal_count"]
                        if displacement["focal_count"] else None)
        candidate_key = _selection_key(selection, score, joint_minfde, focal_minfde, epoch + 1)
        scheduler.step()
        history.append({
            "epoch": epoch + 1, "train_loss": training_total / training_count,
            "validation_loss": score, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_objective": {name: value / training_count for name, value in component_totals.items()},
            "validation_objective": {name: value / validation_count for name, value in validation_components.items()},
            "validation_joint_minFDE": joint_minfde, "validation_focal_minFDE": focal_minfde,
            "validation_joint_scenes": displacement["joint_count"],
            "validation_focal_scenes": displacement["focal_count"],
        })
        improved = best_selection is None or candidate_key < (*best_selection["key"], best_selection["epoch"])
        best_score = min(best_score, score)
        if improved:
            best_selection = {"metric": selection, "key": list(candidate_key[:-1]), "epoch": epoch + 1}
        progress = None
        completed_epochs = epoch + 1
        saved = checkpoint(epoch + 1)
        save_checkpoint(output_dir / "last.pt", saved)
        if improved:
            save_checkpoint(output_dir / "best.pt", saved)
        _atomic_json(output_dir / "history.json", {"manifest_hash": data_hash, "epochs": history})
        last_save = time.monotonic()
        print(json.dumps(history[-1]), flush=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    if not best_path.is_file() and resume is not None:
        best_path = Path(resume).parent / "best.pt"
    return {
        "checkpoint": str(last_path if last_path.is_file() else Path(resume)),
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "completed_epochs": completed_epochs, "best_validation_loss": best_score if math.isfinite(best_score) else None,
        "best_selection": best_selection,
        "budget_exhausted": budget_exhausted,
        "manifest_hash": data_hash, "history": history, "device": str(selected),
        "elapsed_seconds": time.monotonic() - started,
        "training_examples_per_second": examples_seen / training_seconds if training_seconds else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(selected) if selected.type == "cuda" else None,
    }


def calibrate(
    checkpoint: str | Path, cache_dir: str | Path, output_dir: str | Path,
    device: str = "auto", budget_check=None,
) -> dict:
    """Fit calibration on its reserved split; never use the evaluation split."""
    from adp.data import CachedDataset, collate_scenes
    from adp.metrics import risk_labels
    from adp.model import ForecastModel, model_inputs

    checkpoint = Path(checkpoint)
    saved = load_checkpoint(checkpoint)
    data_hash = manifest_hash(cache_dir)
    if saved["manifest_hash"] != data_hash:
        raise ValueError("Calibration refused: the dataset manifest differs from the checkpoint.")
    from adp.data import preprocessing_hash
    if "preprocessing_hash" in saved and saved["preprocessing_hash"] != preprocessing_hash(cache_dir):
        raise ValueError("Calibration refused: preprocessing differs from the checkpoint.")
    selected = resolve_device(device)
    model = ForecastModel(saved["config"].get("model", saved["config"])).to(selected)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    data = CachedDataset(cache_dir, "calibration")
    if not len(data):
        raise ValueError("The calibration split is empty.")
    loader = DataLoader(data, batch_size=int(saved["config"].get("training", {}).get("batch_size", 4)), collate_fn=collate_scenes)
    mode_logits, mode_labels, risk_logits, risk_targets = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            if budget_check:
                budget_check()
            batch = _move_batch(batch, selected)
            outputs = model(**model_inputs(batch))
            targets, eligible = mode_targets(outputs, batch)
            mode_logits.append(outputs["logits"][eligible].cpu())
            mode_labels.append(targets[eligible].cpu())
            labels, risk_eligible = risk_labels(
                batch["future"], batch["future_mask"], batch["dynamic_mask"] & batch["actor_mask"],
            )
            risk_eligible = risk_eligible & batch["actor_mask"].bool()
            risk_logits.append(outputs["risk_logits"][risk_eligible].cpu())
            risk_targets.append(labels[risk_eligible].cpu())
    mode_fit = fit_temperature(torch.cat(mode_logits), torch.cat(mode_labels))
    risk_fit = fit_platt(torch.cat(risk_logits), torch.cat(risk_targets))
    risk_fit["status"] = ("insufficient_positive_events" if risk_fit["positives"] < 50 else
                          "calibrated_proxy" if risk_fit["fitted"] else "not_fitted")
    calibration = {
        "format_version": 1, "mode_temperature": mode_fit["mode_temperature"],
        "risk_temperature": risk_fit["risk_temperature"], "risk_bias": risk_fit["risk_bias"],
        "manifest_hash": data_hash, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "preprocessing_hash": preprocessing_hash(cache_dir),
        "split": "calibration", "mode": mode_fit, "risk": risk_fit,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration.json"
    _atomic_json(path, calibration)
    return {"path": str(path), **calibration}
