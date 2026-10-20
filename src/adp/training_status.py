"""Read-only progress from two configured study roots; no inference or GPU telemetry."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shutil
import threading
import time
import zipfile

MAX_CHECKPOINT_BYTES = 64 * 1024**2
MAX_JSON_BYTES = 8 * 1024**2
_CONFIGURATIONS = ("full", "no_graph", "no_map", "world_ce")
_OBJECTIVES = ("loss", "nll", "trajectory", "risk", "offroad", "shared_world_ce")


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def _integer(value, default=None):
    number = _number(value)
    return max(0, int(number)) if number is not None else default


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _history(value):
    result = {}
    for row in value[-1000:] if isinstance(value, list) else []:
        if not isinstance(row, dict) or not _integer(row.get("epoch")):
            continue
        epoch = _integer(row["epoch"])
        entry = {"epoch": epoch, "train_loss": _number(row.get("train_loss")),
            "validation_loss": _number(row.get("validation_loss")),
            "joint_minFDE": _number(row.get("validation_joint_minFDE")),
            "focal_minFDE": _number(row.get("validation_focal_minFDE"))}
        for key in ("train_objective", "validation_objective"):
            values = row.get(key) if isinstance(row.get(key), dict) else {}
            entry[key] = {name: _number(values[name]) for name in _OBJECTIVES if name in values}
        result[epoch] = entry
    return [result[epoch] for epoch in sorted(result)]


def _checkpoint_metadata(content):
    import torch
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if len(archive.infolist()) > 2000 or sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise ValueError("Expanded checkpoint exceeds the supported limit")
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") not in (1, 2):
        raise ValueError("Unsupported training checkpoint")
    progress = saved.get("training_progress") if isinstance(saved.get("training_progress"), dict) else {}
    order = progress.get("order")
    settings = saved.get("config", {}).get("training", {})
    statistics = saved.get("statistics") if isinstance(saved.get("statistics"), dict) else {}
    examples, seconds = _number(statistics.get("examples_seen")), _number(statistics.get("training_seconds"))
    return {"completed_epochs": _integer(saved.get("completed_epochs"), 0),
        "partial": bool(saved.get("partial_epoch") or progress),
        "epoch": _integer(progress.get("epoch")), "cursor": _integer(progress.get("cursor")),
        "training_examples": len(order) if isinstance(order, (list, tuple)) else None,
        "batch_size": _integer(settings.get("batch_size"), 4), "global_step": _integer(saved.get("global_step")),
        "examples_per_second": _number(examples / seconds) if examples is not None and seconds is not None and seconds > 0 else None,
        "history": _history(saved.get("history"))}


class TrainingStatus:
    def __init__(self, study_root="D:/InteractionLab/study-v1", direction_root="D:/InteractionLab/direction-study-v1",
                 compute_hosts_path="D:/InteractionLab/remote-compute/hosts.json"):
        self.roots = {"original": Path(study_root), "direction": Path(direction_root)}
        self.compute_hosts_path = Path(compute_hosts_path)
        self._cache = {}
        self._failed = {}
        self._lock = threading.RLock()

    def _read(self, path, checkpoint=False):
        previous = self._cache.get(path)
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            failed = self._failed.get(path)
            if failed and failed["signature"] == signature and time.monotonic() - failed["at"] < 30:
                return (previous["value"] if previous else {}), failed["issue"], (previous["mtime"] if previous else None)
            if previous and previous["signature"] == signature:
                return previous["value"], previous.get("issue"), previous["mtime"]
            limit = MAX_CHECKPOINT_BYTES if checkpoint else MAX_JSON_BYTES
            if not 0 < stat.st_size <= limit:
                raise ValueError("File exceeds supported metadata size")
            # Close the file before deserializing so Windows can replace last.pt atomically.
            with path.open("rb") as stream:
                content = stream.read(limit + 1)
            after = path.stat()
            if len(content) > limit or (after.st_mtime_ns, after.st_size) != signature:
                raise OSError("Metadata changed during read")
            value = _checkpoint_metadata(content) if checkpoint else json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("Metadata must contain an object")
            self._cache[path] = {"signature": signature, "value": value, "mtime": stat.st_mtime, "issue": None}
            self._failed.pop(path, None)
            return value, None, stat.st_mtime
        except FileNotFoundError:
            issue = "Temporarily missing; showing last readable metadata" if previous else None
        except Exception as error:
            issue = f"Unreadable metadata ({type(error).__name__}); " + ("showing last readable metadata" if previous else "progress is unavailable")
            if "signature" in locals():
                self._failed[path] = {"signature": signature, "at": time.monotonic(), "issue": issue}
        if previous:
            return previous["value"], issue, previous["mtime"]
        return {}, issue, None

    def _json(self, root, name, issues):
        value, issue, _ = self._read(root / name)
        if issue:
            issues.append(f"{name}: {issue}")
        return value

    @staticmethod
    def _message(value, roots):
        if not isinstance(value, str):
            return None
        for root in roots.values():
            value = value.replace(str(root), "[configured study]").replace(str(root).replace("\\", "/"), "[configured study]")
        # Errors may contain unrelated absolute paths; status responses expose no filesystem arguments.
        if ":\\" in value or ":/" in value or value.startswith("/") or " /" in value:
            return "The study reported a filesystem or configuration error; inspect its local log."
        return value[:600]

    def _run_row(self, root, name, configuration, seed, planned, training_count, ledger_run, active, study_status, now):
        known = configuration is not None
        directory = root / "runs" / name
        metadata, issue, updated = self._read(directory / "last.pt", checkpoint=True) if known else ({}, None, None)
        history_json, history_issue, _ = self._read(directory / "history.json") if known else ({}, None, None)
        history = _history(history_json.get("epochs"))
        if len(metadata.get("history", [])) > len(history):
            history = metadata["history"]
        if planned is not None:
            history = [entry for entry in history if entry["epoch"] <= planned]
        result = ledger_run.get("result") if isinstance(ledger_run.get("result"), dict) else {}
        completed = metadata.get("completed_epochs")
        if completed is None:
            completed = _integer(result.get("completed_epochs"), max((entry["epoch"] for entry in history), default=0))
        completed_clamped = planned is not None and completed > planned
        completed = min(completed, planned) if planned is not None else completed
        count = metadata.get("training_examples") or training_count
        cursor = metadata.get("cursor")
        clamped = cursor is not None and count is not None and cursor > count
        cursor = min(cursor, count) if cursor is not None and count is not None else cursor
        fraction = cursor / count if cursor is not None and count else None
        complete = bool(planned is not None and completed >= planned)
        state = "completed" if complete else "running" if active and study_status == "running" else "stopped" if active and study_status == "stopped" else "pending"
        if not metadata and issue and not history:
            state = "unavailable"
        partial = bool(metadata.get("partial"))
        stage = "completed" if complete else "validation_pending" if partial and fraction == 1 else "train" if partial else "epoch_boundary" if completed else "pending"
        epoch_number = None if complete or planned is None else min(completed + 1, planned)
        batch = metadata.get("batch_size") or 4
        age = max(0., now - updated) if updated is not None else None
        problems = [item for item in (issue, history_issue,
            "Committed cursor exceeds epoch length; progress was clamped" if clamped else None,
            "Completed epochs exceed the frozen schedule; progress was clamped" if completed_clamped else None,
            "Checkpoint is older than 10 minutes; the status file does not prove a worker is live" if active and not complete and age is not None and age > 600 else None) if item]
        throughput = metadata.get("examples_per_second")
        if throughput is None:
            throughput = _number(result.get("training_examples_per_second"))
        throughput = throughput if throughput is not None and throughput >= 0 else None
        peak = _number(result.get("peak_cuda_allocated_bytes"))
        peak = peak if peak is not None and peak >= 0 else None
        return {"id": name, "name": name if known else f"Selected configuration · seed {seed}",
            "configuration": configuration, "seed": seed, "state": state, "planned_epochs": planned,
            "completed_epochs": completed, "epoch_number": epoch_number, "committed_examples": cursor,
            "training_examples": count, "committed_fraction": fraction, "stage": stage,
            "committed_batches": math.ceil(cursor / batch) if cursor is not None else None,
            "batches_per_epoch": math.ceil(count / batch) if count else None,
            "global_step": metadata.get("global_step"), "examples_per_second": throughput,
            "peak_vram_mib": peak / 1024**2 if peak is not None else None,
            "checkpoint": {"updated_at": _iso(updated) if updated is not None else None, "age_seconds": age,
                           "stale": bool(issue or (age is not None and age > 600))},
            "history": history, "issue": "; ".join(problems) or None}

    def _study(self, identifier, root, now):
        issues = []
        original = identifier == "original"
        schedule = self._json(root, "schedule.json", issues)
        planned = _integer(schedule.get("epochs"))
        planned = planned if planned else None
        pipeline = self._json(root, "pipeline_status.json", issues) if original else {}
        followup = self._json(root, "followup_status.json", issues) if not original else {}
        experiment = self._json(root, "experiment.json", issues) if not original else {}
        ledger = self._json(root, "study_ledger.json" if original else "direction_ledger.json", issues)
        manifest = self._json(root, "cache/manifest.json", issues) if original else {}
        malformed = False
        for value, key, expected, label in ((schedule, "configs", list, "schedule.json"),
                (ledger, "runs", dict, "study_ledger.json"), (ledger, "operations", list, "direction_ledger.json"),
                (experiment, "runs", dict, "experiment.json"), (manifest, "splits", dict, "cache manifest")):
            if key in value and not isinstance(value[key], expected):
                issues.append(f"{label}: invalid {key} metadata")
                malformed = True
        splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
        if "train" in splits and not isinstance(splits["train"], list):
            issues.append("cache manifest: invalid train membership metadata")
            malformed = True
        training_ids = splits.get("train") if isinstance(splits.get("train"), list) else []
        training_count = len(training_ids) or None
        source = pipeline if original else followup or experiment
        raw_status = source.get("status")
        status = ("completed" if raw_status == "completed" else "running" if raw_status == "running" else
            "waiting" if raw_status in ("waiting", "waiting_for_reference") else
            "stopped" if raw_status in ("stopped", "failed", "interrupted", "reference_stopped", "incomplete_experiment") else
            "not_started" if not source and not schedule else "unavailable")
        phase = source.get("phase") if isinstance(source.get("phase"), str) else None
        phase = phase if phase and len(phase) <= 80 and all(c.isalnum() or c == "_" for c in phase) else ("train" if experiment and experiment.get("status") != "completed" else "not_started")
        message = self._message(source.get("reason") or source.get("error"), self.roots)
        run_states = {name: record for name, record in ledger.get("runs", {}).items() if isinstance(record, dict)} if isinstance(ledger.get("runs"), dict) else {}
        if isinstance(ledger.get("runs"), dict) and len(run_states) != len(ledger["runs"]):
            issues.append("study_ledger.json: invalid run metadata")
            malformed = True
        if malformed or (not source and issues):
            status = "unavailable"
        active = None
        if original:
            frozen = self._json(root, "frozen_selection.json", issues)
            winner = frozen.get("configuration") if frozen.get("configuration") in _CONFIGURATIONS else None
            if winner is None:
                winner = next((config for config in _CONFIGURATIONS if any(f"{config}-{seed}" in run_states for seed in (43, 44))), None)
            specifications = [(f"{config}-42", config, 42) for config in _CONFIGURATIONS]
            specifications += [(f"{winner}-{seed}" if winner else f"winner-{seed}", winner, seed) for seed in (43, 44)]
            active = next((name for name, _, _ in specifications if name in run_states and not run_states[name].get("completed")), None)
        else:
            configurations = schedule.get("configs") if isinstance(schedule.get("configs"), list) else []
            valid = {f"{variant}-{seed}" for variant in ("control", "directional") for seed in (42, 43, 44)}
            names = [item.get("name") for item in configurations if isinstance(item, dict) and item.get("name") in valid]
            if set(names) != valid:
                names = [f"{variant}-{seed}" for seed in (42, 43, 44) for variant in ("control", "directional")]
            specifications = [(name, name.rsplit("-", 1)[0], int(name.rsplit("-", 1)[1])) for name in names]
            operations = ledger.get("operations") if isinstance(ledger.get("operations"), list) else []
            last_operation = operations[-1] if operations and isinstance(operations[-1], dict) else {}
            label = last_operation.get("label", "")
            if isinstance(label, str) and label.startswith("train-") and label.removeprefix("train-") in valid:
                active = label.removeprefix("train-")
            experiment_runs = experiment.get("runs") if isinstance(experiment.get("runs"), dict) else {}
            run_states = {name: {"completed": bool(record.get("completed"))} for name, record in experiment_runs.items() if isinstance(record, dict)}
        if original and phase != "train":
            active = None
        if not original and followup and phase not in ("direction_study", "train"):
            active = None
        if not original and experiment.get("status") == "completed" and not followup:
            phase = "finished"
        rows = [self._run_row(root, name, config, seed, planned, training_count, run_states.get(name, {}), name == active, status, now)
                for name, config, seed in specifications]
        if not active and status in ("running", "stopped") and phase in ("train", "direction_study"):
            active = next((row["id"] for row in rows if row["state"] != "completed" and row["checkpoint"]["updated_at"]), None)
        if active:
            for row in rows:
                if row["id"] == active and row["state"] not in ("completed", "unavailable"):
                    row["state"] = "running" if status == "running" else "stopped" if status == "stopped" else "pending"
        if not original and phase == "direction_study" and rows and all(row["state"] == "completed" for row in rows):
            active = None
            phase = "selection_scoring"
            if message is None:
                message = "All six training runs are complete; CPU selection scoring and reporting may still be running."
        charged = _number(ledger.get("elapsed_seconds"))
        charged = max(0., charged) if charged is not None else None
        reserved = None
        if original and active and status == "running":
            reserved = _number(run_states.get(active, {}).get("reserved_seconds"))
        elif not original:
            operations = ledger.get("operations", [])
            last = operations[-1] if isinstance(operations, list) and operations and isinstance(operations[-1], dict) else {}
            if last.get("status") == "reserved":
                reserved = _number(last.get("reserved_seconds"))
        reserved = max(0., reserved) if reserved is not None else None
        limit = 40 if original else 12
        completed = sum(row["completed_epochs"] for row in rows)
        total = planned * 6 if planned else None
        current = next((row for row in rows if row["id"] == active), None)
        partial_fraction = current.get("committed_fraction") if current and current["stage"] in ("train", "validation_pending") else None
        disk = root
        while not disk.exists() and disk.parent != disk:
            disk = disk.parent
        try:
            free = shutil.disk_usage(disk).free / 1024**3
            free = max(0., free)
        except OSError:
            free = None
            issues.append("Storage availability could not be read")
        if free is not None and free < 8 and message is None:
            message = "Available storage is below the 8 GiB reserve; the runner pauses at its next storage check."
        if status == "unavailable" and message is None:
            message = "Current state is unavailable; readable checkpoint progress may be stale."
        if not original and status == "waiting" and message is None:
            message = "Waiting for the original frozen study to finish; no follow-up training has started."
        issues.extend(f"{row['name']}: {row['issue']}" for row in rows if row["issue"])
        return {"id": identifier, "title": "Original frozen study" if original else "Map-direction experiment",
            "status": status, "phase": phase, "message": message, "issues": issues, "current_run_id": active,
            "progress": {"planned_epochs": total, "completed_epochs": completed,
                "committed_epoch_fraction": partial_fraction if partial_fraction is not None else 0.,
                "overall_fraction": min(1., completed / total) if total else None,
                "completed_runs": sum(row["state"] == "completed" for row in rows), "total_runs": 6},
            "budget": {"limit_hours": limit, "charged_hours": charged / 3600 if charged is not None else None,
                "reserved_hours": reserved / 3600 if reserved is not None else None,
                "remaining_hours": max(0., limit - charged / 3600) if charged is not None else None,
                "note": "Ledger charges include current and crash reservations; charged time is not exact active GPU duration."},
            "storage": {"free_gib": free, "minimum_free_gib": 8, "below_reserve": free < 8 if free is not None else None},
            "runs": rows}

    def snapshot(self):
        with self._lock:
            from adp.compute_status import read_compute_hosts
            now = time.time()
            studies = [self._study(identifier, root, now) for identifier, root in self.roots.items()]
            return {"version": 1, "generated_at": _iso(now), "poll_seconds": 5,
                    "studies": studies,
                    "compute_hosts": read_compute_hosts(self.compute_hosts_path, studies=studies, now=now)}
