"""Local inference service. Scenario IDs are resolved through a fixed cache index."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
import csv
import platform
import secrets
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from adp.calibration import apply_calibration
from adp.data import collate_scenes, load_scene, manifest_hash, preprocessing_hash, public_scene, read_json
from adp.model import ForecastModel, constant_velocity, model_inputs
from adp.stress import apply_stress
from adp.training import load_checkpoint, resolve_device


class StressSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["position_noise", "dropout", "braking", "cut_in"]
    severity: float = Field(default=.35, ge=0, le=1)
    actor_id: str | None = None
    seed: int = Field(default=42, ge=0, le=2**32-1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    model: Literal["trained", "constant_velocity"] = "constant_velocity"
    stress: StressSpec | None = None


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_token: str = Field(min_length=20, max_length=200)


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["prediction_json", "report_csv", "report_png", "diagnostics"]
    prediction_id: str | None = None
    comparison_prediction_id: str | None = None


def risk_calibration_status(calibration, evaluation, applicable=True):
    if not applicable:
        return "population_applicability_unknown"
    risk = (calibration or {}).get("risk", {})
    if risk.get("fitted") is not True:
        return "not_fitted"
    cal_positive = risk.get("positives")
    eval_positive = (evaluation or {}).get("risk", {}).get("positive_count")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 50 for value in (cal_positive, eval_positive)):
        return "insufficient_positive_events"
    evaluation_id = evaluation.get("checkpoint_sha256") or evaluation.get("checkpoint_id")
    identities = {calibration.get("checkpoint_sha256"), calibration.get("source_checkpoint_sha256")} - {None}
    if (not evaluation_id or evaluation_id not in identities or evaluation.get("manifest_hash") != calibration.get("manifest_hash")
            or evaluation.get("split") != "evaluation" or evaluation.get("model") != "trained"):
        return "evaluation_evidence_unverified"
    return "calibrated_proxy"


def calibration_evidence(calibration, evaluation):
    fitted = (calibration or {}).get("risk", {})
    measured = (evaluation or {}).get("risk", {})
    return {"calibration_positives": fitted.get("positives"), "calibration_count": fitted.get("count"),
            "evaluation_positives": measured.get("positive_count"), "evaluation_count": measured.get("eligible_count"),
            "minimum_positives": 50}


def _cuda_runtime_failure(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(
        word in str(error).lower() for word in ("cuda", "cublas", "cudnn", "nvidia", "gpu driver"))


def _actor_errors(record):
    """Inspector errors use available reference steps; these are not benchmark aggregates."""
    import numpy as np
    rows = []
    trajectories = np.asarray(record["trajectories"])
    for i, actor in enumerate(record["scenario"]["actors"]):
        reference = actor["reference_future"]
        mask = np.array([point is not None for point in reference])
        ade = fde = None
        if mask.any():
            truth = np.array([point or [0, 0] for point in reference])
            distances = np.linalg.norm(trajectories[:, i] - truth, axis=-1)
            ade = float(distances[:, mask].mean(-1).min())
            if mask[-1]:
                fde = float(distances[:, -1].min())
        rows.append({"prediction_id": record["prediction_id"], "scenario_id": record["scenario_id"],
                     "model": record["model"]["kind"], "stress": (record.get("stress") or {}).get("kind", "original"),
                     "actor_id": actor["id"], "actor_type": actor["type"], "reference_steps": int(mask.sum()),
                     "minADE_available_m": ade, "minFDE_at_6s_m": fde,
                     "raw_risk": record["risk"]["raw"][i],
                     "calibrated_risk": record["risk"]["calibrated"][i] if record["risk"]["calibrated"] else None,
                     "risk_status": record["risk"]["status"]})
    return rows


def export_record(target, kind, records, health, state_dir=None):
    """The destination comes only from the native save dialog, never an HTTP path."""
    target = Path(target)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        if kind == "prediction_json":
            payload = {"format_version": 1, "predictions": records,
                       "inspector_errors": [row for record in records for row in _actor_errors(record)],
                       "metric_scope": "Per actor, best of six, available reference steps; not a benchmark report",
                       "safety_claim": False}
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        elif kind == "report_csv":
            rows = [row for record in records for row in _actor_errors(record)]
            with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                for row in rows:
                    for key, value in row.items():
                        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                            row[key] = "'" + value
                    writer.writerow(row)
        elif kind == "report_png":
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            figure = Figure(figsize=(12, 5), constrained_layout=True)
            axes = figure.subplots(1, len(records), squeeze=False)[0]
            for axis, record in zip(axes, records):
                actors = record["scenario"]["actors"]
                index = next((i for i, actor in enumerate(actors) if actor["focal"]), 0)
                actor = actors[index]
                observed = np.array([point for point in actor["history"] if point is not None])
                if len(observed):
                    axis.plot(*observed.T, color="#5c6770", label="Observed")
                reference = np.array([point if point is not None else [np.nan, np.nan] for point in actor["reference_future"]])
                if np.isfinite(reference).any():
                    axis.plot(*reference.T, "--", color="#b86b1e", label="Reference")
                for mode, probability in enumerate(record["probabilities"]):
                    path = np.array(record["trajectories"][mode][index])
                    axis.plot(*path.T, alpha=.25 + .7 * probability, label=f"World {mode + 1}: {probability:.1%}")
                error = _actor_errors(record)[index]
                ade = "unavailable" if error["minADE_available_m"] is None else f"{error['minADE_available_m']:.2f} m"
                axis.set(title=f"{record['model']['kind']} · {(record.get('stress') or {}).get('kind', 'original')}\nFocal minADE over available references: {ade}",
                         xlabel="Local x (m)", ylabel="Local y (m)")
                axis.axis("equal")
                axis.legend(fontsize=7)
            figure.suptitle("Interaction Lab · experimental inspection · no driving-safety claim", fontsize=11)
            figure.savefig(temporary, format="png", dpi=150)
        elif kind == "diagnostics":
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("diagnostics.json", json.dumps({"health": health, "python": platform.python_version(),
                    "platform": platform.platform(), "torch": torch.__version__}, indent=2))
                log = Path(state_dir) / "logs" / "app.log" if state_dir else None
                if log and log.is_file():
                    archive.write(log, "app.log")
        else:
            raise ValueError("Unknown export format")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def create_app(cache_dir="data/cache", checkpoint=None, calibration=None, device="auto", frontend_dir="frontend/dist",
               import_manager=None, desktop_token=None, native_dialogs=None, state_dir=None,
               bundle_metadata=None, evaluation=None, reports_dir=None, data_root=None):
    app = FastAPI(title="Autonomous Driving Prediction", version="0.1.0")
    cache_dir = Path(cache_dir)
    selected_device = resolve_device(device)
    device_warning = None
    bundle_metadata = bundle_metadata or {}
    if evaluation is None and checkpoint:
        candidate = Path(checkpoint).parent / "evaluation" / "metrics.json"
        evaluation = read_json(candidate) if candidate.is_file() else bundle_metadata.get("evaluation")
    index = {}
    if (cache_dir / "manifest.json").exists():
        for split, ids in read_json(cache_dir / "manifest.json")["splits"].items():
            for scenario_id in ids:
                path = cache_dir / split / f"{scenario_id}.npz"
                if path.exists():
                    index[scenario_id] = (path, split, "demo" if bundle_metadata.get("demo") else "benchmark", cache_dir)
    index_lock = threading.RLock()

    def refresh_imports():
        if import_manager is None:
            return
        with index_lock:
            for imported_cache in import_manager.caches():
                for sid in read_json(imported_cache / "manifest.json")["splits"]["imported"]:
                    path = imported_cache / "imported" / f"{sid}.npz"
                    if path.is_file():
                        index[f"import:{imported_cache.name}:{sid}"] = (path, "imported", "imported", imported_cache)
    refresh_imports()
    network, calibration_data, checkpoint_id = None, None, None
    risk_supervised = True
    if checkpoint:
        saved = load_checkpoint(checkpoint)
        if "input_schema" in saved:
            from adp.bundles import input_schema, schemas_compatible
            if not schemas_compatible(saved["input_schema"], input_schema(cache_dir)):
                raise ValueError("Model input schema and scenario cache do not match")
        else:
            if not index or saved["manifest_hash"] != manifest_hash(cache_dir / "manifest.json"):
                raise ValueError("Checkpoint and cache manifest do not match")
            if "preprocessing_hash" in saved and saved["preprocessing_hash"] != preprocessing_hash(cache_dir):
                raise ValueError("Checkpoint and cache preprocessing do not match")
        audit = saved.get("data_audit", {})
        if not audit and (cache_dir/"audit.json").exists():
            audit = read_json(cache_dir/"audit.json")
        if audit:
            risk_supervised = audit.get("train", {}).get("risk_eligible_actors", 0) > 0
        if "risk_supervision_observations" in saved:
            risk_supervised = saved["risk_supervision_observations"] > 0
        network = ForecastModel(saved["config"].get("model", saved["config"]))
        network.load_state_dict(saved["model_state_dict"])
        network.eval()
        checkpoint_id = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if calibration:
        calibration_data = read_json(calibration)
        if not network or calibration_data.get("checkpoint_sha256") != checkpoint_id:
            raise ValueError("Calibration belongs to a different checkpoint")
    if network is not None:
        try:
            network.to(selected_device)
            if selected_device.type == "cuda" and index:
                warm_batch = collate_scenes([load_scene(next(iter(index.values()))[0])])
                warm_inputs = {k: v.to(selected_device) for k, v in model_inputs(warm_batch).items()}
                with torch.inference_mode():
                    network(**warm_inputs)
                torch.cuda.synchronize(selected_device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if device != "auto" or selected_device.type != "cuda" or not _cuda_runtime_failure(error):
                raise
            selected_device = torch.device("cpu")
            network.to(selected_device)
            torch.cuda.empty_cache()
            device_warning = f"GPU initialization failed; using CPU. {error}"
    prediction_lock = threading.Lock()
    records = OrderedDict()
    record_lock = threading.Lock()

    def require_desktop(token):
        if not desktop_token or native_dialogs is None:
            raise HTTPException(403, "This action is available only inside the desktop application.")
        if not token or not token.isascii() or not secrets.compare_digest(token, desktop_token):
            raise HTTPException(403, "Invalid desktop session token. Reopen the application.")

    def get_scene(scenario_id):
        refresh_imports()
        if scenario_id not in index:
            raise HTTPException(404, "Scenario not found in the selected cache manifest. Run adp download and adp preprocess.")
        path, split, source, scene_cache = index[scenario_id]
        scene = load_scene(path)
        scene["metadata"].update(id=scenario_id, split=split, source=source)
        return scene

    @app.get("/api/health")
    def health():
        return {"device": str(selected_device), "checkpoint_loaded": network is not None,
                "checkpoint_id": checkpoint_id, "calibration_loaded": calibration_data is not None,
                "scenario_count": len(index), "risk_head_supervised": risk_supervised if network else False,
                "desktop": bool(desktop_token), "import_available": bool(desktop_token and import_manager),
                "device_warning": device_warning, "data_root": str(data_root) if data_root else None,
                "risk_calibration_status": risk_calibration_status(calibration_data, evaluation),
                "risk_definition": "another retained dynamic actor within 3m during 6s",
                "baseline": "constant_velocity; heuristic risk is uncalibrated"}

    @app.get("/api/scenarios")
    def scenarios():
        refresh_imports()
        items = []
        if (cache_dir / "catalog.json").exists():
            catalog = read_json(cache_dir / "catalog.json")
            items = [{**item, "source": index[item["id"]][2]} for item in catalog["scenarios"] if item["id"] in index]
        if import_manager is not None:
            for imported_cache in import_manager.caches():
                for item in read_json(imported_cache / "catalog.json")["scenarios"]:
                    sid = f"import:{imported_cache.name}:{item['id']}"
                    if sid in index:
                        items.append({**item, "id": sid, "source": "imported"})
        listed = {item["id"] for item in items}
        with index_lock:
            current = list(index.items())
        for scenario_id, (path, split, source, _) in current:
            if scenario_id in listed:
                continue
            scene = load_scene(path)
            items.append({"id": scenario_id, "city": scene["metadata"]["city"], "split": split,
                          "source": source, "actor_count": len(scene["actor_ids"]), "focal_type": public_scene(scene)["actors"][0]["type"]})
        return {"scenarios": items, "total": len(items)}

    @app.get("/api/scenarios/{scenario_id}")
    def scenario(scenario_id: str):
        return public_scene(get_scene(scenario_id))

    @app.post("/api/predict")
    def predict(request: PredictRequest):
        if request.model == "trained" and network is None:
            raise HTTPException(409, "No trained checkpoint loaded. Select constant velocity, or reopen the application with a trained checkpoint.")
        scene = get_scene(request.scenario_id)
        if request.model == "trained" and scene["metadata"].get("source") == "imported":
            from adp.bundles import input_schema, schemas_compatible
            expected = saved["input_schema"] if "input_schema" in saved else input_schema(cache_dir)
            if not schemas_compatible(expected, input_schema(index[request.scenario_id][3])):
                raise HTTPException(422, "Imported data uses an incompatible model input schema. Select the baseline or import it again with this model.")
        if request.stress:
            try:
                scene = apply_stress(scene, request.stress.model_dump())
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
        # Strip labels before model invocation; reference futures are display/evaluation only.
        batch = collate_scenes([scene])
        prediction_device = selected_device if request.model == "trained" else torch.device("cpu")
        inputs = {k: v.to(prediction_device) for k, v in model_inputs(batch).items()}
        start = time.perf_counter()
        population_matches = (scene["metadata"].get("source") != "imported" and not request.stress and calibration_data is not None
                              and (cache_dir / "manifest.json").is_file()
                              and calibration_data.get("manifest_hash") == manifest_hash(cache_dir / "manifest.json"))
        risk_status = risk_calibration_status(calibration_data, evaluation, population_matches)
        with prediction_lock, torch.inference_mode():
            outputs = network(**inputs) if request.model == "trained" else constant_velocity(inputs)
            applied = apply_calibration(outputs, calibration_data) if population_matches and request.model == "trained" else outputs
            probabilities = applied["logits"].softmax(-1)[0].cpu().tolist()
            means = outputs["means"][0].cpu().tolist()
            scales = outputs["scales"][0].cpu().tolist()
            raw_risk = outputs["risk_logits"][0].sigmoid().cpu().tolist()
            calibrated = applied["risk_logits"][0].sigmoid().cpu().tolist() if risk_status == "calibrated_proxy" and request.model == "trained" else None
        status = "uncalibrated_baseline_heuristic" if request.model == "constant_velocity" else "uncalibrated"
        if request.model == "trained":
            status = risk_status
        if request.model == "trained" and not risk_supervised:
            status = "untrained_risk_head"
            calibrated = None
        result = {"scenario_id": request.scenario_id, "actor_ids": scene["actor_ids"], "trajectories": means,
                "scales": scales, "probabilities": probabilities,
                "risk": {"raw": raw_risk, "calibrated": calibrated, "status": status,
                         "applicability": "original_benchmark" if population_matches else "unknown"},
                "model": {"id": checkpoint_id if request.model == "trained" else "constant_velocity_v1", "kind": request.model},
                "calibration_id": hashlib.sha256(json.dumps(calibration_data, sort_keys=True).encode()).hexdigest() if calibrated is not None else None,
                "coordinate_frame": scene["metadata"]["coordinate_frame"], "scenario": public_scene(scene),
                "stress": scene.get("stress"), "elapsed_ms": round((time.perf_counter()-start)*1000, 2)}
        result["source"] = scene["metadata"].get("source", "benchmark")
        result["calibration_applicability"] = "original_benchmark" if population_matches else "unknown"
        result["prediction_id"] = uuid.uuid4().hex
        with record_lock:
            records[result["prediction_id"]] = result
            while len(records) > 8:
                records.popitem(last=False)
        return result

    @app.post("/api/imports")
    def start_import(request: ImportRequest, x_interaction_token: str | None = Header(default=None)):
        require_desktop(x_interaction_token)
        if import_manager is None:
            raise HTTPException(403, "Imports are disabled for this service.")
        try:
            return import_manager.start(request.source_token)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/imports/{job_id}")
    def import_status(job_id: str, x_interaction_token: str | None = Header(default=None)):
        require_desktop(x_interaction_token)
        try:
            return import_manager.status(job_id)
        except (KeyError, AttributeError) as error:
            raise HTTPException(404, "Import job not found") from error

    @app.post("/api/imports/{job_id}/cancel")
    def cancel_import(job_id: str, x_interaction_token: str | None = Header(default=None)):
        require_desktop(x_interaction_token)
        try:
            return import_manager.cancel(job_id)
        except (KeyError, AttributeError) as error:
            raise HTTPException(404, "Import job not found") from error

    @app.get("/api/reports")
    def reports():
        return {"evaluation": evaluation, "model": bundle_metadata.get("model"),
                "provenance": bundle_metadata.get("provenance"),
                "risk_calibration_status": risk_calibration_status(calibration_data, evaluation),
                "risk_calibration_evidence": calibration_evidence(calibration_data, evaluation),
                "evaluation_provenance_verified": bundle_metadata.get("evaluation_provenance_verified", bool(
                    evaluation and calibration_data and evaluation.get("checkpoint_sha256") in
                    {calibration_data.get("checkpoint_sha256"), calibration_data.get("source_checkpoint_sha256")} - {None})),
                "experimental": risk_calibration_status(calibration_data, evaluation) != "calibrated_proxy"}

    @app.get("/api/reports/{name}.png")
    def report_image(name: Literal["evaluation", "training"]):
        path = Path(reports_dir) / f"{name}.png" if reports_dir else None
        if path is None or not path.is_file():
            raise HTTPException(404, "This model bundle has no saved report image.")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/exports")
    def export(request: ExportRequest, x_interaction_token: str | None = Header(default=None)):
        require_desktop(x_interaction_token)
        selected = []
        if request.kind != "diagnostics":
            with record_lock:
                for record_id in [request.prediction_id, request.comparison_prediction_id]:
                    if record_id is not None:
                        if record_id not in records:
                            raise HTTPException(404, "Prediction expired. Run the comparison again before exporting.")
                        selected.append(records[record_id])
            if not selected:
                raise HTTPException(422, "Run a prediction before exporting it.")
            if len(selected) == 2 and selected[0]["scenario_id"] != selected[1]["scenario_id"]:
                raise HTTPException(422, "Comparison predictions must belong to the same scenario.")
        target = native_dialogs.save_path(request.kind)
        if not target:
            return {"saved": False, "cancelled": True}
        try:
            export_record(Path(target), request.kind, selected, health(), state_dir)
        except (ValueError, OSError) as error:
            raise HTTPException(422, f"Export failed: {error}") from error
        return {"saved": True, "cancelled": False, "filename": Path(target).name}

    frontend_dir = Path(frontend_dir)
    if (frontend_dir / "index.html").exists():
        assets = frontend_dir / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/", include_in_schema=False)
        def frontend():
            return FileResponse(frontend_dir / "index.html")
        icon = frontend_dir/"favicon.svg"
        if icon.exists():
            @app.get("/favicon.svg", include_in_schema=False)
            def favicon():
                return FileResponse(icon)
    return app
