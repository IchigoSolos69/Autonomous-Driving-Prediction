"""Local inference service. Scenario IDs are resolved through a fixed cache index."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
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


def create_app(cache_dir="data/cache", checkpoint=None, calibration=None, device="auto", frontend_dir="frontend/dist"):
    app = FastAPI(title="Autonomous Driving Prediction", version="0.1.0")
    cache_dir = Path(cache_dir)
    selected_device = resolve_device(device)
    index = {}
    if (cache_dir / "manifest.json").exists():
        for split, ids in read_json(cache_dir / "manifest.json")["splits"].items():
            for scenario_id in ids:
                path = cache_dir / split / f"{scenario_id}.npz"
                if path.exists():
                    index[scenario_id] = (path, split)
    network, calibration_data, checkpoint_id = None, None, None
    risk_supervised = True
    if checkpoint:
        saved = load_checkpoint(checkpoint)
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
        network.to(selected_device).eval()
        checkpoint_id = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if calibration:
        calibration_data = read_json(calibration)
        if not network or calibration_data.get("checkpoint_sha256") != checkpoint_id:
            raise ValueError("Calibration belongs to a different checkpoint")
        if calibration_data.get("manifest_hash") != manifest_hash(cache_dir / "manifest.json"):
            raise ValueError("Calibration and cache manifest do not match")
        if "preprocessing_hash" in calibration_data and calibration_data["preprocessing_hash"] != preprocessing_hash(cache_dir):
            raise ValueError("Calibration and cache preprocessing do not match")
    if network is not None and selected_device.type == "cuda":
        # Pay CUDA's first-kernel initialization cost before announcing readiness.
        warm_batch = collate_scenes([load_scene(next(iter(index.values()))[0])])
        warm_inputs = {k: v.to(selected_device) for k, v in model_inputs(warm_batch).items()}
        with torch.inference_mode():
            network(**warm_inputs)
        torch.cuda.synchronize(selected_device)
    prediction_lock = threading.Lock()

    def get_scene(scenario_id):
        if scenario_id not in index:
            raise HTTPException(404, "Scenario not found in the selected cache manifest. Run adp download and adp preprocess.")
        return load_scene(index[scenario_id][0])

    @app.get("/api/health")
    def health():
        return {"device": str(selected_device), "checkpoint_loaded": network is not None,
                "checkpoint_id": checkpoint_id, "calibration_loaded": calibration_data is not None,
                "scenario_count": len(index), "risk_head_supervised": risk_supervised if network else False,
                "risk_definition": "another retained dynamic actor within 3m during 6s",
                "baseline": "constant_velocity; heuristic risk is uncalibrated"}

    @app.get("/api/scenarios")
    def scenarios():
        if (cache_dir / "catalog.json").exists():
            catalog = read_json(cache_dir / "catalog.json")
            items = [item for item in catalog["scenarios"] if item["id"] in index]
            return {"scenarios": items, "total": len(items)}
        items = []
        for scenario_id, (path, split) in index.items():
            scene = load_scene(path)
            items.append({"id": scenario_id, "city": scene["metadata"]["city"], "split": split,
                          "actor_count": len(scene["actor_ids"]), "focal_type": public_scene(scene)["actors"][0]["type"]})
        return {"scenarios": items, "total": len(items)}

    @app.get("/api/scenarios/{scenario_id}")
    def scenario(scenario_id: str):
        return public_scene(get_scene(scenario_id))

    @app.post("/api/predict")
    def predict(request: PredictRequest):
        if request.model == "trained" and network is None:
            raise HTTPException(409, "No trained checkpoint loaded. Select constant velocity, or reopen the application with a trained checkpoint.")
        scene = get_scene(request.scenario_id)
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
        with prediction_lock, torch.inference_mode():
            outputs = network(**inputs) if request.model == "trained" else constant_velocity(inputs)
            applied = apply_calibration(outputs, calibration_data) if calibration_data and request.model == "trained" else outputs
            probabilities = applied["logits"].softmax(-1)[0].cpu().tolist()
            means = outputs["means"][0].cpu().tolist()
            scales = outputs["scales"][0].cpu().tolist()
            raw_risk = outputs["risk_logits"][0].sigmoid().cpu().tolist()
            calibrated = applied["risk_logits"][0].sigmoid().cpu().tolist() if calibration_data and request.model == "trained" else None
        status = "uncalibrated_baseline_heuristic" if request.model == "constant_velocity" else "uncalibrated"
        if calibrated is not None:
            status = calibration_data.get("risk", {}).get("status", "calibrated_proxy")
        if request.model == "trained" and not risk_supervised:
            status = "untrained_risk_head"
        return {"scenario_id": request.scenario_id, "actor_ids": scene["actor_ids"], "trajectories": means,
                "scales": scales, "probabilities": probabilities,
                "risk": {"raw": raw_risk, "calibrated": calibrated, "status": status},
                "model": {"id": checkpoint_id if request.model == "trained" else "constant_velocity_v1", "kind": request.model},
                "calibration_id": hashlib.sha256(json.dumps(calibration_data, sort_keys=True).encode()).hexdigest() if calibrated is not None else None,
                "coordinate_frame": scene["metadata"]["coordinate_frame"], "scenario": public_scene(scene),
                "stress": scene.get("stress"), "elapsed_ms": round((time.perf_counter()-start)*1000, 2)}

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
