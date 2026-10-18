from fastapi.testclient import TestClient
import pytest

from adp.api import create_app


def test_missing_data_checkpoint_and_untrusted_ids(tmp_path):
    client = TestClient(create_app(tmp_path, device="cpu", frontend_dir=tmp_path/"absent"))
    assert client.get("/api/health").json()["checkpoint_loaded"] is False
    assert client.get("/api/scenarios").json() == {"scenarios": [], "total": 0}
    assert client.post("/api/predict", json={"scenario_id":"../../etc/passwd", "model":"constant_velocity"}).status_code == 404
    assert client.post("/api/predict", json={"scenario_id":"none", "model":"trained"}).status_code == 409
    assert client.post("/api/predict", json={"scenario_id":"none", "stress":{"kind":"braking", "severity":2}}).status_code == 422


def test_cached_scenario_prediction_and_stress_share_inputs(tmp_path):
    import json
    from adp.data import preprocess
    from test_data import av2_fixture

    raw = tmp_path/'raw'
    folder = raw/'train'/'fixture'
    folder.mkdir(parents=True)
    av2_fixture(folder)
    (raw/'manifest.json').write_text(json.dumps({'splits': {'train': ['fixture'], 'validation': [], 'calibration': [], 'evaluation': []}}))
    cache = tmp_path/'cache'
    preprocess(raw, cache)
    client = TestClient(create_app(cache, device='cpu', frontend_dir=tmp_path/'absent'))
    original = client.post('/api/predict', json={'scenario_id': 'fixture'}).json()
    stressed = client.post('/api/predict', json={'scenario_id': 'fixture', 'stress': {'kind': 'position_noise', 'severity': .35, 'actor_id': 'focal'}}).json()
    assert len(original['trajectories']) == 6
    assert len(original['trajectories'][0]) == 3
    assert len(original['trajectories'][0][0]) == 60
    assert abs(sum(original['probabilities'])-1) < 1e-6
    assert original['risk']['status'] == 'uncalibrated_baseline_heuristic'
    assert original['scenario']['actors'][0]['history'] != stressed['scenario']['actors'][0]['history']
    assert original['scenario']['actors'][0]['reference_future'] == stressed['scenario']['actors'][0]['reference_future']
    assert original['trajectories'] != stressed['trajectories']
    assert stressed['stress']['label'] == 'observation_corruption'


def prepared_cache(tmp_path):
    import json
    from adp.data import preprocess
    from test_data import av2_fixture
    raw = tmp_path / "raw"
    source = raw / "train" / "fixture"
    source.mkdir(parents=True)
    av2_fixture(source)
    (raw / "manifest.json").write_text(json.dumps({"splits": {"train": ["fixture"], "validation": [], "calibration": [], "evaluation": []}}))
    cache = tmp_path / "cache"
    preprocess(raw, cache, workers=1)
    return cache, source


def trained_artifacts(tmp_path, cache, positives=50, fitted=True):
    import hashlib
    import torch
    from adp.data import manifest_hash, preprocessing_hash, write_json
    from adp.model import ForecastModel
    config = {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 0, "use_graph": False, "use_map": False}
    checkpoint = tmp_path / "weights.pt"
    torch.save({"format_version": 1, "config": {"model": config}, "model_state_dict": ForecastModel(config).state_dict(),
                "manifest_hash": manifest_hash(cache / "manifest.json"), "preprocessing_hash": preprocessing_hash(cache),
                "risk_supervision_observations": 100}, checkpoint)
    calibration = tmp_path / "calibration.json"
    values = {"mode_temperature": 2., "risk_temperature": 2., "risk_bias": .5,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "manifest_hash": manifest_hash(cache / "manifest.json"),
              "risk": {"fitted": fitted, "count": 100, "positives": positives}}
    write_json(calibration, values)
    evaluation = {"manifest_hash": values["manifest_hash"], "checkpoint_sha256": values["checkpoint_sha256"], "split": "evaluation", "model": "trained",
                  "risk": {"eligible_count": 100, "positive_count": 50}}
    return checkpoint, calibration, evaluation


@pytest.mark.parametrize("cal_positives,eval_positives,fitted,expected", [
    (49, 50, True, "insufficient_positive_events"), (50, 49, True, "insufficient_positive_events"),
    (50, 50, False, "not_fitted"), (50, 50, True, "calibrated_proxy"),
])
def test_calibrated_risk_requires_both_populations_and_fit(tmp_path, cal_positives, eval_positives, fitted, expected):
    cache, _ = prepared_cache(tmp_path)
    checkpoint, calibration, evaluation = trained_artifacts(tmp_path, cache, cal_positives, fitted)
    evaluation["risk"]["positive_count"] = eval_positives
    client = TestClient(create_app(cache, checkpoint, calibration, device="cpu", evaluation=evaluation))
    result = client.post("/api/predict", json={"scenario_id": "fixture", "model": "trained"}).json()
    assert result["risk"]["status"] == expected
    assert (result["risk"]["calibrated"] is not None) == (expected == "calibrated_proxy")
    evidence = client.get("/api/reports").json()["risk_calibration_evidence"]
    assert evidence["calibration_positives"] == cal_positives
    assert evidence["evaluation_positives"] == eval_positives


def test_native_import_tokens_namespace_and_unknown_population(tmp_path):
    from types import SimpleNamespace
    from adp.imports import ImportManager
    from test_imports import finish
    cache, source = prepared_cache(tmp_path)
    checkpoint, calibration, evaluation = trained_artifacts(tmp_path, cache)
    manager = ImportManager(tmp_path / "imports")
    token = "a" * 32
    try:
        client = TestClient(create_app(cache, checkpoint, calibration, device="cpu", evaluation=evaluation,
                            import_manager=manager, desktop_token=token, native_dialogs=SimpleNamespace()))
        grant = manager.grant_directory(source)
        assert client.post("/api/imports", json=grant).status_code == 422  # extra fields are rejected
        payload = {"source_token": grant["source_token"]}
        assert client.post("/api/imports", json=payload).status_code == 403
        started = client.post("/api/imports", json=payload, headers={"X-Interaction-Token": token}).json()
        assert finish(manager, started["id"])["status"] == "completed"
        assert client.get(f"/api/imports/{started['id']}").status_code == 403
        assert client.get(f"/api/imports/{started['id']}", headers={"X-Interaction-Token": token}).json()["status"] == "completed"
        catalog = client.get("/api/scenarios").json()["scenarios"]
        imported, = [item for item in catalog if item["source"] == "imported"]
        assert imported["id"].startswith("import:") and imported["id"] != "fixture"
        original = client.post("/api/predict", json={"scenario_id": "fixture", "model": "trained"}).json()
        result = client.post("/api/predict", json={"scenario_id": imported["id"], "model": "trained"}).json()
        assert original["risk"]["calibrated"] is not None
        assert result["risk"]["calibrated"] is None
        assert result["risk"]["status"] == "population_applicability_unknown"
        assert result["source"] == "imported" and result["scenario"]["id"] == imported["id"]
        assert result["trajectories"] == original["trajectories"]
        assert result["probabilities"] != original["probabilities"]  # imported worlds use raw logits
        stressed = client.post("/api/predict", json={"scenario_id": "fixture", "model": "trained", "stress": {"kind": "position_noise", "severity": 0}}).json()
        assert stressed["risk"]["calibrated"] is None
        from adp.data import read_json, write_json
        imported_cache, = manager.caches()
        schema = read_json(imported_cache / "preprocessing.json")
        write_json(imported_cache / "preprocessing.json", {**schema, "max_actors": 64})
        assert client.post("/api/predict", json={"scenario_id": imported["id"], "model": "trained"}).status_code == 422
        assert client.post("/api/predict", json={"scenario_id": imported["id"], "model": "constant_velocity"}).status_code == 200
    finally:
        manager.close()


def test_exports_need_native_token_dialog_and_keep_comparisons(tmp_path):
    import json
    import zipfile
    from types import SimpleNamespace
    cache, _ = prepared_cache(tmp_path)
    chosen = [tmp_path / "prediction.json", tmp_path / "report.csv", tmp_path / "report.png", tmp_path / "diagnostics.zip", None]
    dialogs = SimpleNamespace(save_path=lambda kind: chosen.pop(0))
    token = "a" * 32
    client = TestClient(create_app(cache, device="cpu", desktop_token=token, native_dialogs=dialogs, state_dir=tmp_path))
    first = client.post("/api/predict", json={"scenario_id": "fixture"}).json()
    second = client.post("/api/predict", json={"scenario_id": "fixture", "stress": {"kind": "position_noise", "severity": .2}}).json()
    payload = {"kind": "prediction_json", "prediction_id": second["prediction_id"], "comparison_prediction_id": first["prediction_id"]}
    assert client.post("/api/exports", json=payload).status_code == 403
    assert client.post("/api/exports", json={**payload, "path": "untrusted"}, headers={"X-Interaction-Token": token}).status_code == 422
    for kind in ("prediction_json", "report_csv", "report_png", "diagnostics"):
        assert client.post("/api/exports", json={**payload, "kind": kind}, headers={"X-Interaction-Token": token}).json()["saved"] is True
    saved = json.loads((tmp_path / "prediction.json").read_text())
    assert [item["prediction_id"] for item in saved["predictions"]] == [second["prediction_id"], first["prediction_id"]]
    assert all(row["minADE_available_m"] < 1e-5 and row["minFDE_at_6s_m"] < 1e-5 for row in saved["inspector_errors"] if row["prediction_id"] == first["prediction_id"])
    assert len((tmp_path / "report.csv").read_text(encoding="utf-8-sig").splitlines()) == 7
    assert (tmp_path / "report.png").read_bytes().startswith(b"\x89PNG")
    assert "diagnostics.json" in zipfile.ZipFile(tmp_path / "diagnostics.zip").namelist()
    assert client.post("/api/exports", json=payload, headers={"X-Interaction-Token": token}).json()["cancelled"] is True
    for _ in range(8):
        client.post("/api/predict", json={"scenario_id": "fixture"})
    assert client.post("/api/exports", json=payload, headers={"X-Interaction-Token": token}).status_code == 404
    from adp.api import export_record
    first["scenario"]["actors"][0]["id"] = "=1+1"
    export_record(tmp_path / "safe.csv", "report_csv", [first], {})
    assert "'=1+1" in (tmp_path / "safe.csv").read_text(encoding="utf-8-sig")


def test_gpu_runtime_fallback_is_auto_only_and_schema_failures_stay_hard(tmp_path, monkeypatch):
    import torch
    from adp.model import ForecastModel
    cache, _ = prepared_cache(tmp_path)
    checkpoint, _, _ = trained_artifacts(tmp_path, cache)
    monkeypatch.setattr("adp.api.resolve_device", lambda device: torch.device("cuda"))
    original_to = ForecastModel.to

    def gpu_failure(self, device):
        if torch.device(device).type == "cuda":
            raise RuntimeError("CUDA kernel is not supported")
        return original_to(self, device)

    monkeypatch.setattr(ForecastModel, "to", gpu_failure)
    client = TestClient(create_app(cache, checkpoint, device="auto"))
    assert client.get("/api/health").json()["device"] == "cpu"
    assert "using CPU" in client.get("/api/health").json()["device_warning"]
    with pytest.raises(RuntimeError, match="CUDA kernel"):
        create_app(cache, checkpoint, device="cuda")

    def schema_failure(self, device):
        raise RuntimeError("size mismatch in input schema")

    monkeypatch.setattr(ForecastModel, "to", schema_failure)
    with pytest.raises(RuntimeError, match="size mismatch"):
        create_app(cache, checkpoint, device="auto")


def test_missing_evaluation_evidence_never_exposes_calibrated_risk(tmp_path):
    cache, _ = prepared_cache(tmp_path)
    checkpoint, calibration, _ = trained_artifacts(tmp_path, cache)
    client = TestClient(create_app(cache, checkpoint, calibration, device="cpu"))
    result = client.post("/api/predict", json={"scenario_id": "fixture", "model": "trained"}).json()
    assert result["risk"]["calibrated"] is None
    assert result["risk"]["status"] == "insufficient_positive_events"
    evidence = client.get("/api/reports").json()["risk_calibration_evidence"]
    assert evidence["evaluation_positives"] is None and evidence["evaluation_count"] is None
