from copy import deepcopy
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from adp.api import create_app
from adp.desktop import DesktopBridge
from test_api import prepared_cache


def test_saved_cases_are_read_only_exportable_and_leave_live_model_unchanged(tmp_path, monkeypatch):
    cache, _ = prepared_cache(tmp_path)
    baseline = TestClient(create_app(cache, device="cpu", frontend_dir=tmp_path)).post(
        "/api/predict", json={"scenario_id": "fixture"}).json()
    baseline.pop("prediction_id")
    predictions = []
    for label in ("control", "directional"):
        prediction = deepcopy(baseline)
        prediction.update(analysis_label=label, analysis_seed=42, development_only=True)
        predictions.append(prediction)
    stored = {"analysis_id": "saved", "scenario": baseline["scenario"], "predictions": predictions}
    calls = []

    class Store:
        def list(self):
            return [{"id": "saved", "development_only": True}]

        def detail(self, key, filters):
            if key != "saved":
                raise KeyError(key)
            calls.append(filters)
            return {"id": key, "counts": {"scenarios": 1}, "filters": filters}

        def case(self, key, scenario):
            if key != "saved" or scenario != "fixture":
                raise KeyError(key)
            return stored

    selected = tmp_path / "saved Ω.csv"
    client = TestClient(create_app(cache, device="cpu", frontend_dir=tmp_path,
        analysis_store=Store(), desktop_token="native-secret", native_dialogs=SimpleNamespace(save_path=lambda kind: selected)))
    monkeypatch.setattr("adp.api.model_inputs", lambda *args: pytest.fail("Recorded replay entered live inference"))
    assert client.get("/api/analyses").json()["analyses"][0]["id"] == "saved"
    assert client.get("/api/analyses/saved?city=PIT&role=focal").status_code == 200
    assert calls == [{"city": "PIT", "role": "focal"}]
    assert client.get("/api/analyses/saved?path=foreign").status_code == 422
    assert client.get("/api/analyses/saved?city=PIT&city=MIA").status_code == 422
    assert client.get("/api/analyses/missing").status_code == 404
    response = client.get("/api/analyses/saved/cases/fixture").json()
    ids = [item["prediction_id"] for item in response["predictions"]]
    assert len(set(ids)) == 2 and all(response["predictions"][i]["analysis_id"] == "saved" for i in range(2))
    assert "prediction_id" not in stored["predictions"][0]
    exported = client.post("/api/exports", json={"kind": "report_csv", "prediction_id": ids[0], "comparison_prediction_id": ids[1]},
                           headers={"X-Interaction-Token": "native-secret"})
    assert exported.json()["saved"]
    text = selected.read_text(encoding="utf-8-sig")
    assert "analysis_label,analysis_seed,development_only" in text
    assert "directional,42,True" in text
    assert client.get("/api/health").json()["checkpoint_loaded"] is False
    for _ in range(4):
        client.get("/api/analyses/saved/cases/fixture")
    assert client.post("/api/exports", json={"kind": "prediction_json", "prediction_id": ids[0]},
                       headers={"X-Interaction-Token": "native-secret"}).status_code == 404


def test_no_analysis_store_has_actionable_empty_and_missing_states(tmp_path):
    client = TestClient(create_app(tmp_path, device="cpu", frontend_dir=tmp_path))
    assert client.get("/api/analyses").json() == {"analyses": []}
    assert client.get("/api/analyses/none").status_code == 404
    assert client.get("/api/analyses/none/cases/fixture").status_code == 404


def test_native_analysis_selection_cancel_and_install(tmp_path, monkeypatch):
    import sys
    chosen = iter([None, (str(tmp_path / "analysis Ω"),)])
    sources = []
    store = SimpleNamespace(install=lambda path: sources.append(path) or {"id": "verified", "development_only": True})
    bridge = DesktopBridge(tmp_path, tmp_path, {}, "token", None, store)
    bridge._window = SimpleNamespace(create_file_dialog=lambda *args, **kwargs: next(chosen))
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(FileDialog=SimpleNamespace(FOLDER=20)))
    assert bridge.choose_analysis_directory() is None
    assert sources == []
    assert bridge.choose_analysis_directory()["id"] == "verified"
    assert sources == [str(tmp_path / "analysis Ω")]
