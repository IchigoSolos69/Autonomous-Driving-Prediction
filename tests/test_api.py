from fastapi.testclient import TestClient

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
