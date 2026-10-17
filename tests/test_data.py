import json

import numpy as np
import pandas as pd
import pytest

from adp.data import collate_scenes, load_av2, preprocess, preprocessing_hash, public_scene, to_local, to_world


def av2_fixture(tmp_path):
    rows = []
    for actor, offset in [("focal", 0), ("AV", 10), ("other", 20)]:
        for t in range(110):
            rows.append(dict(scenario_id="fixture", focal_track_id="focal", city="PIT", track_id=actor,
                             object_type="vehicle", object_category=3 if actor == "focal" else 2,
                             timestep=t, observed=t < 50, position_x=t*.1+offset, position_y=0.,
                             velocity_x=1., velocity_y=0., heading=0.))
    path = tmp_path/"scenario_fixture.parquet"
    pd.DataFrame(rows).to_parquet(path)
    map_path = tmp_path/"log_map_archive_fixture.json"
    points = lambda y: [{"x": x, "y": y, "z": 0} for x in [-100, 100]]
    map_path.write_text(json.dumps({"lane_segments": {"1": {"id": 1, "left_lane_boundary": points(2), "right_lane_boundary": points(-2)}},
                                    "drivable_areas": {"1": {"area_boundary": [{"x": x,"y": y,"z": 0} for x,y in [(-100,-3),(100,-3),(100,3),(-100,3)]]}}}))
    return path, map_path


def test_coordinate_round_trip():
    points = np.array([[10., 5.], [-4., 3.]])
    np.testing.assert_allclose(to_world(to_local(points, [4, 2], 1.2), [4, 2], 1.2), points, atol=1e-5)


def test_selection_and_coordinates_do_not_read_futures(tmp_path):
    path, map_path = av2_fixture(tmp_path)
    first = load_av2(path, map_path, max_actors=2)
    df = pd.read_parquet(path)
    df.loc[df.timestep >= 50, ["position_x", "position_y"]] = 10000
    df.to_parquet(path)
    second = load_av2(path, map_path, max_actors=2)
    assert first["actor_ids"] == ["focal", "AV"] == second["actor_ids"]
    np.testing.assert_array_equal(first["history"], second["history"])
    np.testing.assert_array_equal(first["map_points"], second["map_points"])
    assert first["sdf"][0,128,128] > 0
    assert first["sdf"][0,150,128] < 0
    assert public_scene(first)["selection"]["total_dynamic"] == 3
    batch = collate_scenes([first, load_av2(path, map_path, max_actors=3)])
    assert batch["history"].shape == (2,3,50,8)
    assert not batch["actor_mask"][0,2]


def test_preprocessing_equivalent_options_preserve_fingerprint(tmp_path):
    raw, cache = tmp_path/"raw", tmp_path/"cache"
    raw.mkdir()
    (raw/"manifest.json").write_text(json.dumps({"splits": {"train": []}}), encoding="utf-8")
    preprocess(raw, cache, workers=1)
    config_path = cache/"preprocessing.json"
    assert json.loads(config_path.read_text()) == {
        "version": 1, "max_actors": 32, "max_polylines": 128, "map_radius": 100,
    }
    assert '"map_radius": 100.0' not in config_path.read_text()
    original = config_path.read_bytes()
    fingerprint = preprocessing_hash(cache)
    preprocess(raw, cache, workers=1, max_actors=32, max_polylines=128, map_radius=100.0)
    assert config_path.read_bytes() == original
    assert preprocessing_hash(cache) == fingerprint
    with pytest.raises(ValueError, match="options changed"):
        preprocess(raw, cache, workers=1, map_radius=101)
    assert config_path.read_bytes() == original


def test_preprocessing_legacy_defaults_keep_original_bytes(tmp_path):
    raw, cache = tmp_path/"raw", tmp_path/"cache"
    raw.mkdir()
    cache.mkdir()
    (raw/"manifest.json").write_text(json.dumps({"splits": {"train": []}}), encoding="utf-8")
    config_path = cache/"preprocessing.json"
    for original in (b'{ "version": 1 }\n', b'{ "map_radius": 100.0, "version": 1 }\n'):
        config_path.write_bytes(original)
        fingerprint = preprocessing_hash(cache)
        preprocess(raw, cache, workers=1, map_radius=100)
        assert config_path.read_bytes() == original
        assert preprocessing_hash(cache) == fingerprint
