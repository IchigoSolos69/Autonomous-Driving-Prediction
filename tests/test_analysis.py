"""Failure reports preserve complete-future rules and portable replay boundaries."""
import copy
import gzip
import json
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import torch

from adp.analysis import (AnalysisStore, SLICE_DEFINITIONS, _bounded_json, _cohort, _detail,
                          _models, _preview_order, _write_json, actor_descriptors,
                          actor_metrics, analyze_failures, validate_analysis)
from adp.data import collate_scenes, load_av2, manifest_hash, write_json
from adp.model import ForecastModel, constant_velocity, model_inputs
from test_data import av2_fixture


@pytest.fixture(autouse=True)
def builder_storage_headroom(monkeypatch):
    monkeypatch.setattr("adp.analysis.ensure_space", lambda *args, **kwargs: None)


@pytest.fixture
def scene(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    return load_av2(*av2_fixture(source))


@pytest.fixture
def reference_study(tmp_path, scene):
    root = tmp_path / "study"
    cache = root / "cache"
    write_json(cache / "manifest.json", {"splits": {"train": ["train-only"], "validation": ["fixture"],
                                                  "calibration": ["never-open-cal"], "evaluation": ["never-open-test"]}})
    write_json(cache / "preprocessing.json", {"version": 1})
    cache.joinpath("validation").mkdir()
    metadata = {key: value for key, value in scene.items() if not isinstance(value, np.ndarray)}
    np.savez_compressed(cache / "validation" / "fixture.npz",
        **{key: value for key, value in scene.items() if isinstance(value, np.ndarray)}, metadata_json=json.dumps(metadata))
    config = {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 0, "use_graph": False, "use_map": False}
    checkpoint = root / "best.pt"
    torch.manual_seed(42)
    torch.save({"format_version": 2, "config": {"model": config}, "model_state_dict": ForecastModel(config).state_dict(),
        "manifest_hash": manifest_hash(cache / "manifest.json"), "preprocessing_hash": manifest_hash(cache / "preprocessing.json")}, checkpoint)
    write_json(root / "pipeline_status.json", {"status": "completed"})
    write_json(root / "frozen_selection.json", {"deployment_run": "full-42", "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest_hash(checkpoint)})
    return root


def test_descriptors_use_observations_with_exact_boundaries_and_missing_endpoints(scene):
    scene["history"][0, 49, 2] = 1
    scene["history"][1, 49, 2] = 5
    scene["history"][2, 49, 2] = 15
    scene["history"][1, 49, 4:6] = [np.sin(np.pi/6), np.cos(np.pi/6)]
    scene["history"][2, 49, 4:6] = [np.sin(-np.pi/6), np.cos(-np.pi/6)]
    scene["history_mask"][2, :11] = False
    descriptors = actor_descriptors(scene)
    assert [actor["speed_bin"] for actor in descriptors] == ["1-<5", "5-<15", ">=15"]
    assert [actor["turn_bin"] for actor in descriptors] == ["straight", "left", "right"]
    assert descriptors[0]["density_bin"] == "1-3"
    assert descriptors[2]["history_bin"] == "<80%"
    scene["future"][:] = 1e6
    scene["future_mask"][:] = False
    assert actor_descriptors(scene) == descriptors
    scene["history_mask"][1, 39] = False
    assert actor_descriptors(scene)[1]["turn_bin"] == "unknown"


def test_turning_is_heading_change_independent_of_cutoff_speed(scene):
    scene["history"][0, 49, 2:4] = 0
    scene["history"][0, 39, 4:6] = [0, 1]
    scene["history"][0, 49, 4:6] = [np.sin(np.pi/6), np.cos(np.pi/6)]
    assert actor_descriptors(scene)[0]["turn_bin"] == "left"


def test_actor_metrics_exclude_incomplete_and_preserve_per_actor_best_modes(scene):
    batch = collate_scenes([scene])
    outputs = constant_velocity(model_inputs(batch))
    outputs["means"][:, :, 0] = torch.as_tensor(scene["future"][0]) + 3
    outputs["means"][:, 1, 0] = torch.as_tensor(scene["future"][0])
    scene["future_mask"][1, 12] = False
    metrics = actor_metrics(scene, outputs)
    assert metrics[0]["minADE"] == metrics[0]["minFDE"] == metrics[0]["miss_rate"] == 0
    assert metrics[1]["requested"] and metrics[1]["excluded_incomplete_future"]
    assert metrics[1]["minFDE"] is None
    assert metrics[2]["eligible"]


def test_filters_intersect_same_actor_and_aggregate_full_records_not_previews(scene):
    descriptors = actor_descriptors(scene)
    for index, actor in enumerate(descriptors):
        actor["metrics"] = {"reference": {"eligible": True, "requested": True, "excluded_incomplete_future": False,
            "minADE": index+1., "minFDE": index+2., "miss_rate": float(index > 0)}}
    record = {"scenario_id": "one", "actors": descriptors, "source_selection": {"total_scored": 6, "retained_scored": 3},
        "metrics": {"reference": {"jointADE": 4., "jointFDE": 5.}}}
    records = [record, {**copy.deepcopy(record), "scenario_id": "two"}]
    counts, metrics, _, _ = _cohort(records, ["reference"], {"role": "focal"})
    assert counts == {"actors": 2, "scenarios": 2, "small_sample": True}
    assert metrics["reference"]["minFDE"] == 2
    assert metrics["reference"]["jointFDE"] == 5
    assert metrics["reference"]["retained_target_coverage"] == .5
    metadata = {"id": "a"*24, "title": "test", "split": "validation", "development_only": True,
        "model_labels": ["reference"], "provenance": {}, "slice_definitions": SLICE_DEFINITIONS,
        "cases": [{"scenario_id": "one"}]}
    detail = _detail(metadata, records, {"role": "focal", "actor_type": "vehicle"})
    assert detail["counts"]["scenarios"] == 2 and detail["counts"]["cases"] == 1
    assert len(detail["cases"][0]["actors"]) == 1
    assert _detail(metadata, records, {"role": "focal", "actor_type": "pedestrian"})["counts"]["actors"] == 0
    with pytest.raises(ValueError, match="Unknown"):
        _cohort(records, ["reference"], {"path": "secret"})


def test_preview_ranking_deduplicates_and_fills_deterministically():
    labels = [f"{variant}-{seed}" for variant in ("control", "directional") for seed in (42, 43, 44)]
    rows = []
    for index in range(100):
        rows.append({"scenario_id": f"s{index:03}", "metrics": {label: {
            "jointFDE": float(index) if label.startswith("directional") else 0.,
            "focal_minFDE": float(100-index)} for label in labels}})
    selected = _preview_order(rows, "direction-study", labels)
    assert len(selected) == 64 and len({row["scenario_id"] for row in selected}) == 64
    assert selected[0]["scenario_id"] == "s099" and selected[32]["scenario_id"] == "s000"
    assert _preview_order(list(reversed(rows)), "direction-study", labels) == selected


def test_build_validate_import_offline_replay_and_corruption(reference_study, tmp_path):
    report = tmp_path / "解析 – reference"
    result = analyze_failures(reference_study, report)
    assert result["counts"]["scenarios"] == result["counts"]["cases"] == 1
    metadata, records = validate_analysis(report)
    assert metadata["provenance"]["labels_used"] == "selection_only"
    assert metadata["model_labels"] == ["reference", "constant_velocity"]
    assert (report / "summary.csv").is_file() and (report / "failures.png").is_file()
    store = AnalysisStore(tmp_path / "installed")
    disk = type("Disk", (), {"free": 20*1024**3})()
    with patch("adp.analysis.shutil.disk_usage", return_value=disk):
        summary = store.install(report)
        assert store.install(report) == summary
    assert store.list() == [summary]
    detail = store.detail(summary["id"], {"role": "focal"})
    assert detail["counts"]["actors"] == 1
    replay = store.case(summary["id"], "fixture")
    assert replay["predictions"][0]["analysis_label"] == "reference"
    assert replay["predictions"][0]["risk"]["calibrated"] is None
    assert replay["predictions"][0]["analysis_provenance"]["labels_used"] == "selection_only"
    with pytest.raises(KeyError):
        store.case(summary["id"], "../../secret")
    with pytest.raises(KeyError):
        store.detail("../secret")
    source_bytes = (report / "analysis.json").read_bytes()
    (report / "records.json.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        store.install(report)
    assert (report / "analysis.json").read_bytes() == source_bytes
    assert store.case(summary["id"], "fixture")["scenario"]["id"] == "fixture"


def test_rejects_unfinished_reference_before_loading_labels(reference_study):
    write_json(reference_study / "pipeline_status.json", {"status": "running"})
    with patch("adp.analysis.load_scene", side_effect=AssertionError("No label read")):
        with pytest.raises(ValueError, match="complete"):
            _models(reference_study)


def test_bounded_decompression_and_nonfinite_json(tmp_path):
    path = tmp_path / "compressed.json.gz"
    path.write_bytes(gzip.compress(b"[" + b"0,"*1000+b"0]"))
    with pytest.raises(ValueError, match="Expanded"):
        _bounded_json(path, 100, True)
    path.write_bytes(gzip.compress(b'{"x":NaN}'))
    with pytest.raises(ValueError, match="Non-finite"):
        _bounded_json(path, 100, True)


def test_rejects_extra_weights_and_path_traversal_manifest(reference_study, tmp_path):
    report = tmp_path / "report"
    analyze_failures(reference_study, report)
    (report / "weights.pt").write_bytes(b"no executable model files")
    with pytest.raises(ValueError, match="files differ"):
        validate_analysis(report)
    (report / "weights.pt").unlink()
    metadata = json.loads((report / "analysis.json").read_text())
    metadata["files"]["../../outside.json"] = {"bytes": 0, "sha256": "0"*64}
    _write_json(report / "analysis.json", metadata)
    with pytest.raises(ValueError):
        validate_analysis(report)


def test_import_storage_guard_never_publishes_partial(reference_study, tmp_path):
    report = tmp_path / "report"
    analyze_failures(reference_study, report)
    store = AnalysisStore(tmp_path / "installed")
    disk = type("Disk", (), {"free": 8*1024**3})()
    with patch("adp.analysis.shutil.disk_usage", return_value=disk):
        with pytest.raises(ValueError, match="8 GiB"):
            store.install(report)
    assert list(store.root.iterdir()) == []


def test_builder_storage_guard_stops_before_inference_or_publication(reference_study, tmp_path):
    output = tmp_path / "blocked"
    with patch("adp.analysis.ensure_space", side_effect=RuntimeError("Storage paused")), \
            patch("adp.analysis._load_models") as models:
        with pytest.raises(RuntimeError, match="Storage paused"):
            analyze_failures(reference_study, output)
    models.assert_not_called()
    assert not output.exists()


def test_case_validation_rejects_rehashed_bad_probabilities_and_calibration(reference_study, tmp_path):
    report = tmp_path / "report"
    analyze_failures(reference_study, report)
    metadata = json.loads((report / "analysis.json").read_text())
    path = report / "cases" / "fixture.json.gz"
    original = _bounded_json(path, 24*1024**2, True)
    for mutation in ("probabilities", "calibration_id", "scenario"):
        payload = copy.deepcopy(original)
        prediction = payload["predictions"][0]
        if mutation == "probabilities":
            prediction["probabilities"] = [1., 1., 0., 0., 0., 0.]
        elif mutation == "calibration_id":
            prediction["calibration_id"] = "unsupported-validation-calibrator"
        else:
            prediction["scenario"]["actors"][0]["history"][0] = [100., 100.]
        _write_json(path, payload, True)
        metadata["files"]["cases/fixture.json.gz"] = {"bytes": path.stat().st_size, "sha256": manifest_hash(path)}
        _write_json(report / "analysis.json", metadata)
        with pytest.raises(ValueError, match="probabilities|alignment"):
            validate_analysis(report)


def test_cache_revalidates_changed_installed_metadata(reference_study, tmp_path):
    report = tmp_path / "report"
    result = analyze_failures(reference_study, report)
    store = AnalysisStore(tmp_path / "installed")
    with patch("adp.analysis.shutil.disk_usage", return_value=type("Disk", (), {"free": 20*1024**3})()):
        store.install(report)
    store.detail(result["id"])
    installed = store.root / result["id"] / "records.json.gz"
    installed.write_bytes(b"changed externally")
    with pytest.raises(ValueError, match="hash"):
        store.detail(result["id"])


def test_rehashed_replays_reject_ui_metadata_masks_geometry_and_nested_risk(reference_study, tmp_path):
    report = tmp_path / "report"
    analyze_failures(reference_study, report)
    metadata = json.loads((report / "analysis.json").read_text())
    path = report / "cases" / "fixture.json.gz"
    original = _bounded_json(path, 24*1024**2, True)
    for mutation in ("city", "missing_selection", "selection_count", "focal_flag", "actor_type", "history_mask",
                     "future_mask", "origin", "geometry", "risk_shape", "risk_type", "probability_type", "trajectory_type", "duration", "baseline_seed"):
        payload = copy.deepcopy(original)
        scene = payload["scenario"]
        if mutation == "city":
            scene["city"] = {}
        elif mutation == "missing_selection":
            scene.pop("selection")
        elif mutation == "selection_count":
            scene["selection"]["retained_dynamic"] = 0
        elif mutation == "focal_flag":
            scene["actors"][0]["focal"] = False
        elif mutation == "actor_type":
            scene["actors"][0]["type"] = "pedestrian"
        elif mutation == "history_mask":
            scene["actors"][0]["history"][0] = None
        elif mutation == "future_mask":
            scene["actors"][0]["reference_future"][0] = None
        elif mutation == "origin":
            scene["coordinate_frame"]["origin"][0] = "0"
        elif mutation == "geometry":
            scene["map"]["lanes"][0]["points"][0] = [False, 0]
        elif mutation == "risk_shape":
            payload["predictions"][0]["risk"]["raw"] = [[0.1, 0.2]]*len(scene["actors"])
        elif mutation == "risk_type":
            payload["predictions"][0]["risk"]["raw"] = ["0.1"]*len(scene["actors"])
        elif mutation == "probability_type":
            payload["predictions"][0]["probabilities"] = [str(value) for value in payload["predictions"][0]["probabilities"]]
        elif mutation == "trajectory_type":
            payload["predictions"][0]["trajectories"][0][0][0][0] = "0"
        elif mutation == "duration":
            payload["predictions"][0]["elapsed_ms"] = "0"
        else:
            payload["predictions"][-1]["analysis_seed"] = 42
        for prediction in payload["predictions"]:
            prediction["scenario"] = scene
            prediction["coordinate_frame"] = scene["coordinate_frame"]
        _write_json(path, payload, True)
        metadata["files"]["cases/fixture.json.gz"] = {"bytes": path.stat().st_size, "sha256": manifest_hash(path)}
        _write_json(report / "analysis.json", metadata)
        with pytest.raises(ValueError):
            validate_analysis(report)


def test_malformed_metadata_containers_return_validation_errors(reference_study, tmp_path):
    report = tmp_path / "report"
    analyze_failures(reference_study, report)
    metadata = json.loads((report / "analysis.json").read_text())
    metadata["provenance"] = []
    _write_json(report / "analysis.json", metadata)
    with pytest.raises(ValueError, match="schema is incomplete"):
        validate_analysis(report)


def test_direction_report_records_six_final_runs_with_independent_seed_labels(reference_study, tmp_path):
    root = tmp_path / "direction"
    runs = {}
    for variant in ("control", "directional"):
        for seed in (42, 43, 44):
            name = f"{variant}-{seed}"
            checkpoint = root / "runs" / name / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            shutil.copyfile(reference_study / "best.pt", checkpoint)
            runs[name] = {"checkpoint": str(checkpoint), "last_sha256": manifest_hash(checkpoint), "completed": True}
    write_json(root / "protocol.json", {"kind": "direction-study", "cache_dir": str(reference_study / "cache"),
        "selection_ids": ["fixture"], "seeds": [42, 43, 44]})
    write_json(root / "experiment.json", {"status": "completed", "runs": runs})
    result = analyze_failures(root, tmp_path / "direction-report")
    metadata, records = validate_analysis(result["output_dir"])
    assert len(metadata["model_labels"]) == 7
    payload = _bounded_json(tmp_path / "direction-report" / "cases" / "fixture.json.gz", 24*1024**2, True)
    assert [(item["analysis_label"], item["analysis_seed"]) for item in payload["predictions"]] == [
        (variant, seed) for variant in ("control", "directional") for seed in (42, 43, 44)] + [("constant_velocity", None)]
    assert len(records[0]["actors"][0]["metrics"]) == 7
    runs["control-42"]["completed"] = False
    write_json(root / "experiment.json", {"status": "completed", "runs": runs})
    with pytest.raises(ValueError, match="incomplete"):
        _models(root)
