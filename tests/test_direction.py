"""Geometry, matched initialization and budget/selection boundaries of the direction study."""
import copy
from unittest.mock import patch

import pytest
import torch

from adp.data import read_json, write_json
from adp.direction import (DirectionBudgetExhausted, SEEDS, VARIANTS, audit_lane_order,
    charged_operation, compare_seed_records, direction_configs, freeze_epochs, run_direction_study)
from adp.model import ForecastModel, lane_tangents, model_inputs


def test_tangents_preserve_sign_and_centered_differences():
    points = torch.tensor([[[[0., 0.], [1., 0.], [1., 1.], [2., 1.]]]])
    features, mask = torch.tensor([[[1., 0., 0., 0.]]]), torch.tensor([[True]])
    result = lane_tangents(points, features, mask)
    expected = torch.tensor([[[[1., 0.], [2**-.5, 2**-.5], [2**-.5, 2**-.5], [1., 0.]]]])
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(lane_tangents(points.flip(-2), features, mask), -result.flip(-2))


def test_duplicate_points_crosswalks_and_padding_have_zero_tangents():
    points = torch.zeros(1, 4, 20, 2)
    points[:, :, :, 0] = torch.arange(20)
    points[:, 0] = 2
    features = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]]])
    mask = torch.tensor([[True, True, False, True]])
    result = lane_tangents(points, features, mask)
    assert not result[:, :3].any()
    torch.testing.assert_close(result[:, 3, :, 0], torch.ones(1, 20))
    torch.testing.assert_close(result[:, 3, :, 1], torch.zeros(1, 20))
    assert torch.isfinite(result).all()


def test_legacy_loading_and_identical_eight_input_initialization():
    config = {"model": {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 0}, "seed": 1}
    legacy = ForecastModel(config["model"])
    restored = ForecastModel(config["model"])
    restored.load_state_dict(legacy.state_dict())
    assert restored.map_encoder[0].in_features == 6
    candidates = direction_configs(config)
    for number in SEEDS:
        pair = [item for item in candidates if item["config"]["seed"] == number]
        models = []
        for item in pair:
            torch.manual_seed(number)
            models.append(ForecastModel(item["config"]["model"]))
        assert models[0].map_encoder[0].in_features == models[1].map_encoder[0].in_features == 8
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
        left, right = copy.deepcopy(pair[0]["config"]), copy.deepcopy(pair[1]["config"])
        left["model"].pop("map_point_encoding")
        right["model"].pop("map_point_encoding")
        assert left == right
    with pytest.raises(ValueError, match="map_point_encoding"):
        ForecastModel({"map_point_encoding": "invalid"})


def test_directional_inference_observed_only_and_world_shape():
    from adp.training import _smoke_batch
    batch = _smoke_batch(torch.device("cpu"))
    batch["map_points"][..., 0] = torch.arange(20)
    batch["map_features"][..., 0] = 1
    model = ForecastModel({"width": 16, "heads": 4, "temporal_layers": 1,
                           "graph_layers": 0, "map_point_encoding": "xy_tangent"}).eval()
    with torch.inference_mode():
        first = model(**model_inputs(batch))
        batch["future"][:] = 100000
        batch["future_mask"][:] = False
        second = model(**model_inputs(batch))
    assert first["means"].shape == (1, 6, 2, 60, 2)
    torch.testing.assert_close(first["means"], second["means"], rtol=0, atol=0)


def test_inference_bundle_validation_accepts_legacy_and_both_eight_input_encoders(tmp_path):
    from adp.bundles import _weights, input_schema
    write_json(tmp_path / "preprocessing.json", {"version": 1})
    for encoding in (None, "xy_zero_tangent", "xy_tangent"):
        config = {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 0}
        if encoding:
            config["map_point_encoding"] = encoding
        model = ForecastModel(config)
        checkpoint = {"format_version": 1, "config": {"model": config},
            "model_state_dict": model.state_dict(), "input_schema": input_schema(tmp_path)}
        path = tmp_path / f"{encoding or 'legacy'}.pt"
        torch.save(checkpoint, path)
        saved = _weights(path)
        assert saved["model_state_dict"]["map_encoder.0.weight"].shape[-1] == (8 if encoding else 6)


def test_uniform_epoch_schedule_uses_slower_pilot_and_reserve():
    schedule = freeze_epochs({"control": {"examples_per_second": 50}, "directional": {"examples_per_second": 43}}, 100)
    assert schedule["epochs"] == 5
    assert schedule["slower_examples_per_second"] == 43
    assert schedule["projected_training_seconds"] <= 10.5 * 3600
    with pytest.raises(DirectionBudgetExhausted):
        freeze_epochs({"control": {"examples_per_second": .01}, "directional": {"examples_per_second": .01}}, 100)


def test_budget_reserves_before_work_refunds_normal_return_and_rejects_limit(tmp_path):
    with patch("adp.direction.time.monotonic", side_effect=[100, 105, 105]):
        with charged_operation(tmp_path, "pilot", 300):
            assert read_json(tmp_path / "direction_ledger.json")["elapsed_seconds"] == 300
    state = read_json(tmp_path / "direction_ledger.json")
    assert state["elapsed_seconds"] == 5
    assert state["operations"][0]["status"] == "returned"
    write_json(tmp_path / "direction_ledger.json", {"version": 1, "elapsed_seconds": 12 * 3600, "operations": []})
    with pytest.raises(DirectionBudgetExhausted):
        with charged_operation(tmp_path, "never starts"):
            pytest.fail("GPU work entered after the ceiling")


def test_budget_expiration_is_charged_and_stops_operation(tmp_path):
    write_json(tmp_path / "direction_ledger.json", {"version": 1, "elapsed_seconds": 12 * 3600 - 2, "operations": []})
    with patch("adp.direction.time.monotonic", side_effect=[100, 103, 103]):
        with pytest.raises(DirectionBudgetExhausted):
            with charged_operation(tmp_path, "last slice") as (_, check):
                check()
    assert read_json(tmp_path / "direction_ledger.json")["elapsed_seconds"] == 12 * 3600 + 1


def test_waiting_reference_never_touches_cache_or_starts_training(tmp_path):
    reference, followup = tmp_path / "reference", tmp_path / "followup"
    write_json(reference / "pipeline_status.json", {"status": "running", "phase": "train"})
    with patch("adp.direction._run") as start:
        result = run_direction_study(reference, followup, "cpu")
    assert result["status"] == "waiting_for_reference"
    assert not followup.exists()
    start.assert_not_called()
    with pytest.raises(ValueError, match="separate"):
        run_direction_study(reference, reference / "nested", "cpu")


def test_scenario_bootstrap_keeps_all_three_paired_seeds_together():
    records = {}
    for number in SEEDS:
        records[f"control-{number}"] = [{"scenario_id": "a", "jointFDE": 10}, {"scenario_id": "b", "jointFDE": 20}]
        records[f"directional-{number}"] = [{"scenario_id": "a", "jointFDE": 7 + (number - 42)},
                                            {"scenario_id": "b", "jointFDE": 17 + (number - 42)}]
    result = compare_seed_records(records, 1000, 42)
    assert result["metrics"]["jointFDE"] == {"delta": -2., "interval95": [-2., -2.], "eligible_scenarios": 2}
    assert result["individual_seeds"]["42"]["metrics"]["jointFDE"]["delta"] == -3
    assert result["individual_seeds"]["44"]["metrics"]["jointFDE"]["delta"] == -1
    records["control-44"].append({"scenario_id": "a", "jointFDE": 1})
    with pytest.raises(ValueError, match="one record"):
        compare_seed_records(records)


def test_successor_audit_detects_reversed_raw_lane(tmp_path):
    def lane(identifier, start, end, successors):
        return {"id": identifier, "left_lane_boundary": [[start, 1, 0], [end, 1, 0]],
            "right_lane_boundary": [[start, -1, 0], [end, -1, 0]], "successors": successors}
    path = tmp_path / "pilot" / "raw" / "train" / "scene" / "log_map_archive_scene.json"
    write_json(path, {"lane_segments": {"1": lane(1, 0, 10, [2]), "2": lane(2, 10, 20, [])}})
    assert audit_lane_order(tmp_path, ["scene"])["qualified"]
    write_json(path, {"lane_segments": {"1": lane(1, 10, 0, [2]), "2": lane(2, 20, 10, [])}})
    audit = audit_lane_order(tmp_path, ["scene"])
    assert not audit["qualified"] and audit["reverse"] == 1


def test_reference_selection_uses_only_complete_map_enabled_screening_runs(tmp_path):
    from adp.data import manifest_hash
    from adp.direction import _prepare
    reference, output = tmp_path / "reference", tmp_path / "direction"
    training, selection = [f"t{i}" for i in range(40000)], [f"v{i}" for i in range(2000)]
    write_json(reference / "cache" / "manifest.json", {"splits": {
        "train": training, "validation": selection, "calibration": ["untouched-c"], "evaluation": ["untouched-e"]}})
    write_json(reference / "cache" / "preprocessing.json", {"version": 1})
    write_json(reference / "pilot" / "cache" / "manifest.json", {"splits": {"train": training[:1000], "validation": []}})
    write_json(reference / "pilot" / "cache" / "preprocessing.json", {"version": 1})
    write_json(reference / "pipeline_status.json", {"phase": "finished", "status": "completed"})
    configs = []
    for name, score, focal, epoch in (("full", 2, 3, 2), ("no_graph", 2, 2, 1), ("world_ce", 2, 2, 2), ("no_map", 0, 0, 1)):
        config = {"seed": 42, "model": {"use_map": name != "no_map"}, "training": {"checkpoint_selection": "joint_minFDE"}}
        configs.append([name, config])
        directory = reference / "runs" / f"{name}-42"
        directory.mkdir(parents=True)
        checkpoint = {"format_version": 2, "config": config, "completed_epochs": 2, "training_progress": None,
            "manifest_hash": manifest_hash(reference / "cache" / "manifest.json"),
            "preprocessing_hash": manifest_hash(reference / "cache" / "preprocessing.json"),
            "best_selection": {"metric": "joint_minFDE", "key": [score, focal], "epoch": epoch}}
        for filename in ("last.pt", "best.pt"):
            torch.save(checkpoint, directory / filename)
    write_json(reference / "schedule.json", {"epochs": 2, "configs": configs})
    protocol = _prepare(reference, output)
    assert protocol["selected_screening_run"] == "no_graph-42"
    assert protocol["train_ids"] == training and protocol["selection_ids"] == selection
    assert not protocol["heldout_labels_used"]
    assert _prepare(reference, output) == protocol
    assert not (reference / "cache" / "calibration").exists()
    assert not (reference / "cache" / "evaluation").exists()


def test_complete_runner_resumes_committed_batches_and_compares_final_checkpoints(tmp_path):
    from adp.direction import _run
    from adp.research import config_fingerprint
    from adp.training import load_checkpoint
    reference, output = tmp_path / "reference", tmp_path / "direction"
    output.mkdir()
    config = {"seed": 42, "model": {"width": 16, "heads": 4, "use_map": True}, "training": {"batch_size": 4}}
    protocol = {"base_config": config, "pilot_ids": ["pilot"], "train_ids": ["t1", "t2", "t3", "t4"],
        "selection_ids": ["s1", "s2"], "cache_dir": str(reference / "cache"),
        "manifest_sha256": "manifest", "preprocessing_sha256": "preprocessing"}
    def prepare(*_):
        write_json(output / "protocol.json", protocol)
        return protocol
    def profile(_, candidate, device, check):
        check()
        return {"examples_per_second": 43, "config_sha256": config_fingerprint(candidate)}
    training_calls = []
    def train(candidate, cache, directory, epochs, device, resume, max_runtime_seconds):
        training_calls.append((directory.name, resume))
        directory.mkdir(parents=True, exist_ok=True)
        partial = len(training_calls) == 1
        if len(training_calls) == 2:
            assert resume == directory / "last.pt"
            assert load_checkpoint(resume)["training_progress"]["cursor"] == 2
        saved = {"format_version": 2, "config": candidate, "manifest_hash": "manifest",
            "preprocessing_hash": "preprocessing", "completed_epochs": 0 if partial else epochs,
            "training_progress": {"cursor": 2} if partial else None}
        torch.save(saved, directory / "last.pt")
        return {"completed_epochs": saved["completed_epochs"]}
    def score(checkpoint, cache, directory, ids):
        assert checkpoint.name == "last.pt"
        assert cache == reference / "cache"
        value = 1 if "directional" in directory.name else 2
        return [{"scenario_id": sid, "jointFDE": value} for sid in ids], {"scenario_count": 2}
    with patch("adp.direction._prepare", side_effect=prepare), patch("adp.direction.ensure_space"), \
         patch("adp.direction.audit_lane_order", return_value={"qualified": True}), \
         patch("adp.direction._profile", side_effect=profile), patch("adp.training.train", side_effect=train), \
         patch("adp.reporting.plot_training"), patch("adp.direction._score", side_effect=score):
        result = _run(reference, output, "cpu")
        assert result["status"] == "completed"
        assert len(training_calls) == 7
        assert len(result["runs"]) == 6
        assert read_json(output / "comparison.json")["metrics"]["jointFDE"]["delta"] == -1
        spent = read_json(output / "direction_ledger.json")["elapsed_seconds"]
        _run(reference, output, "cpu")
        assert len(training_calls) == 7
        assert read_json(output / "direction_ledger.json")["elapsed_seconds"] == spent


def test_exhausted_runner_records_incomplete_instead_of_comparing(tmp_path):
    from adp.direction import _run
    protocol = {"base_config": {"model": {}, "training": {}}, "pilot_ids": ["p"]}
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "direction_ledger.json", {"version": 1, "elapsed_seconds": 12 * 3600, "operations": []})
    with patch("adp.direction._prepare", return_value=protocol), patch("adp.direction.ensure_space"), \
         patch("adp.direction.audit_lane_order", return_value={"qualified": True}), \
         patch("adp.training.train") as train:
        result = _run(tmp_path / "reference", tmp_path, "cpu")
    assert result["status"] == "incomplete_experiment" and result["budget_exhausted"]
    assert result["checkpoints_preserved"] and not result["resumable"]
    assert not (tmp_path / "comparison.json").exists()
    train.assert_not_called()
