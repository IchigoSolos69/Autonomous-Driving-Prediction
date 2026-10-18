import copy
from unittest.mock import patch
import uuid

import numpy as np
import pytest

from adp.data import manifest_hash, read_json, write_json
from adp.protocol import create_protocol, paired_comparison, validate_benchmark
from adp.research import ensure_space, prepare_study, study_configs
from adp.metrics import matched_overlap_diagnostics


def test_protocol_allows_unseen_compatible_reference_and_rejects_leakage(tmp_path):
    cache = tmp_path / "cache"
    splits = {"train": ["new-train"], "validation": ["new-select"], "calibration": ["new-cal"], "evaluation": ["new-test"]}
    write_json(cache / "manifest.json", {"splits": splits})
    write_json(cache / "preprocessing.json", {"version": 1})
    old = tmp_path / "old.json"
    write_json(old, {"splits": {"train": ["old-train"], "validation": ["old-select"], "calibration": ["old-cal"]}})
    protocol = create_protocol(cache, tmp_path / "protocol.json", [old])
    checkpoint = {"manifest_hash": manifest_hash(old), "preprocessing_hash": manifest_hash(cache / "preprocessing.json")}
    assert validate_benchmark(checkpoint, cache, protocol)["training_selection_calibration_disjoint"]
    leaked = copy.deepcopy(protocol)
    leaked["provenance_manifests"][checkpoint["manifest_hash"]]["splits"]["calibration"] = ["new-test"]
    with pytest.raises(ValueError, match="leaks"):
        validate_benchmark(checkpoint, cache, leaked)
    changed = copy.deepcopy(checkpoint)
    changed["preprocessing_hash"] = "unknown"
    with pytest.raises(ValueError, match="preprocessing"):
        validate_benchmark(changed, cache, protocol)
    with pytest.raises(ValueError, match="immutable"):
        create_protocol(cache, tmp_path / "protocol.json", [])
    changed = copy.deepcopy(protocol)
    changed["split_hashes"]["evaluation"] = "tampered"
    with pytest.raises(ValueError, match="membership"):
        validate_benchmark(checkpoint, cache, changed)


def test_paired_bootstrap_matches_hand_computed_scenarios_and_rejects_unmatched():
    candidate = [{"scenario_id": str(i), "jointFDE": 1., "minFDE": [0., 2.]} for i in range(4)]
    reference = [{"scenario_id": str(i), "jointFDE": 3., "minFDE": [2., 4.]} for i in range(4)]
    result = paired_comparison(candidate, reference, 100, 42)
    assert result["jointFDE_gate_passed"]
    assert result["metrics"]["jointFDE"]["delta"] == -2
    assert result["metrics"]["jointFDE"]["interval95"] == [-2., -2.]
    assert result["metrics"]["minFDE"]["delta"] == -2
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(candidate, reference[:-1])
    with pytest.raises(ValueError, match="one record"):
        paired_comparison(candidate+[candidate[0]], reference)


def test_full_listing_selection_excludes_all_inspected_and_is_immutable(tmp_path):
    uid = lambda n: str(uuid.UUID(int=n))
    project = tmp_path / "project"
    write_json(project / "artifacts" / "smoke" / "manifest.json", {"splits": {"train": [uid(1)], "evaluation": [uid(101)]}})
    listings = {"train": [uid(n) for n in range(1, 11)], "val": [uid(n) for n in range(101, 121)]}
    counts = {"train": 4, "validation": 2, "calibration": 2, "evaluation": 2}
    root = tmp_path / "study"
    def listing(split, directory):
        write_json(directory / f"{split}.json", {"ids": listings[split]})
        return listings[split]
    with patch("adp.research._listing", side_effect=listing), patch("adp.research.ensure_space"):
        manifest = prepare_study(root, project, counts)
        assert prepare_study(root, project, counts) == manifest
    val = [sid for split, ids in manifest["splits"].items() if split != "train" for sid in ids]
    assert uid(101) not in val and len(set(val)) == 6
    assert manifest["excluded_validation_ids"] == [uid(101)]
    assert read_json(root / "pilot" / "raw" / "manifest.json")["splits"]["evaluation"] == []
    with pytest.raises(ValueError, match="immutable"):
        prepare_study(root, project, counts, seed=43)
    (root / "pilot" / "raw" / "manifest.json").unlink()
    assert prepare_study(root, project, counts) == manifest
    assert (root / "pilot" / "raw" / "manifest.json").is_file()


def test_storage_guard_and_configuration_grid(tmp_path):
    with patch("adp.research.shutil.disk_usage") as disk:
        disk.return_value.free = 8 * 1024**3
        ensure_space(tmp_path)
        with pytest.raises(RuntimeError, match="Storage paused"):
            ensure_space(tmp_path, additional_bytes=1)
    original = {"seed": 99, "model": {"width": 128}, "training": {"batch_size": 4}}
    grid = dict(study_configs(original))
    assert grid["world_ce"]["model"]["shared_world_ce_weight"] == .1
    assert not grid["no_map"]["model"]["use_map"]
    assert not grid["no_graph"]["model"]["use_graph"]
    assert original["model"] == {"width": 128}
    from adp.research import remaining_storage
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "partial.bin").write_bytes(b"x" * 12)
    assert remaining_storage(tmp_path, 20) == 8


def test_overlap_matches_reference_masks_and_separates_existing_from_new():
    initial = np.array([[0., 0.], [0., 0.]])
    headings = np.zeros(2)
    truth = np.zeros((2, 2, 2))
    prediction = np.zeros((1, 2, 2, 2))
    kwargs = dict(actor_types=np.array([0, 0]), dynamic=np.array([True, True]),
                  probabilities=np.array([1.]), initial_positions=initial, initial_headings=headings)
    result = matched_overlap_diagnostics(prediction, truth, np.ones((2, 2), dtype=bool), **kwargs)
    assert result["reference_overlap_scene"] == 1
    assert result["predicted_matched_overlap_scene"] == 1
    assert result["reference_new_overlap_scene"] == 0
    assert result["initial_overlap_pair_rate"] == 1
    kwargs["initial_positions"] = np.array([[0., 0.], [10., 0.]])
    kwargs["actor_types"] = np.array([1, 1])
    kwargs["initial_headings"] = np.zeros(2)
    result = matched_overlap_diagnostics(prediction, truth, np.ones((2, 2), dtype=bool), **kwargs)
    assert result["reference_new_overlap_scene"] == 1
    result = matched_overlap_diagnostics(prediction, truth, np.zeros((2, 2), dtype=bool), **kwargs)
    assert result == {"matched_pair_count": 0, "matched_pair_timestamps": 0}


def test_study_refuses_unmeasured_configuration_before_training(tmp_path):
    from adp.research import config_fingerprint, run_study
    measured = {"model": {"width": 128}, "training": {"batch_size": 4}}
    write_json(tmp_path / "pilot.json", {"config_sha256": config_fingerprint(measured)})
    with pytest.raises(ValueError, match="configuration differs"):
        run_study(tmp_path, {"model": {"width": 512}}, "cpu")


def test_finalization_rejects_a_truncated_cached_holdout_before_promotion(tmp_path):
    from adp.research import finalize_study
    root = tmp_path / "study"
    write_json(root / "cache" / "manifest.json", {"splits": {"evaluation": ["a", "b"]}})
    write_json(root / "protocol.json", {"version": 1})
    write_json(root / "cv_scale.json", {"scale_m": 1., "split": "calibration", "manifest_sha256": manifest_hash(root / "cache" / "manifest.json")})
    runs = ["full-42", "full-43", "full-44"]
    for name in runs:
        path = root / "runs" / name / "best.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"frozen fixture weights")
    checkpoint = root / "runs" / runs[0] / "best.pt"
    reference = root / "reference.pt"
    reference.write_bytes(b"reference fixture weights")
    write_json(root / "frozen_selection.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": manifest_hash(checkpoint),
        "all_seed_runs": runs, "deployment_run": runs[0]})
    calibration = root / "calibration" / runs[0] / "calibration.json"
    write_json(calibration, {"fixture": True})
    report = {"checkpoint_sha256": manifest_hash(checkpoint), "manifest_hash": manifest_hash(root / "cache" / "manifest.json"),
        "benchmark_protocol_sha256": manifest_hash(root / "protocol.json"), "calibration_sha256": manifest_hash(calibration),
        "calibration_applied": True, "split": "evaluation", "scenario_limit": None,
        "constant_velocity_scale_m": None, "stress": None, "scenario_count": 1, "bootstrap": {"replicates": 1000}}
    output = root / "evaluation" / runs[0]
    write_json(output / "metrics.json", report)
    write_json(output / "records.json", [{"scenario_id": "a", "jointFDE": 0.}])
    with pytest.raises(ValueError, match="incomplete"):
        finalize_study(root, reference, "cpu")
    assert not (root / "promotion.json").exists()


def test_inference_operations_share_the_training_budget_and_refund_unused_slice(tmp_path):
    from adp.research import metered_operation
    write_json(tmp_path / "study_ledger.json", {"elapsed_seconds": 10., "runs": {}})
    with patch("adp.research.time.monotonic", side_effect=[0., .5, 1.]):
        with metered_operation(tmp_path, "cuda") as check:
            pass
    assert read_json(tmp_path / "study_ledger.json")["elapsed_seconds"] == 11.
    write_json(tmp_path / "study_ledger.json", {"elapsed_seconds": 40*3600-1, "runs": {}})
    with patch("adp.research.time.monotonic", side_effect=[0., 2., 2.]):
        with pytest.raises(RuntimeError, match="ceiling"):
            with metered_operation(tmp_path, "cuda") as check:
                check()
    with pytest.raises(RuntimeError, match="ceiling"):
        with metered_operation(tmp_path, "cuda"):
            pass


def test_pipeline_restores_windows_sleep_state_after_failure(tmp_path):
    from unittest.mock import MagicMock, call
    from adp.research import run_pipeline
    native = MagicMock()
    native.SetThreadExecutionState.return_value = 0x80000000
    with patch("sys.platform", "win32"), patch("ctypes.WinDLL", return_value=native, create=True), \
            patch("adp.research._run_pipeline", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            run_pipeline(tmp_path, tmp_path, {}, tmp_path / "reference.pt")
    assert native.SetThreadExecutionState.call_args_list == [call(0x80000001), call(0x80000000)]
