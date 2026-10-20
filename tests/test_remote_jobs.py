"""Remote development jobs isolate data, ownership, checkpoints and compute caps."""
import json
from pathlib import Path
import uuid
from unittest.mock import patch

import pytest

from adp.data import read_json, write_json
from adp.remote_jobs import _owner_lock, main, prepare_job, run_job, select_ids


@pytest.fixture
def staged(tmp_path):
    root = tmp_path / "remote-\u03b4"
    cache = root / "cache"
    splits = {"train": [str(uuid.UUID(int=1)), str(uuid.UUID(int=2))],
              "validation": [str(uuid.UUID(int=3))]}
    write_json(cache / "manifest.json", {"splits": splits})
    write_json(cache / "preprocessing.json", {"observed_steps": 50, "future_steps": 60})
    for split, ids in splits.items():
        (cache / split).mkdir()
        for sid in ids:
            (cache / split / f"{sid}.npz").write_bytes(b"stand-in cache; never opened as model input")
    write_json(root / "source-manifest.json", {"splits": {
        **splits, "calibration": [str(uuid.UUID(int=4))], "evaluation": [str(uuid.UUID(int=5))]}})
    return root


def freeze(root, **kwargs):
    return prepare_job(root, {"seed": 42, "model": {"graph": True},
        "training": {"batch_size": 4, "amp": True, "learning_rate": .0003}}, **kwargs)


def test_subset_selection_is_order_independent_and_does_not_repeat():
    assert select_ids(["c", "a", "b"], 2) == select_ids(["b", "a", "c"], 2)
    assert select_ids(["a"], 5) == ["a"]
    with pytest.raises(ValueError, match="duplicate"):
        select_ids(["a", "a"], 1)
    with pytest.raises(ValueError, match="positive"):
        select_ids(["a"], True)


def test_prepare_freezes_portable_data_and_conservative_940mx_configuration(staged):
    protocol = freeze(staged)
    assert protocol == freeze(staged)
    assert protocol["split_counts"] == {"train": 2, "validation": 1}
    assert protocol["maximum_compute_seconds"] == 1800
    assert "not frozen-study" in protocol["purpose"]
    settings = read_json(staged / "config.json")["training"]
    assert settings == {"batch_size": 1, "accumulation_steps": 32, "amp": False,
                        "learning_rate": .0003, "num_workers": 0, "checkpoint_seconds": 300,
                        "minimum_free_gib": 8}
    assert len(protocol["cache_files"]) == 5
    assert "training.py" in protocol["source_files"]
    with pytest.raises(ValueError, match="immutable"):
        freeze(staged, epochs=2)


def test_prepare_rejects_foreign_or_overlapping_scenarios_and_extra_data(staged):
    path = staged / "cache" / "manifest.json"
    manifest = read_json(path)
    manifest["splits"]["calibration"] = []
    write_json(path, manifest)
    with pytest.raises(ValueError, match="selection validation only"):
        freeze(staged)
    del manifest["splits"]["calibration"]
    manifest["splits"]["validation"] = manifest["splits"]["train"][:1]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="unique and disjoint"):
        freeze(staged)
    manifest["splits"]["validation"] = [str(uuid.UUID(int=3))]
    write_json(path, manifest)
    source = read_json(staged / "source-manifest.json")
    source["splits"]["validation"] = []
    write_json(staged / "source-manifest.json", source)
    with pytest.raises(ValueError, match="originate"):
        freeze(staged)
    (staged / "cache" / "calibration").mkdir()
    with pytest.raises(ValueError, match="Unexpected"):
        freeze(staged)


def test_prepare_rejects_traversal_and_unlisted_files(staged):
    (staged / "cache" / "unexpected.npz").write_bytes(b"never train this")
    with pytest.raises(ValueError, match="files differ"):
        freeze(staged)
    (staged / "cache" / "unexpected.npz").unlink()
    path = staged / "cache" / "manifest.json"
    manifest = read_json(path)
    manifest["splits"]["train"] = ["../../secret"]
    write_json(path, manifest)
    with pytest.raises(ValueError):
        freeze(staged)


def test_preflight_detects_changed_source_config_and_cache_before_training(staged):
    protocol = freeze(staged)
    with patch("adp.remote_jobs._source_files", return_value={}), patch("adp.training.train") as train:
        with pytest.raises(ValueError, match="source differs"):
            run_job(staged)
        train.assert_not_called()
    config_path = staged / "config.json"
    config = read_json(config_path)
    write_json(config_path, {**config, "seed": 43})
    with pytest.raises(ValueError, match="configuration"):
        run_job(staged)
    write_json(config_path, config)
    (staged / "cache" / next(name for name in protocol["cache_files"] if name.endswith(".npz"))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="data provenance"):
        run_job(staged)
    assert not (staged / "compute-ledger.json").exists()
    assert not (staged / ".remote-job.lock").exists()


def test_resume_uses_own_checkpoint_and_never_restarts_completed_job(staged):
    freeze(staged, max_runtime_seconds=20)
    invocations = []

    def train(config, cache, output, **kwargs):
        invocations.append(kwargs)
        output.mkdir(exist_ok=True)
        (output / "last.pt").write_bytes(b"committed optimizer state")
        return {"completed_epochs": len(invocations) - 1, "budget_exhausted": len(invocations) == 1,
                "training_examples_per_second": 12.5}

    with patch("adp.training.train", side_effect=train), patch("adp.research.ensure_space") as space, \
            patch("adp.remote_jobs.time.monotonic", side_effect=[10, 14, 20, 25]):
        first = run_job(staged, device="cpu", slice_seconds=8)
        second = run_job(staged, device="cpu", slice_seconds=8)
        assert run_job(staged, device="cpu", slice_seconds=8) == second
    assert first["state"] == "paused" and second["state"] == "completed"
    assert invocations[0]["resume"] is None
    assert invocations[1]["resume"] == staged / "run" / "last.pt"
    assert all(call["max_runtime_seconds"] == 8 for call in invocations)
    assert second["charged_seconds"] == 9 and second["remaining_seconds"] == 11
    ledger = read_json(staged / "compute-ledger.json")
    assert [entry["charged_seconds"] for entry in ledger["slices"]] == [4, 5]
    assert space.call_count == 2
    assert not (staged / ".remote-job.lock").exists()


def test_failure_charges_actual_time_and_keeps_partial_checkpoint(staged):
    freeze(staged)

    def interrupted(*args, **kwargs):
        (staged / "run").mkdir()
        (staged / "run" / "last.pt").write_bytes(b"last committed group")
        raise KeyboardInterrupt("operator interrupted")

    with patch("adp.training.train", side_effect=interrupted), patch("adp.research.ensure_space"), \
            patch("adp.remote_jobs.time.monotonic", side_effect=[10, 13]):
        with pytest.raises(KeyboardInterrupt):
            run_job(staged)
    assert read_json(staged / "compute-ledger.json")["charged_seconds"] == 3
    assert read_json(staged / "status.json")["state"] == "failed"
    assert (staged / "run" / "last.pt").read_bytes() == b"last committed group"
    assert not (staged / ".remote-job.lock").exists()


def test_crash_reservation_cannot_be_reused_and_foreign_ledger_is_rejected(staged):
    protocol = freeze(staged)
    ledger = {"version": 1, "protocol_sha256": protocol["protocol_sha256"], "charged_seconds": 1800,
              "slices": [{"state": "reserved", "reserved_seconds": 1800, "charged_seconds": 1800}]}
    write_json(staged / "compute-ledger.json", ledger)
    with patch("adp.training.train") as train:
        result = run_job(staged)
        assert result["state"] == "budget_exhausted"
        train.assert_not_called()
    write_json(staged / "compute-ledger.json", {**ledger, "protocol_sha256": "foreign"})
    with pytest.raises(ValueError, match="another protocol"):
        run_job(staged)


def test_owner_lock_prevents_duplicate_worker_and_preserves_existing_owner(staged):
    freeze(staged)
    with _owner_lock(staged):
        owner = (staged / ".remote-job.lock").read_bytes()
        with pytest.raises(RuntimeError, match="already has an owner"):
            run_job(staged)
        assert (staged / ".remote-job.lock").read_bytes() == owner
    assert not (staged / ".remote-job.lock").exists()


def test_bad_budget_and_device_are_rejected_without_gpu_access(staged):
    for seconds in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="budget"):
            freeze(staged, max_runtime_seconds=seconds)
        with pytest.raises(ValueError, match="duration"):
            run_job(staged, slice_seconds=seconds)
    with pytest.raises(ValueError, match="explicitly"):
        run_job(staged, device="auto")


def test_cli_prepare_and_run_dispatch(staged, capsys):
    config = staged / "input.json"
    write_json(config, {"model": {}, "seed": 42})
    main(["prepare", "--root", str(staged), "--config", str(config)])
    assert json.loads(capsys.readouterr().out)["epochs"] == 1
    with patch("adp.remote_jobs.run_job", return_value={"state": "paused"}) as run:
        main(["run", "--root", str(staged), "--device", "cpu", "--slice-seconds", "10"])
    run.assert_called_once_with(staged, device="cpu", slice_seconds=10.)
    assert json.loads(capsys.readouterr().out) == {"state": "paused"}
