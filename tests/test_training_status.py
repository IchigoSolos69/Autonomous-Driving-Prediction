"""Read-only checkpoint progress is bounded, CPU-only, cached, finite and explicit about stale state."""
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from adp.data import write_json
from adp.training_status import TrainingStatus, _checkpoint_metadata


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "original"
    write_json(root / "schedule.json", {"epochs": 15, "configs": []})
    write_json(root / "pipeline_status.json", {"status": "running", "phase": "train"})
    write_json(root / "study_ledger.json", {"elapsed_seconds": 7200, "runs": {
        "full-42": {"completed": False, "reserved_seconds": 900}}})
    write_json(root / "cache" / "manifest.json", {"splits": {"train": list(range(8))}})
    _checkpoint(root)
    return root


def _checkpoint(root, completed=2, cursor=4, name="full-42", history=None):
    path = root / "runs" / name / "last.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "completed_epochs": completed,
        "partial_epoch": cursor is not None, "training_progress": {"epoch": completed, "cursor": cursor, "order": list(range(8))} if cursor is not None else None,
        "global_step": 5, "config": {"training": {"batch_size": 2}},
        "statistics": {"examples_seen": 40, "training_seconds": 20},
        "history": history or [{"epoch": 1, "train_loss": 3., "validation_loss": 4.,
            "validation_joint_minFDE": 5., "validation_focal_minFDE": 6., "train_objective": {"nll": 2.}}],
        "model_state_dict": {"NEVER_EXPOSE_THIS_WEIGHT": torch.ones(3)},
        "rng_state": {"torch": torch.get_rng_state()}}, path)
    return path


@pytest.fixture(autouse=True)
def readable_storage(monkeypatch):
    monkeypatch.setattr("adp.training_status.shutil.disk_usage", lambda _: SimpleNamespace(free=12 * 1024**3))


def test_checkpoint_based_progress_and_history_are_truthful_and_finite(study, tmp_path):
    snapshot = TrainingStatus(study, tmp_path / "direction").snapshot()
    row = snapshot["studies"][0]
    assert row["id"] == "original" and row["status"] == "running" and row["phase"] == "train"
    assert row["progress"] == {"planned_epochs": 90, "completed_epochs": 2,
        "committed_epoch_fraction": .5, "overall_fraction": 2 / 90, "completed_runs": 0, "total_runs": 6}
    current = row["runs"][0]
    assert current["epoch_number"] == 3 and current["completed_epochs"] == 2
    assert current["committed_examples"] == 4 and current["training_examples"] == 8
    assert current["committed_batches"] == 2 and current["batches_per_epoch"] == 4
    assert current["examples_per_second"] == 2 and current["peak_vram_mib"] is None
    assert current["history"][0]["joint_minFDE"] == 5
    assert current["history"][0]["train_objective"] == {"nll": 2}
    assert row["budget"]["charged_hours"] == 2 and row["budget"]["reserved_hours"] == .25
    assert row["budget"]["remaining_hours"] == 38 and "not exact" in row["budget"]["note"]
    assert row["runs"][4]["configuration"] is None and row["runs"][4]["id"] == "winner-43"
    payload = json.dumps(snapshot, allow_nan=False)
    assert "NEVER_EXPOSE_THIS_WEIGHT" not in payload and str(study) not in payload


def test_combined_compute_hosts_use_configured_file_and_local_checkpoint_progress(study, tmp_path):
    from datetime import datetime, timezone
    host_file = tmp_path / "host-status.json"
    write_json(host_file, {"version": 1, "hosts": [{"id": "private", "status": "completed",
        "updated_at": datetime.now(timezone.utc).isoformat(), "gpu": "940MX",
        "training": {"completed_epochs": 1, "planned_epochs": 1}}]})
    snapshot = TrainingStatus(study, tmp_path / "direction", compute_hosts_path=host_file).snapshot()
    assert [host["id"] for host in snapshot["compute_hosts"]] == ["local", "private", "colab", "kaggle"]
    assert snapshot["compute_hosts"][0]["training"]["planned_epochs"] == 90
    assert snapshot["compute_hosts"][1]["status"] == "completed"
    assert snapshot["studies"][0]["progress"]["completed_epochs"] == 2


def test_cpu_checkpoint_load_uses_closed_bytes_and_is_not_repeated_while_unchanged(study, tmp_path):
    provider = TrainingStatus(study, tmp_path / "direction")
    original = torch.load
    def load(content, **kwargs):
        assert isinstance(content, io.BytesIO)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return original(content, **kwargs)
    with patch("torch.load", side_effect=load) as called, patch("torch.cuda.init", side_effect=AssertionError("GPU initialized")):
        provider.snapshot()
        provider.snapshot()
    called.assert_called_once()
    assert all("model_state_dict" not in value["value"] for path, value in provider._cache.items() if path.name == "last.pt")


def test_changed_checkpoint_reloads_but_corruption_preserves_last_good_metadata(study, tmp_path):
    provider = TrainingStatus(study, tmp_path / "direction")
    path = study / "runs" / "full-42" / "last.pt"
    with patch("adp.training_status._checkpoint_metadata", wraps=_checkpoint_metadata) as decode:
        provider.snapshot()
        previous = path.stat().st_mtime
        _checkpoint(study, completed=3)
        os.utime(path, (previous + 2, previous + 2))
        assert provider.snapshot()["studies"][0]["runs"][0]["completed_epochs"] == 3
        path.write_bytes(b"corrupt checkpoint")
        os.utime(path, (previous + 4, previous + 4))
        value = provider.snapshot()["studies"][0]["runs"][0]
        assert value["completed_epochs"] == 3 and value["issue"] and value["checkpoint"]["stale"]
        provider.snapshot()
        assert decode.call_count == 3


def test_validation_pending_does_not_count_as_completed_epoch(study, tmp_path):
    _checkpoint(study, completed=2, cursor=8)
    value = TrainingStatus(study, tmp_path / "direction").snapshot()["studies"][0]
    assert value["runs"][0]["stage"] == "validation_pending"
    assert value["progress"]["completed_epochs"] == 2
    assert value["progress"]["committed_epoch_fraction"] == 1
    assert value["progress"]["overall_fraction"] == 2 / 90


def test_progress_budget_and_nonfinite_metrics_are_clamped(study, tmp_path):
    _checkpoint(study, completed=999, cursor=999, history=[{"epoch": 1, "train_loss": float("nan"), "validation_loss": float("inf")}])
    write_json(study / "study_ledger.json", {"elapsed_seconds": 41 * 3600, "runs": {"full-42": {"completed": False}}})
    row = TrainingStatus(study, tmp_path / "direction").snapshot()["studies"][0]
    assert row["progress"]["completed_epochs"] == 15 and row["progress"]["overall_fraction"] == 1 / 6
    assert row["runs"][0]["committed_fraction"] == 1 and "clamped" in row["runs"][0]["issue"]
    assert row["runs"][0]["history"][0]["train_loss"] is None
    assert row["budget"]["remaining_hours"] == 0
    json.dumps(row, allow_nan=False)


def test_missing_and_corrupt_files_do_not_write_or_crash(tmp_path):
    original, direction = tmp_path / "missing-original", tmp_path / "missing-direction"
    provider = TrainingStatus(original, direction)
    snapshot = provider.snapshot()
    assert [item["status"] for item in snapshot["studies"]] == ["not_started", "not_started"]
    assert not original.exists() and not direction.exists()
    original.mkdir()
    (original / "pipeline_status.json").write_text("{not valid JSON")
    row = provider.snapshot()["studies"][0]
    assert row["status"] == "unavailable" and row["issues"]


def test_malformed_valid_json_containers_are_actionable_instead_of_api_errors(study, tmp_path):
    write_json(study / "study_ledger.json", {"runs": {"full-42": "invalid"}})
    write_json(study / "cache" / "manifest.json", {"splits": ["invalid"]})
    write_json(study / "schedule.json", {"epochs": 15, "configs": "invalid"})
    row = TrainingStatus(study, tmp_path / "direction").snapshot()["studies"][0]
    assert row["status"] == "unavailable"
    assert len(row["issues"]) >= 3
    json.dumps(row, allow_nan=False)
    write_json(study / "cache" / "manifest.json", {"splits": {"train": "invalid"}})
    assert TrainingStatus(study, tmp_path / "direction").snapshot()["studies"][0]["status"] == "unavailable"


def test_stopped_storage_floor_and_stale_checkpoint_are_visible(study, tmp_path):
    write_json(study / "pipeline_status.json", {"status": "stopped", "phase": "train", "error": "Storage paused: preserve 8 GiB"})
    path = study / "runs" / "full-42" / "last.pt"
    os.utime(path, (100, 100))
    with patch("adp.training_status.shutil.disk_usage", return_value=SimpleNamespace(free=3.37 * 1024**3)), \
         patch("adp.training_status.time.time", return_value=1000):
        row = TrainingStatus(study, tmp_path / "direction").snapshot()["studies"][0]
    assert row["status"] == "stopped" and "Storage paused" in row["message"]
    assert row["storage"]["below_reserve"] and row["storage"]["free_gib"] == 3.37
    assert row["runs"][0]["state"] == "stopped"
    assert row["runs"][0]["checkpoint"]["age_seconds"] == 900
    assert any("does not prove" in issue for issue in row["issues"])


def test_direction_waiting_has_unknown_schedule_and_separate_cap(tmp_path):
    direction = tmp_path / "direction"
    write_json(direction / "followup_status.json", {"status": "waiting", "phase": "waiting_for_reference"})
    row = TrainingStatus(tmp_path / "original", direction).snapshot()["studies"][1]
    assert row["status"] == "waiting" and row["progress"]["planned_epochs"] is None
    assert row["budget"]["limit_hours"] == 12 and len(row["runs"]) == 6
    assert row["current_run_id"] is None and "no follow-up training" in row["message"]


def test_direction_reservation_and_actual_run_memory_are_reported_without_pilot_substitution(tmp_path):
    direction = tmp_path / "direction"
    write_json(direction / "schedule.json", {"epochs": 5, "configs": []})
    write_json(direction / "followup_status.json", {"status": "running", "phase": "direction_study"})
    write_json(direction / "direction_ledger.json", {"elapsed_seconds": 3600, "operations": [
        {"label": "train-control-42", "status": "reserved", "reserved_seconds": 900}]})
    _checkpoint(direction, completed=1, name="control-42")
    write_json(direction / "pilot.json", {"peak_torch_allocated_bytes": 999999999})
    row = TrainingStatus(tmp_path / "original", direction).snapshot()["studies"][1]
    assert row["progress"]["planned_epochs"] == 30 and row["current_run_id"] == "control-42"
    assert row["budget"]["reserved_hours"] == .25 and row["runs"][0]["peak_vram_mib"] is None
    assert row["runs"][0]["training_examples"] == 8


def test_selected_seed_rows_and_completed_epoch_totals_use_real_identifiers(study, tmp_path):
    write_json(study / "schedule.json", {"epochs": 2, "configs": []})
    names = ["full-42", "no_graph-42", "no_map-42", "world_ce-42", "no_graph-43", "no_graph-44"]
    write_json(study / "study_ledger.json", {"elapsed_seconds": 100, "runs": {
        name: {"completed": True, "result": {"peak_cuda_allocated_bytes": 128 * 1024**2}} for name in names}})
    write_json(study / "pipeline_status.json", {"status": "running", "phase": "finalize"})
    for name in names:
        _checkpoint(study, completed=2, cursor=None, name=name)
    row = TrainingStatus(study, tmp_path / "direction").snapshot()["studies"][0]
    assert row["progress"]["planned_epochs"] == row["progress"]["completed_epochs"] == 12
    assert row["progress"]["completed_runs"] == 6 and row["progress"]["overall_fraction"] == 1
    assert row["current_run_id"] is None and row["phase"] == "finalize"
    assert row["runs"][4]["id"] == "no_graph-43"
    assert row["runs"][0]["peak_vram_mib"] == 128
