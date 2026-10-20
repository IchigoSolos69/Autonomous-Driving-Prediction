"""The queued workflow never trains before successful reference completion."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from adp.data import read_json, write_json
from adp.followup import _analysis, _prevent_idle_sleep, run_direction_followup


@pytest.fixture(autouse=True)
def storage_headroom(monkeypatch):
    monkeypatch.setattr("adp.followup.ensure_space", lambda *args, **kwargs: None)


def _reference(tmp_path, status="completed", phase="finished"):
    root = tmp_path / "reference"
    write_json(root / "pipeline_status.json", {"status": status, "phase": phase})
    return root


def test_wait_does_not_charge_gpu_or_analyze_before_reference_completion(tmp_path):
    reference = _reference(tmp_path, "running", "train")
    output = tmp_path / "direction"
    order = []
    def finish_reference(seconds):
        assert seconds == 30
        assert order == []
        assert not (output / "direction_ledger.json").exists()
        assert read_json(output / "followup_status.json")["status"] == "waiting"
        write_json(reference / "pipeline_status.json", {"status": "completed", "phase": "finished"})
    def analyze(study, path):
        order.append("reference analysis" if study == reference else "direction analysis")
        return {"output_dir": str(path)}
    def train(*args):
        order.append("direction study")
        return {"status": "completed"}
    with patch("adp.followup._prevent_idle_sleep", return_value=nullcontext()), \
         patch("adp.followup.time.sleep", side_effect=finish_reference) as sleep, \
         patch("adp.followup._analysis", side_effect=analyze), \
         patch("adp.direction.run_direction_study", side_effect=train):
        result = run_direction_followup(reference, output)
    assert result["status"] == "completed"
    assert order == ["reference analysis", "direction study", "direction analysis"]
    sleep.assert_called_once_with(30)
    assert (output / "followup_request.json").exists()


@pytest.mark.parametrize("status,phase", [("stopped", "train"), ("failed", "acquire"),
    ("incomplete_experiment", "train"), ("completed", "train"), ("unknown", "train")])
def test_terminal_or_invalid_reference_stops_without_followup_work(tmp_path, status, phase):
    reference = _reference(tmp_path, status, phase)
    with patch("adp.followup._prevent_idle_sleep", return_value=nullcontext()), \
         patch("adp.followup.time.sleep") as sleep, patch("adp.followup._analysis") as analysis, \
         patch("adp.direction.run_direction_study") as train:
        result = run_direction_followup(reference, tmp_path / "direction")
    assert result["status"] == "reference_stopped"
    assert result["reference_status"]["status"] == status
    assert not result["followup_training_allowed"]
    analysis.assert_not_called()
    train.assert_not_called()
    sleep.assert_not_called()


def test_incomplete_direction_experiment_preserves_reference_report_and_skips_final_analysis(tmp_path):
    reference = _reference(tmp_path)
    with patch("adp.followup._prevent_idle_sleep", return_value=nullcontext()), \
         patch("adp.followup._analysis", return_value={"id": "reference-report"}) as analysis, \
         patch("adp.direction.run_direction_study", return_value={"status": "incomplete_experiment", "reason": "12-hour cap"}):
        result = run_direction_followup(reference, tmp_path / "direction", device="cpu")
    assert result["status"] == "incomplete_experiment" and result["reason"] == "12-hour cap"
    assert result["reference_analysis"]["id"] == "reference-report"
    assert "direction_analysis" not in result
    analysis.assert_called_once_with(reference, tmp_path / "direction" / "reference-analysis")


def test_verified_existing_analysis_is_reused_and_other_provenance_rejected(tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    provenance = {"kind": "reference", "checkpoint_sha256": {"reference": "hash"}}
    metadata = {"id": "report-id", "provenance": provenance}
    with patch("adp.analysis._models", return_value=(None, None, provenance)), \
         patch("adp.analysis.validate_analysis", return_value=(metadata, [])) as validate, \
         patch("adp.analysis.analyze_failures") as build:
        result = _analysis(tmp_path / "study", folder)
        assert result["reused"]
        validate.assert_called_once_with(folder)
        build.assert_not_called()
    with patch("adp.analysis._models", return_value=(None, None, {"kind": "different"})), \
         patch("adp.analysis.validate_analysis", return_value=(metadata, [])), \
         patch("adp.analysis.analyze_failures") as build:
        with pytest.raises(ValueError, match="different inputs"):
            _analysis(tmp_path / "study", folder)
        build.assert_not_called()


def test_corrupt_existing_analysis_stops_without_replacing_it(tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    sentinel = folder / "user-note.txt"
    sentinel.write_text("preserve")
    with patch("adp.analysis._models", return_value=(None, None, {})), \
         patch("adp.analysis.validate_analysis", side_effect=ValueError("hash mismatch")), \
         patch("adp.analysis.analyze_failures") as build:
        with pytest.raises(ValueError, match="hash mismatch"):
            _analysis(tmp_path / "study", folder)
    assert sentinel.read_text() == "preserve"
    build.assert_not_called()


def test_non_windows_workflow_makes_no_windows_power_calls(tmp_path):
    reference = _reference(tmp_path, "stopped", "train")
    with patch("adp.followup.sys.platform", "linux"), patch("ctypes.WinDLL", create=True) as windows:
        result = run_direction_followup(reference, tmp_path / "direction")
    assert result["status"] == "reference_stopped"
    windows.assert_not_called()


def test_windows_idle_sleep_flags_are_restored_on_error():
    execution_state = Mock(side_effect=[0x80000000, 0x80000001])
    with patch("adp.followup.sys.platform", "win32"), \
         patch("ctypes.WinDLL", return_value=SimpleNamespace(SetThreadExecutionState=execution_state), create=True):
        with pytest.raises(RuntimeError, match="simulated"):
            with _prevent_idle_sleep():
                raise RuntimeError("simulated")
    assert execution_state.call_args_list == [call(0x80000001), call(0x80000000)]


def test_workflow_records_interruption_for_resume(tmp_path):
    reference = _reference(tmp_path, "running", "train")
    output = tmp_path / "direction"
    with patch("adp.followup._prevent_idle_sleep", return_value=nullcontext()), \
         patch("adp.followup.time.sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_direction_followup(reference, output)
    status = read_json(output / "followup_status.json")
    assert status["status"] == "interrupted" and status["resumable"]
    assert not (output / "direction_ledger.json").exists()


def test_source_change_during_wait_stops_before_analysis_or_training(tmp_path):
    reference = _reference(tmp_path, "running", "train")
    output = tmp_path / "direction"
    def complete(_):
        write_json(reference / "pipeline_status.json", {"status": "completed", "phase": "finished"})
    with patch("adp.followup._prevent_idle_sleep", return_value=nullcontext()), \
         patch("adp.followup._source_hash", side_effect=["queued-source", "changed-source"]), \
         patch("adp.followup.time.sleep", side_effect=complete), patch("adp.followup._analysis") as analysis, \
         patch("adp.direction.run_direction_study") as train:
        with pytest.raises(ValueError, match="source changed"):
            run_direction_followup(reference, output)
    assert read_json(output / "followup_status.json")["status"] == "stopped"
    analysis.assert_not_called()
    train.assert_not_called()


@pytest.mark.parametrize("interval", [0, 61, float("nan"), True])
def test_invalid_poll_interval_is_rejected_before_work(tmp_path, interval):
    reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="poll_seconds"):
        run_direction_followup(reference, tmp_path / "direction", interval)
    assert not (tmp_path / "direction").exists()
