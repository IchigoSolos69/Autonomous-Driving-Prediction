from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adp.cli import _evaluate
from adp.stress import StressEligibilityError


def evaluation_args(tmp_path, **overrides):
    values = dict(device="cpu", cache_dir=tmp_path / "cache", split="evaluation",
                  calibration=None, checkpoint=None, limit=None, bootstrap=0, seed=42,
                  kind="cut_in", severity=0.35, output_dir=tmp_path / "report")
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("invalid", [{"severity": 2}, {"severity": float("nan")},
                                    {"severity": float("inf")}, {"kind": "unknown"},
                                    {"seed": -1}, {"seed": 2**32}])
def test_invalid_stress_fails_before_loading_data_or_writing_reports(tmp_path, invalid):
    args = evaluation_args(tmp_path, **invalid)
    with patch("adp.data.CachedDataset") as dataset, patch("adp.reporting.write_report") as write:
        with pytest.raises(ValueError):
            _evaluate(args, stress=True)
        dataset.assert_not_called()
        write.assert_not_called()
    assert not args.output_dir.exists()


def test_valid_but_unsupported_counterfactual_is_recorded_as_skipped(tmp_path):
    dataset = [{"id": "no-adjacent-lane"}]
    with patch("adp.data.CachedDataset") as cached, patch("adp.reporting.write_report") as write, \
            patch("adp.stress.apply_stress", side_effect=StressEligibilityError("No adjacent lane")):
        cached.return_value.__len__.return_value = len(dataset)
        cached.return_value.__getitem__.side_effect = dataset.__getitem__
        cached.return_value.manifest_hash = "manifest"
        report = _evaluate(evaluation_args(tmp_path), stress=True)
    assert report["scenario_count"] == 0
    assert report["skipped"] == [{"scenario_id": "no-adjacent-lane", "reason": "No adjacent lane"}]
    assert report["risk"]["raw_source"] == "constant_velocity_heuristic"
    write.assert_called_once()


def test_stress_programming_error_is_not_silently_skipped(tmp_path):
    with patch("adp.data.CachedDataset") as cached, patch("adp.reporting.write_report") as write, \
            patch("adp.stress.apply_stress", side_effect=ValueError("Malformed history")):
        cached.return_value.__len__.return_value = 1
        cached.return_value.__getitem__.return_value = {"id": "broken"}
        with pytest.raises(ValueError, match="Malformed history"):
            _evaluate(evaluation_args(tmp_path), stress=True)
        write.assert_not_called()
