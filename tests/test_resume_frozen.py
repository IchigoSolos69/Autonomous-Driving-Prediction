"""The one-off resume wrapper preserves exact benchmark provenance and original code."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from adp.data import manifest_hash, read_json, write_json
from adp.protocol import create_protocol

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "resume-frozen-study.py"
_SPEC = importlib.util.spec_from_file_location("resume_frozen_study_script", _PATH)
helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helper)


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "study"
    manifest = {"splits": {"train": ["a"], "validation": ["b"], "calibration": ["c"], "evaluation": ["d"]}}
    write_json(root / "cache" / "manifest.json", manifest)
    (root / "raw").mkdir()
    (root / "raw" / "manifest.json").write_bytes((root / "cache" / "manifest.json").read_bytes())
    old = root / "old-manifest.json"
    write_json(old, {"splits": {"train": ["old-a"], "validation": ["old-b"]}})
    snapshot = root / "provenance" / f"{manifest_hash(old)}.json"
    snapshot.parent.mkdir()
    snapshot.write_bytes(old.read_bytes())
    create_protocol(root / "cache", root / "protocol.json", [snapshot])
    extra = root / "provenance" / "new-test-manifest.json"
    write_json(extra, {"splits": {"train": ["synthetic-test"]}})
    config, checkpoint = tmp_path / "config.json", tmp_path / "best.pt"
    write_json(config, {"seed": 42})
    checkpoint.write_bytes(b"reference-checkpoint-hash-only-fixture")
    return root, config, checkpoint, snapshot, extra


def test_resume_ignores_new_manifests_but_original_protocol_equality_and_unrelated_calls_remain(frozen, tmp_path):
    import adp.protocol as protocol
    root, config, checkpoint, snapshot, extra = frozen
    original, protocol_bytes = protocol.create_protocol, (root / "protocol.json").read_bytes()
    project = Path(__file__).resolve().parents[1]
    def pipeline(study, source, settings, reference, device, workers):
        assert study == root and source == project and settings == {"seed": 42}
        assert reference == checkpoint and device == "cpu" and workers == 8
        # This is exactly the call made by the unmodified original pipeline.
        value = protocol.create_protocol(root / "cache", root / "protocol.json", (root / "provenance").glob("*.json"))
        assert set(value["provenance_manifests"]) == {manifest_hash(root / "cache" / "manifest.json"), manifest_hash(snapshot)}
        other = tmp_path / "unrelated-protocol.json"
        protocol.create_protocol(root / "cache", other, [snapshot, extra])
        assert len(read_json(other)["provenance_manifests"]) == 3
        return {"promotion_gate_passed": False}
    with patch("adp.research.run_pipeline", side_effect=pipeline), patch("adp.research.ensure_space"):
        result = helper.resume_frozen_study(root, project, config, checkpoint, "cpu")
    assert protocol.create_protocol is original
    assert (root / "protocol.json").read_bytes() == protocol_bytes
    assert snapshot.is_file() and extra.is_file()
    receipt = read_json(result["receipt"])
    assert receipt["status"] == "completed" and receipt["protocol_unchanged"] and receipt["original_source_unchanged"]
    assert receipt["additional_at_start"]["count"] == 1
    assert receipt["protocol_calls"][0]["ignored_additional"]["files"][0]["sha256"] == manifest_hash(extra)


@pytest.mark.parametrize("changed", ["snapshot", "cache", "raw", "policy"])
def test_changed_frozen_inputs_refuse_before_pipeline(frozen, changed):
    root, config, checkpoint, snapshot, _ = frozen
    if changed == "snapshot":
        snapshot.write_text('{"splits":{}}')
    elif changed in ("cache", "raw"):
        write_json(root / changed / "manifest.json", {"splits": {"train": ["changed"]}})
    else:
        value = read_json(root / "protocol.json")
        value["policy"] = "altered policy"
        write_json(root / "protocol.json", value)
    with patch("adp.research.run_pipeline") as pipeline, patch("adp.research.ensure_space"):
        with pytest.raises(ValueError):
            helper.resume_frozen_study(root, Path(__file__).resolve().parents[1], config, checkpoint, "cpu")
    pipeline.assert_not_called()
    assert not (root / "resume-receipts").exists()


def test_wrapper_restored_and_failure_receipt_written_when_pipeline_stops(frozen):
    import adp.protocol as protocol
    root, config, checkpoint, _, _ = frozen
    original = protocol.create_protocol
    with patch("adp.research.run_pipeline", side_effect=RuntimeError("storage floor")), patch("adp.research.ensure_space"):
        with pytest.raises(RuntimeError, match="storage floor"):
            helper.resume_frozen_study(root, Path(__file__).resolve().parents[1], config, checkpoint, "cpu")
    assert protocol.create_protocol is original
    receipt = read_json(next((root / "resume-receipts").glob("*.json")))
    assert receipt["status"] == "stopped" and receipt["protocol_unchanged"] and receipt["original_source_unchanged"]
