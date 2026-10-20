from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from adp import compute_queue, drive_evidence as drive


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(drive._bytes(value))


@pytest.fixture
def queue(tmp_path):
    root, saved = tmp_path / "runtime", tmp_path / "Drive/InteractionLab/compute-v1/colab/development-v1"
    expected, protocol = drive.expected_plan("colab")
    jobs = [{"id":identifier, "spec":spec, "state":"queued"} for identifier, spec in expected.items()]
    jobs[0]["state"] = "completed"
    first = jobs[0]
    _write(root / "jobs" / first["id"] / "run/history.json", {"epochs":[{}] * 20})
    provenance = {"queue_job_id":first["id"], "queue_job_spec":first["spec"],
        "payload_sha256":drive.PAYLOAD_SHA256, "queue_protocol_sha256":protocol}
    files = {"best.pt":b"saved best checkpoint bytes", "last.pt":b"saved final checkpoint bytes"}
    core = {"provenance":provenance, "files":{name:{"sha256":drive._digest(data), "size_bytes":len(data)} for name, data in files.items()}}
    identifier = drive._digest(drive._bytes(core))
    archive = root / "exports" / (identifier + ".zip")
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as stream:
        for name, data in files.items():
            stream.writestr(name, data)
        stream.writestr("snapshot.json", drive._bytes({**core,"snapshot_id":identifier}))
    record = {"job_id":first["id"], "snapshot_id":identifier, "archive_name":archive.name,
        "bytes":archive.stat().st_size, "sha256":drive._sha(archive)}
    _write(root / "worker-status.json", {"updated_at":"2026-10-19T00:00:00+00:00", "state":"paused", "exports":[record]})
    _write(root / "controller/queue.json", {"jobs":jobs,"hosts":{"colab":{"charged_seconds":300.}}})
    return root, saved, record


def test_saved_protocol_matches_existing_immutable_queue_exactly():
    for host in ("colab", "kaggle"):
        specs = [compute_queue.job_spec(host, config, seed) for config in compute_queue.CONFIGURATIONS for seed in compute_queue.SEEDS]
        expected, protocol = drive.expected_plan(host)
        assert expected == {compute_queue.spec_hash(spec):spec for spec in specs}
        assert protocol == compute_queue.spec_hash({"version":1,"host_seconds":21600,"jobs":specs})


def test_publish_is_atomic_idempotent_source_preserving_and_never_claims_live_gpu(queue):
    root, saved, record = queue
    source_before = {str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
    value = drive.publish_saved_queue(root, saved)
    target = saved / "exports" / record["snapshot_id"] / record["archive_name"]
    assert target.read_bytes() == (root / "exports" / record["archive_name"]).read_bytes()
    stamp = target.stat().st_mtime_ns
    drive.publish_saved_queue(root, saved)
    assert target.stat().st_mtime_ns == stamp
    assert source_before == {str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert not list(saved.glob(".publish-*")) and not list((saved / "exports").glob(".copy-*"))
    row = drive.read_saved_queue(saved)
    assert value["completed_jobs"] == 1 and row["completed_epochs"] == 20
    assert row["available_archives"] == row["expected_archives"] == 1
    assert row["source_observed_at"] == "2026-10-19T00:00:00+00:00"
    assert "not live GPU proof" in row["message"] and "strict CPU" in row["message"]


def test_archive_collision_preserves_existing_bytes_and_prior_evidence(queue):
    root, saved, record = queue
    drive.publish_saved_queue(root, saved)
    target = saved / "exports" / record["snapshot_id"] / record["archive_name"]
    target.write_bytes(b"preserve collision")
    status = (saved / "status.json").read_bytes()
    with pytest.raises(ValueError, match="collision"):
        drive.publish_saved_queue(root, saved)
    assert target.read_bytes() == b"preserve collision" and (saved / "status.json").read_bytes() == status


@pytest.mark.parametrize("corruption", ["payload", "protocol", "epochs", "duplicate-job", "charge", "traversal", "missing-export"])
def test_local_reader_rejects_checksums_or_provenance_without_executing_models(queue, corruption):
    root, saved, _ = queue
    value = drive.publish_saved_queue(root, saved)
    bad = deepcopy(value)
    if corruption == "payload":
        bad["payload_sha256"] = "a" * 64
    elif corruption == "protocol":
        bad["queue_protocol_sha256"] = "a" * 64
    elif corruption == "epochs":
        bad["completed_epochs"] = 240
    elif corruption == "duplicate-job":
        bad["jobs"][1] = bad["jobs"][0]
    elif corruption == "charge":
        bad["charged_gpu_seconds"] = 21601
    elif corruption == "traversal":
        bad["exports"][0]["snapshot_id"] = "../outside"
    else:
        bad["exports"] = []
    _write(saved / "status.json", {"sha256":drive._digest(drive._bytes(bad)), "payload":bad})
    with pytest.raises(ValueError):
        drive.read_saved_queue(saved)


def test_missing_local_archive_is_availability_evidence_not_a_completed_transfer(queue):
    root, saved, record = queue
    drive.publish_saved_queue(root, saved)
    (saved / "exports" / record["snapshot_id"] / record["archive_name"]).unlink()
    row = drive.read_saved_queue(saved)
    assert row["available_archives"] == 0 and row["completed_jobs"] == 1
    assert "strict CPU verification is still required" in row["message"]


def test_settings_require_existing_folder_and_offline_read_preserves_configuration(queue, tmp_path):
    root, saved, _ = queue
    config = tmp_path / "settings/cloud-folders.json"
    with pytest.raises(ValueError, match="existing"):
        drive.configure_folder(saved, config=config)
    drive.publish_saved_queue(root, saved)
    drive.configure_folder(saved, config=config)
    assert drive.configured_evidence(config)["colab"]["completed_jobs"] == 1
    original = config.read_bytes()
    (saved / "status.json").unlink()
    assert drive.configured_evidence(config)["colab"]["phase"] == "saved_source_unavailable"
    assert config.read_bytes() == original


def test_bounded_json_and_source_archive_mutation_fail_closed(queue):
    root, saved, record = queue
    source = root / "exports" / record["archive_name"]
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="differs"):
        drive.publish_saved_queue(root, saved)
    assert not saved.exists()
    (root / "worker-status.json").write_bytes(b" " * (drive.MAX_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="bound"):
        drive.publish_saved_queue(root, saved)


def test_relative_paths_and_linked_saved_sources_are_rejected(queue, monkeypatch):
    root, saved, _ = queue
    drive.publish_saved_queue(root, saved)
    with pytest.raises(ValueError, match="paths"):
        drive.read_saved_queue(Path("relative/folder"))
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == saved.parent)
    with pytest.raises(ValueError, match="paths"):
        drive.read_saved_queue(saved)


def test_foreground_wrapper_only_watches_existing_process_and_keeps_its_budget(monkeypatch, tmp_path):
    events, waits = [], []
    class Process:
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise drive.subprocess.TimeoutExpired("already-owned", timeout)
            self.returncode = 0
    monkeypatch.setattr(drive.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Wrapper cannot start another job"))
    monkeypatch.setattr(drive, "publish_saved_queue", lambda *args: events.append(args) or {"completed_jobs":1,"completed_epochs":20})
    result = drive.watch_foreground(Process(), tmp_path, tmp_path / "saved")
    assert result["completed_jobs"] == 1 and waits == [60, 60] and len(events) == 3


def test_exact_drive_probe_recognizes_connection_without_inventing_training_progress(tmp_path):
    saved = tmp_path / "Drive/My Drive/InteractionLab/compute-v1/colab/development-v1"
    _write(saved / "connection-check.json", drive.CONNECTION_PROBE)
    config = tmp_path / "settings/cloud-folders.json"
    drive.configure_folder(saved, config=config)
    before = (saved / "connection-check.json").read_bytes()
    row = drive.configured_evidence(config)["colab"]
    assert row["status"] == "checked" and row["phase"] == "connected_awaiting_evidence"
    assert row["scope"] == "drive_connection_probe"
    assert not any(key in row for key in ("source_observed_at", "completed_jobs", "completed_epochs", "saved_run_status"))
    assert (saved / "connection-check.json").read_bytes() == before


@pytest.mark.parametrize("probe", [None, {"nonce":"wrong","purpose":drive.CONNECTION_PROBE["purpose"]},
    {**drive.CONNECTION_PROBE,"extra":"unapproved"}])
def test_missing_or_mismatched_drive_probe_remains_unavailable(tmp_path, probe):
    saved = tmp_path / "Drive/colab"
    saved.mkdir(parents=True)
    if probe is not None:
        _write(saved / "connection-check.json", probe)
    config = tmp_path / "settings.json"
    drive.configure_folder(saved, config=config)
    assert drive.configured_evidence(config)["colab"]["phase"] == "saved_source_unavailable"


def test_valid_probe_never_masks_corrupt_existing_status(queue, tmp_path):
    root, saved, _ = queue
    drive.publish_saved_queue(root, saved)
    _write(saved / "connection-check.json", drive.CONNECTION_PROBE)
    config = tmp_path / "settings.json"
    drive.configure_folder(saved, config=config)
    (saved / "status.json").write_bytes(b"corrupt status must not become a successful connection probe")
    row = drive.configured_evidence(config)["colab"]
    assert row["status"] == "unavailable" and row["phase"] == "saved_source_unavailable"
