"""Host snapshots do not confuse old evidence, completed work or controller liveness."""
from datetime import datetime, timezone
import json

import pytest

from adp.compute_status import MAX_HOSTS_BYTES, read_compute_hosts


NOW = 1_800_000_000


def iso(seconds=0):
    return datetime.fromtimestamp(NOW + seconds, timezone.utc).isoformat()


def profile(path, hosts):
    path.write_text(json.dumps({"version": 1, "hosts": hosts}), encoding="utf-8")
    return path


def host(identifier="private", **values):
    return {"id": identifier, "status": "running", "updated_at": iso(-30),
            "phase": "training", "gpu": "GeForce 940MX", **values}


def study(**values):
    return {"id": "original", "status": "running", "phase": "train", "current_run_id": "full-42",
            "progress": {"completed_epochs": 2, "planned_epochs": 90}, "runs": [
                {"id": "full-42", "committed_examples": 32, "training_examples": 40000,
                 "examples_per_second": 3.9, "checkpoint": {"updated_at": iso(-60)}}], **values}


def test_combined_host_schema_and_controller_observation_are_read_only(tmp_path):
    path = profile(tmp_path / "hosts.json", [
        host("local", status="completed", gpu="Qualified local GPU"),
        host(training={"completed_epochs": 0, "planned_epochs": 1, "committed_examples": 256,
                       "total_examples": 1000, "examples_per_second": 18.5}),
        host("colab", status="qualifying", gpu="NVIDIA T4"),
        host("kaggle", status="waiting", message="Waiting for a free accelerator.")])
    before = path.read_bytes()
    result = read_compute_hosts(path, studies=[study()], now=NOW)
    assert [row["id"] for row in result] == ["local", "private", "colab", "kaggle"]
    assert result[0]["status"] == "running" and result[0]["gpu"] == "Qualified local GPU"
    assert result[0]["training"] == {"completed_epochs": 2, "planned_epochs": 90,
        "committed_examples": 32, "total_examples": 40000, "examples_per_second": 3.9}
    assert result[1]["training"]["committed_examples"] == 256
    assert result[1]["freshness"]["age_seconds"] == 30 and not result[1]["stale"]
    assert result[2]["status"] == "qualifying" and result[3]["status"] == "waiting"
    assert all(row["freshness"]["checked_at"] == iso() for row in result)
    assert path.read_bytes() == before
    json.dumps(result, allow_nan=False)


def test_missing_host_file_does_not_create_it_or_claim_execution(tmp_path):
    path = tmp_path / "absent" / "hosts.json"
    result = read_compute_hosts(path, now=NOW)
    assert not path.exists() and not path.parent.exists()
    assert all(row["status"] == "unknown" and row["stale"] for row in result)
    assert all("controller" in row["message"] for row in result[1:])


def test_stale_execution_downgrades_but_completed_work_remains_known(tmp_path):
    path = profile(tmp_path / "hosts.json", [host(updated_at=iso(-4 * 3600 - 1)),
        host("colab", status="completed", updated_at=iso(-2 * 86400))])
    private, colab = read_compute_hosts(path, now=NOW)[1:3]
    assert private["status"] == "unknown" and private["last_observed_status"] == "running"
    assert private["stale"] and "execution is unconfirmed" in private["message"]
    assert colab["status"] == "completed" and colab["stale"]
    assert colab["last_observed_status"] == "completed"


def test_fresh_observation_can_confirm_an_older_checkpoint_state(tmp_path):
    path = profile(tmp_path / "hosts.json", [host(updated_at=iso(-86400), observed_at=iso(-15))])
    private = read_compute_hosts(path, now=NOW)[1]
    assert private["status"] == "running" and private["updated_at"] == iso(-86400)
    assert private["observed_at"] == iso(-15) and private["freshness"]["age_seconds"] == 15


def test_undated_execution_is_unconfirmed_and_future_clock_is_invalid(tmp_path):
    path = profile(tmp_path / "hosts.json", [host(updated_at=None), host("colab", observed_at=iso(301))])
    result = read_compute_hosts(path, now=NOW)
    assert result[1]["status"] == "unknown" and result[1]["stale"]
    assert result[2]["status"] == "unavailable" and "invalid" in result[2]["message"]


@pytest.mark.parametrize("value", [[], {"version": True, "hosts": []}, {"version": 1, "hosts": {}},
    {"version": 1, "hosts": [{"id": "../private"}]}, {"version": 1, "hosts": [host(), host()]}])
def test_invalid_container_is_actionable_and_preserves_local_progress(tmp_path, value):
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps(value))
    rows = read_compute_hosts(path, studies=[study()], now=NOW)
    assert rows[0]["status"] == "running"
    assert all(row["status"] == "unavailable" and "unreadable" in row["message"] for row in rows[1:])


@pytest.mark.parametrize("changes", [
    {"training": []}, {"training": {"completed_epochs": True}},
    {"training": {"completed_epochs": 2, "planned_epochs": 1}},
    {"training": {"committed_examples": 1001, "total_examples": 1000}},
    {"training": {"examples_per_second": -1}}, {"updated_at": "2026-10-19T12:00:00"},
    {"message": "bad\ncontrol"}, {"gpu": {"model": "unrecognized shape"}},
    {"transfer": {"status": "verified", "verified_at": iso(-10)}}, {"cleanup": []},
    {"status": "invented_status"},
])
def test_bad_individual_evidence_does_not_discard_other_hosts(tmp_path, changes):
    path = profile(tmp_path / "hosts.json", [host(**changes), host("colab", status="completed")])
    rows = read_compute_hosts(path, now=NOW)
    assert rows[1]["status"] == "unavailable" and "invalid" in rows[1]["message"]
    assert rows[2]["status"] == "completed"


def test_oversized_corrupt_and_nonfinite_json_cannot_escape_the_api(tmp_path):
    path = tmp_path / "hosts.json"
    for data in (b"x" * (MAX_HOSTS_BYTES + 1), b"{not JSON", b'{"version":1,"hosts":[],"untrusted":NaN}'):
        path.write_bytes(data)
        rows = read_compute_hosts(path, now=NOW)
        assert all(row["status"] == "unavailable" for row in rows[1:])
        json.dumps(rows, allow_nan=False)


def test_transfer_receipts_and_cleanup_survive_stale_job_completion(tmp_path):
    receipt = "D:/InteractionLab/remote-results/résultats/receipt.json"
    path = profile(tmp_path / "hosts.json", [host(status="completed", updated_at=iso(-86400),
        transfer={"status": "verified", "verified_at": iso(-7200), "local_path": receipt},
        cleanup={"status": "completed", "message": "Deleted only the acknowledged export; active checkpoints preserved."})])
    private = read_compute_hosts(path, now=NOW)[1]
    assert private["status"] == "completed" and private["stale"]
    assert private["transfer"] == {"status": "verified", "verified_at": iso(-7200), "local_path": receipt}
    assert private["cleanup"]["status"] == "completed" and "preserved" in private["cleanup"]["message"]


def test_local_freshness_comes_from_checkpoint_without_modifying_study_progress(tmp_path):
    value = study()
    value["runs"][0]["checkpoint"]["updated_at"] = iso(-601)
    original = json.loads(json.dumps(value))
    local = read_compute_hosts(tmp_path / "missing", studies=[value], now=NOW)[0]
    assert local["status"] == "unknown" and local["last_observed_status"] == "running" and local["stale"]
    assert local["training"]["completed_epochs"] == 2
    assert value == original


def test_local_cleanup_evidence_cannot_spoof_local_study_progress_or_freshness(tmp_path):
    path = profile(tmp_path / "hosts.json", [host("local", status="completed", phase="finished",
        observed_at=iso(), updated_at=iso(), training={"completed_epochs": 99, "planned_epochs": 99},
        cleanup={"status": "waiting_for_dependencies",
                 "message": "Original study and direction follow-up still consume the cache; no deletion performed."})])
    local = read_compute_hosts(path, studies=[study()], now=NOW)[0]
    assert local["cleanup"] == {"status": "waiting_for_dependencies",
        "message": "Original study and direction follow-up still consume the cache; no deletion performed."}
    assert local["status"] == "running" and local["phase"] == "train"
    assert local["training"]["completed_epochs"] == 2 and local["training"]["planned_epochs"] == 90
    assert local["updated_at"] == local["observed_at"] == iso(-60)
    assert local["freshness"]["age_seconds"] == 60
