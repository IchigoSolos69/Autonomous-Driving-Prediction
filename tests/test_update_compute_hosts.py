"""Refreshing one sync preserves other hosts and never promotes unverified delivery."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from adp.compute_status import read_compute_hosts


NOW = 1_800_000_000
STAMP = datetime.fromtimestamp(NOW - 30, timezone.utc).isoformat()
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/update-compute-hosts.py"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("update_compute_hosts", SCRIPT)
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    result_root, identifier = tmp_path / "résultats", "a" * 64
    directory, protocol = result_root / identifier, "b" * 64
    metadata = {"snapshot_id": identifier, "files": {"last.pt": {"sha256": "c" * 64, "size_bytes": 5},
                "job.json": {}, "result.json": {}}, "provenance": {"job_protocol_sha256": protocol}}
    validated = {"manifest": metadata, "manifest_sha256": "d" * 64,
                 "checkpoint_metadata": {"last.pt": {"completed_epochs": 1}}}
    validator = Mock(return_value=validated)
    monkeypatch.setattr(updater, "validate_snapshot", validator)
    receipt_path = result_root / "receipts" / (identifier + ".json")
    receipt = {"format_version": 1, "verified": True, "checkpoint_validation": "cpu_weights_only",
               "snapshot_id": identifier, "files": metadata["files"], "manifest_sha256": "d" * 64,
               "local_directory": str(directory), "verified_at": STAMP}
    write(receipt_path, receipt)
    write(directory / "job.json", {"epochs": 1, "split_counts": {"train": 1000}})
    write(directory / "result.json", {"state": "completed", "updated_at": STAMP,
        "result": {"completed_epochs": 1, "training_examples_per_second": 17.5}})
    sync = {"updated_at": STAMP, "before": {"state": "completed", "protocol_sha256": protocol,
        "process_alive": False, "training_on_gpu": False, "training": {}},
        "snapshot": {"state": "snapshot", "snapshot_id": identifier},
        "delivered": {"snapshot_id": identifier, "path": str(directory), "receipt_path": str(receipt_path), "receipt": receipt},
        "resume": {"state": "not_resumed", "reason": "completed"}}
    sync_path, hosts = result_root / "last-sync.json", tmp_path / "hosts.json"
    write(sync_path, sync)
    other = [{"id": "local", "cleanup": {"status": "protected", "message": "Frozen studies still need this data."}},
             {"id": "colab", "status": "qualifying", "observed_at": STAMP, "gpu": "NVIDIA T4"},
             {"id": "kaggle", "status": "waiting", "message": "Private notebook input upload pending."}]
    write(hosts, {"version": 1, "hosts": deepcopy(other)})
    return updater, sync_path, hosts, sync, receipt_path, receipt, validator, other


def test_refresh_preserves_other_hosts_and_publishes_only_verified_evidence(evidence):
    updater, sync_path, hosts, _, receipt_path, _, validator, others = evidence
    value = updater.update_hosts(sync_path, hosts, now=NOW)
    document = json.loads(hosts.read_text())
    assert document["hosts"][:3] == others
    assert value["status"] == "completed" and value["training"] == {
        "completed_epochs": 1, "planned_epochs": 1, "committed_examples": 1000,
        "total_examples": 1000, "examples_per_second": 17.5}
    assert value["transfer"] == {"status": "verified", "verified_at": STAMP, "local_path": str(receipt_path)}
    assert value["cleanup"]["status"] == "completed"
    validator.assert_called_once()
    rows = read_compute_hosts(hosts, now=NOW)
    assert rows[1]["status"] == "completed" and rows[1]["freshness"]["age_seconds"] == 30
    assert rows[0]["cleanup"]["status"] == "protected"
    assert not list(hosts.parent.glob(".hosts-update-*"))


def test_repeating_old_sync_is_deterministic_without_refreshing_observation_time(evidence):
    updater, sync_path, hosts, *_ = evidence
    updater.update_hosts(sync_path, hosts, now=NOW)
    before = hosts.read_bytes()
    updater.update_hosts(sync_path, hosts, now=NOW + 5 * 3600)
    assert hosts.read_bytes() == before
    row = read_compute_hosts(hosts, now=NOW + 5 * 3600)[1]
    assert row["observed_at"] == STAMP and row["stale"] and row["status"] == "completed"


@pytest.mark.parametrize("mutation", ["receipt", "paths", "checkpoint", "protocol"])
def test_invalid_delivery_cannot_replace_any_existing_host_evidence(evidence, mutation):
    updater, sync_path, hosts, sync, receipt_path, receipt, validator, _ = evidence
    if mutation == "receipt":
        receipt["verified"] = False
        sync["delivered"]["receipt"] = receipt
        write(receipt_path, receipt)
    elif mutation == "paths":
        sync["delivered"]["path"] = str(hosts.parent / "unrelated")
    elif mutation == "protocol":
        sync["before"]["protocol_sha256"] = "e" * 64
    else:
        validator.side_effect = ValueError("Corrupt checkpoint")
    write(sync_path, sync)
    before = hosts.read_bytes()
    with pytest.raises(ValueError):
        updater.update_hosts(sync_path, hosts, now=NOW)
    assert hosts.read_bytes() == before


def test_unchanged_sync_revalidates_the_existing_receipt(evidence):
    updater, sync_path, hosts, sync, *_ = evidence
    sync["snapshot"]["state"], sync["delivered"] = "unchanged", None
    write(sync_path, sync)
    assert updater.update_hosts(sync_path, hosts, now=NOW)["transfer"]["status"] == "verified"


@pytest.mark.parametrize("path_kind", ["receipt", "directory"])
def test_cleaned_remote_checkpoint_preserves_revalidated_local_completion(evidence, path_kind):
    updater, sync_path, hosts, sync, receipt_path, _, validator, others = evidence
    original = updater.update_hosts(sync_path, hosts, now=NOW)
    document = json.loads(hosts.read_text())
    private = next(row for row in document["hosts"] if row["id"] == "private")
    if path_kind == "directory":
        private["transfer"]["local_path"] = sync["delivered"]["path"]
    private["cleanup"] = {"status": "completed", "message": "Owned remote cache and checkpoint copies were removed after verification."}
    write(hosts, document)
    sync["snapshot"], sync["delivered"] = {"state": "no_checkpoint"}, None
    sync["before"]["state"], sync["before"]["training"] = "not_started", {}
    sync["updated_at"] = datetime.fromtimestamp(NOW, timezone.utc).isoformat()
    write(sync_path, sync)
    value = updater.update_hosts(sync_path, hosts, now=NOW)
    assert value["status"] == "completed" and "local archive" in value["message"]
    assert value["training"] == original["training"]
    assert value["updated_at"] == original["updated_at"] and value["observed_at"] == sync["updated_at"]
    assert value["transfer"] == original["transfer"] and value["transfer"]["local_path"] == str(receipt_path)
    assert value["cleanup"] == private["cleanup"]
    assert json.loads(hosts.read_text())["hosts"][:3] == others
    assert validator.call_count == 2


def test_cleaned_remote_checkpoint_cannot_hide_invalid_archived_receipt(evidence):
    updater, sync_path, hosts, sync, receipt_path, receipt, _, _ = evidence
    updater.update_hosts(sync_path, hosts, now=NOW)
    sync["snapshot"], sync["delivered"] = {"state": "no_checkpoint"}, None
    write(sync_path, sync)
    receipt["verified"] = False
    write(receipt_path, receipt)
    before = hosts.read_bytes()
    with pytest.raises(ValueError, match="provenance differs"):
        updater.update_hosts(sync_path, hosts, now=NOW)
    assert hosts.read_bytes() == before


def test_new_resume_and_dead_worker_do_not_claim_gpu_training(evidence):
    updater, sync_path, hosts, sync, *_ = evidence
    sync["before"]["state"] = "running"
    write(sync_path, sync)
    assert updater.update_hosts(sync_path, hosts, now=NOW)["status"] == "unknown"
    sync["resume"] = {"state": "launched", "pid": 123}
    write(sync_path, sync)
    assert updater.update_hosts(sync_path, hosts, now=NOW)["status"] == "waiting"


def test_stronger_cleanup_policy_is_preserved(evidence):
    updater, sync_path, hosts, *_ = evidence
    document = json.loads(hosts.read_text())
    cleanup = {"status": "waiting_for_dependencies", "message": "Separate cleanup checks must pass."}
    document["hosts"].append({"id": "private", "cleanup": cleanup})
    write(hosts, document)
    assert updater.update_hosts(sync_path, hosts, now=NOW)["cleanup"] == cleanup


def test_failed_atomic_publish_keeps_previous_profile_and_removes_temporary(evidence, monkeypatch):
    updater, sync_path, hosts, *_ = evidence
    before = hosts.read_bytes()
    monkeypatch.setattr(updater.os, "replace", Mock(side_effect=OSError("file in use")))
    with pytest.raises(OSError, match="file in use"):
        updater.update_hosts(sync_path, hosts, now=NOW)
    assert hosts.read_bytes() == before and not list(hosts.parent.glob(".hosts-update-*"))


def test_detected_concurrent_change_is_preserved(evidence, monkeypatch):
    updater, sync_path, hosts, *_ = evidence
    original = json.loads(hosts.read_text())
    original["hosts"][1]["message"] = "New browser observation arrived."
    monkeypatch.setattr(updater.os, "fsync", lambda _: write(hosts, original))
    with pytest.raises(RuntimeError, match="changed during refresh"):
        updater.update_hosts(sync_path, hosts, now=NOW)
    assert json.loads(hosts.read_text()) == original


@pytest.mark.parametrize("transfer_status", ["waiting", "verified"])
def test_old_pilot_refresh_preserves_development_queue_with_independent_pilot_proof(evidence, transfer_status):
    updater, sync_path, hosts, sync, receipt_path, _, validator, others = evidence
    snapshot = validator.return_value["manifest"]
    write(sync_path.parent / snapshot["snapshot_id"] / "snapshot.json", snapshot)
    queue = {"id": "private", "status": "running", "phase": "development queue", "observed_at": STAMP,
             "message": "Full seed 43: 7/20 epochs committed.", "training": {"completed_epochs": 27, "planned_epochs": 240},
             "transfer": {"status": transfer_status, "verified_at": STAMP if transfer_status == "verified" else None,
                          "local_path": str(hosts.parent / "queue-receipts/queue-proof.json") if transfer_status == "verified" else None}}
    write(hosts, {"version": 1, "hosts": [*others, queue]})
    sync["snapshot"], sync["delivered"] = {"state": "no_checkpoint"}, None
    write(sync_path, sync)
    before = hosts.read_bytes()
    assert updater.update_hosts(sync_path, hosts, now=NOW) == queue
    assert hosts.read_bytes() == before
    validator.assert_called_once_with(receipt_path.parent.parent / snapshot["snapshot_id"])


def test_development_queue_does_not_hide_corrupt_historical_pilot_receipt(evidence):
    updater, sync_path, hosts, sync, receipt_path, receipt, validator, others = evidence
    snapshot = validator.return_value["manifest"]
    write(sync_path.parent / snapshot["snapshot_id"] / "snapshot.json", snapshot)
    queue = {"id": "private", "status": "running", "phase": "development queue", "observed_at": STAMP,
             "transfer": {"status": "verified", "verified_at": STAMP, "local_path": "unrelated queue receipt"}}
    write(hosts, {"version": 1, "hosts": [*others, queue]})
    sync["snapshot"], sync["delivered"] = {"state": "no_checkpoint"}, None
    receipt["verified"] = False
    write(receipt_path, receipt)
    write(sync_path, sync)
    before = hosts.read_bytes()
    with pytest.raises(ValueError, match="provenance differs"):
        updater.update_hosts(sync_path, hosts, now=NOW)
    assert hosts.read_bytes() == before
