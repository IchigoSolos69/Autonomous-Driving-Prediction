"""Completed queue jobs remain unacknowledged until pinned, strict CPU verification."""
import hashlib
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import uuid
import zipfile

import pytest
import torch

from adp.compute_cleanup import _hash
from adp.compute_queue import job_spec, spec_hash
from adp.data import read_json, write_json
from adp.model import ForecastModel
from adp.queue_results import expected_configuration, import_queue_result, queue_protocol_hash
from adp.remote_jobs import prepare_job
from adp.remote_transfer import _json_bytes, create_snapshot


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    root = tmp_path_factory.mktemp("queue-input")
    splits = {"train": [str(uuid.UUID(int=value)) for value in range(1, 1001)],
              "validation": [str(uuid.UUID(int=value)) for value in range(1001, 1065)]}
    write_json(root / "cache/manifest.json", {"splits": splits})
    write_json(root / "cache/preprocessing.json", {"version": 1})
    write_json(root / "source-manifest.json", {"splits": {**splits, "calibration": [], "evaluation": []}})
    for split, ids in splits.items():
        (root / "cache" / split).mkdir()
        for sid in ids:
            (root / "cache" / split / (sid + ".npz")).write_bytes(b"immutable training-only input")
    source = root / "source/adp"
    source.mkdir(parents=True)
    for name in ("training.py", "remote_jobs.py"):
        (source / name).write_text("# pinned source", encoding="utf-8")
    initial = {"model": {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1},
               "seed": 42, "training": {"checkpoint_selection": "joint_minFDE"}}
    write_json(root / "initial-config.json", initial)
    metadata = {"initial-config.json": (root / "initial-config.json").read_bytes()}
    # Use the real job protocol builder, including real independent scenario membership.
    temporary_spec = {"seed": 42, "configuration": "full"}
    config = expected_configuration(metadata, temporary_spec)
    protocol = prepare_job(root, config, epochs=20, max_runtime_seconds=1800, source_root=source)
    members = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*")
               if path.is_file() and path.name not in {"config.json", "job.json"}}
    index = {name: {"size": len(data), "sha256": _sha(data)} for name, data in members.items()}
    members["package-files.json"] = _json_bytes(index)
    archive = root / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
    return archive, _sha(archive.read_bytes()), metadata, protocol


@pytest.fixture
def queued(tmp_path, payload, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    archive, digest, metadata, protocol = payload
    monkeypatch.setattr("adp.compute_queue.PAYLOAD_SHA256", digest)
    monkeypatch.setattr("adp.queue_results.PAYLOAD_SHA256", digest)

    def build(host="private", configuration="full", seed=42):
        spec = job_spec(host, configuration, seed)
        config = expected_configuration(metadata, spec)
        job = {**protocol, "config_sha256": _hash(config), "maximum_compute_seconds": 900. if host == "kaggle" else 1800.}
        job["protocol_sha256"] = _hash({key: value for key, value in job.items() if key != "protocol_sha256"})
        run = tmp_path / (host + configuration + str(seed)) / "run"
        run.mkdir(parents=True)
        write_json(run / "config.json", config)
        write_json(run / "job.json", job)
        write_json(run / "result.json", {"state": "completed", "protocol_sha256": job["protocol_sha256"],
            "result": {"completed_epochs": 20}})
        write_json(run / "history.json", {"epochs": [{"epoch": epoch} for epoch in range(1, 21)]})
        for name, epoch in (("last.pt", 20), ("best.pt", 12)):
            torch.save({"format_version": 2, "config": config, "completed_epochs": epoch, "global_step": epoch * 32,
                "partial_epoch": False, "manifest_hash": job["cache_files"]["manifest.json"],
                "preprocessing_hash": job["cache_files"]["preprocessing.json"],
                "model_state_dict": ForecastModel(config["model"]).state_dict()}, run / name)
        provenance = {"queue_job_spec": spec, "queue_job_id": spec_hash(spec), "payload_sha256": digest,
            "queue_protocol_sha256": queue_protocol_hash(host), "allocated_device_count": 2 if host == "kaggle" else 1,
            "job_protocol_sha256": job["protocol_sha256"], "config_sha256": _hash(config),
            "manifest_sha256": job["cache_files"]["manifest.json"],
            "preprocessing_sha256": job["cache_files"]["preprocessing.json"]}
        snapshot = create_snapshot(run, run.parent / "snapshots", spec_hash(spec), provenance)
        source = Path(snapshot["path"])
        data = {path.name: path.read_bytes() for path in source.iterdir()}
        exported = tmp_path / (host + configuration + str(seed) + ".zip")
        _zip(exported, data)
        return {"archive": exported, "payload": archive, "spec": spec,
                "output": tmp_path / "本地 results", "data": data}
    return build


def _zip(path, data):
    with zipfile.ZipFile(path, "w") as zipped:
        for name, content in data.items():
            zipped.writestr(name, content)


def _import(case, **overrides):
    return import_queue_result(case["archive"], case["output"], payload_archive=case["payload"],
        job_spec=case["spec"], expected_archive_sha256=overrides.get("sha", _sha(case["archive"].read_bytes())))


def _rewrite(case, *, change_provenance=None, change_member=None):
    data = case["data"]
    manifest = json.loads(data["snapshot.json"])
    if change_provenance:
        change_provenance(manifest["provenance"])
    if change_member:
        name, content = change_member(data)
        data[name] = content
        manifest["files"][name] = {"sha256": _sha(content), "size_bytes": len(content)}
        if name.endswith(".pt"):
            saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            manifest["checkpoint_metadata"][name].update(completed_epochs=saved["completed_epochs"],
                global_step=saved["global_step"], partial_epoch=saved["partial_epoch"])
    manifest["snapshot_id"] = _sha(_json_bytes({key: value for key, value in manifest.items() if key != "snapshot_id"}))
    data["snapshot.json"] = _json_bytes(manifest)
    _zip(case["archive"], data)


def test_completed_cpu_import_is_durable_idempotent_and_preserves_sources(queued, monkeypatch):
    case = queued()
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("Imports must use CPU only"))
    result = _import(case)
    receipt = result["queue_receipt"]
    assert receipt["verified"] and receipt["job_id"] == spec_hash(case["spec"])
    assert receipt["checkpoint_validation"] == "strict_cpu_observed_only_completed"
    assert receipt["allocated_device_count"] == 1
    assert receipt["queue_protocol_sha256"] == queue_protocol_hash("private")
    assert read_json(result["queue_receipt_path"]) == receipt
    assert read_json(result["receipt_path"])["verified"]
    assert _import(case)["queue_receipt"] == receipt
    assert case["archive"].is_file() and case["payload"].is_file()
    assert not list(case["output"].glob(".queue-receive-*"))


@pytest.mark.parametrize("configuration,seed", [("no_graph", 43), ("no_map", 44), ("world_ce", 42)])
def test_exact_approved_configuration_and_seed_are_supported(queued, configuration, seed):
    case = queued(configuration=configuration, seed=seed)
    assert _import(case)["queue_receipt"]["job_spec"] == case["spec"]


def test_kaggle_two_allocated_gpus_require_half_wall_time_protocol(queued):
    case = queued(host="kaggle")
    assert _import(case)["queue_receipt"]["allocated_device_count"] == 2


@pytest.mark.parametrize("field,value", [("queue_job_id", "f" * 64), ("queue_protocol_sha256", "f" * 64),
    ("payload_sha256", "f" * 64), ("allocated_device_count", 2), ("allocated_device_count", True),
    ("job_protocol_sha256", "f" * 64), ("config_sha256", "f" * 64)])
def test_wrong_provenance_never_publishes_or_acknowledges(queued, field, value):
    case = queued()
    _rewrite(case, change_provenance=lambda provenance: provenance.update({field: value}))
    with pytest.raises(ValueError):
        _import(case)
    assert not list(case["output"].iterdir())


@pytest.mark.parametrize("corruption", ["source", "config", "wall-budget", "paused", "epochs", "last-epoch", "partial-epoch", "wrong-shape"])
def test_self_consistent_archive_cannot_change_pinned_recipe_or_complete_partial_training(queued, corruption):
    case = queued()
    def changed(data):
        if corruption in {"source", "wall-budget"}:
            value = json.loads(data["job.json"])
            if corruption == "source":
                value["source_files"]["training.py"] = "f" * 64
            else:
                value["maximum_compute_seconds"] = 21600
            value["protocol_sha256"] = _hash({key: item for key, item in value.items() if key != "protocol_sha256"})
            return "job.json", _json_bytes(value)
        if corruption == "config":
            value = json.loads(data["config.json"])
            value["seed"] = 44
            return "config.json", _json_bytes(value)
        if corruption in {"paused", "epochs"}:
            value = json.loads(data["result.json"])
            if corruption == "paused":
                value["state"] = "paused"
            else:
                value["result"]["completed_epochs"] = 19
            return "result.json", _json_bytes(value)
        saved = torch.load(io.BytesIO(data["last.pt"]), map_location="cpu", weights_only=True)
        if corruption == "last-epoch":
            saved["completed_epochs"] = 19
        elif corruption == "partial-epoch":
            saved["partial_epoch"] = True
        else:
            name = next(iter(saved["model_state_dict"]))
            saved["model_state_dict"][name] = torch.zeros(1)
        stream = io.BytesIO()
        torch.save(saved, stream)
        return "last.pt", stream.getvalue()
    _rewrite(case, change_member=changed)
    with pytest.raises(ValueError):
        _import(case)
    assert not list(case["output"].iterdir())


@pytest.mark.parametrize("corruption", ["hash", "missing", "traversal", "duplicate", "symlink"])
def test_zip_trust_boundary_rejects_corruption_and_paths(queued, corruption):
    case = queued()
    if corruption == "missing":
        case["data"].pop("best.pt")
        _zip(case["archive"], case["data"])
    elif corruption != "hash":
        with zipfile.ZipFile(case["archive"], "a") as archive:
            if corruption == "symlink":
                entry = zipfile.ZipInfo("metrics.json")
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(entry, "outside")
            else:
                archive.writestr("../outside" if corruption == "traversal" else "last.pt", b"unexpected")
    with pytest.raises(ValueError):
        _import(case, **({"sha": "f" * 64} if corruption == "hash" else {}))
    assert not case["output"].exists()


def test_reserve_and_corrupt_existing_receipt_fail_closed(queued, monkeypatch):
    case = queued()
    with monkeypatch.context() as scoped:
        scoped.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=8 * 1024**3))
        with pytest.raises(OSError, match="8 GiB"):
            _import(case)
    assert not list(case["output"].iterdir())
    result = _import(case)
    receipt = Path(result["queue_receipt_path"])
    receipt.write_text('{"corrupt":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="receipt is corrupt"):
        _import(case)
    assert receipt.read_text() == '{"corrupt":true}'


def test_changed_archive_during_verification_and_wrong_original_payload_never_publish(queued, monkeypatch, tmp_path):
    case = queued()
    wrong = tmp_path / "wrong-inputs.zip"
    wrong.write_bytes(case["payload"].read_bytes() + b"unapproved")
    with pytest.raises(ValueError, match="pinned payload"):
        _import({**case, "payload": wrong})
    assert not case["output"].exists()
    from adp import queue_results
    validate = queue_results.validate_snapshot
    def changed_archive(path):
        loaded = validate(path)
        with case["archive"].open("ab") as stream:
            stream.write(b"changed after verification began")
        return loaded
    monkeypatch.setattr(queue_results, "validate_snapshot", changed_archive)
    with pytest.raises(ValueError, match="changed during verification"):
        _import(case)
    assert not list(case["output"].iterdir())


def test_receipt_write_failure_does_not_return_an_acknowledgement(queued, monkeypatch):
    case = queued()
    def failed_receipt(*_):
        raise OSError("simulated durable receipt failure")
    monkeypatch.setattr("adp.queue_results._write", failed_receipt)
    with pytest.raises(OSError, match="durable receipt failure"):
        _import(case)
    # The verified model is recoverable, while the acknowledgement is absent.
    assert len(list(case["output"].glob("*/last.pt"))) == 1
    assert not list((case["output"] / "queue-receipts").iterdir())
    assert case["archive"].is_file() and case["payload"].is_file()

