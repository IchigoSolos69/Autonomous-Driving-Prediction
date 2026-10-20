"""Cloud exports are independently pinned, bounded and CPU-loadable before publication."""
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import uuid
import zipfile

import pytest
import torch

from adp.cloud_results import FILES, PURPOSE, import_cloud_results
from adp.data import read_json, write_json
from adp.model import ForecastModel
from adp.remote_jobs import prepare_job


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    monkeypatch.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    root = tmp_path / "runtime"
    splits = {"train": [str(uuid.UUID(int=1))], "validation": [str(uuid.UUID(int=2))]}
    write_json(root / "cache" / "manifest.json", {"splits": splits})
    write_json(root / "cache" / "preprocessing.json", {"version": 1})
    write_json(root / "source-manifest.json", {"splits": {**splits, "calibration": [], "evaluation": []}})
    for split, ids in splits.items():
        (root / "cache" / split).mkdir()
        (root / "cache" / split / f"{ids[0]}.npz").write_bytes(b"pinned independent pilot input")
    source = root / "source" / "adp"
    source.mkdir(parents=True)
    (source / "training.py").write_text("# immutable pilot source")
    (source / "remote_jobs.py").write_text("# immutable job runner")
    config = {"model": {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1}, "seed": 42}
    write_json(root / "initial-config.json", config)
    index = {path.relative_to(root).as_posix(): {"size": path.stat().st_size, "sha256": _sha(path.read_bytes())}
             for path in root.rglob("*") if path.is_file()}
    write_json(root / "package-files.json", index)
    payload = tmp_path / "pinned-inputs.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    job = prepare_job(root, config, source_root=source)
    config = read_json(root / "config.json")
    (root / "run").mkdir()
    for name in ("last.pt", "best.pt"):
        torch.save({"format_version": 2, "config": config, "completed_epochs": 1, "global_step": 1,
            "partial_epoch": False, "manifest_hash": job["cache_files"]["manifest.json"],
            "preprocessing_hash": job["cache_files"]["preprocessing.json"],
            "model_state_dict": ForecastModel(config["model"]).state_dict()}, root / "run" / name)
    write_json(root / "run" / "history.json", {"epochs": [{"epoch": 1}]})
    write_json(root / "status.json", {"state": "completed", "protocol_sha256": job["protocol_sha256"], "result": {"completed_epochs": 1}})
    write_json(root / "compute-ledger.json", {"protocol_sha256": job["protocol_sha256"], "charged_seconds": 3., "slices": [{"charged_seconds": 1.}, {"charged_seconds": 2.}]})
    write_json(root / "qualification.json", {"protocol_sha256": job["protocol_sha256"], "status": "passed", "cpu_gpu_agreement": "passed"})
    write_json(root / "environment.json", {"python": "3.12", "packages": {"torch": "2.13.0"}})
    data = {name: (root / name).read_bytes() for name in FILES}
    archive, receipt = _export(tmp_path, data, _sha(payload.read_bytes()))
    return archive, receipt, payload, tmp_path / "本地 cloud models", data


def _export(parent, data, payload_hash, verification=None):
    archive = parent / "cloud-export.zip"
    receipt = {"version": 1, "purpose": PURPOSE, "payload_sha256": payload_hash,
               "files": {name: {"sha256": _sha(content), "bytes": len(content)} for name, content in data.items()}}
    if verification is not None:
        receipt["input_verification"] = verification
    with zipfile.ZipFile(archive, "w") as exported:
        for name, content in data.items():
            exported.writestr(name, content)
        exported.writestr("export-manifest.json", json.dumps(receipt, indent=2) + "\n")
    receipt["archive_sha256"] = _sha(archive.read_bytes())
    return archive, receipt


def _import(pilot, **overrides):
    archive, receipt, payload, output, _ = pilot
    return import_cloud_results(overrides.get("archive", archive), overrides.get("receipt", receipt), output,
        payload_archive=payload, expected_payload_sha256=_sha(payload.read_bytes()))


def test_import_strict_cpu_portable_results_and_all_audit_files_atomically(pilot, monkeypatch):
    archive, receipt, payload, output, data = pilot
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("Cloud import must remain CPU-only"))
    result = _import(pilot)
    installed = Path(result["path"])
    assert result["receipt"]["verified"] and result["receipt"]["checkpoint_validation"] == "strict_cpu_observed_only"
    assert result["receipt"]["split_counts"] == {"train": 1, "validation": 1}
    assert all((installed / name).read_bytes() == content for name, content in data.items())
    assert (installed / "import-receipt.json").is_file()
    assert not list(output.glob(".cloud-import-*"))
    assert _import(pilot)["already_imported"]
    assert archive.is_file() and payload.is_file()


@pytest.mark.parametrize("corruption", ["receipt", "missing", "member-hash", "wrong-checkpoint", "wrong-payload-members", "status", "ledger", "traversal", "duplicate", "symlink"])
def test_corrupt_cloud_results_never_publish(pilot, corruption):
    archive, receipt, payload, output, data = pilot
    if corruption == "receipt":
        receipt = {**receipt, "archive_sha256": "f" * 64}
    elif corruption == "missing":
        data.pop("run/best.pt")
        archive, receipt = _export(archive.parent, data, _sha(payload.read_bytes()))
    elif corruption == "wrong-checkpoint":
        data["run/best.pt"] = b"not a torch checkpoint"
        archive, receipt = _export(archive.parent, data, _sha(payload.read_bytes()))
    elif corruption == "wrong-payload-members":
        data["initial-config.json"] = b'{"changed":true}'
        archive, receipt = _export(archive.parent, data, _sha(payload.read_bytes()))
    elif corruption in {"status", "ledger"}:
        name = "status.json" if corruption == "status" else "compute-ledger.json"
        value = json.loads(data[name])
        value.update({"state": "paused"} if corruption == "status" else {"charged_seconds": 1000.})
        data[name] = json.dumps(value).encode()
        archive, receipt = _export(archive.parent, data, _sha(payload.read_bytes()))
    else:
        with zipfile.ZipFile(archive, "a") as exported:
            if corruption == "member-hash":
                info = exported.getinfo("run/history.json")
                # Rebuild to change bytes while keeping the original manifest's hash.
                members = {name: exported.read(name) for name in exported.namelist()}
            elif corruption == "symlink":
                info = zipfile.ZipInfo("unexpected-link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                exported.writestr(info, "outside")
            else:
                exported.writestr("../outside.txt" if corruption == "traversal" else "status.json", b"unexpected")
        if corruption == "member-hash":
            members["run/history.json"] = b'{"changed":true}'
            with zipfile.ZipFile(archive, "w") as rebuilt:
                for name, content in members.items():
                    rebuilt.writestr(name, content)
        receipt = {**receipt, "archive_sha256": _sha(archive.read_bytes())}
    with pytest.raises(ValueError):
        _import(pilot, archive=archive, receipt=receipt)
    assert not output.exists() or not list(output.iterdir())


def test_extracted_input_receipt_must_pin_the_approved_package_index(pilot):
    archive, _, payload, _, data = pilot
    index_hash = _sha(data["package-files.json"])
    verification = {"kind": "verified_indexed_extracted_members", "package_index_sha256": index_hash,
                    "verified_member_count": len(json.loads(data["package-files.json"]))}
    archive, receipt = _export(archive.parent, data, _sha(payload.read_bytes()), verification)
    assert _import(pilot, archive=archive, receipt=receipt)["receipt"]["verified"]
    archive, receipt = _export(archive.parent, data, _sha(payload.read_bytes()), {**verification, "package_index_sha256": "f" * 64})
    with pytest.raises(ValueError, match="extracted-input"):
        _import(pilot, archive=archive, receipt=receipt)


def test_storage_reserve_and_existing_local_corruption_block_import(pilot, monkeypatch):
    _, _, _, output, _ = pilot
    with monkeypatch.context() as scoped:
        scoped.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=8 * 1024**3))
        with pytest.raises(OSError, match="8 GiB"):
            _import(pilot)
    assert not list(output.iterdir())
    installed = Path(_import(pilot)["path"])
    (installed / "run" / "best.pt").write_bytes(b"changed local weights")
    with pytest.raises(ValueError, match="Existing local"):
        _import(pilot)
