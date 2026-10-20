"""Snapshot import/pruning uses verified bytes and never touches active checkpoints."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from adp.model import ForecastModel
from adp.remote_transfer import (MINIMUM_FREE_BYTES, create_snapshot, import_snapshot,
                                 prune_snapshot, validate_snapshot)


@pytest.fixture(autouse=True)
def sufficient_storage(monkeypatch):
    monkeypatch.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))


def _save(path, version=2, **overrides):
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1}
    saved = {"format_version": version, "config": {"model": config, "training": {"seed": 42}},
             "manifest_hash": "a" * 64, "preprocessing_hash": "b" * 64,
             "model_state_dict": ForecastModel(config).state_dict(), "optimizer_state_dict": {"state": {}},
             "completed_epochs": 1, "global_step": 5, "partial_epoch": False}
    torch.save({**saved, **overrides}, path)


@pytest.fixture
def snapshot(tmp_path):
    run = tmp_path / "active-run"
    _save(run / "last.pt")
    _save(run / "best.pt", version=1)
    (run / "history.json").write_text('{"epochs":[{"epoch":1,"train_loss":2.0}]}', encoding="utf-8")
    (run / "unrelated-private.txt").write_text("Never exported", encoding="utf-8")
    exported = create_snapshot(run, tmp_path / "exports", "control-42", {"manifest_sha256": "a" * 64})
    return run, exported


def test_cpu_validation_atomic_unicode_import_receipt_and_safe_pruning(snapshot, tmp_path, monkeypatch):
    run, exported = snapshot
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("Transfer must not initialize CUDA"))
    forward = ForecastModel.forward
    checked_calls = []
    def observed_only_forward(self, **inputs):
        assert not any("future" in key for key in inputs)
        assert all(value.device.type == "cpu" for value in inputs.values())
        checked_calls.append(True)
        return forward(self, **inputs)
    monkeypatch.setattr(ForecastModel, "forward", observed_only_forward)
    rng_before = torch.get_rng_state().clone()
    original_last = (run / "last.pt").read_bytes()
    original_best = (run / "best.pt").read_bytes()
    received = import_snapshot(exported["path"], tmp_path / "本地 results")
    local = Path(received["path"])
    assert validate_snapshot(local)["manifest_sha256"] == exported["manifest_sha256"]
    assert set(exported["manifest"]["files"]) == {"last.pt", "best.pt", "history.json"}
    assert received["receipt"]["checkpoint_validation"] == "cpu_weights_only"
    assert json.loads(Path(received["receipt_path"]).read_text()) == received["receipt"]
    assert not list(local.parent.glob(".transfer-*"))
    assert import_snapshot(exported["path"], local.parent)["snapshot_id"] == received["snapshot_id"]
    assert prune_snapshot(Path(exported["path"]).parent, exported["snapshot_id"], received["receipt"])["deleted"]
    assert local.is_dir()
    assert (run / "last.pt").read_bytes() == original_last
    assert (run / "best.pt").read_bytes() == original_best
    assert checked_calls
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert prune_snapshot(Path(exported["path"]).parent, exported["snapshot_id"], received["receipt"])["already_absent"]


@pytest.mark.parametrize("corruption", ["hash", "missing", "extra", "traversal", "invalid-checkpoint"])
def test_corrupt_transfers_never_publish_or_produce_receipts(snapshot, tmp_path, corruption):
    _, exported = snapshot
    root = Path(exported["path"])
    if corruption == "hash":
        (root / "history.json").write_text('{"modified":true}', encoding="utf-8")
    elif corruption == "missing":
        (root / "last.pt").unlink()
    elif corruption == "extra":
        (root / "execute.py").write_text("bad", encoding="utf-8")
    else:
        metadata = json.loads((root / "snapshot.json").read_text())
        if corruption == "traversal":
            metadata["files"]["../outside.txt"] = metadata["files"].pop("history.json")
        else:
            (root / "last.pt").write_bytes(b"not a checkpoint")
            metadata["files"]["last.pt"] = {"size_bytes": 16, "sha256": hashlib.sha256(b"not a checkpoint").hexdigest()}
        _rewrite_metadata(root, metadata)
    with pytest.raises(ValueError):
        import_snapshot(root, tmp_path / "local")
    assert not list((tmp_path / "local").glob(".transfer-*"))
    assert not (tmp_path / "local" / "receipts").exists()
    assert not (tmp_path / "local" / exported["snapshot_id"]).exists()


def _rewrite_metadata(root, metadata):
    core = {key: value for key, value in metadata.items() if key != "snapshot_id"}
    canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    metadata["snapshot_id"] = hashlib.sha256(canonical(core)).hexdigest()
    (root / "snapshot.json").write_bytes(canonical(metadata))


def test_reserve_includes_incoming_bytes_before_any_copy(snapshot, tmp_path, monkeypatch):
    _, exported = snapshot
    monkeypatch.setattr("adp.remote_transfer.shutil.disk_usage", lambda _: SimpleNamespace(free=MINIMUM_FREE_BYTES + 1))
    with pytest.raises(OSError, match="8 GiB"):
        import_snapshot(exported["path"], tmp_path / "local")
    assert not list((tmp_path / "local").iterdir())
    with pytest.raises(OSError, match="8 GiB"):
        create_snapshot(tmp_path / "active-run", tmp_path / "new-exports", "run", {})
    assert not list((tmp_path / "new-exports").glob(".transfer-*"))


@pytest.mark.parametrize("field,value", [("verified", False), ("snapshot_id", "f" * 64),
    ("manifest_sha256", "f" * 64), ("files", {}), ("checkpoint_validation", "not_checked")])
def test_remote_deletion_requires_exact_verified_receipt(snapshot, tmp_path, field, value):
    _, exported = snapshot
    receipt = import_snapshot(exported["path"], tmp_path / "local")["receipt"]
    with pytest.raises(ValueError):
        prune_snapshot(Path(exported["path"]).parent, exported["snapshot_id"], {**receipt, field: value})
    assert Path(exported["path"]).is_dir()


def test_prune_revalidates_remote_bytes_and_rejects_active_root_or_traversal(snapshot, tmp_path):
    run, exported = snapshot
    receipt = import_snapshot(exported["path"], tmp_path / "local")["receipt"]
    with pytest.raises(ValueError, match="opaque"):
        prune_snapshot(Path(exported["path"]).parent, "../active-run", receipt)
    with pytest.raises(ValueError):
        prune_snapshot(run, exported["snapshot_id"], receipt)
    (Path(exported["path"]) / "history.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="mismatch"):
        prune_snapshot(Path(exported["path"]).parent, exported["snapshot_id"], receipt)
    assert Path(exported["path"]).exists()
    assert (run / "last.pt").is_file()


def test_snapshot_rejects_inconsistent_checkpoint_provenance_and_missing_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="last.pt"):
        create_snapshot(run, tmp_path / "exports", "run", {})
    _save(run / "last.pt")
    _save(run / "best.pt", manifest_hash="f" * 64)
    with pytest.raises(ValueError, match="inconsistent"):
        create_snapshot(run, tmp_path / "exports", "run", {})
    assert not list((tmp_path / "exports").glob(".transfer-*"))
    (run / "best.pt").unlink()
    with pytest.raises(ValueError, match="provenance"):
        create_snapshot(run, tmp_path / "exports", "run", {"manifest_sha256": "c" * 64})


def test_links_and_nested_active_export_roots_are_rejected(snapshot, tmp_path):
    run, exported = snapshot
    with pytest.raises(ValueError, match="separate"):
        create_snapshot(run, run / "exports", "run", {})
    target = Path(exported["path"]) / "history.json"
    target.unlink()
    try:
        target.symlink_to(run / "history.json")
    except OSError:
        pytest.skip("Creating symbolic links requires permission on this Windows host")
    with pytest.raises(ValueError, match="unlinked"):
        import_snapshot(exported["path"], tmp_path / "local")


def test_existing_local_corruption_blocks_remote_prune_receipt(snapshot, tmp_path):
    _, exported = snapshot
    received = import_snapshot(exported["path"], tmp_path / "local")
    (Path(received["path"]) / "last.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        import_snapshot(exported["path"], tmp_path / "local")
    assert Path(exported["path"]).is_dir()


@pytest.mark.parametrize("corruption", ["shape", "missing-key", "dtype", "oversized-config"])
def test_checkpoint_must_strictly_load_the_actual_cpu_model(tmp_path, corruption):
    run = tmp_path / "run"
    _save(run / "last.pt")
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    if corruption == "shape":
        saved["model_state_dict"]["history_embedding.weight"] = torch.zeros(16, 7)
    elif corruption == "missing-key":
        saved["model_state_dict"].pop("history_embedding.weight")
    elif corruption == "dtype":
        saved["model_state_dict"]["history_embedding.weight"] = saved["model_state_dict"]["history_embedding.weight"].double()
    else:
        saved["config"]["model"]["width"] = 1_000_000
    torch.save(saved, run / "last.pt")
    with pytest.raises(ValueError, match="configuration|strict CPU"):
        create_snapshot(run, tmp_path / "exports", "run", {})
    assert not list((tmp_path / "exports").glob(".transfer-*"))
