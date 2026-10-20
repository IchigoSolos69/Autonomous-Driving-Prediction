"""Run notebook trust-boundary helpers without requiring a hosted GPU session."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import uuid
import zipfile

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "train_remote.ipynb"


def notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    scope = {"Path": Path, "hashlib": hashlib, "json": json, "re": re,
             "shutil": shutil, "time": time, "uuid": uuid, "zipfile": zipfile}
    for cell in notebook["cells"]:
        if set(cell["metadata"].get("tags", [])) & {"cache-helper", "export-helper"}:
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), scope)
    return scope


def cache_fixture(path, splits=None):
    path.mkdir()
    splits = splits or {"train": ["train-1"], "validation": ["select-1"]}
    (path / "manifest.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    (path / "preprocessing.json").write_text('{"schema": 1}', encoding="utf-8")
    for split, ids in splits.items():
        (path / split).mkdir(exist_ok=True)
        for scenario in ids:
            if isinstance(scenario, str) and "/" not in scenario:
                (path / split / (scenario + ".npz")).write_bytes(b"trusted fixture cache")
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def test_notebook_compiles_and_is_unexecuted():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"remote-cell-{index}", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            code.append(source)
    all_code = "\n".join(code)
    assert "'calibrate'" not in all_code and "'evaluate'" not in all_code
    assert "'study'" not in all_code and "'direction-study'" not in all_code
    assert "--max-runtime-seconds" in all_code
    assert "weights_only=True" in all_code and "rtol=1e-4" in all_code


def test_stage_selection_only_with_unicode_and_resume(tmp_path, monkeypatch):
    helpers = notebook_helpers()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20 * 1024**3})())
    source, target = tmp_path / "données", tmp_path / "runtime" / "cache"
    digest = cache_fixture(source)
    (source / "evaluation").mkdir()
    (source / "evaluation" / "locked.npz").write_bytes(b"never copy")
    assert helpers["stage_development_cache"](source, target, digest) == target
    assert not (target / "evaluation").exists()
    assert (source / "train" / "train-1.npz").exists()
    assert helpers["stage_development_cache"](source, target, digest) == target
    (target / "train" / "train-1.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        helpers["stage_development_cache"](source, target, digest)


@pytest.mark.parametrize("splits", [
    {"train": ["a"], "validation": ["b"], "calibration": []},
    {"train": ["a"], "validation": ["a"]},
    {"train": ["../escape"], "validation": ["b"]},
    {"train": [1], "validation": ["b"]},
    {"train": [], "validation": ["b"]},
])
def test_stage_refuses_unsafe_manifests(tmp_path, splits):
    helpers = notebook_helpers()
    source = tmp_path / "source"
    digest = cache_fixture(source, splits)
    with pytest.raises(ValueError):
        helpers["stage_development_cache"](source, tmp_path / "cache", digest)
    assert not (tmp_path / "cache").exists()


def test_stage_checks_identity_and_storage_before_publication(tmp_path, monkeypatch):
    helpers = notebook_helpers()
    source = tmp_path / "source"
    digest = cache_fixture(source)
    with pytest.raises(ValueError, match="SHA-256"):
        helpers["stage_development_cache"](source, tmp_path / "cache", "0" * 64)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 8 * 1024**3})())
    with pytest.raises(RuntimeError, match="8 GiB"):
        helpers["stage_development_cache"](source, tmp_path / "cache", digest)
    assert not (tmp_path / "cache").exists()


def test_export_is_verified_and_preserves_active_checkpoints(tmp_path, monkeypatch):
    helpers = notebook_helpers()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20 * 1024**3})())
    source, run = tmp_path / "cache", tmp_path / "run"
    cache_fixture(source)
    run.mkdir()
    for name in ("last.pt", "best.pt", "config.json", "history.json"):
        (run / name).write_bytes(("trusted " + name).encode())
    active = {p.name: p.read_bytes() for p in run.iterdir()}
    target, receipt_path = helpers["export_snapshot"](run, source, tmp_path / "exports", "source-hash")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["archive_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert "snapshot.json" in archive.namelist()
        for name, info in receipt["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
    assert {p.name: p.read_bytes() for p in run.iterdir()} == active
    assert not list(target.parent.glob("*.partial"))


def test_export_refuses_missing_checkpoint(tmp_path):
    helpers = notebook_helpers()
    with pytest.raises(ValueError, match="checkpoint"):
        helpers["export_snapshot"](tmp_path / "run", tmp_path / "cache", tmp_path / "exports", "source")
