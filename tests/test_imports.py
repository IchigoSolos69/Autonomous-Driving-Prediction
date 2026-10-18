import hashlib
import json
import threading
import time

import pytest

from adp.imports import ImportManager
from test_data import av2_fixture


def finish(manager, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = manager.status(job_id)
        if state["status"] in ("completed", "cancelled", "failed"):
            return state
        time.sleep(.01)
    raise AssertionError("Import did not complete")


def test_import_validates_then_publishes_without_modifying_sources(tmp_path):
    source = tmp_path / "source with spaces"
    source.mkdir()
    av2_fixture(source)
    (source / "scenario_broken.parquet").write_bytes(b"invalid")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    manager = ImportManager(tmp_path / "imports")
    try:
        grant = manager.grant_directory(source)
        assert "source_token" in grant and "path" not in grant
        started = manager.start(grant["source_token"])
        done = finish(manager, started["id"])
        assert done["status"] == "completed"
        assert done["total"] == 2 and done["processed"] == 2 and done["imported"] == 1
        assert done["errors"][0]["scenario"] == "broken"
        cache, = manager.caches()
        assert json.loads((cache / "manifest.json").read_text())["splits"] == {"imported": ["fixture"]}
        assert (cache / "imported" / "fixture.npz").is_file()
        assert not list(manager.root.glob(".partial-*"))
        assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
        with pytest.raises(ValueError, match="expired"):
            manager.start(grant["source_token"])
    finally:
        manager.close()


def test_cancel_during_conversion_never_publishes_partial_cache(tmp_path, monkeypatch):
    import adp.imports as imports
    source = tmp_path / "source"
    source.mkdir()
    av2_fixture(source)
    entered, release = threading.Event(), threading.Event()
    original = imports.load_av2

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(imports, "load_av2", paused)
    manager = ImportManager(tmp_path / "imports")
    try:
        started = manager.start(manager.grant_directory(source)["source_token"])
        assert entered.wait(5)
        assert manager.caches() == []
        with pytest.raises(ValueError, match="already running"):
            manager.start(manager.grant_directory(source)["source_token"])
        manager.cancel(started["id"])
        release.set()
        assert finish(manager, started["id"])["status"] == "cancelled"
    finally:
        release.set()
        manager.close()
    assert manager.caches() == []
    assert not list(manager.root.glob(".partial-*"))


def test_invalid_or_excessive_selection_fails_without_publication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "scenario_bad.parquet").write_bytes(b"bad")
    manager = ImportManager(tmp_path / "imports", max_scenarios=1)
    try:
        first = manager.start(manager.grant_directory(source)["source_token"])
        assert finish(manager, first["id"])["status"] == "failed"
        assert manager.caches() == []
        (source / "scenario_extra.parquet").write_bytes(b"bad")
        second = manager.start(manager.grant_directory(source)["source_token"])
        assert "exceeds" in finish(manager, second["id"])["error"]
        assert manager.caches() == []
    finally:
        manager.close()
