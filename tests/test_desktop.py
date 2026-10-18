import argparse
import socket
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from fastapi import FastAPI

from adp.desktop import LocalBackend, add_arguments, resolve_paths, status_page


def options(root, *arguments):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(["--project-dir", str(root), *arguments])


def test_desktop_paths_ignore_working_directory_and_label_missing_weights(tmp_path, monkeypatch):
    root = tmp_path / "workspace with spaces"
    (root / "frontend/dist").mkdir(parents=True)
    (root / "frontend/dist/index.html").write_text("app")
    monkeypatch.chdir(tmp_path)
    paths = resolve_paths(options(root))
    assert paths["cache_dir"] == root / "data/cache"
    assert paths["checkpoint"] is None
    artifact = root / "artifacts/experiment"
    artifact.mkdir(parents=True)
    (artifact / "best.pt").touch()
    (artifact / "calibration.json").touch()
    paths = resolve_paths(options(root))
    assert paths["checkpoint"] == artifact / "best.pt"
    assert paths["calibration"] == artifact / "calibration.json"
    assert resolve_paths(options(root, "--baseline"))["checkpoint"] is None
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_paths(options(root, "--checkpoint", "absent.pt"))
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_paths(options(root, "--baseline", "--checkpoint", "absent.pt"))


def test_missing_frontend_and_safe_error_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="not been built"):
        resolve_paths(options(tmp_path))
    page = status_page("Startup", "<script>alert('bad')</script>", tmp_path)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_local_service_owns_port_and_rejects_foreign_origins():
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"ready": True}

    backend = LocalBackend()
    try:
        origin = backend.start(lambda: app)
        assert origin.startswith("http://127.0.0.1:")
        assert b'"ready":true' in urlopen(origin + "/api/health", timeout=5).read()
        with pytest.raises(HTTPError) as foreign:
            urlopen(Request(origin + "/api/health", headers={"Origin": "https://example.org"}), timeout=5)
        assert foreign.value.code == 403
        with pytest.raises(HTTPError) as host:
            urlopen(Request(origin + "/api/health", headers={"Host": "example.org"}), timeout=5)
        assert host.value.code == 400
        port = backend.listener.getsockname()[1]
    finally:
        backend.close()
    assert not backend.thread.is_alive()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_closing_during_model_startup_does_not_start_server():
    backend = LocalBackend()
    initializing, finish = threading.Event(), threading.Event()
    result = []

    def make_app():
        initializing.set()
        assert finish.wait(5)
        return FastAPI()

    worker = threading.Thread(target=lambda: result.append(backend.start(make_app)))
    worker.start()
    assert initializing.wait(5)
    backend.close()
    finish.set()
    worker.join(5)
    assert result == [None]
    assert backend.server is None


def test_startup_failure_can_be_closed_without_leaking_socket():
    backend = LocalBackend()
    with pytest.raises(ValueError, match="incompatible model"):
        backend.start(lambda: (_ for _ in ()).throw(ValueError("incompatible model")))
    backend.close()
    assert backend.listener is None


def test_legacy_renderer_shows_install_guidance_before_loading_model(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from adp.desktop import run
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-appdata"))
    monkeypatch.setattr("adp.desktop.SingleInstance.acquire", lambda self: True)

    class Hook:
        def __iadd__(self, callback):
            self.callback = callback
            return self

    shown, closed = Hook(), Hook()
    displayed = threading.Event()
    pages = []

    def load_html(page):
        pages.append(page)
        displayed.set()

    window = SimpleNamespace(events=SimpleNamespace(shown=shown, closed=closed), load_html=load_html)

    def start(**kwargs):
        shown.callback()
        assert displayed.wait(5)
        closed.callback()

    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(settings={}, renderer="mshtml",
                         create_window=lambda *a, **k: window, start=start))
    run(options(tmp_path))
    assert "WebView2 Runtime is required" in pages[0]
    assert "has not been built" not in pages[0]


def test_desktop_settings_keep_user_data_separate_from_resources(tmp_path, monkeypatch):
    from adp.desktop import application_state_dir, data_directory, read_settings, save_settings
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    state = application_state_dir()
    assert state == tmp_path / "appdata" / "Interaction Lab"
    settings = {"data_root": str(tmp_path / "another drive"), "model_bundle": str(tmp_path / "model")}
    save_settings(state, settings)
    assert read_settings(state) == settings
    assert data_directory(options(tmp_path), state, settings) == tmp_path / "another drive"
    assert data_directory(options(tmp_path, "--data-root", str(tmp_path / "override")), state, settings) == tmp_path / "override"


def test_native_import_grant_and_save_use_only_dialog_paths(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from adp.desktop import DesktopBridge
    from adp.imports import ImportManager
    source = tmp_path / "source"
    source.mkdir()
    manager = ImportManager(tmp_path / "data" / "imports")
    bridge = DesktopBridge(tmp_path / "state", tmp_path / "data", {}, "secret", manager)
    chosen = iter([(str(source),), (str(tmp_path / "chosen.json"),), None])
    bridge._window = SimpleNamespace(create_file_dialog=lambda *a, **k: next(chosen))
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(FileDialog=SimpleNamespace(FOLDER=20, SAVE=30)))
    try:
        grant = bridge.choose_import_directory()
        assert grant["display_name"] == "source" and "path" not in grant
        assert bridge.save_path("prediction_json") == str(tmp_path / "chosen.json")
        assert bridge.choose_import_directory() is None
        with pytest.raises(ValueError, match="Unknown export"):
            bridge.save_path("../../arbitrary")
        assert bridge.desktop_context()["token"] == "secret"
    finally:
        manager.close()


def test_frozen_paths_read_bundled_assets_not_working_directory(tmp_path, monkeypatch):
    import sys
    import adp.desktop as desktop
    module = tmp_path / "installed" / "adp" / "desktop.py"
    resources = module.parent / "resources"
    (resources / "frontend").mkdir(parents=True)
    (resources / "frontend" / "index.html").write_text("built interface")
    monkeypatch.setattr(desktop, "__file__", str(module))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    paths = resolve_paths(options(tmp_path / "arbitrary cwd", "--baseline"))
    assert paths["frontend_dir"] == resources / "frontend"
    assert paths["cache_dir"] == resources / "demo" / "cache"
    assert paths["checkpoint"] is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex behavior")
def test_existing_instance_is_restored_and_mutex_handle_is_closed(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from adp.desktop import SingleInstance, TITLE
    calls = []

    class Call:
        def __init__(self, name, result):
            self.name, self.result = name, result

        def __call__(self, *args):
            calls.append((self.name, args))
            return self.result

    kernel = SimpleNamespace(CreateMutexW=Call("mutex", 100), CloseHandle=Call("close", 1))
    user = SimpleNamespace(FindWindowW=Call("find", 200), ShowWindow=Call("show", 1), SetForegroundWindow=Call("focus", 1))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 183)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=user))
    instance = SingleInstance()
    assert instance.acquire() is False
    assert ("find", (None, TITLE)) in calls
    assert ("show", (200, 9)) in calls
    assert ("focus", (200,)) in calls
    assert ("close", (100,)) in calls
    assert instance.handle is None
