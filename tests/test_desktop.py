import argparse
import socket
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
