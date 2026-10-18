"""Native Windows window with an application-owned local inference service."""
from __future__ import annotations

import argparse
import html
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time


TITLE = "Interaction Lab — Autonomous Driving Prediction"


def add_arguments(parser):
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--checkpoint")
    parser.add_argument("--calibration")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline", action="store_true", help="Use the explicitly labelled constant-velocity baseline")
    parser.add_argument("--data-root", type=Path, help="Writable imported data/model storage; logs and settings remain in Local AppData")
    parser.add_argument("--model-bundle", type=Path, help="Validated inference bundle directory")


def application_state_dir():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Interaction Lab"


def read_settings(state_dir):
    import json
    path = Path(state_dir) / "settings.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(key not in ("data_root", "model_bundle") or not isinstance(item, str) for key, item in value.items()):
        raise ValueError("Application settings are invalid. Remove settings.json from the Interaction Lab Local AppData folder.")
    return value


def save_settings(state_dir, settings):
    from adp.data import write_json
    write_json(Path(state_dir) / "settings.json", settings)


def data_directory(args, state_dir, settings):
    return Path(getattr(args, "data_root", None) or settings.get("data_root") or Path(state_dir) / "data").resolve()


def resolve_paths(args):
    root = args.project_dir.resolve()

    def local(path):
        return (root / path).resolve()

    frozen = bool(getattr(sys, "frozen", False))
    resources = Path(__file__).resolve().parent / "resources"
    frontend = resources / "frontend" if frozen else root / "frontend" / "dist"
    if not (frontend / "index.html").is_file():
        if frozen:
            raise FileNotFoundError("The application interface is missing. Repair or reinstall Interaction Lab.")
        raise FileNotFoundError("The application interface has not been built. Run npm.cmd ci and npm.cmd run build in the frontend folder, then reopen the application.")
    if args.baseline and (args.checkpoint or args.calibration or getattr(args, "model_bundle", None)):
        raise ValueError("--baseline cannot be combined with a checkpoint or calibration file.")
    checkpoint = None
    if not args.baseline:
        checkpoint = local(args.checkpoint) if args.checkpoint else next(
            (root / "artifacts" / name / "best.pt" for name in ("experiment", "run")
             if (root / "artifacts" / name / "best.pt").is_file()), None)
    calibration = local(args.calibration) if args.calibration else None
    if checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.calibration and not checkpoint:
        raise ValueError("Calibration requires a trained checkpoint.")
    if checkpoint and not calibration and checkpoint.with_name("calibration.json").is_file():
        calibration = checkpoint.with_name("calibration.json")
    if calibration and not calibration.is_file():
        raise FileNotFoundError(f"Calibration not found: {calibration}")
    cache = resources / "demo" / "cache" if frozen and args.cache_dir == "data/cache" else local(args.cache_dir)
    result = dict(cache_dir=cache, checkpoint=checkpoint, calibration=calibration, device=args.device, frontend_dir=frontend,
                  bundle_metadata={"demo": frozen})
    selected_bundle = getattr(args, "model_bundle", None)
    if selected_bundle is None and frozen and (resources / "demo" / "bundle" / "bundle.json").is_file() and not args.baseline:
        selected_bundle = resources / "demo" / "bundle"
    if selected_bundle and not args.baseline:
        from adp.bundles import load_bundle
        bundle = load_bundle(selected_bundle)
        result.update(checkpoint=bundle["paths"]["checkpoint"], calibration=bundle["paths"]["calibration"],
                      bundle_metadata={**bundle["metadata"], "demo": frozen},
                      evaluation=bundle["metadata"].get("evaluation"), reports_dir=bundle["paths"].get("reports_dir"))
    if checkpoint and not frozen and not selected_bundle:
        result["reports_dir"] = checkpoint.parent / "evaluation"
    return result


class SingleInstance:
    """A per-user Windows mutex; a second launch restores the existing window."""
    def __init__(self):
        self.handle = None

    def acquire(self):
        if sys.platform != "win32":
            return True
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel.CreateMutexW.restype = wintypes.HANDLE
        self.handle = kernel.CreateMutexW(None, False, "Local\\InteractionLab.ResearchBeta")
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            user = ctypes.windll.user32
            user.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
            user.FindWindowW.restype = wintypes.HWND
            user.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
            user.ShowWindow.restype = wintypes.BOOL
            user.SetForegroundWindow.argtypes = (wintypes.HWND,)
            user.SetForegroundWindow.restype = wintypes.BOOL
            window = user.FindWindowW(None, TITLE)
            if window:
                user.ShowWindow(window, 9)
                user.SetForegroundWindow(window)
            self.close()
            return False
        return True

    def close(self):
        if self.handle:
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle(self.handle)
            self.handle = None


class DesktopBridge:
    def __init__(self, state_dir, data_root, settings, token, imports):
        self._state_dir, self._data_root = Path(state_dir), Path(data_root)
        self._settings, self._token, self._imports = settings, token, imports
        self._window = None
        self._dialog_lock = threading.Lock()

    def desktop_context(self):
        return {"desktop": True, "token": self._token, "data_root": str(self._data_root), "settings": dict(self._settings)}

    def _choose(self, dialog_type, **options):
        if self._window is None:
            raise RuntimeError("Application window is not ready")
        with self._dialog_lock:
            selected = self._window.create_file_dialog(dialog_type, allow_multiple=False, **options)
        return selected[0] if isinstance(selected, (list, tuple)) and selected else selected or None

    def choose_import_directory(self):
        import webview
        directory = self._choose(webview.FileDialog.FOLDER)
        return self._imports.grant_directory(directory) if directory else None

    def choose_data_directory(self):
        import webview
        directory = self._choose(webview.FileDialog.FOLDER)
        if not directory:
            return None
        import tempfile
        destination = Path(directory).resolve(strict=True)
        # Test the chosen location before saving a setting that could prevent the next startup.
        with tempfile.TemporaryFile(prefix=".interaction-lab-write-test-", dir=destination):
            pass
        self._settings["data_root"] = str(destination)
        save_settings(self._state_dir, self._settings)
        return {"data_root": self._settings["data_root"], "restart_required": True}

    def choose_model_bundle(self, archive=False):
        import webview
        from adp.bundles import install_bundle
        if not isinstance(archive, bool):
            raise ValueError("Archive selection must be true or false")
        source = self._choose(webview.FileDialog.OPEN, file_types=("Model bundle (*.zip)",)) if archive else self._choose(webview.FileDialog.FOLDER)
        if not source:
            return None
        bundle = install_bundle(source, self._data_root / "models")
        self._settings["model_bundle"] = str(bundle["paths"]["checkpoint"].parent)
        save_settings(self._state_dir, self._settings)
        return {"bundle_id": bundle["metadata"]["bundle_id"], "restart_required": True}

    def save_path(self, kind):
        import webview
        extensions = {"prediction_json": ("json", "Prediction JSON (*.json)"), "report_csv": ("csv", "Report CSV (*.csv)"),
                      "report_png": ("png", "Report PNG (*.png)"), "diagnostics": ("zip", "Diagnostics ZIP (*.zip)")}
        if kind not in extensions:
            raise ValueError("Unknown export format")
        extension, label = extensions[kind]
        return self._choose(webview.FileDialog.SAVE, save_filename=f"interaction-lab-{kind}.{extension}", file_types=(label,))


class LocalBackend:
    """Own one socket/server; closing during model initialization cannot start a late server."""

    def __init__(self):
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.server = self.thread = self.listener = None
        self.failure = None

    def start(self, app_factory, timeout=30):
        if self.closed.is_set():
            return None
        app = app_factory()
        import uvicorn
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        from starlette.responses import PlainTextResponse

        with self.lock:
            if self.closed.is_set():
                return None
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(128)
            port = self.listener.getsockname()[1]
            origin = f"http://127.0.0.1:{port}"
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1"])

            @app.middleware("http")
            async def same_origin(request, call_next):
                if request.headers.get("origin") not in (None, origin):
                    return PlainTextResponse("This local service accepts application requests only.", status_code=403)
                return await call_next(request)

            self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                       log_config=None, access_log=False))

            def serve():
                try:
                    self.server.run(sockets=[self.listener])
                except BaseException as error:
                    self.failure = error
                    logging.exception("Local inference service failed")

            self.thread = threading.Thread(target=serve, name="adp-inference", daemon=True)
            self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.closed.is_set():
            if self.server.started:
                logging.info("Desktop inference ready at %s", origin)
                return origin
            if not self.thread.is_alive():
                raise RuntimeError(f"The inference service could not start: {self.failure or 'see application log'}")
            if time.monotonic() >= deadline:
                raise TimeoutError("The local inference service did not become ready within 30 seconds.")
            self.closed.wait(.05)
        return None

    def request_stop(self):
        with self.lock:
            self.closed.set()
            if self.server:
                self.server.should_exit = True

    def close(self):
        self.request_stop()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.server.force_exit = True
                self.thread.join(timeout=1)
        if self.listener:
            self.listener.close()


def status_page(title, message, log_path=None):
    detail = f"<p class='log'>Application log: {html.escape(str(log_path))}</p>" if log_path else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>
        body{{margin:0;background:#f3f5f4;color:#24332f;font:14px 'Segoe UI',sans-serif;display:grid;min-height:100vh;place-items:center}}
        main{{max-width:650px;padding:32px}} .mark{{color:#247c70;font:600 14px Consolas,monospace;letter-spacing:2px}}
        h1{{font-size:24px;font-weight:600}} p{{line-height:1.65;white-space:pre-wrap}} .log{{font:12px Consolas,monospace;color:#5c6964;overflow-wrap:anywhere}}
        </style></head><body><main><div class='mark'>INTERACTION LAB</div><h1>{html.escape(title)}</h1>
        <p>{html.escape(message)}</p>{detail}</main></body></html>"""


def run(args):
    instance = SingleInstance()
    if not instance.acquire():
        return
    state_dir = application_state_dir()
    log_path = state_dir / "logs" / "app.log"
    backend = LocalBackend()
    imports = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")],
                            format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
        settings = read_settings(state_dir)
        data_root = data_directory(args, state_dir, settings)
        data_root.mkdir(parents=True, exist_ok=True)
        if not getattr(args, "model_bundle", None) and settings.get("model_bundle") and not args.baseline:
            args.model_bundle = Path(settings["model_bundle"])
        from adp.imports import ImportManager
        imports = ImportManager(data_root / "imports")
        token = secrets.token_urlsafe(32)
        bridge = DesktopBridge(state_dir, data_root, settings, token, imports)
        import webview
        webview.settings["ALLOW_DOWNLOADS"] = False
        webview.settings["ALLOW_FILE_URLS"] = False
        webview.settings["SHOW_DEFAULT_MENUS"] = False
        window = webview.create_window(TITLE, html=status_page("Starting the application", "Loading the local scenario library and forecasting model.\nThe first GPU startup may take a few seconds."),
                                       width=1440, height=900, min_size=(860, 600), background_color="#f3f5f4", js_api=bridge)
        bridge._window = window
        window.events.closed += backend.request_stop

        def initialize():
            try:
                # pywebview can silently fall back to Internet Explorer when WebView2 is missing.
                if webview.renderer != "edgechromium":
                    raise RuntimeError("Microsoft Edge WebView2 Runtime is required. Install it from https://developer.microsoft.com/microsoft-edge/webview2/ then reopen Interaction Lab.")
                paths = resolve_paths(args)
                from adp.bundles import input_schema
                if Path(paths["cache_dir"]).joinpath("preprocessing.json").is_file():
                    schema = input_schema(paths["cache_dir"])
                    imports.options = {"max_actors": schema["max_actors"], "max_polylines": schema["max_polylines"], "map_radius": schema["map_radius_m"]}
                os.environ.setdefault("OMP_NUM_THREADS", "2")

                def make_app():
                    from adp.api import create_app
                    import torch
                    torch.set_num_threads(2)
                    return create_app(**paths, import_manager=imports, desktop_token=token, native_dialogs=bridge, state_dir=state_dir, data_root=data_root)

                origin = backend.start(make_app)
                if origin and not backend.closed.is_set():
                    window.load_url(origin)
            except Exception as error:
                logging.exception("Application startup failed")
                still_open = not backend.closed.is_set()
                backend.close()
                # The window remains available to read the error until the user closes it.
                if still_open:
                    window.load_html(status_page("Unable to start Interaction Lab", str(error) + "\nClose the application, fix the issue above, then reopen it.", log_path))

        def begin():
            threading.Thread(target=initialize, name="adp-startup", daemon=True).start()

        window.events.shown += begin
        webview.start(gui="edgechromium", debug=False, private_mode=True)
    except Exception as error:
        logging.exception("Desktop window failed")
        message = (f"{error}\n\nCheck that Microsoft Edge WebView2 Runtime is installed, then reopen the application.\n\nLog: {log_path}")
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "Interaction Lab — startup error", 0x10)
        else:
            raise RuntimeError(message) from error
    finally:
        backend.close()
        if imports is not None:
            imports.close()
        instance.close()


def main():
    parser = argparse.ArgumentParser(description="Launch the Interaction Lab Windows desktop application")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
