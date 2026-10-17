"""Native Windows window with an application-owned local inference service."""
from __future__ import annotations

import argparse
import html
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
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


def resolve_paths(args):
    root = args.project_dir.resolve()

    def local(path):
        return (root / path).resolve()

    frontend = root / "frontend" / "dist"
    if not (frontend / "index.html").is_file():
        raise FileNotFoundError("The application interface has not been built. Run npm.cmd ci and npm.cmd run build in the frontend folder, then reopen the application.")
    if args.baseline and (args.checkpoint or args.calibration):
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
    return dict(cache_dir=local(args.cache_dir), checkpoint=checkpoint,
                calibration=calibration, device=args.device, frontend_dir=frontend)


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
    root = args.project_dir.resolve()
    log_dir = root / "artifacts" / "desktop"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")],
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    backend = LocalBackend()
    try:
        import webview
        webview.settings["ALLOW_DOWNLOADS"] = False
        webview.settings["ALLOW_FILE_URLS"] = False
        webview.settings["SHOW_DEFAULT_MENUS"] = False
        window = webview.create_window(TITLE, html=status_page("Starting the application", "Loading the local scenario library and forecasting model.\nThe first GPU startup may take a few seconds."),
                                       width=1440, height=900, min_size=(860, 600), background_color="#f3f5f4")
        window.events.closed += backend.request_stop

        def initialize():
            try:
                # pywebview can silently fall back to Internet Explorer when WebView2 is missing.
                if webview.renderer != "edgechromium":
                    raise RuntimeError("Microsoft Edge WebView2 Runtime is required. Install it from https://developer.microsoft.com/microsoft-edge/webview2/ then reopen Interaction Lab.")
                paths = resolve_paths(args)
                os.environ.setdefault("OMP_NUM_THREADS", "2")

                def make_app():
                    from adp.api import create_app
                    import torch
                    torch.set_num_threads(2)
                    return create_app(**paths)

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
        message = (f"{error}\n\nInstall Microsoft Edge WebView2 Runtime and run scripts\\setup.ps1, then reopen the application.\n\nLog: {log_path}")
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "Interaction Lab — startup error", 0x10)
        else:
            raise RuntimeError(message) from error
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description="Launch the Interaction Lab Windows desktop application")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
