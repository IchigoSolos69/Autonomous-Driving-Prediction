"""Independent local transfers, guarded cleanup and explicitly approved queue dispatch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from filelock import FileLock, Timeout

from adp.compute_cleanup import _alive, _safe, _write
from adp.compute_status import LABELS, MAX_HOSTS_BYTES, _host
from adp.remote_control import _config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("D:/InteractionLab/remote-compute/private-server.json")
DEFAULT_STATE_DIR = Path("D:/InteractionLab/remote-compute/monitor")
DEFAULT_HOSTS = Path("D:/InteractionLab/remote-compute/hosts.json")
DEFAULT_INTERVAL = 5 * 60
MINIMUM_FREE_BYTES = 10 * 1024**3
MAX_TRANSFER_STAGING_BYTES = 3 * 512 * 1024**2
SYNC_TIMEOUT = 900
MAX_OUTPUT_BYTES = 1024**2


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    path = _safe(path)
    if not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError("Expected bounded, regular monitor metadata")
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite monitor metadata")))
    if not isinstance(value, dict):
        raise ValueError("Monitor metadata must contain an object")
    return value, raw


def _brief(value, limit=600):
    return " ".join(str(value).split())[-limit:]


def _diagnostic(value):
    lines = []
    for line in str(value or "").splitlines():
        lower = line.casefold()
        if any(token in lower for token in ("command", "identityfile", "private-key", "private_key", "password", "token=", "-----begin", "ssh -i", "'-i'", '"-i"')):
            lines.append("Command or credential details omitted.")
        else:
            lines.append(line)
    return _brief(" ".join(lines[-12:]), 3000)


def _python():
    executable = Path(sys.executable)
    if executable.name.casefold() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    if not executable.is_file():
        raise ValueError("The qualified Python runtime is missing its console executable")
    return str(executable)


def _command(command, *, timeout):
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"),
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUTF8": "1"}
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, text=True,
            encoding="utf-8", errors="replace", capture_output=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Child operation timed out after {timeout} seconds", "returncode": None}
    except OSError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error.strerror or 'child could not start'}", "returncode": None}
    if result.returncode:
        return {"ok": False, "error": f"Child exited with code {result.returncode}. {_diagnostic(result.stderr)}", "returncode": result.returncode}
    if len(result.stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return {"ok": False, "error": "Child response exceeds the supported metadata bound", "returncode": 0}
    try:
        value = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite response")))
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
    except (ValueError, TypeError):
        return {"ok": False, "error": "Child returned invalid JSON metadata", "returncode": 0}
    return {"ok": True, "result": value, "returncode": 0}


def _publish_host(hosts_path, identifier, changes):
    path = _safe(hosts_path)
    if path.exists():
        document, original = _read(path)
    else:
        document, original = {"version": 1, "hosts": []}, None
    rows = document.get("hosts")
    if (type(document.get("version")) is not int or document["version"] != 1
            or not isinstance(rows, list) or len(rows) > 4
            or any(not isinstance(row, dict) or row.get("id") not in LABELS for row in rows)
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("Existing host metadata is invalid; it was preserved")
    previous = next((row for row in rows if row["id"] == identifier), {"id": identifier})
    updated = {**previous, **changes}
    _host(updated, time.time())
    document["hosts"] = [updated if row["id"] == identifier else row for row in rows]
    if not any(row["id"] == identifier for row in rows):
        document["hosts"].append(updated)
    if len(json.dumps(document, allow_nan=False).encode()) > MAX_HOSTS_BYTES:
        raise ValueError("Combined compute-host metadata is too large")
    current = path.read_bytes() if path.exists() else None
    if current != original:
        raise RuntimeError("Host evidence changed during publication; existing records were preserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, document)


def _log(root):
    for name in ("monitor.log", "monitor.log.1"):
        _safe(root / name, root)
    logger = logging.getLogger(f"adp.compute_monitor.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(root / "monitor.log", maxBytes=1024**2, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def _event(logger, event, **details):
    logger.info(json.dumps({"at": _now(), "event": event, **details}, allow_nan=False))


def run_cycle(config_path, hosts_path, *, phase=lambda _: None, logger=None, postcheck=None):
    """Transfer without resume; refresh evidence; always check guarded local cleanup."""
    outcomes = {}
    phase("cloud_connections")
    from adp.cloud_monitor import check_cloud_connections
    for identifier, evidence in check_cloud_connections().items():
        outcomes[identifier + "_connection"] = evidence
        try:
            _publish_host(hosts_path, identifier, {"connection": {**evidence, "checked_at": _now()}})
        except Exception as error:
            outcomes[identifier + "_publish_error"] = _brief(_diagnostic(error))
    try:
        connection = _config(config_path)
        for name in ("local_root", "local_staging"):
            directory = _safe(Path(connection[name]))
            while not directory.exists():
                directory = directory.parent
            if shutil.disk_usage(directory).free - MAX_TRANSFER_STAGING_BYTES < MINIMUM_FREE_BYTES:
                raise OSError("Private synchronization paused to preserve the dynamic 10 GiB local reserve, including bounded transfer copies.")
        phase("private_transfer")
        sync = _command([_python(), str(PROJECT_ROOT / "scripts" / "sync-private-server.py"),
                         "--config", str(config_path), "--no-resume"], timeout=SYNC_TIMEOUT)
        outcomes["private_sync"] = {key: value for key, value in sync.items() if key != "result"}
        if not sync["ok"]:
            raise RuntimeError(sync["error"])
        phase("private_evidence")
        update = _command([_python(), str(PROJECT_ROOT / "scripts" / "update-compute-hosts.py"),
            "--last-sync", str(Path(connection["local_root"]) / "last-sync.json"),
            "--hosts", str(hosts_path)], timeout=300)
        outcomes["private_update"] = {key: value for key, value in update.items() if key != "result"}
        if not update["ok"]:
            raise RuntimeError(update["error"])
    except Exception as error:
        message = _brief(_diagnostic(error))
        outcomes["private_error"] = message
        try:
            _publish_host(hosts_path, "private", {"status": "unavailable", "phase": "sync_unavailable",
                "message": _brief("Private synchronization could not be verified. " + message + " Previous local results and cleanup records are retained."),
                "observed_at": _now()})
        except Exception as publish_error:
            outcomes["private_publish_error"] = _brief(_diagnostic(publish_error))
    phase("local_cleanup")
    # No timeout: model archival and a 40,000-file guarded cleanup must finish safely.
    cleanup = _command([_python(), "-m", "adp.study_cleanup", "--archive-completed", "--execute"], timeout=None)
    outcomes["local_cleanup"] = {key: value for key, value in cleanup.items() if key != "result"}
    if cleanup["ok"]:
        result = cleanup["result"]
        state = result.get("state")
        if state == "completed":
            changes = {"status": "completed", "message": "Processed training NPZ cleanup completed; model archive, raw sources and held-out inputs remain protected."}
        elif state == "blocked":
            changes = {"status": "protected", "message": _brief(result.get("reason", "Training caches remain protected until all completion checks pass."))}
        else:
            changes = {"status": "blocked", "message": "Local cleanup returned an unexpected state; previous datasets and archives remain protected."}
        outcomes["local_cleanup"].update(state=state, message=changes["message"])
    else:
        changes = {"status": "blocked", "message": _brief("Local cleanup could not be verified. " + cleanup["error"])}
    try:
        _publish_host(hosts_path, "local", {"cleanup": changes})
    except Exception as error:
        outcomes["local_publish_error"] = _brief(_diagnostic(error))
    if postcheck is not None:
        try:
            phase("postcheck")
            value = postcheck()
            outcomes["postcheck"] = {"state": _brief(value.get("state", "unknown"), 80),
                                     "message": _brief(value.get("message", ""))}
        except Exception as error:
            outcomes["postcheck"] = {"state": "error", "message": _brief(_diagnostic(error))}
    if logger:
        _event(logger, "cycle_finished", outcomes=outcomes)
    return outcomes


def _owned(root):
    try:
        owner = FileLock(str(_safe(root / "monitor.lock", root)), timeout=0)
        owner.acquire()
    except Timeout:
        return True
    else:
        owner.release()
        return False


def monitor_status(state_dir=DEFAULT_STATE_DIR):
    root = _safe(state_dir)
    if not root.is_dir():
        return {"running": False, "state": "not_started", "pid_stale": False}
    saved = _read(root / "status.json")[0] if (root / "status.json").exists() else {}
    owned = _owned(root)
    alive = _alive(saved.get("pid"))
    return {**saved, "running": owned, "pid_alive": alive,
        "pid_stale": bool(saved.get("pid")) and not (owned and alive),
        "state": saved.get("state", "starting") if owned else "stopped", "observed_at": _now()}


def request_stop(state_dir=DEFAULT_STATE_DIR):
    root = _safe(state_dir)
    if not root.is_dir() or not _owned(root):
        return {"state": "not_running", "running": False}
    _write(root / "stop.json", {"version": 1, "requested_epoch": time.time(), "requested_at": _now()})
    return {"state": "stop_requested", "message": "Monitor will finish its current cycle safely, then stop. Study and application processes are unaffected."}


def _stop_requested(root, started_epoch):
    path = root / "stop.json"
    if not path.exists():
        return False
    value = _read(path)[0]
    requested = value.get("requested_epoch")
    return value.get("version") == 1 and isinstance(requested, (int, float)) and requested >= started_epoch


def run_monitor(config_path=DEFAULT_CONFIG, state_dir=DEFAULT_STATE_DIR, hosts_path=DEFAULT_HOSTS,
                *, interval_seconds=DEFAULT_INTERVAL, once=False, postcheck=None):
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or not 60 <= interval_seconds <= 86400:
        raise ValueError("Monitor interval must be between 60 seconds and one day")
    root = _safe(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_safe(root / "monitor.lock", root)), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return {"state": "already_running", "running": True}
    logger = None
    state = {"version": 1, "pid": os.getpid(), "state": "starting", "phase": "starting",
             "started_at": _now(), "started_epoch": time.time(), "interval_seconds": interval_seconds,
             "storage_reserve_bytes": MINIMUM_FREE_BYTES,
             "cycles_completed": 0, "last_cycle": None, "next_check_at": None}
    try:
        logger = _log(root)
        _write(root / "status.json", state)
        _event(logger, "monitor_started", pid=state["pid"], interval_seconds=interval_seconds)
        def phase(name):
            state.update(state="running", phase=name, updated_at=_now())
            _write(root / "status.json", state)
        def stopping():
            try:
                return _stop_requested(root, state["started_epoch"])
            except Exception as error:
                message = _brief(_diagnostic(error))
                if state.get("control_error") != message:
                    state["control_error"] = message
                    _event(logger, "control_error", error=message)
                return False
        while not stopping():
            try:
                state["last_cycle"] = run_cycle(config_path, hosts_path, phase=phase, logger=logger, postcheck=postcheck)
            except Exception as error:
                state["last_cycle"] = {"unexpected_error": _brief(_diagnostic(error))}
                _event(logger, "cycle_error", error=state["last_cycle"]["unexpected_error"])
            state["cycles_completed"] += 1
            state.update(state="waiting", phase="waiting", updated_at=_now())
            deadline = time.monotonic() + interval_seconds
            state["next_check_at"] = datetime.fromtimestamp(time.time() + interval_seconds, timezone.utc).isoformat()
            _write(root / "status.json", state)
            if once:
                break
            while time.monotonic() < deadline and not stopping():
                time.sleep(min(1., max(0., deadline - time.monotonic())))
        state.update(state="stopped", phase="stopped", stopped_at=_now(), next_check_at=None)
        _write(root / "status.json", state)
        _event(logger, "monitor_stopped", cycles_completed=state["cycles_completed"])
        return state
    finally:
        if logger:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--hosts", type=Path, default=DEFAULT_HOSTS)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--queue-plan", type=Path, help="Dispatch only this explicitly approved finite development queue")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--once", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--stop", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.status:
            result = monitor_status(args.state_dir)
        elif args.stop:
            result = request_stop(args.state_dir)
        else:
            postcheck = None
            if args.queue_plan:
                from adp.private_queue import cycle
                postcheck = lambda: cycle(args.config, args.queue_plan)
            result = run_monitor(args.config, args.state_dir, args.hosts,
                                 interval_seconds=args.interval_seconds, once=args.once, postcheck=postcheck)
        if sys.stdout is not None:
            print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        if sys.stderr is not None:
            print("Compute monitor: " + _brief(_diagnostic(error)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
