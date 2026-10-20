import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from adp import compute_monitor


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/start-compute-monitor.py"
    spec = importlib.util.spec_from_file_location("monitor_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    executable = runtime / ("pythonw.exe" if os.name == "nt" else "python")
    executable.write_bytes(b"test executable")
    monkeypatch.setattr(module.sys, "executable", str(executable))
    monkeypatch.setattr(module, "DEFAULT_QUEUE_PLAN", tmp_path / "absent-plan.json")
    monkeypatch.setattr(compute_monitor, "monitor_status", lambda _: {"running": False})
    return module


def test_launcher_does_not_duplicate_running_monitor(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(compute_monitor, "monitor_status", lambda _: {"running": True, "pid": 123})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate launch"))
    assert launcher.launch(tmp_path) == {"state": "already_running", "pid": 123}


def test_launcher_detaches_without_shell_or_inherited_streams(launcher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "Popen",
                        lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(pid=456))
    assert launcher.launch(tmp_path) == {"state": "started", "pid": 456}
    args, options = calls[0]
    assert args[0][-2:] == ["--state-dir", str(tmp_path)]
    assert args[0][2:4] == ["--interval-seconds", "300"]
    assert options["stdin"] == options["stdout"] == options["stderr"] == launcher.subprocess.DEVNULL
    assert options["close_fds"] and not options.get("shell", False)
    if os.name == "nt":
        assert options["creationflags"] & launcher.subprocess.CREATE_BREAKAWAY_FROM_JOB
        assert options["creationflags"] & launcher.subprocess.DETACHED_PROCESS
    else:
        assert options["start_new_session"]


def test_missing_runtime_refuses_start(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "executable", str(tmp_path / "missing/python.exe"))
    with pytest.raises(FileNotFoundError):
        launcher.launch(tmp_path)


def test_failed_breakaway_does_not_silently_launch_attached(launcher, monkeypatch, tmp_path, capsys):
    calls = []
    def denied(*args, **kwargs):
        calls.append(args)
        raise PermissionError("breakaway denied")
    monkeypatch.setattr(launcher.subprocess, "Popen", denied)
    assert launcher.main(["--state-dir", str(tmp_path)]) == 1
    assert len(calls) == 1
    assert "could not start independently" in capsys.readouterr().err


def test_approved_queue_plan_is_forwarded_to_independent_monitor(launcher, monkeypatch, tmp_path):
    plan = tmp_path / "approved-plan.json"
    plan.write_text('{}')
    monkeypatch.setattr(launcher, "DEFAULT_QUEUE_PLAN", plan)
    calls = []
    monkeypatch.setattr(launcher.subprocess, "Popen",
                        lambda *a, **kw: calls.append(a[0]) or SimpleNamespace(pid=456))
    launcher.launch(tmp_path)
    assert calls[0][-2:] == ["--queue-plan", str(plan)]
