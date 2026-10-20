"""The development shortcut launcher uses explicit paths and never starts a test GUI."""
import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch-qualified-desktop.py"


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("qualified_desktop_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "projet étiqueté"
    for name in ("scripts/launch-qualified-desktop.py", "src/adp/__main__.py", "frontend/dist/index.html"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    interpreter = tmp_path / "qualified python/pythonw.exe"
    interpreter.parent.mkdir()
    interpreter.write_bytes(b"qualified fixture")
    cache = tmp_path / "scenario cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    monkeypatch.setattr(module, "__file__", str(project / "scripts/launch-qualified-desktop.py"))
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired("desktop", 3)
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    args = module.arguments(["--python", str(interpreter), "--cache-dir", str(cache),
                             "--checkpoint", str(tmp_path / "optional missing weights.pt"),
                             "--data-root", str(tmp_path / "user data"), "--training"])
    return module, project, args, process, popen


def test_launcher_argv_env_unicode_paths_and_cpu_only(launcher):
    module, project, args, process, popen = launcher
    assert module.launch(args) is process
    argv = popen.call_args.args[0]
    assert argv[:4] == [str(args.python.resolve()), "-m", "adp", "desktop"]
    assert argv[argv.index("--project-dir") + 1] == str(project.resolve())
    assert argv[argv.index("--cache-dir") + 1] == str(args.cache_dir.resolve())
    assert argv[argv.index("--device") + 1] == "cpu" and argv[-1] == "--training"
    options = popen.call_args.kwargs
    assert "shell" not in options and options["cwd"] == str(project.resolve())
    assert options["env"]["PYTHONPATH"] == str(project / "src")
    assert options["env"]["OMP_NUM_THREADS"] == options["env"]["MKL_NUM_THREADS"] == "2"
    assert options["env"]["LOCALAPPDATA"] == str(project / "artifacts/native-qualification-state")
    assert options["stdout"].name.endswith("desktop.stdout.log")
    assert options["stderr"].name.endswith("desktop.stderr.log")
    assert options["creationflags"] == (subprocess.CREATE_NO_WINDOW if module.os.name == "nt" else 0)
    process.wait.assert_called_once_with(timeout=3)


@pytest.mark.parametrize("missing", ["python", "cache", "source", "frontend"])
def test_missing_required_path_reports_actionable_error_before_spawn(launcher, missing):
    module, project, args, _, popen = launcher
    path = {"python": args.python, "cache": args.cache_dir / "manifest.json",
            "source": project / "src/adp/__main__.py", "frontend": project / "frontend/dist/index.html"}[missing]
    path.unlink()
    with pytest.raises(FileNotFoundError, match="qualified interpreter|prepared scenario cache|workspace source|frontend assets"):
        module.launch(args)
    popen.assert_not_called()


def test_training_flag_is_optional_and_single_instance_handoff_can_exit_zero(launcher):
    module, _, args, process, popen = launcher
    args.training = False
    process.wait.side_effect = None
    process.wait.return_value = 0
    assert module.launch(args) is process
    assert "--training" not in popen.call_args.args[0]


def test_actual_early_failure_points_to_error_log(launcher):
    module, project, args, process, _ = launcher
    process.wait.side_effect = None
    process.wait.return_value = 7
    with pytest.raises(RuntimeError, match="exit 7.*desktop.stderr.log"):
        module.launch(args)
    assert (project / "artifacts/launcher/desktop.stderr.log").is_file()


def test_cli_error_has_no_gui_dialog(launcher, monkeypatch):
    module, _, _, _, popen = launcher
    monkeypatch.setattr(module, "launch", Mock(side_effect=RuntimeError("real startup error")))
    with pytest.raises(SystemExit, match="Interaction Lab could not start.*") as caught:
        module.main([])
    assert "real startup error" in str(caught.value)
    popen.assert_not_called()
