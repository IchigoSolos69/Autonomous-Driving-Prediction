"""Open the qualified workspace application; this is not an installed release launcher."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = Path("C:/Users/Adi/Documents/GitHub/Autonomous Driving Prediction")
    parser.add_argument("--python", type=Path, default=Path("D:/InteractionLab/runtime-cu130/Scripts/pythonw.exe"))
    parser.add_argument("--cache-dir", type=Path, default=source / "data/cache")
    parser.add_argument("--checkpoint", type=Path, default=source / "artifacts/experiment/best.pt")
    parser.add_argument("--data-root", type=Path, default=Path("D:/InteractionLab/validation-failure-smoke"))
    parser.add_argument("--training", action="store_true")
    return parser.parse_args(argv)


def launch(args):
    root = Path(__file__).resolve().parents[1]
    executable, cache = args.python.expanduser().resolve(), args.cache_dir.expanduser().resolve()
    for path, hint in ((executable, "Select the qualified interpreter with --python"),
                       (root / "src/adp/__main__.py", "Restore the workspace source"),
                       (root / "frontend/dist/index.html", "Build the frontend assets first"),
                       (cache / "manifest.json", "Select a prepared scenario cache with --cache-dir")):
        if not path.is_file():
            raise FileNotFoundError(f"{hint}: {path}")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       LOCALAPPDATA=str(root / "artifacts/native-qualification-state"))
    command = [str(executable), "-m", "adp", "desktop", "--project-dir", str(root),
               "--cache-dir", str(cache), "--checkpoint", str(args.checkpoint.expanduser().resolve()),
               "--data-root", str(args.data_root.expanduser().resolve()), "--device", "cpu"]
    if args.training:
        command.append("--training")
    logs = root / "artifacts/launcher"
    logs.mkdir(parents=True, exist_ok=True)
    error_log = logs / "desktop.stderr.log"
    with (logs / "desktop.stdout.log").open("ab") as stdout, error_log.open("ab") as stderr:
        process = subprocess.Popen(command, cwd=str(root), env=environment, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        code = process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        return process
    if code:
        raise RuntimeError(f"Application startup failed (exit {code}). Inspect {error_log}")
    return process  # A successful single-instance handoff can exit immediately.


def main(argv=None):
    try:
        launch(arguments(argv))
    except (OSError, ValueError, RuntimeError) as error:
        message = f"Interaction Lab could not start.\n\n{error}"
        if os.name == "nt" and sys.stderr is None:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "Interaction Lab startup", 0x10)
            raise SystemExit(1)
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
