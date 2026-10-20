"""Start the monitor independently of the caller's Windows process job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_QUEUE_PLAN = Path("D:/InteractionLab/remote-compute/queues/development-v1/approved-plan.json")

def launch(state_dir):
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from adp.compute_monitor import DEFAULT_INTERVAL, monitor_status

    state = monitor_status(state_dir)
    if state["running"]:
        return {"state": "already_running", "pid": state.get("pid")}
    interpreter = Path(sys.executable)
    if os.name == "nt":
        interpreter = interpreter.with_name("pythonw.exe")
    entry = project / "scripts/compute-monitor.py"
    if not interpreter.is_file() or not entry.is_file():
        raise FileNotFoundError("Restore the qualified Python runtime and monitor entry point")
    environment = {**os.environ, "PYTHONPATH": str(project / "src"),
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUTF8": "1"}
    flags = 0
    if os.name == "nt":
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_BREAKAWAY_FROM_JOB)
    command = [str(interpreter), str(entry), "--interval-seconds", str(DEFAULT_INTERVAL), "--state-dir", str(state_dir)]
    queue_plan = DEFAULT_QUEUE_PLAN
    if queue_plan.is_file():
        command += ["--queue-plan", str(queue_plan)]
    process = subprocess.Popen(command,
        cwd=project, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags,
        start_new_session=os.name != "nt")
    return {"state": "started", "pid": process.pid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("D:/InteractionLab/remote-compute/monitor"))
    args = parser.parse_args(argv)
    try:
        result = launch(args.state_dir)
    except (OSError, ValueError, RuntimeError) as error:
        message = f"The compute monitor could not start independently.\n\n{error}"
        if os.name == "nt" and sys.stderr is None:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "Interaction Lab Compute Monitor", 0x10)
        elif sys.stderr is not None:
            print(message, file=sys.stderr)
        return 1
    if sys.stdout is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
