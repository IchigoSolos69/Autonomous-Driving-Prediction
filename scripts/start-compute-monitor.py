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


def status_message(result, state_dir):
    from adp.compute_monitor import monitor_status

    try:
        state = monitor_status(state_dir)
    except (OSError, ValueError, RuntimeError):
        state = None
    if state is None:
        lines = ["The launch request succeeded, but monitor status could not be read.",
                 "Check the saved status and log below."]
    elif (state.get("running") and not state.get("pid_stale")
          and state.get("pid_alive") is not False and state.get("state") != "stopped"):
        lines = ["The compute monitor is running in the background.",
                 f"Worker PID: {state.get('pid') or 'not yet published'}",
                 f"Phase: {state.get('phase') or 'starting'}",
                 f"Completed checks: {state.get('cycles_completed', 0)}",
                 f"Last {'completed check' if state.get('phase') == 'waiting' else 'check update'}: {state.get('updated_at') or 'not yet published'}",
                 f"Next check: {state.get('next_check_at') or 'scheduled after the current check finishes'}"]
    elif result.get("state") == "started" or state.get("running"):
        # The detached worker may not have acquired its lock or written status yet.
        lines = ["The compute monitor is starting in the background.",
                 f"Startup process PID: {result.get('pid') if result.get('state') == 'started' else 'not yet confirmed'}",
                 "The first check and worker status have not been published yet."]
    else:
        lines = ["The compute monitor is not currently running.",
                 "It may have stopped after the shortcut checked it. Launch the shortcut again."]
    lines += ["", f"Status: {Path(state_dir) / 'status.json'}",
              f"Log: {Path(state_dir) / 'monitor.log'}", "",
              "Closing this dialog does not stop the background monitor.",
              "Open Interaction Lab's Training screen for training and cloud evidence."]
    return "\n".join(lines)


def show_status(result, state_dir):
    message = status_message(result, state_dir)
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "Interaction Lab Compute Monitor", 0x40)
    elif sys.stdout is not None:
        print(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("D:/InteractionLab/remote-compute/monitor"))
    parser.add_argument("--show-status", action="store_true", help="Show background status after launch; omit for quiet sign-in startup")
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
    if args.show_status:
        show_status(result, args.state_dir)
    elif sys.stdout is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
