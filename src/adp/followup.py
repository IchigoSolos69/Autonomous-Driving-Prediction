"""One process waits for the frozen study, then runs the approved development follow-up."""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import sys
import time

from adp.data import manifest_hash, read_json, write_json
from adp.research import config_fingerprint, ensure_space


def _source_hash():
    return config_fingerprint({name: manifest_hash(Path(__file__).parent / name) for name in
        ("followup.py", "direction.py", "analysis.py", "model.py", "training.py", "data.py", "metrics.py")})


def _check_source(root):
    if read_json(root / "followup_request.json")["source_sha256"] != _source_hash():
        raise ValueError("Follow-up source changed while the workflow was active; no further phase will start")


@contextmanager
def _prevent_idle_sleep():
    # Scoped to this thread; the display and persistent power settings stay unchanged.
    keep_awake, previous = None, 0
    if sys.platform == "win32":
        import ctypes
        keep_awake = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
        keep_awake.argtypes = [ctypes.c_uint32]
        keep_awake.restype = ctypes.c_uint32
        previous = keep_awake(0x80000001)
        if not previous:
            raise RuntimeError("Windows could not prevent idle sleep; the follow-up did not start")
    try:
        yield
    finally:
        if keep_awake is not None:
            keep_awake(previous)


def _analysis(study_root, output):
    from adp.analysis import _models, analyze_failures, validate_analysis
    _, _, expected = _models(study_root)
    reused = output.exists()
    if not reused:
        analyze_failures(study_root, output)
    metadata, _ = validate_analysis(output)
    if metadata["provenance"] != expected:
        raise ValueError(f"Existing analysis belongs to different inputs or source: {output}; preserve it outside this output path before resuming")
    return {"id": metadata["id"], "output_dir": str(output), "reused": reused, "development_only": True}


def run_direction_followup(reference_root, root="D:/InteractionLab/direction-study-v1", poll_seconds=30, device="cuda"):
    from filelock import FileLock, Timeout
    if isinstance(poll_seconds, bool) or not math.isfinite(poll_seconds) or not 1 <= poll_seconds <= 60:
        raise ValueError("poll_seconds must be between 1 and 60 seconds")
    reference_root, root = Path(reference_root).resolve(), Path(root).resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError("Reference study directory does not exist")
    if root == reference_root or root in reference_root.parents or reference_root in root.parents:
        raise ValueError("Use a separate follow-up root outside the reference study")
    if root.exists() and not root.is_dir():
        raise ValueError("Follow-up root must be a directory")
    ensure_space(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / "followup.lock"), timeout=0), _prevent_idle_sleep():
            request = {"version": 1, "reference_root": str(reference_root), "root": str(root),
                "source_sha256": _source_hash()}
            request_path = root / "followup_request.json"
            if request_path.exists() and read_json(request_path) != request:
                raise ValueError("Follow-up inputs or source changed; preserve the recorded workflow and use a separate root")
            if not request_path.exists():
                write_json(request_path, request)
            return _run(reference_root, root, poll_seconds, device)
    except Timeout as error:
        raise RuntimeError("A follow-up workflow already owns this root") from error


def _run(reference_root, root, poll_seconds, device):
    path = root / "followup_status.json"
    state = {"status": "waiting", "phase": "waiting_for_reference", "development_only": True,
             "reference_root": str(reference_root), "device": device, "followup_training_allowed": False}
    try:
        while True:
            reference_path = reference_root / "pipeline_status.json"
            reference = read_json(reference_path) if reference_path.exists() else {"status": "not_started"}
            status = reference.get("status")
            state["reference_status"] = reference
            if status == "completed" and reference.get("phase") == "finished":
                break
            if status not in ("running", "not_started"):
                state.update(status="reference_stopped", reason="The original study is not successfully completed; no follow-up GPU work started")
                write_json(path, state)
                return state
            write_json(path, state)
            time.sleep(poll_seconds)
        state.update(status="running", phase="reference_analysis")
        write_json(path, state)
        _check_source(root)
        state["reference_analysis"] = _analysis(reference_root, root / "reference-analysis")
        _check_source(root)
        state["phase"] = "direction_study"
        state["followup_training_allowed"] = True
        write_json(path, state)
        from adp.direction import run_direction_study
        result = run_direction_study(reference_root, root, device)
        state["direction_result"] = result
        state["followup_training_allowed"] = result.get("status") != "waiting_for_reference"
        if result.get("status") != "completed":
            state.update(status="incomplete_experiment", reason=result.get("reason", "The paired experiment has not completed; final failure analysis was not built"))
            write_json(path, state)
            return state
        state["phase"] = "direction_analysis"
        write_json(path, state)
        _check_source(root)
        state["direction_analysis"] = _analysis(root, root / "direction-analysis")
        _check_source(root)
        state.update(status="completed", phase="finished")
        write_json(path, state)
        return state
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "stopped",
                     reason=str(error) or type(error).__name__, resumable=True)
        write_json(path, state)
        raise
