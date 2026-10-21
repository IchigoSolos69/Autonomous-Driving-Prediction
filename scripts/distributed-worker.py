"""Explicitly start one finite host allocation; runtime loss never resets its ledger."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

BASE_SHA = "c04f4cf43cf6423546e20fb9e87a54bfb9598a6b700a1dcfe0853abec31f982e"


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run(args):
    from filelock import FileLock
    root = args.root.absolute()
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in (root, *root.parents)):
        raise ValueError("Worker paths must be unlinked")
    base_path = root / "worker-base.py"
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Frozen base worker differs")
    base = load(base_path, "distributed_worker_base")
    module = load(root / "distributed_plan.py", "distributed_plan")
    resources = base.read_json(root / "resources.json")
    if resources["protocol_sha256"] != module.protocol_hash(args.host):
        raise ValueError("Follow-up resource protocol differs")
    for name, digest in resources["files"].items():
        if (Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name
                or base.sha(base.safe_path(root / name)) != digest):
            raise ValueError("Pinned worker resource differs")
    if base.sha(root / "inputs.zip") != module.PAYLOAD_SHA256:
        raise ValueError("Only the approved pilot data may be used")
    # The source helper is unchanged; the stricter operational reserve applies here.
    base.MINIMUM_FREE = 10 * 1024**3
    path = root / "allocation.json"
    with FileLock(str(root / ".distributed.lock"), timeout=0):
        state = module.validate_state(base.read_json(path), args.host) if path.exists() else module.initial_state(args.host)
        if state["active"] is not None or any(a["state"] == "reserved" for a in state["attempts"]):
            raise RuntimeError("An interrupted allocation is conservatively charged; preserve it for process-death recovery")
        def save():
            state["observed_at"] = datetime.now(timezone.utc).isoformat()
            state["pid"] = os.getpid()
            module.validate_state(state, args.host)
            base.write_json(path, state)
        save()
        helper = load(root / "pilot-helper.py", "distributed_pilot_helper")
        base.check_space(root, 5 * 1024**3)
        count = 2 if args.host == "kaggle" else 1
        for spec in module.plan(args.host)["jobs"]:
            identifier = module.spec_hash(spec)
            if identifier in {a["job_id"] for a in state["attempts"]}:
                continue
            if spec["maximum_seconds"] > module.HOST_SECONDS - state["charged_seconds"]:
                state["state"] = "budget_exhausted"; save(); break
            attempt = {"job_id": identifier, "state": "reserved", "charged_seconds": spec["maximum_seconds"],
                "started_at": datetime.now(timezone.utc).isoformat()}
            state["attempts"].append(attempt)
            state["active"] = identifier
            state["charged_seconds"] += attempt["charged_seconds"]
            state["state"] = "running"; save()
            began = time.monotonic()
            deadline = began + spec["maximum_seconds"] / count
            checkpoint = None
            exit_verified = True
            try:
                job = base.stage_job(root, helper, root / "inputs.zip", spec, identifier)
                hardware = json.loads(base.run_child([sys.executable, "-c", base.QUALIFICATION], job,
                    min(90., deadline-time.monotonic()-60), capture=True))
                if hardware["device_count"] != count:
                    raise ValueError("Unexpected allocated device count")
                base.write_json(job / "qualification.json", hardware)
                base.run_child([sys.executable, "-m", "adp.remote_jobs", "prepare", "--root", str(job),
                    "--config", str(job / "initial-config.json"), "--epochs", str(spec["epochs"]),
                    "--max-runtime-seconds", str(spec["maximum_seconds"] / count)], job, deadline-time.monotonic()-60)
                base.run_child([sys.executable, "-m", "adp.remote_jobs", "run", "--root", str(job),
                    "--device", "cuda", "--slice-seconds", str(deadline-time.monotonic()-90)], job,
                    deadline-time.monotonic()-45)
                status = base.read_json(job / "status.json")
                outcome = "completed" if status.get("state") == "completed" and status["result"]["completed_epochs"] == spec["epochs"] else "paused"
                checkpoint = base.sha(job / "run/last.pt")
                export_args = argparse.Namespace(root=root, control_source=root / "control")
                artifact = base.export_job(export_args, module, job, {"id": identifier, "spec": spec}, count,
                    module.protocol_hash(args.host))
                state["exports"].append(artifact)
                attempt["state"] = outcome
                attempt["checkpoint_sha256"] = checkpoint
                if outcome == "completed":
                    state["completed"].append(identifier)
            except BaseException as error:
                attempt["error"] = f"{type(error).__name__}: {error}"[:600]
                if isinstance(error, base.ChildExitUnverified):
                    exit_verified = False
                else:
                    attempt["state"] = "failed"
            finally:
                if exit_verified:
                    elapsed = (time.monotonic()-began) * count
                    state["charged_seconds"] += elapsed - attempt["charged_seconds"]
                    attempt["charged_seconds"] = elapsed
                    attempt["wall_seconds"] = elapsed / count
                    attempt["device_count"] = count
                    state["active"] = None
                    if elapsed > spec["maximum_seconds"]:
                        state["completed"] = [i for i in state["completed"] if i != identifier]
                        attempt["state"] = "failed"
                        attempt["error"] = "Measured execution exceeded its reservation"
                state["state"] = "running" if attempt["state"] == "completed" else attempt["state"]
                save()
            print(json.dumps({"host": args.host, "completed_jobs": len(state["completed"]),
                "charged_seconds": state["charged_seconds"], "state": state["state"]}), flush=True)
            if attempt["state"] != "completed":
                break
        else:
            state["state"] = "completed"; save()
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", choices=("private", "colab", "kaggle"), required=True)
    print(json.dumps(run(parser.parse_args()), allow_nan=False))
