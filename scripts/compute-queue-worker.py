"""Run the finite, development-only pilot queue; never start either frozen study.

Cloud use is an explicitly started notebook process, not a remote polling service.
An interrupted owner lease requires positive process-death proof before recovery.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

PAYLOAD_SHA256 = "f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa"
MINIMUM_FREE = 8 * 1024**3
EXPORT_MARGIN = 60.
SHA = re.compile(r"[0-9a-f]{64}")


class ChildExitUnverified(RuntimeError):
    """Do not release a lease when its owned process tree may still be alive."""


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def safe_path(path, *, exists=True):
    path = Path(path).expanduser().absolute()
    if any(linked(item) for item in (path, *path.parents)):
        raise ValueError("Queue paths cannot contain links or junctions")
    if exists and not path.exists():
        raise ValueError("Required queue resource is missing: " + str(path))
    return path.resolve()


def read_json(path, limit=2 * 1024**2):
    path = safe_path(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Expected bounded JSON file: " + str(path))
    return json.loads(path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def write_json(path, value, *, private=False):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=".worker-", dir=path.parent)
    temporary = Path(name)
    try:
        if private:
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_plan(module, host):
    return {"version": 1, "host_seconds": 21600,
            "jobs": [module.job_spec(host, config, seed)
                     for config in module.CONFIGURATIONS for seed in module.SEEDS]}


def verify_resources(args, module):
    root = safe_path(args.root)
    resources = read_json(root / "resources.json")
    plan_hash = module.spec_hash(finite_plan(module, args.host))
    if resources.get("payload_sha256") != PAYLOAD_SHA256 or resources.get("plan_sha256") != plan_hash:
        raise ValueError("Resource manifest does not bind the approved payload and finite host plan")
    files = resources.get("files")
    if not isinstance(files, dict) or not 3 <= len(files) <= 256:
        raise ValueError("Missing bounded controller resource checksum index")
    validated = set()
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or ":" in name or not SHA.fullmatch(str(digest)):
            raise ValueError("Invalid controller resource checksum entry")
        target = safe_path(root / name)
        if not target.is_file() or sha(target) != digest:
            raise ValueError("Controller resource checksum differs: " + name)
        validated.add(target)
    for required in (__file__, args.queue_module, args.pilot_helper):
        if safe_path(required) not in validated:
            raise ValueError("Worker, queue and pilot helper must be pinned controller resources")
    payload = safe_path(args.payload)
    if payload.is_file() and sha(payload) != PAYLOAD_SHA256:
        raise ValueError("Payload differs from the approved train/selection pilot")
    for forbidden in ("job.json", "pipeline_status.json", "followup_request.json"):
        if (root / forbidden).exists():
            raise ValueError("Use a fresh queue namespace, separate from existing jobs and studies")
    return plan_hash


def control_module(source, name):
    source = safe_path(source)
    if not (source / "adp" / (name + ".py")).is_file():
        raise ValueError("Qualified control package is incomplete: " + name)
    sys.path.insert(0, str(source))
    module = __import__("adp." + name, fromlist=[name])
    if Path(module.__file__).resolve() != (source / "adp" / (name + ".py")).resolve():
        raise ValueError("A different adp package was already imported; start a fresh worker process")
    return module


def local_dependencies(control_source):
    cleanup = control_module(control_source, "study_cleanup")
    selection = read_json(cleanup.STUDY_ROOT / "frozen_selection.json")
    digest = selection.get("checkpoint_sha256")
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ValueError("Local study deployment checkpoint is not frozen")
    proof = read_json(cleanup.MODEL_ARCHIVE_ROOT / digest / "cleanup-proof.json")
    result = cleanup.plan_study_cleanup(cleanup.STUDY_ROOT, cleanup.DIRECTION_ROOT, proof)
    if result.get("state") != "completed":
        raise ValueError("Local GPU queue waits for original study, direction study, CPU reports and completed training-cache cleanup: " + str(result.get("reason", result.get("state"))))
    return ("original_study", "direction_study", "cpu_reports", "training_cache_cleanup")


def check_space(root, projected=0):
    if shutil.disk_usage(root).free - projected < MINIMUM_FREE:
        raise RuntimeError("Queue paused to preserve the 8 GiB free-space reserve")


def recipe(initial, spec):
    result = deepcopy(initial)
    result["seed"] = spec["seed"]
    result.setdefault("model", {}).update(use_graph=spec["configuration"] != "no_graph",
        use_map=spec["configuration"] != "no_map",
        shared_world_ce_weight=.1 if spec["configuration"] == "world_ce" else 0.)
    return result


def child_environment(job):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(job / "source"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return env


def stop_owned_child(process):
    """Stop only this worker's subprocess tree, including Windows venv launchers."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)


def run_child(argv, job, timeout, *, capture=False):
    if timeout <= 0:
        raise TimeoutError("No wall-time allowance remains for a child process")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (job / "worker-child.log").open("ab") as log:
        process = subprocess.Popen(argv, cwd=job, env=child_environment(job), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else log, stderr=log, **options)
        try:
            output, _ = process.communicate(timeout=timeout)
        except BaseException:
            try:
                stop_owned_child(process)
            except BaseException as error:
                raise ChildExitUnverified("Could not prove owned child process-tree exit; preserve the active lease") from error
            raise
        if process.returncode:
            raise RuntimeError(f"Owned child exited {process.returncode}; see {job / 'worker-child.log'}")
    return output


QUALIFICATION = """import json,sys,torch
assert sys.version_info[:2]==(3,12), 'Use the isolated Python 3.12 runtime'
assert torch.__version__.split('+')[0]=='2.13.0', 'Use qualified PyTorch 2.13.0'
assert torch.cuda.is_available(), 'CUDA initialization failed; this GPU queue does not fall back silently'
count=torch.cuda.device_count()
x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x).square().mean().backward(); torch.cuda.synchronize()
print(json.dumps({'device_count':count,'gpu':[torch.cuda.get_device_name(i) for i in range(count)],'python':sys.version.split()[0],'torch':torch.__version__}))
"""


def stage_job(root, helper, payload, spec, job_id):
    base = root / "input"
    if Path(payload).is_dir():
        helper.verify_extracted_and_stage(payload, base)
    else:
        helper.verify_and_stage(payload, base, PAYLOAD_SHA256)
    initial = recipe(read_json(base / "initial-config.json"), spec)
    job = root / "jobs" / job_id
    check_space(root, 512 * 1024**2)
    if not job.exists():
        job.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".job-", dir=job.parent))
        try:
            for name in ("cache", "source"):
                shutil.copytree(base / name, temporary / name)
            shutil.copy2(base / "source-manifest.json", temporary / "source-manifest.json")
            write_json(temporary / "initial-config.json", initial)
            temporary.rename(job)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    else:
        safe_path(job)
        if read_json(job / "initial-config.json") != initial:
            raise ValueError("Existing queue job configuration differs from its immutable recipe")
    # Recheck every copied source/cache file before the child imports or unpickles anything.
    index = read_json(base / "package-files.json")
    expected_source = set()
    for name, info in index.items():
        if name.startswith(("source/", "cache/")) or name == "source-manifest.json":
            target = safe_path(job / name)
            if not target.is_file() or sha(target) != info["sha256"]:
                raise ValueError("Job source/cache differs from the pinned payload: " + name)
            if name.startswith("source/"):
                expected_source.add(target)
    if {item.resolve() for item in (job / "source").rglob("*") if item.is_file()} != expected_source:
        raise ValueError("The frozen job source must contain exactly the eight approved files")
    return job


def export_job(args, module, job, record, count, plan_hash):
    run = job / "run"
    protocol = read_json(job / "job.json")
    for name in ("job.json", "config.json"):
        shutil.copy2(job / name, run / name)
    shutil.copy2(job / "qualification.json", run / "qualification.json")
    shutil.copy2(job / "status.json", run / "result.json")
    provenance = {"queue_job_spec": record["spec"], "queue_job_id": record["id"],
        "payload_sha256": PAYLOAD_SHA256, "queue_protocol_sha256": plan_hash,
        "allocated_device_count": count, "job_protocol_sha256": protocol["protocol_sha256"],
        "config_sha256": protocol["config_sha256"],
        "manifest_sha256": protocol["cache_files"]["manifest.json"],
        "preprocessing_sha256": protocol["cache_files"]["preprocessing.json"]}
    transfer = control_module(args.control_source, "remote_transfer")
    snapshot = transfer.create_snapshot(run, args.root / "snapshots", record["id"], provenance)
    identifier = snapshot["snapshot_id"]
    folder = Path(snapshot["path"])
    metadata = read_json(folder / "snapshot.json", 8 * 1024**2)
    check_space(args.root, sum(item["size_bytes"] for item in metadata["files"].values()) + 8 * 1024**2)
    exports = args.root / "exports"
    exports.mkdir(exist_ok=True)
    temporary = exports / (identifier + ".partial")
    destination = exports / (identifier + ".zip")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
        for name in ["snapshot.json", *sorted(metadata["files"])]:
            archive.write(folder / name, name)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != {"snapshot.json", *metadata["files"]}:
            raise ValueError("Result archive membership differs from its snapshot")
        for name, info in metadata["files"].items():
            data = archive.read(name)
            if len(data) != info["size_bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError("Result archive checksum differs: " + name)
    os.replace(temporary, destination)
    return {"job_id": record["id"], "snapshot_id": identifier, "archive_name": destination.name,
            "bytes": destination.stat().st_size, "sha256": sha(destination)}


def public_status(root, module, host):
    path = root / "controller" / "queue.json"
    queue = module.ComputeQueue(path.parent).snapshot() if path.exists() else None
    status = read_json(root / "worker-status.json") if (root / "worker-status.json").exists() else {}
    return {"version": 1, "host": host, "queue": queue, **status}


def publish_status(root, queue, host, exports, state, *, current=None, message=None, last=None):
    jobs = [job for job in queue.snapshot()["jobs"] if job["spec"]["host"] == host]
    write_json(root / "worker-status.json", {"updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "state": state, "current_job_id": current, "last_job_id": last,
        "completed_jobs": sum(job["state"] == "completed" for job in jobs), "planned_jobs": len(jobs),
        "message": message, "exports": exports})


def stop_requested(root):
    flag = root / "stop-after-current.json"
    if not flag.exists() and not linked(flag):
        return False
    if read_json(flag, 1024) != {"version": 1, "stop": True}:
        raise ValueError("Invalid stop-after-current flag; preserve it and inspect the bounded control file")
    return True


def run_worker(args):
    from filelock import FileLock
    args.root = safe_path(args.root)
    module = load_module(args.queue_module, "interaction_queue")
    if args.status:
        return public_status(args.root, module, args.host)
    plan_hash = verify_resources(args, module)
    if getattr(args, "stop_after_current", False):
        flag = args.root / "stop-after-current.json"
        if flag.exists() or linked(flag):
            stop_requested(args.root)
        write_json(flag, {"version": 1, "stop": True})
        return {**public_status(args.root, module, args.host), "stop_after_current": True}
    lock = args.root / ".worker.lock"
    if linked(lock):
        raise ValueError("Worker lock cannot be linked")
    with FileLock(str(lock), timeout=0):
        if getattr(args, "resume_queue", False) and stop_requested(args.root):
            (args.root / "stop-after-current.json").unlink()
        dependencies = local_dependencies(args.control_source) if args.host == "local" else ()
        # Twelve independent cache copies plus checkpoints/snapshots/ZIPs, retaining 8 GiB.
        check_space(args.root, 5 * 1024**3)
        helper = load_module(args.pilot_helper, "interaction_pilot_helper")
        queue = module.ComputeQueue(args.root / "controller")
        queue.initialize(); queue.plan(args.host)
        prior = public_status(args.root, module, args.host)
        exports = prior.get("exports", [])
        if any(job["spec"]["host"] == args.host and job["lease"] for job in prior["queue"]["jobs"]):
            raise RuntimeError("An existing worker lease requires verified process-death recovery; it cannot be reset by restart")
        publish_status(args.root, queue, args.host, exports, "running")
        owner = f"{socket.gethostname()[:60]}:{os.getpid()}"
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        while True:
            if stop_requested(args.root):
                publish_status(args.root, queue, args.host, exports, "paused", message="Stopped before the next job; current committed results and pending jobs are preserved")
                break
            started = time.monotonic()
            record = queue.claim(args.host, owner, dependencies_verified=dependencies)
            if record is None:
                jobs = [job for job in queue.snapshot()["jobs"] if job["spec"]["host"] == args.host]
                state = "completed" if all(job["state"] == "completed" for job in jobs) else "blocked" if any(job["state"] == "blocked" for job in jobs) else "failed"
                publish_status(args.root, queue, args.host, exports, state)
                break
            publish_status(args.root, queue, args.host, exports, "running", current=record["id"])
            token = record["lease_token"]
            write_json(args.root / ".worker-lease.json", {"job_id": record["id"], "lease_token": token,
                "owner_id": owner, "pid": os.getpid()}, private=True)
            count, began, artifact, outcome, message, checkpoint = None, False, None, "failed", None, None
            exit_verified = True
            try:
                job = stage_job(args.root, helper, args.payload, record["spec"], record["id"])
                # Qualification itself is charged to this job, including a real CUDA backward pass.
                expected_count = 2 if args.host == "kaggle" else 1
                qualification_allowance = min(90., started + record["lease"]["reserved_seconds"] / expected_count - time.monotonic() - EXPORT_MARGIN)
                hardware = json.loads(run_child([str(python), "-c", QUALIFICATION], job, qualification_allowance, capture=True))
                count = hardware["device_count"]
                if type(count) is not int or count != (2 if args.host == "kaggle" else 1):
                    raise ValueError("Allocated GPU count differs from this host's qualified queue contract")
                queue.start(record["id"], token, owner, device_count=count); began = True
                write_json(job / "qualification.json", hardware)
                maximum = record["spec"]["maximum_seconds"] / count
                deadline = started + record["lease"]["reserved_seconds"] / count
                if record["lease"]["resume_checkpoint_sha256"]:
                    if sha(job / "run" / "last.pt") != record["lease"]["resume_checkpoint_sha256"]:
                        raise ValueError("Resume checkpoint differs from the committed queue receipt")
                run_child([str(python), "-m", "adp.remote_jobs", "prepare", "--root", str(job),
                    "--config", str(job / "initial-config.json"), "--epochs", "20",
                    "--max-runtime-seconds", str(maximum)], job, deadline - time.monotonic() - EXPORT_MARGIN)
                allowance = deadline - time.monotonic() - EXPORT_MARGIN
                run_child([str(python), "-m", "adp.remote_jobs", "run", "--root", str(job),
                    "--device", "cuda", "--slice-seconds", str(allowance)], job,
                    deadline - time.monotonic() - EXPORT_MARGIN / 2)
                status = read_json(job / "status.json")
                checkpoint = sha(job / "run" / "last.pt")
                outcome = "completed" if status.get("state") == "completed" and status.get("result", {}).get("completed_epochs") == 20 else "paused"
                artifact = export_job(args, module, job, record, count, plan_hash)
                if artifact not in exports:
                    exports.append(artifact)
            except BaseException as error:
                message = f"{type(error).__name__}: {error}"[:600]
                if isinstance(error, ChildExitUnverified):
                    exit_verified = False
                elif not began:
                    # No surviving child: run_child kills and waits for its owned process tree.
                    queue.recover(record["id"], token, verify_exit=lambda _: True)
                else:
                    outcome = "failed"
            finally:
                if began and exit_verified:
                    wall = time.monotonic() - started
                    settled = queue.finish(record["id"], token, outcome=outcome, elapsed_seconds=wall * count,
                        wall_seconds=wall, checkpoint_sha256=checkpoint, message=message)
                    outcome = settled["state"]
                jobs = [job for job in queue.snapshot()["jobs"] if job["spec"]["host"] == args.host]
                state = "completed" if all(job["state"] == "completed" for job in jobs) else "waiting" if outcome == "completed" and args.once else "running" if outcome == "completed" else "paused" if outcome == "queued" else outcome
                publish_status(args.root, queue, args.host, exports, state,
                    current=record["id"] if not exit_verified else None, message=message, last=record["id"])
                if exit_verified:
                    (args.root / ".worker-lease.json").unlink(missing_ok=True)
            if args.once or outcome != "completed":
                break
        return public_status(args.root, module, args.host)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", choices=("local", "private", "colab", "kaggle"), required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--queue-module", type=Path, required=True)
    parser.add_argument("--pilot-helper", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--once", action="store_true")
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--stop-after-current", action="store_true", help="Request a pause before the next job without killing training")
    control.add_argument("--resume-queue", action="store_true", help="Clear the validated stop flag after acquiring the exclusive worker lock")
    args = parser.parse_args(argv)
    print(json.dumps(run_worker(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
