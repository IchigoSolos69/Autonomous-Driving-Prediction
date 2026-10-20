"""Finite development jobs; budget seconds are wall time times allocated GPU count."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

from filelock import FileLock

HOSTS = ("local", "private", "colab", "kaggle")
CONFIGURATIONS = ("full", "no_graph", "no_map", "world_ce")
SEEDS = (42, 43, 44)
HOST_SECONDS = 21600
JOB_SECONDS = 1800
PAYLOAD_SHA256 = "f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa"
LOCAL_DEPENDENCIES = ("original_study", "direction_study", "cpu_reports", "training_cache_cleanup")
MAX_BYTES = 2 * 1024**2
ACTIVE = {"running", "manual_start_required"}
STATES = ACTIVE | {"queued", "completed", "failed", "blocked"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def spec_hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _host(host):
    if host not in HOSTS:
        raise ValueError("Unsupported compute host")
    return host


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2 * HOST_SECONDS:
        raise ValueError("Elapsed seconds must be finite, nonnegative and bounded")
    return float(value)


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value):
        raise ValueError("Expected a bounded owner or worker identity")
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 identity")
    return value


def job_spec(host, configuration, seed):
    _host(host)
    if configuration not in CONFIGURATIONS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only the fixed development configurations and seeds are supported")
    return {"version": 1, "purpose": "development_only", "budget_unit": "allocated_gpu_seconds", "host": host, "recipe": "pilot-seed",
        "configuration": configuration, "seed": seed, "epochs": 20, "maximum_seconds": JOB_SECONDS,
        "payload_sha256": PAYLOAD_SHA256, "train_count": 1000, "selection_count": 64,
        "dependencies": list(LOCAL_DEPENDENCIES) if host == "local" else []}


def validate_spec(value):
    if not isinstance(value, dict) or value != job_spec(value.get("host"), value.get("configuration"), value.get("seed")):
        raise ValueError("Job specification differs from the approved finite recipe")
    return value


class ComputeQueue:
    def __init__(self, root):
        path = Path(root).expanduser().absolute()
        if any(item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()) for item in (path, *path.parents)):
            raise ValueError("Queue storage must not use links or junctions")
        self.root = path.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "queue.json"
        self.lock = self.root / ".compute-queue.lock"

    def _read(self):
        if self.path.is_symlink() or (hasattr(self.path, "is_junction") and self.path.is_junction()):
            raise ValueError("Queue file must not be linked")
        with self.path.open("rb") as stream:
            content = stream.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise ValueError("Queue metadata exceeds its size bound")
        value = json.loads(content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite queue number")))
        self._validate(value)
        return value

    @staticmethod
    def _validate(value):
        try:
            if set(value) != {"version", "host_seconds", "hosts", "jobs"} or value["version"] != 1 or value["host_seconds"] != HOST_SECONDS or set(value["hosts"]) != set(HOSTS) or len(value["jobs"]) > 48:
                raise ValueError("Unsupported queue schema or budget")
            ids, active = set(), set()
            for job in value["jobs"]:
                if set(job) != {"id", "spec_sha256", "spec", "state", "checkpoint_sha256", "charged_seconds", "lease", "attempts", "message", "block_reason"}:
                    raise ValueError("Unexpected job fields")
                spec = validate_spec(job["spec"])
                if job["id"] != spec_hash(spec) or job["spec_sha256"] != job["id"] or job["id"] in ids or job["state"] not in STATES or len(job["attempts"]) > 32:
                    raise ValueError("Invalid or repeated immutable job")
                ids.add(job["id"])
                if job["checkpoint_sha256"] is not None:
                    _sha(job["checkpoint_sha256"])
                costs = sum(_seconds(attempt["charged_seconds"]) for attempt in job["attempts"])
                for attempt in job["attempts"]:
                    if set(attempt) != {"token_sha256", "owner_id", "worker_id", "device_count", "reserved_seconds", "claimed_at", "started_at", "resume_checkpoint_sha256", "charged_seconds", "wall_seconds", "outcome", "finished_at"} or attempt["outcome"] not in {"completed", "failed", "paused"}:
                        raise ValueError("Unexpected attempt fields or outcome")
                    _sha(attempt["token_sha256"]); _identity(attempt["owner_id"])
                    if attempt["worker_id"] is not None:
                        _identity(attempt["worker_id"])
                    if not 0 < _seconds(attempt["reserved_seconds"]) <= JOB_SECONDS:
                        raise ValueError("Invalid attempt reservation")
                    if attempt["wall_seconds"] is not None and (type(attempt["device_count"]) is not int or not 1 <= attempt["device_count"] <= 8
                            or not math.isclose(attempt["charged_seconds"], _seconds(attempt["wall_seconds"]) * attempt["device_count"], rel_tol=1e-9, abs_tol=1e-6)):
                        raise ValueError("Measured GPU charge differs from allocated GPU count times wall duration")
                if not math.isclose(_seconds(job["charged_seconds"]), costs, abs_tol=1e-8):
                    raise ValueError("Job charges differ from its attempts")
                if job["state"] == "completed" and (job["checkpoint_sha256"] is None or not job["attempts"]
                        or job["attempts"][-1]["outcome"] != "completed" or job["charged_seconds"] > JOB_SECONDS
                        or job["attempts"][-1]["charged_seconds"] > job["attempts"][-1]["reserved_seconds"]):
                    raise ValueError("Completed jobs require a settled, in-budget checkpoint result")
                lease = job["lease"]
                if (job["state"] in ACTIVE) != (lease is not None):
                    raise ValueError("Active jobs require exactly one lease")
                if lease:
                    if spec["host"] in active or set(lease) != {"token_sha256", "owner_id", "worker_id", "device_count", "reserved_seconds", "claimed_at", "started_at", "resume_checkpoint_sha256"}:
                        raise ValueError("More than one active lease on a host")
                    active.add(spec["host"])
                    _sha(lease["token_sha256"]); _identity(lease["owner_id"])
                    if lease["worker_id"] is not None:
                        _identity(lease["worker_id"])
                    if lease["device_count"] is not None and (type(lease["device_count"]) is not int or not 1 <= lease["device_count"] <= 8):
                        raise ValueError("Invalid allocated GPU count")
                    if not 0 < _seconds(lease["reserved_seconds"]) <= JOB_SECONDS - job["charged_seconds"]:
                        raise ValueError("Reservation exceeds the remaining job allowance")
                    if lease["resume_checkpoint_sha256"] != job["checkpoint_sha256"]:
                        raise ValueError("Resume checkpoint identity changed")
            for host, ledger in value["hosts"].items():
                if set(ledger) != {"charged_seconds", "overheads"} or len(ledger["overheads"]) > 128:
                    raise ValueError("Invalid host ledger")
                total = sum(job["charged_seconds"] for job in value["jobs"] if job["spec"]["host"] == host)
                planned = [job["id"] for job in value["jobs"] if job["spec"]["host"] == host]
                if planned and planned != [spec_hash(job_spec(host, configuration, seed)) for configuration in CONFIGURATIONS for seed in SEEDS]:
                    raise ValueError("Host plan is incomplete or reordered")
                if any(set(entry) != {"seconds", "reason", "recorded_at"} or not isinstance(entry["reason"], str) or not 1 <= len(entry["reason"]) <= 200 for entry in ledger["overheads"]):
                    raise ValueError("Invalid overhead evidence")
                total += sum(_seconds(entry["seconds"]) for entry in ledger["overheads"])
                if not math.isclose(_seconds(ledger["charged_seconds"]), total, abs_tol=1e-8):
                    raise ValueError("Host charges differ from committed job and overhead costs")
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("Incomplete queue metadata") from error

    def _write(self, value):
        self._validate(value)
        content = _bytes(value)
        if len(content) > MAX_BYTES:
            raise ValueError("Queue metadata exceeds its size bound")
        descriptor, name = tempfile.mkstemp(prefix=".queue-", dir=self.root)
        path = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(path, self.path)
            if os.name != "nt":
                descriptor = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            path.unlink(missing_ok=True)

    def _change(self, action):
        with self._mutex():
            value = self._read()
            result = action(value)
            self._write(value)
            return deepcopy(result)

    def _mutex(self):
        if self.lock.is_symlink() or (self.lock.exists() and (not self.lock.is_file() or self.lock.stat().st_nlink != 1)):
            raise ValueError("Queue lock must be a regular unshared file")
        return FileLock(str(self.lock), timeout=0)

    def initialize(self):
        with self._mutex():
            if self.path.exists():
                return self._read()
            value = {"version": 1, "host_seconds": HOST_SECONDS,
                "hosts": {host: {"charged_seconds": 0., "overheads": []} for host in HOSTS}, "jobs": []}
            self._write(value)
            return value

    def snapshot(self):
        with self._mutex():
            return self._read()

    def plan(self, host):
        _host(host)
        def action(value):
            existing = {job["id"] for job in value["jobs"]}
            for configuration in CONFIGURATIONS:
                for seed in SEEDS:
                    spec = job_spec(host, configuration, seed)
                    identifier = spec_hash(spec)
                    if identifier not in existing:
                        value["jobs"].append({"id": identifier, "spec_sha256": identifier, "spec": spec,
                            "state": "queued", "checkpoint_sha256": None, "charged_seconds": 0.,
                            "lease": None, "attempts": [], "message": None, "block_reason": None})
            return [job for job in value["jobs"] if job["spec"]["host"] == host]
        return self._change(action)

    def claim(self, host, owner_id, *, dependencies_verified=()):
        _host(host); _identity(owner_id)
        def action(value):
            jobs = [job for job in value["jobs"] if job["spec"]["host"] == host]
            if any(job["state"] in ACTIVE for job in jobs):
                return None
            job = next((job for job in jobs if job["state"] in {"queued", "blocked"}), None)
            if job is None:
                return None
            if not set(job["spec"]["dependencies"]) <= set(dependencies_verified):
                job.update(state="blocked", block_reason="dependencies", message="External study, CPU report and cache-cleanup dependencies must be verified by the runner")
                return None
            allowance = JOB_SECONDS - job["charged_seconds"]
            if allowance <= 0 or len(job["attempts"]) >= 32:
                job.update(state="failed", block_reason="job_budget", message="The immutable job allowance is exhausted")
                return None
            if allowance > HOST_SECONDS - value["hosts"][host]["charged_seconds"]:
                for pending in jobs:
                    if pending["state"] in {"queued", "blocked"}:
                        pending.update(state="blocked", block_reason="host_budget", message="The six GPU-hour host budget cannot fund this reservation")
                return None
            token = uuid.uuid4().hex + uuid.uuid4().hex
            job.update(state="manual_start_required" if host in {"colab", "kaggle"} else "running", block_reason=None, message=None,
                lease={"token_sha256": hashlib.sha256(token.encode()).hexdigest(), "owner_id": owner_id,
                    "worker_id": None, "device_count": None, "reserved_seconds": allowance, "claimed_at": _now(), "started_at": None,
                    "resume_checkpoint_sha256": job["checkpoint_sha256"]})
            return {**job, "lease_token": token}
        return self._change(action)

    @staticmethod
    def _leased(value, identifier, token):
        _sha(identifier); _sha(token)
        job = next((job for job in value["jobs"] if job["id"] == identifier), None)
        if not job or job["state"] not in ACTIVE or job["lease"]["token_sha256"] != hashlib.sha256(token.encode()).hexdigest():
            raise ValueError("No matching active job lease")
        return job

    def start(self, job_id, lease_token, worker_id, device_count=1):
        _identity(worker_id)
        if type(device_count) is not int or not 1 <= device_count <= 8:
            raise ValueError("Allocated GPU count must be an integer from one to eight")
        def action(value):
            job = self._leased(value, job_id, lease_token)
            if job["lease"]["worker_id"] not in (None, worker_id) or job["lease"]["device_count"] not in (None, device_count):
                raise ValueError("This lease already belongs to another worker")
            job["lease"].update(worker_id=worker_id, device_count=device_count, started_at=job["lease"]["started_at"] or _now())
            job["state"] = "running"
            return job
        return self._change(action)

    @staticmethod
    def _settle(value, job, outcome, elapsed, checkpoint, message, wall_seconds=None):
        lease = job["lease"]
        overrun = elapsed > lease["reserved_seconds"]
        job["attempts"].append({**lease, "charged_seconds": elapsed, "wall_seconds": wall_seconds, "outcome": outcome, "finished_at": _now()})
        job["charged_seconds"] += elapsed
        value["hosts"][job["spec"]["host"]]["charged_seconds"] += elapsed
        job.update(lease=None, checkpoint_sha256=checkpoint or job["checkpoint_sha256"],
            state="queued" if outcome == "paused" and job["charged_seconds"] < JOB_SECONDS else "completed" if outcome == "completed" and not overrun else "failed",
            block_reason=None, message="Measured execution exceeded its reserved allowance" if overrun else message)
        return job

    def finish(self, job_id, lease_token, *, outcome, elapsed_seconds, checkpoint_sha256=None, message=None, wall_seconds=None):
        """Settle charged GPU seconds; a measured wall duration is mandatory for multiple GPUs."""
        elapsed = _seconds(elapsed_seconds)
        if outcome not in {"completed", "failed", "paused"} or (outcome != "failed" and checkpoint_sha256 is None):
            raise ValueError("Completion or pause requires a checkpoint; select a supported outcome")
        if checkpoint_sha256 is not None:
            _sha(checkpoint_sha256)
        if message is not None and (not isinstance(message, str) or len(message) > 600):
            raise ValueError("Result message is excessive")
        def action(value):
            job = self._leased(value, job_id, lease_token)
            if job["state"] != "running" or job["lease"]["started_at"] is None:
                raise ValueError("Interactive handoff must start before reporting execution")
            count = job["lease"]["device_count"]
            if wall_seconds is None and count != 1:
                raise ValueError("Measured wall seconds are required for multiple allocated GPUs")
            wall = elapsed if wall_seconds is None else _seconds(wall_seconds)
            if not math.isclose(elapsed, wall * count, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("Charged GPU seconds must equal wall seconds times allocated GPU count")
            return self._settle(value, job, outcome, elapsed, checkpoint_sha256, message, wall)
        return self._change(action)

    def recover(self, job_id, lease_token, *, checkpoint_sha256=None, verify_exit):
        """An authenticated runner must positively verify exit; timestamps are insufficient."""
        if checkpoint_sha256 is not None:
            _sha(checkpoint_sha256)
        def action(value):
            job = self._leased(value, job_id, lease_token)
            if verify_exit(deepcopy(job)) is not True:
                raise ValueError("Explicit worker-exit proof is required; stale heartbeats do not prove death")
            return self._settle(value, job, "failed", job["lease"]["reserved_seconds"], checkpoint_sha256,
                "Worker exit verified; unknown execution conservatively charged its full reservation")
        return self._change(action)

    def charge_overhead(self, host, seconds, reason):
        _host(host); seconds = _seconds(seconds)
        if not isinstance(reason, str) or not 1 <= len(reason) <= 200:
            raise ValueError("Overhead requires a bounded reason")
        def action(value):
            ledger = value["hosts"][host]
            reserved = sum(job["lease"]["reserved_seconds"] for job in value["jobs"] if job["spec"]["host"] == host and job["lease"])
            if seconds > HOST_SECONDS - ledger["charged_seconds"] - reserved:
                raise ValueError("Overhead exceeds the shared six-hour host allowance")
            ledger["overheads"].append({"seconds": seconds, "reason": reason, "recorded_at": _now()})
            ledger["charged_seconds"] += seconds
            return ledger
        return self._change(action)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init"); commands.add_parser("status")
    plan = commands.add_parser("plan"); plan.add_argument("host", choices=HOSTS)
    claim = commands.add_parser("claim"); claim.add_argument("host", choices=HOSTS); claim.add_argument("--owner", required=True)
    for name in ("start", "finish"):
        action = commands.add_parser(name); action.add_argument("job_id"); action.add_argument("--lease", required=True)
        if name == "start":
            action.add_argument("--worker", required=True)
            action.add_argument("--device-count", type=int, default=1)
        else:
            action.add_argument("--outcome", choices=("completed", "failed", "paused"), required=True)
            action.add_argument("--elapsed-seconds", type=float, required=True); action.add_argument("--checkpoint-sha256")
            action.add_argument("--wall-seconds", type=float)
    args = parser.parse_args(argv)
    queue = ComputeQueue(args.root)
    if args.command == "init": result = queue.initialize()
    elif args.command == "status": result = queue.snapshot()
    elif args.command == "plan": result = queue.plan(args.host)
    elif args.command == "claim": result = queue.claim(args.host, args.owner)
    elif args.command == "start": result = queue.start(args.job_id, args.lease, args.worker, device_count=args.device_count)
    else: result = queue.finish(args.job_id, args.lease, outcome=args.outcome, elapsed_seconds=args.elapsed_seconds, checkpoint_sha256=args.checkpoint_sha256, wall_seconds=args.wall_seconds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
