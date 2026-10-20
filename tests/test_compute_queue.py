"""Finite plans cannot reset ownership, completed work or aggregate host budgets."""
import importlib.util
import json
from pathlib import Path

from filelock import FileLock, Timeout
import pytest

from adp.compute_queue import (ComputeQueue, CONFIGURATIONS, HOST_SECONDS, HOSTS, JOB_SECONDS,
    LOCAL_DEPENDENCIES, PAYLOAD_SHA256, SEEDS, job_spec, main, spec_hash, validate_spec)


@pytest.fixture
def queue(tmp_path):
    result = ComputeQueue(tmp_path / "queue")
    result.initialize()
    return result


def claim_started(queue, host="private", dependencies=()):
    job = queue.claim(host, "controller-1", dependencies_verified=dependencies)
    assert job is not None
    queue.start(job["id"], job["lease_token"], "worker-123")
    return job


def test_plan_has_exact_twelve_immutable_new_development_jobs_and_is_idempotent(queue):
    for host in HOSTS:
        jobs = queue.plan(host)
        assert len(jobs) == 12
        assert [(job["spec"]["configuration"], job["spec"]["seed"]) for job in jobs] == [(configuration, seed) for configuration in CONFIGURATIONS for seed in SEEDS]
        assert all(job["id"] == job["spec_sha256"] == spec_hash(job["spec"]) for job in jobs)
        assert all(job["spec"]["epochs"] == 20 and job["spec"]["maximum_seconds"] == 1800 and job["spec"]["payload_sha256"] == PAYLOAD_SHA256 for job in jobs)
        assert queue.plan(host) == jobs
    assert len({job["id"] for job in queue.snapshot()["jobs"]}) == 48
    before = queue.path.read_bytes()
    queue.initialize()
    assert queue.path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("epochs", 1), ("seed", 99), ("configuration", "custom"), ("payload_sha256", "0"*64),
                                        ("train_count", 50000), ("selection_count", 2000), ("maximum_seconds", 21600), ("command", "arbitrary executable")])
def test_spec_rejects_unapproved_inputs_and_arbitrary_execution_fields(field, value):
    spec = job_spec("private", "full", 42)
    spec[field] = value
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_local_claim_waits_for_all_external_gates_without_modifying_studies(queue):
    queue.plan("local")
    assert queue.claim("local", "controller") is None
    assert queue.snapshot()["jobs"][0]["state"] == "blocked"
    assert queue.claim("local", "controller", dependencies_verified=LOCAL_DEPENDENCIES[:-1]) is None
    job = queue.claim("local", "controller", dependencies_verified=LOCAL_DEPENDENCIES)
    assert job["state"] == "running"
    assert job["spec"]["dependencies"] == list(LOCAL_DEPENDENCIES)
    assert list(queue.root.parent.iterdir()) == [queue.root]


def test_claim_is_sequential_per_host_and_reserves_before_start(queue):
    queue.plan("private"); queue.plan("local")
    job = queue.claim("private", "controller")
    assert job["lease"]["reserved_seconds"] == JOB_SECONDS
    assert queue.claim("private", "another-controller") is None
    assert "lease_token" not in queue.path.read_text()
    assert queue.snapshot()["hosts"]["private"]["charged_seconds"] == 0
    assert queue.claim("local", "local-controller", dependencies_verified=LOCAL_DEPENDENCIES) is not None
    with pytest.raises(ValueError, match="another worker"):
        queue.start(job["id"], job["lease_token"], "worker-1")
        queue.start(job["id"], job["lease_token"], "worker-2")


@pytest.mark.parametrize("host", ["colab", "kaggle"])
def test_cloud_requires_explicit_interactive_start_and_has_no_gpu_launcher(queue, host):
    queue.plan(host)
    job = queue.claim(host, "notebook-controller")
    assert job["state"] == "manual_start_required"
    with pytest.raises(ValueError, match="handoff"):
        queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=10, checkpoint_sha256="a"*64)
    started = queue.start(job["id"], job["lease_token"], "interactive-notebook-1")
    assert started["state"] == "running"
    done = queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=10, checkpoint_sha256="a"*64)
    assert done["state"] == "completed"


def test_finished_job_never_runs_again_and_measured_unused_reservation_is_refunded(queue):
    queue.plan("private")
    job = claim_started(queue)
    finished = queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=73, checkpoint_sha256="a"*64)
    assert finished["state"] == "completed"
    assert finished["charged_seconds"] == 73 and finished["lease"] is None
    before = queue.path.read_bytes()
    with pytest.raises(ValueError, match="lease"):
        queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=73, checkpoint_sha256="a"*64)
    assert queue.path.read_bytes() == before
    next_job = queue.claim("private", "controller-2")
    assert next_job["id"] != job["id"]
    assert queue.snapshot()["hosts"]["private"]["charged_seconds"] == 73


def test_all_twelve_jobs_end_the_plan_even_when_host_budget_remains(queue):
    queue.plan("private")
    completed = []
    for index in range(12):
        job = claim_started(queue)
        completed.append(job["id"])
        queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=100,
                     checkpoint_sha256=f"{index + 1:064x}")
    assert len(set(completed)) == 12
    assert queue.snapshot()["hosts"]["private"]["charged_seconds"] == 1200
    assert all(job["state"] == "completed" for job in queue.plan("private"))
    assert queue.claim("private", "another-controller") is None


def test_partial_resume_keeps_exact_identity_checkpoint_and_job_remaining_allowance(queue):
    queue.plan("private")
    original = claim_started(queue)
    queue.finish(original["id"], original["lease_token"], outcome="paused", elapsed_seconds=300, checkpoint_sha256="b"*64)
    resumed = queue.claim("private", "resume-controller")
    assert resumed["id"] == original["id"]
    assert resumed["lease_token"] != original["lease_token"]
    assert resumed["lease"]["reserved_seconds"] == 1500
    assert resumed["lease"]["resume_checkpoint_sha256"] == "b"*64
    with pytest.raises(ValueError):
        queue.start(original["id"], original["lease_token"], "stale-worker")


def test_unknown_crash_keeps_reservation_and_recovery_requires_positive_exit_proof(queue):
    queue.plan("private")
    job = claim_started(queue)
    assert queue.claim("private", "restart-controller") is None
    before = queue.path.read_bytes()
    with pytest.raises(ValueError, match="stale heartbeats"):
        queue.recover(job["id"], job["lease_token"], verify_exit=lambda _job: False)
    assert queue.path.read_bytes() == before
    recovered = queue.recover(job["id"], job["lease_token"], checkpoint_sha256="c"*64, verify_exit=lambda record: record["lease"]["worker_id"] == "worker-123")
    assert recovered["state"] == "failed" and recovered["charged_seconds"] == 1800
    assert recovered["checkpoint_sha256"] == "c"*64
    assert queue.snapshot()["hosts"]["private"]["charged_seconds"] == 1800


def test_overhead_and_active_reservations_share_six_hour_allowance(queue):
    queue.plan("private")
    queue.charge_overhead("private", HOST_SECONDS - JOB_SECONDS, "Previously measured qualification")
    job = claim_started(queue)
    before = queue.path.read_bytes()
    with pytest.raises(ValueError, match="shared six-hour"):
        queue.charge_overhead("private", 1, "Must not consume reserved job time")
    assert queue.path.read_bytes() == before
    queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=1800, checkpoint_sha256="a"*64)
    assert queue.claim("private", "controller-2") is None
    assert all(record["state"] == "blocked" for record in queue.snapshot()["jobs"][1:])


def test_actual_overrun_is_recorded_honestly_even_past_host_ceiling(queue):
    queue.plan("private")
    queue.charge_overhead("private", HOST_SECONDS - JOB_SECONDS, "Measured earlier overhead")
    job = claim_started(queue)
    done = queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=1801, checkpoint_sha256="d"*64)
    assert done["state"] == "failed" and "exceeded" in done["message"]
    assert queue.snapshot()["hosts"]["private"]["charged_seconds"] == HOST_SECONDS + 1
    assert queue.claim("private", "controller-2") is None


def test_two_allocated_gpus_charge_twice_wall_time_with_durable_evidence(queue):
    queue.plan("kaggle")
    job = queue.claim("kaggle", "notebook-controller")
    queue.start(job["id"], job["lease_token"], "two-t4-runtime", device_count=2)
    before = queue.path.read_bytes()
    with pytest.raises(ValueError, match="wall seconds"):
        queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=600, checkpoint_sha256="a"*64)
    with pytest.raises(ValueError, match="GPU seconds"):
        queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=300, wall_seconds=300, checkpoint_sha256="a"*64)
    assert queue.path.read_bytes() == before
    done = queue.finish(job["id"], job["lease_token"], outcome="completed", elapsed_seconds=600, wall_seconds=300, checkpoint_sha256="a"*64)
    assert done["charged_seconds"] == 600
    assert done["attempts"][0]["device_count"] == 2 and done["attempts"][0]["wall_seconds"] == 300
    assert done["spec"]["budget_unit"] == "allocated_gpu_seconds"


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -1, True])
def test_invalid_measured_costs_cannot_change_ledger(queue, seconds):
    queue.plan("private")
    job = claim_started(queue)
    before = queue.path.read_bytes()
    with pytest.raises(ValueError):
        queue.finish(job["id"], job["lease_token"], outcome="failed", elapsed_seconds=seconds)
    assert queue.path.read_bytes() == before


def test_os_lock_and_atomic_write_failure_preserve_existing_state(queue, monkeypatch):
    queue.plan("private")
    before = queue.path.read_bytes()
    with FileLock(str(queue.lock), timeout=0):
        with pytest.raises(Timeout):
            queue.claim("private", "racing-controller")
    monkeypatch.setattr("adp.compute_queue.os.replace", lambda *_args: (_ for _ in ()).throw(OSError("write interrupted")))
    with pytest.raises(OSError):
        queue.claim("private", "controller")
    assert queue.path.read_bytes() == before
    assert not list(queue.root.glob(".queue-*"))


@pytest.mark.parametrize("mutation", ["spec", "budget", "charge", "extra", "reorder", "fake_completion"])
def test_tampered_queue_fails_closed(queue, mutation):
    queue.plan("private")
    value = queue.snapshot()
    if mutation == "spec": value["jobs"][0]["spec"]["payload_sha256"] = "0"*64
    elif mutation == "budget": value["host_seconds"] = 999999
    elif mutation == "charge": value["hosts"]["private"]["charged_seconds"] = 100
    elif mutation == "extra": value["jobs"][0]["command"] = "arbitrary command"
    elif mutation == "fake_completion": value["jobs"][0]["state"] = "completed"
    else: value["jobs"].reverse()
    queue.path.write_text(json.dumps(value))
    before = queue.path.read_bytes()
    with pytest.raises(ValueError):
        queue.claim("private", "controller")
    assert queue.path.read_bytes() == before


def test_module_is_portable_without_other_adp_imports_and_cli_never_launches_compute(tmp_path, capsys):
    path = Path(__file__).resolve().parents[1] / "src/adp/compute_queue.py"
    spec = importlib.util.spec_from_file_location("standalone_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "portable"
    main(["--root", str(root), "init"])
    assert json.loads(capsys.readouterr().out)["host_seconds"] == 21600
    main(["--root", str(root), "plan", "colab"])
    assert len(json.loads(capsys.readouterr().out)) == 12
    main(["--root", str(root), "claim", "colab", "--owner", "notebook-controller"])
    assert json.loads(capsys.readouterr().out)["state"] == "manual_start_required"
    assert module.ComputeQueue(root).snapshot()["jobs"][0]["lease"]["worker_id"] is None
