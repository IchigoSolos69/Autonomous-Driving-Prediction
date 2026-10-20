"""The portable adapter owns only finite development jobs, never study workers."""
from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adp import compute_queue


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compute-queue-worker.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("compute_queue_worker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def args(tmp_path):
    return Namespace(root=tmp_path, host="private", payload=tmp_path / "inputs.zip",
        queue_module=tmp_path / "compute_queue.py", pilot_helper=tmp_path / "pilot-helper.py",
        control_source=tmp_path / "control", status=False, once=False)


def test_recipe_changes_only_approved_seed_and_three_model_fields(worker):
    initial = {"seed": 42, "model": {"width": 128, "use_map": True, "use_graph": True},
               "training": {"learning_rate": .0003, "checkpoint_selection": "joint_minFDE"}}
    for config in compute_queue.CONFIGURATIONS:
        result = worker.recipe(initial, compute_queue.job_spec("private", config, 43))
        assert result["seed"] == 43
        assert result["model"] == {"width": 128, "use_map": config != "no_map",
            "use_graph": config != "no_graph", "shared_world_ce_weight": .1 if config == "world_ce" else 0.}
        assert result["training"] == initial["training"]
    assert initial["seed"] == 42
    assert "shared_world_ce_weight" not in initial["model"]


def test_protocol_matches_verifier_finite_plan(worker):
    plan = worker.finite_plan(compute_queue, "kaggle")
    assert plan == {"version": 1, "host_seconds": 21600, "jobs": [compute_queue.job_spec("kaggle", c, s)
        for c in compute_queue.CONFIGURATIONS for s in compute_queue.SEEDS]}


def test_resources_validate_before_any_job_and_reject_tamper(worker, args, monkeypatch):
    runner = args.root / "worker.py"
    monkeypatch.setattr(worker, "__file__", str(runner))
    resources = {}
    for path in (runner, args.queue_module, args.pilot_helper):
        path.write_bytes(b"trusted controller source")
        resources[path.name] = worker.sha(path)
    args.payload.mkdir()  # Extracted Kaggle upload is checked by the pinned pilot helper.
    plan_hash = compute_queue.spec_hash(worker.finite_plan(compute_queue, args.host))
    write(args.root / "resources.json", {"files": resources, "payload_sha256": worker.PAYLOAD_SHA256,
                                         "plan_sha256": plan_hash})
    assert worker.verify_resources(args, compute_queue) == plan_hash
    args.queue_module.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum differs"):
        worker.verify_resources(args, compute_queue)
    assert not (args.root / "controller").exists()


@pytest.mark.parametrize("bad_name", ["../worker.py", "C:\\worker.py", "sub/worker.py"])
def test_resource_paths_cannot_escape_namespace(worker, args, bad_name):
    write(args.root / "resources.json", {"files": {bad_name: "a" * 64, "b": "b" * 64, "c": "c" * 64},
        "payload_sha256": worker.PAYLOAD_SHA256,
        "plan_sha256": compute_queue.spec_hash(worker.finite_plan(compute_queue, args.host))})
    with pytest.raises(ValueError, match="checksum entry"):
        worker.verify_resources(args, compute_queue)


def test_status_does_not_initialize_missing_queue_or_invoke_gpu(worker, args, monkeypatch):
    args.status = True
    monkeypatch.setattr(worker, "load_module", lambda *_: compute_queue)
    monkeypatch.setattr(worker, "run_child", Mock(side_effect=AssertionError("GPU must not initialize")))
    result = worker.run_worker(args)
    assert result["queue"] is None
    assert list(args.root.iterdir()) == []


@pytest.fixture
def fake_run(worker, args, monkeypatch):
    monkeypatch.setattr(worker, "verify_resources", lambda *_: "a" * 64)
    monkeypatch.setattr(worker, "check_space", lambda *_: None)
    monkeypatch.setattr(worker, "load_module", lambda path, _: compute_queue if path == args.queue_module else SimpleNamespace())
    calls = []
    settings = {"state": "completed", "device_count": 1, "qualification_error": None}

    def stage(root, helper, payload, spec, job_id):
        job = root / "jobs" / job_id
        (job / "run").mkdir(parents=True, exist_ok=True)
        write(job / "initial-config.json", {"seed": spec["seed"]})
        return job

    def child(argv, job, timeout, capture=False):
        calls.append((argv, timeout, capture))
        assert timeout > 0
        if capture:
            if settings["qualification_error"]:
                raise settings["qualification_error"]
            return json.dumps({"device_count": settings["device_count"], "gpu": ["qualified GPU"]}).encode()
        if "prepare" in argv:
            write(job / "job.json", {"protocol_sha256": "b" * 64})
            write(job / "config.json", {})
        else:
            (job / "run" / "last.pt").write_bytes(b"not loaded in this adapter test")
            write(job / "status.json", {"state": settings["state"], "result": {"completed_epochs": 20 if settings["state"] == "completed" else 1}})

    def export(args, module, job, record, count, plan_hash):
        return {"job_id": record["id"], "snapshot_id": "c" * 64, "archive_name": "c" * 64 + ".zip", "bytes": 100, "sha256": "d" * 64}

    monkeypatch.setattr(worker, "stage_job", stage)
    monkeypatch.setattr(worker, "run_child", child)
    monkeypatch.setattr(worker, "export_job", export)
    return calls, settings


def test_twelve_completed_jobs_advance_and_are_never_repeated(worker, args, fake_run):
    calls, _ = fake_run
    first = worker.run_worker(args)
    assert len(first["exports"]) == 12
    assert first["state"] == "completed" and first["completed_jobs"] == first["planned_jobs"] == 12
    assert first["pid"] == worker.os.getpid() and first["current_job_id"] is None
    assert all(job["state"] == "completed" for job in first["queue"]["jobs"])
    assert len(calls) == 36
    second = worker.run_worker(args)
    assert len(calls) == 36
    assert len(second["exports"]) == 12
    assert not (args.root / ".worker-lease.json").exists()


def test_once_qualifies_one_fresh_job_only(worker, args, fake_run):
    args.once = True
    result = worker.run_worker(args)
    assert len(result["exports"]) == 1
    assert [job["state"] for job in result["queue"]["jobs"]].count("completed") == 1
    assert len(fake_run[0]) == 3
    assert result["state"] == "waiting" and result["completed_jobs"] == 1


def test_paused_checkpoint_stops_without_unbounded_retry(worker, args, fake_run):
    fake_run[1]["state"] = "paused"
    result = worker.run_worker(args)
    assert result["queue"]["jobs"][0]["state"] == "queued"
    assert result["queue"]["jobs"][0]["checkpoint_sha256"]
    assert len(result["exports"]) == 1
    assert len(fake_run[0]) == 3
    assert result["state"] == "paused"


def test_worker_publishes_live_pid_and_current_job_before_each_child(worker, args, fake_run, monkeypatch):
    original = worker.run_child
    observed = []
    def child(*call_args, **kwargs):
        value = read(args.root / "worker-status.json")
        assert value["state"] == "running"
        assert value["pid"] == worker.os.getpid()
        assert len(value["current_job_id"]) == 64
        observed.append(value["current_job_id"])
        return original(*call_args, **kwargs)
    monkeypatch.setattr(worker, "run_child", child)
    worker.run_worker(args)
    assert len(set(observed)) == 12


def test_stop_flag_pauses_before_any_claim_without_cuda(worker, args, fake_run):
    write(args.root / "stop-after-current.json", {"version": 1, "stop": True})
    result = worker.run_worker(args)
    assert result["state"] == "paused"
    assert all(job["state"] == "queued" and job["lease"] is None for job in result["queue"]["jobs"])
    assert fake_run[0] == []


def test_stop_request_does_not_replace_active_status_or_initialize_gpu(worker, args, fake_run):
    args.stop_after_current = True
    write(args.root / "worker-status.json", {"state": "running", "pid": 123, "current_job_id": "a" * 64})
    result = worker.run_worker(args)
    assert result["state"] == "running" and result["pid"] == 123
    assert result["stop_after_current"] is True
    assert fake_run[0] == []
    assert not (args.root / "controller").exists()


def test_stop_after_current_preserves_completed_first_job_and_other_pending_jobs(worker, args, fake_run, monkeypatch):
    original = worker.run_child
    def child(*call_args, **kwargs):
        value = original(*call_args, **kwargs)
        if "run" in call_args[0]:
            write(args.root / "stop-after-current.json", {"version": 1, "stop": True})
        return value
    monkeypatch.setattr(worker, "run_child", child)
    result = worker.run_worker(args)
    assert result["state"] == "paused" and result["completed_jobs"] == 1
    assert len(fake_run[0]) == 3
    assert result["queue"]["jobs"][1]["state"] == "queued"


def test_resume_clears_only_valid_stop_flag_with_exclusive_worker_lock(worker, args, fake_run):
    args.resume_queue, args.once = True, True
    write(args.root / "stop-after-current.json", {"version": 1, "stop": True})
    result = worker.run_worker(args)
    assert result["completed_jobs"] == 1
    assert not (args.root / "stop-after-current.json").exists()


def test_corrupt_stop_flag_is_preserved_and_no_job_is_claimed(worker, args, fake_run):
    write(args.root / "stop-after-current.json", {"version": 1, "stop": False})
    with pytest.raises(ValueError, match="Invalid stop"):
        worker.run_worker(args)
    queue = compute_queue.ComputeQueue(args.root / "controller").snapshot()
    assert all(job["lease"] is None for job in queue["jobs"])
    assert fake_run[0] == []


def test_kaggle_two_allocated_gpus_half_wall_allowance_and_double_charge(worker, args, fake_run):
    args.host, args.once = "kaggle", True
    fake_run[1]["device_count"] = 2
    result = worker.run_worker(args)
    prepare = fake_run[0][1][0]
    assert float(prepare[prepare.index("--max-runtime-seconds") + 1]) == 900
    run = fake_run[0][2][0]
    assert 0 < float(run[run.index("--slice-seconds") + 1]) < 900
    attempt = result["queue"]["jobs"][0]["attempts"][0]
    assert attempt["device_count"] == 2
    assert attempt["charged_seconds"] == attempt["wall_seconds"] * 2


def test_failed_qualification_conservatively_charges_one_reservation(worker, args, fake_run):
    fake_run[1]["qualification_error"] = RuntimeError("CUDA unavailable")
    result = worker.run_worker(args)
    job = result["queue"]["jobs"][0]
    assert job["state"] == "failed"
    assert job["charged_seconds"] == 1800
    assert len(fake_run[0]) == 1


def test_unverified_child_exit_keeps_active_lease_and_blocks_restart(worker, args, fake_run):
    fake_run[1]["qualification_error"] = worker.ChildExitUnverified("unknown process-tree exit")
    result = worker.run_worker(args)
    assert result["queue"]["jobs"][0]["lease"] is not None
    assert (args.root / ".worker-lease.json").exists()
    with pytest.raises(RuntimeError, match="process-death recovery"):
        worker.run_worker(args)
    assert len(fake_run[0]) == 1


def test_local_gate_prevents_queue_initialization_and_cuda_import(worker, args, fake_run, monkeypatch):
    args.host = "local"
    monkeypatch.setattr(worker, "local_dependencies", Mock(side_effect=ValueError("Original study is not completed")))
    with pytest.raises(ValueError, match="Original study"):
        worker.run_worker(args)
    assert not (args.root / "controller").exists()
    assert fake_run[0] == []


def test_local_gates_require_actual_completed_cleanup_proof(worker, args, monkeypatch):
    study, archive = args.root / "study", args.root / "archive"
    write(study / "frozen_selection.json", {"checkpoint_sha256": "e" * 64})
    proof = {"version": 1, "verified": True}
    write(archive / ("e" * 64) / "cleanup-proof.json", proof)
    cleanup = SimpleNamespace(STUDY_ROOT=study, DIRECTION_ROOT=args.root / "direction", MODEL_ARCHIVE_ROOT=archive,
        plan_study_cleanup=Mock(return_value={"state": "blocked", "reason": "CPU report pending"}))
    monkeypatch.setattr(worker, "control_module", lambda *_: cleanup)
    with pytest.raises(ValueError, match="CPU report pending"):
        worker.local_dependencies(args.control_source)
    cleanup.plan_study_cleanup.return_value = {"state": "completed"}
    assert worker.local_dependencies(args.control_source) == compute_queue.LOCAL_DEPENDENCIES
    cleanup.plan_study_cleanup.assert_called_with(study, cleanup.DIRECTION_ROOT, proof)


def test_child_env_contains_only_frozen_job_source_and_cpu_thread_limits(worker, tmp_path):
    env = worker.child_environment(tmp_path)
    assert env["PYTHONPATH"] == str(tmp_path / "source")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "2"


def test_timeout_stops_owned_child_before_propagating(worker, tmp_path, monkeypatch):
    process = Mock()
    process.communicate.side_effect = worker.subprocess.TimeoutExpired(["python"], 1)
    monkeypatch.setattr(worker.subprocess, "Popen", Mock(return_value=process))
    stop = Mock()
    monkeypatch.setattr(worker, "stop_owned_child", stop)
    with pytest.raises(worker.subprocess.TimeoutExpired):
        worker.run_child(["python", "owned-job.py"], tmp_path, 1)
    stop.assert_called_once_with(process)


def test_failed_process_tree_kill_never_claims_verified_exit(worker, tmp_path, monkeypatch):
    process = Mock()
    process.communicate.side_effect = worker.subprocess.TimeoutExpired(["python"], 1)
    monkeypatch.setattr(worker.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(worker, "stop_owned_child", Mock(side_effect=OSError("process still alive")))
    with pytest.raises(worker.ChildExitUnverified):
        worker.run_child(["python", "owned-job.py"], tmp_path, 1)


def test_export_contract_flat_zip_and_bound_provenance(worker, args, monkeypatch):
    import zipfile
    job = args.root / "jobs" / ("f" * 64)
    run = job / "run"
    run.mkdir(parents=True)
    spec = compute_queue.job_spec("private", "no_map", 44)
    identifier = compute_queue.spec_hash(spec)
    protocol = {"protocol_sha256": "a" * 64, "config_sha256": "b" * 64,
                "cache_files": {"manifest.json": "c" * 64, "preprocessing.json": "d" * 64}}
    for name, value in (("job.json", protocol), ("config.json", {}), ("qualification.json", {"device_count": 1}),
                        ("status.json", {"state": "completed", "result": {"completed_epochs": 20}})):
        write(job / name, value)
    snapshot = args.root / "snapshots" / ("e" * 64)
    snapshot.mkdir(parents=True)
    payload = b"trusted checkpoint"
    (snapshot / "last.pt").write_bytes(payload)
    write(snapshot / "snapshot.json", {"files": {"last.pt": {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}})
    creator = Mock(return_value={"snapshot_id": "e" * 64, "path": str(snapshot)})
    monkeypatch.setattr(worker, "control_module", lambda *_: SimpleNamespace(create_snapshot=creator))
    monkeypatch.setattr(worker, "check_space", lambda *_: None)
    result = worker.export_job(args, compute_queue, job, {"id": identifier, "spec": spec}, 1, "f" * 64)
    assert set(result) == {"job_id", "snapshot_id", "archive_name", "bytes", "sha256"}
    assert result["archive_name"] == "e" * 64 + ".zip"
    assert creator.call_args.args[2] == identifier
    provenance = creator.call_args.args[3]
    assert set(provenance) == {"queue_job_spec", "queue_job_id", "payload_sha256", "queue_protocol_sha256",
        "allocated_device_count", "job_protocol_sha256", "config_sha256", "manifest_sha256", "preprocessing_sha256"}
    assert provenance["queue_job_spec"] == spec
    assert read(run / "result.json")["state"] == "completed"
    with zipfile.ZipFile(args.root / "exports" / result["archive_name"]) as archive:
        assert set(archive.namelist()) == {"snapshot.json", "last.pt"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
