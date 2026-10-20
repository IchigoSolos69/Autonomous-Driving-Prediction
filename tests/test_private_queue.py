"""Private dispatch/transfer boundaries, without SSH, GPU work or real cleanup."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from adp import compute_queue, private_queue
from adp.compute_queue import spec_hash
from adp.queue_results import queue_protocol_hash


def _sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def resources(tmp_path, monkeypatch):
    files = {}
    for name in ("worker.py", "compute_queue.py", "queue_cleanup.py", "pilot-helper.py", "inputs.zip"):
        files[name] = tmp_path / name
        files[name].write_bytes(("pinned " + name).encode())
    digest = _sha(files["inputs.zip"].read_bytes())
    monkeypatch.setattr(private_queue, "PAYLOAD", files["inputs.zip"])
    monkeypatch.setattr(private_queue, "PAYLOAD_SHA256", digest)
    monkeypatch.setattr(compute_queue, "PAYLOAD_SHA256", digest)
    monkeypatch.setattr(private_queue, "resource_paths", lambda: files)
    monkeypatch.setattr(private_queue, "LOCAL_ROOT", tmp_path / "local")
    return files


@pytest.fixture
def connection():
    return {"host": "aadi@192.168.29.71", "identityfile": "C:/keys/compute key",
        "remote_python": "/home/aadi/runtime/bin/python", "remote_source": "/home/aadi/interaction-lab/control-v1/src"}


def test_fixed_plan(resources, tmp_path):
    approved = tmp_path / "approved.json"
    first = private_queue.approve_plan(approved)
    assert private_queue.approve_plan(approved) == first
    plan = private_queue._plan(approved)
    assert plan["host_seconds"] == 21600
    assert all(len(jobs) == 12 and sum(job["maximum_seconds"] for job in jobs) == 21600
               for jobs in plan["host_plans"].values())
    assert all(job["epochs"] == 20 for jobs in plan["host_plans"].values() for job in jobs)
    resources["worker.py"].write_bytes(b"modified worker")
    with pytest.raises(ValueError, match="changed"):
        private_queue.approve_plan(approved)
    with pytest.raises(ValueError, match="approved hashes"):
        private_queue._plan(approved)


def _mark_shared(monkeypatch, path):
    original = Path.stat
    def stat(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        return os.stat_result((*value[:3], 2, *value[4:])) if self == path else value
    monkeypatch.setattr(Path, "stat", stat)


def test_plan_rejects_hard_linked_resources_before_any_dispatch(resources, monkeypatch):
    _mark_shared(monkeypatch, resources["worker.py"])
    with pytest.raises(ValueError, match="unshared"):
        private_queue.make_plan()


def test_stage_manifest_matches_actual_worker_finite_plan_contract(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    manifests = []
    def remote(_config, code, *args, stdin=None, **_):
        if code == private_queue.INSPECT:
            return {"exists": True, "files": plan["files"]}
        if stdin:
            manifests.append(json.loads(stdin))
        return {"verified": True}
    monkeypatch.setattr(private_queue, "_remote", remote)
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: pytest.fail("No matching resource should be uploaded again"))
    private_queue.stage(connection, plan)
    assert manifests[0]["plan_sha256"] == queue_protocol_hash("private")
    worker_path = private_queue.PROJECT / "scripts/compute-queue-worker.py"
    spec = importlib.util.spec_from_file_location("private_queue_worker_contract", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.__file__ = str(resources["worker.py"])
    worker.PAYLOAD_SHA256 = private_queue.PAYLOAD_SHA256
    (resources["worker.py"].parent / "resources.json").write_text(json.dumps(manifests[0]))
    args = SimpleNamespace(root=resources["worker.py"].parent, host="private", payload=resources["inputs.zip"],
        queue_module=resources["compute_queue.py"], pilot_helper=resources["pilot-helper.py"])
    assert worker.verify_resources(args, compute_queue) == queue_protocol_hash("private")


@pytest.mark.parametrize("observed", [{"process_alive": True}, {"worker_lock_held": True}, {"state": "completed"},
    {"state": "budget_exhausted"}, {"state": "blocked"}, {"state": "failed"}])
def test_owned_or_terminal_worker_is_not_relaunched(connection, monkeypatch, observed):
    monkeypatch.setattr(private_queue, "status", lambda _: observed)
    monkeypatch.setattr(private_queue, "stage", lambda *_: pytest.fail("Existing or terminal jobs must not be restaged"))
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: pytest.fail("Existing worker must not be relaunched"))
    assert private_queue.start(connection, {}) == observed


def test_launch_uses_new_namespace_and_cli_accepted_by_actual_worker(resources, connection, monkeypatch, tmp_path):
    calls = []
    root = tmp_path / "queue-new"
    root.mkdir()
    monkeypatch.setattr(private_queue, "REMOTE_ROOT", str(root))
    monkeypatch.setattr(private_queue, "status", lambda _: {"state": "not_started"})
    monkeypatch.setattr(private_queue, "stage", lambda *_: {"state": "staged"})
    def remote(_config, code, *args, **_):
        calls.append((code, args))
        return {"state": "starting", "pid": 123}
    monkeypatch.setattr(private_queue, "_remote", remote)
    private_queue.start(connection, private_queue.make_plan())
    code, args = calls[-1]
    commands = []
    monkeypatch.setattr(private_queue.subprocess, "Popen", lambda command, **kwargs: commands.append((command, kwargs)) or SimpleNamespace(pid=123))
    monkeypatch.setattr(sys, "argv", ["-c", *map(str, args)])
    with redirect_stdout(io.StringIO()):
        exec(code, {})
    command, options = commands[0]
    assert "server-pilot-v1" not in " ".join(command)
    assert command[1] == str(root / "worker.py")
    assert options["start_new_session"] and options["close_fds"]
    worker_path = private_queue.PROJECT / "scripts/compute-queue-worker.py"
    spec = importlib.util.spec_from_file_location("private_worker_cli", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    parsed = []
    monkeypatch.setattr(worker, "run_worker", lambda args: parsed.append(args) or {})
    monkeypatch.setattr(sys, "argv", command[1:])
    with redirect_stdout(io.StringIO()):
        worker.main()
    assert parsed[0].host == "private" and parsed[0].root == root
    assert parsed[0].control_source == Path(connection["remote_source"])


def test_ssh_is_noninteractive_pinned_and_quotes_remote_arguments(connection, monkeypatch):
    captured = []
    def run(command, **options):
        captured.append((command, options))
        return SimpleNamespace(returncode=0, stdout=b'{"ok":true}', stderr=b"")
    monkeypatch.setattr(private_queue.subprocess, "run", run)
    assert private_queue._remote(connection, "print('safe')", "a path;$(not shell)")["ok"]
    command, options = captured[0]
    for option in ("IdentitiesOnly=yes", "BatchMode=yes", "StrictHostKeyChecking=yes", "ConnectTimeout=15"):
        assert option in command
    assert shlex.split(command[-1]) == [connection["remote_python"], "-c", "print('safe')", "a path;$(not shell)"]
    assert options["timeout"] == 120 and options["capture_output"]


def test_scp_uses_same_strict_noninteractive_identity_options(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    uploaded = []
    def remote(_config, code, *args, **_):
        if code == private_queue.INSPECT:
            return {"exists": True, "files": {name: digest for name, digest in plan["files"].items() if name != "worker.py"}}
        return {"verified": True}
    monkeypatch.setattr(private_queue, "_remote", remote)
    monkeypatch.setattr(private_queue.subprocess, "run", lambda command, **options: uploaded.append((command, options)) or SimpleNamespace(returncode=0))
    private_queue.stage(connection, plan)
    command, options = uploaded[0]
    assert command[0] == "scp" and len(uploaded) == 1
    for value in ("IdentitiesOnly=yes", "BatchMode=yes", "StrictHostKeyChecking=yes", "ConnectTimeout=15", connection["identityfile"]):
        assert value in command
    assert command[-2] == str(resources["worker.py"])
    assert command[-1].startswith(connection["host"] + ":" + private_queue.REMOTE_ROOT + "/.incoming-")
    assert options["timeout"] == 900


@pytest.mark.parametrize("stdout", [b"[]", b"x" * (private_queue.MAX_STATUS + 1)], ids=["nonobject", "excessive"])
def test_nonobject_or_excessive_ssh_metadata_is_rejected(connection, monkeypatch, stdout):
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=stdout, stderr=b""))
    with pytest.raises(ValueError):
        private_queue._remote(connection, "safe")


def _export_fixture(plan):
    content = b"portable approved queue export"
    job_id = spec_hash(plan["host_plans"]["private"][0])
    snapshot_id = "a" * 64
    return content, {"job_id": job_id, "snapshot_id": snapshot_id, "archive_name": snapshot_id + ".zip",
                     "sha256": _sha(content), "bytes": len(content)}


@pytest.mark.parametrize("mutation", ["traversal", "unknown-job", "size", "boolean-size", "snapshot"])
def test_untrusted_export_identity_path_or_size_is_rejected_before_transfer(resources, connection, monkeypatch, mutation):
    plan = private_queue.make_plan()
    _, export = _export_fixture(plan)
    export.update({"traversal": {"archive_name": "../elsewhere.zip"}, "unknown-job": {"job_id": "b" * 64},
        "size": {"bytes": private_queue.MAX_TOTAL_BYTES + 1}, "boolean-size": {"bytes": True},
        "snapshot": {"snapshot_id": "../escape"}}[mutation])
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: pytest.fail("Unsafe export must not reach SSH"))
    with pytest.raises(ValueError):
        private_queue.sync(connection, plan)


def test_storage_reserve_includes_download_and_two_import_copies(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    _, export = _export_fixture(plan)
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    monkeypatch.setattr(private_queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=private_queue.MINIMUM_FREE_BYTES + 3 * export["bytes"] - 1))
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: pytest.fail("Insufficient storage must block before transfer"))
    assert private_queue.MINIMUM_FREE_BYTES == 10 * 1024**3
    with pytest.raises(OSError, match="10 GiB"):
        private_queue.sync(connection, plan)


def test_cleanup_is_never_called_when_strict_local_import_fails(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    content, export = _export_fixture(plan)
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir(parents=True)
    (downloads / export["archive_name"]).write_bytes(content)
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    def rejected(*_a, **_k):
        raise ValueError("strict CPU checkpoint failed")
    monkeypatch.setattr("adp.queue_results.import_queue_result", rejected)
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: pytest.fail("Remote copies must survive failed local import"))
    with pytest.raises(ValueError, match="strict CPU"):
        private_queue.sync(connection, plan)
    assert (downloads / export["archive_name"]).read_bytes() == content


def test_verified_receipts_are_delivered_before_explicit_remote_cleanup(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    content, export = _export_fixture(plan)
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir(parents=True)
    (downloads / export["archive_name"]).write_bytes(content)
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    calls = []
    imported = {"queue_receipt": {"verified": True, "job_id": export["job_id"]},
                "receipt": {"verified": True, "snapshot_id": export["snapshot_id"]}}
    def verified(*args, **kwargs):
        assert kwargs["job_spec"] == plan["host_plans"]["private"][0]
        assert kwargs["expected_archive_sha256"] == export["sha256"]
        calls.append("durable-local-import")
        return imported
    def cleanup(_config, code, *args, stdin=None, **_):
        assert calls == ["durable-local-import"]
        assert "'--execute'" in code
        assert args == (private_queue.REMOTE_ROOT + "/queue_cleanup.py", private_queue.REMOTE_ROOT, export["job_id"])
        assert json.loads(stdin) == {"queue_receipt": imported["queue_receipt"], "import_receipt": imported["receipt"], "export": export}
        calls.append("guarded-remote-cleanup")
        return {"state": "completed"}
    monkeypatch.setattr("adp.queue_results.import_queue_result", verified)
    monkeypatch.setattr(private_queue, "_remote", cleanup)
    assert private_queue.sync(connection, plan)["imported"][0]["cleanup"] == "completed"
    assert calls == ["durable-local-import", "guarded-remote-cleanup"]
    assert (downloads / export["archive_name"]).read_bytes() == content


def test_another_dispatcher_owner_blocks_before_status_or_dispatch(resources, connection, monkeypatch):
    from filelock import FileLock, Timeout
    private_queue.LOCAL_ROOT.mkdir(parents=True)
    lock = private_queue.LOCAL_ROOT / ".dispatch.lock"
    monkeypatch.setattr(private_queue, "_plan", lambda *_: pytest.fail("Another controller owns this cycle"))
    with FileLock(str(lock), timeout=0):
        with pytest.raises(Timeout):
            private_queue.cycle()


def test_preexisting_hardlinked_partial_cannot_overwrite_another_local_file(resources, connection, monkeypatch, tmp_path):
    plan = private_queue.make_plan()
    content, export = _export_fixture(plan)
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir(parents=True)
    victim = tmp_path / "preserved-model.pt"
    victim.write_bytes(b"preserve this model")
    partial = downloads / (export["archive_name"] + ".incoming")
    partial.write_bytes(b"existing shared partial")
    _mark_shared(monkeypatch, partial)
    original_open = Path.open
    def shared_open(self, *args, **kwargs):
        # Simulate a hardlink without requiring Windows hardlink privileges.
        return original_open(victim if self == partial else self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", shared_open)
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    monkeypatch.setattr(private_queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    def downloaded(_command, **options):
        options["stdout"].write(content)
        return SimpleNamespace(returncode=0, stderr=b"")
    monkeypatch.setattr(private_queue.subprocess, "run", downloaded)
    monkeypatch.setattr("adp.queue_results.import_queue_result", lambda *_a, **_k: {"queue_receipt": {"verified": True}, "receipt": {"verified": True}})
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: {"state": "protected"})
    try:
        private_queue.sync(connection, plan)
    except (ValueError, OSError):
        pass
    assert victim.read_bytes() == b"preserve this model"


def _status_fixture(tmp_path, monkeypatch, connection, rows=None):
    root = tmp_path / "private"
    (root / "controller").mkdir(parents=True)
    specifications = [compute_queue.job_spec("private", "full", seed) for seed in (42, 43, 44)]
    jobs = [{"id": spec_hash(spec), "spec": spec, "state": state}
            for spec, state in zip(specifications, ("completed", "running", "queued"))]
    ledger = root / "controller/queue.json"
    ledger.write_text(json.dumps({"jobs": jobs, "hosts": {"private": {"charged_seconds": 800}}}))
    # The ledger identifies the active run even during a worker-status publication transition.
    (root / "worker-status.json").write_text(json.dumps({"state": "running", "current_job_id": jobs[0]["id"]}))
    history = root / "jobs" / jobs[1]["id"] / "run/history.json"
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps({"epochs": rows if rows is not None else [{"epoch": i} for i in range(1, 8)]}))
    monkeypatch.setattr(private_queue, "REMOTE_ROOT", str(root))
    def remote(_config, code, *args, **_):
        monkeypatch.setattr(sys, "argv", ["-c", *map(str, args)])
        output = io.StringIO()
        with redirect_stdout(output):
            exec(code, {})
        return json.loads(output.getvalue())
    monkeypatch.setattr(private_queue, "_remote", remote)
    return root, ledger, jobs, history


def test_status_sums_committed_epochs_and_reports_active_configuration(connection, tmp_path, monkeypatch):
    _, _, jobs, _ = _status_fixture(tmp_path, monkeypatch, connection)
    observed = private_queue.status(connection)
    assert observed["completed_epochs"] == 27 and observed["planned_epochs"] == 60
    assert observed["completed_jobs"] == 1
    assert observed["current_run"] == {"job_id": jobs[1]["id"], "configuration": "full", "seed": 43,
        "completed_epochs": 7, "planned_epochs": 20, "progress_updated_at": observed["current_run"]["progress_updated_at"]}
    assert observed["current_run"]["progress_updated_at"] and observed["observed_at"]


@pytest.mark.parametrize("rows", ["not a list", [{"epoch": True}], [{"epoch": 2}],
    [{"epoch": 1}, {"epoch": 1}], [{"epoch": i} for i in range(1, 22)]],
    ids=["nonlist", "boolean", "gap", "duplicate", "excess"])
def test_status_rejects_malformed_epoch_history(connection, tmp_path, monkeypatch, rows):
    _status_fixture(tmp_path, monkeypatch, connection, rows)
    with pytest.raises(AssertionError, match="epoch"):
        private_queue.status(connection)


@pytest.mark.parametrize("mutation", ["traversal", "foreign", "shared", "symlink", "oversize", "nonobject"],
    ids=["traversal", "foreign", "shared", "symlink", "oversize", "nonobject"])
def test_status_rejects_unsafe_job_or_history_metadata(connection, tmp_path, monkeypatch, mutation):
    _, ledger, jobs, history = _status_fixture(tmp_path, monkeypatch, connection)
    if mutation in {"traversal", "foreign"}:
        jobs[1]["id"] = "../outside" if mutation == "traversal" else "a" * 64
        ledger.write_text(json.dumps({"jobs": jobs, "hosts": {"private": {"charged_seconds": 800}}}))
    elif mutation == "shared":
        _mark_shared(monkeypatch, history)
    elif mutation == "symlink":
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == history or original(self))
    else:
        history.write_bytes(b"x" * (2 * 1024**2) if mutation == "oversize" else b"[]")
    with pytest.raises(AssertionError):
        private_queue.status(connection)


def _cycle_fixture(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    private_queue.LOCAL_ROOT.mkdir()
    monkeypatch.setattr(private_queue, "_plan", lambda *_: plan)
    monkeypatch.setattr(private_queue, "_config", lambda *_: connection)
    monkeypatch.setattr(private_queue, "start", lambda *_: {})
    current_spec = plan["host_plans"]["private"][1]
    observed = {"state": "running", "process_alive": True, "completed_jobs": 1, "total_jobs": 12,
        "charged_seconds": 1000, "completed_epochs": 27, "observed_at": "2026-10-20T07:30:00+00:00",
        "current_run": {"job_id": spec_hash(current_spec), "configuration": "full", "seed": 43,
                        "completed_epochs": 7, "planned_epochs": 20}}
    monkeypatch.setattr(private_queue, "sync", lambda *_: {"observed": observed, "imported": []})
    published = []
    monkeypatch.setattr(private_queue, "_publish_host", lambda _path, _host, changes: published.append(changes))
    return plan, observed, published


def test_cycle_publishes_live_epochs_and_clears_inherited_pilot_transfer(resources, connection, monkeypatch):
    _, _, published = _cycle_fixture(resources, connection, monkeypatch)
    private_queue._cycle()
    assert published[0]["training"]["completed_epochs"] == 27
    assert published[0]["training"]["planned_epochs"] == 240
    assert "full, seed 43: 7/20 epochs committed" in published[0]["message"]
    assert published[0]["observed_at"] == "2026-10-20T07:30:00+00:00"
    assert published[0]["transfer"] == {"status": "waiting", "verified_at": None, "local_path": None}


@pytest.mark.parametrize("mutation", ["boolean", "excess", "current-excess", "wrong-seed", "mismatch"],
    ids=["boolean", "excess", "current-excess", "wrong-seed", "mismatch"])
def test_cycle_rejects_malformed_committed_progress(resources, connection, monkeypatch, mutation):
    _, observed, published = _cycle_fixture(resources, connection, monkeypatch)
    if mutation == "boolean":
        observed["completed_epochs"] = True
    elif mutation == "excess":
        observed["completed_epochs"] = 241
    elif mutation == "current-excess":
        observed["current_run"]["completed_epochs"] = 21
    elif mutation == "wrong-seed":
        observed["current_run"]["seed"] = 99
    else:
        observed["completed_epochs"] = 20
    with pytest.raises(ValueError, match="progress"):
        private_queue._cycle()
    assert not published


def _queue_receipt(plan):
    spec = plan["host_plans"]["private"][0]
    return {"version": 1, "verified": True, "job_id": spec_hash(spec), "job_spec": spec,
        "snapshot_id": "b" * 64, "payload_sha256": private_queue.PAYLOAD_SHA256,
        "queue_protocol_sha256": queue_protocol_hash("private"),
        "checkpoint_validation": "strict_cpu_observed_only_completed", "verified_at": "2026-10-20T07:25:00+00:00"}


def test_cycle_preserves_verified_queue_transfer_without_new_imports(resources, connection, monkeypatch):
    plan, _, published = _cycle_fixture(resources, connection, monkeypatch)
    receipt = _queue_receipt(plan)
    directory = private_queue.LOCAL_ROOT / "results/queue-receipts"
    directory.mkdir(parents=True)
    path = directory / (receipt["job_id"] + "-" + receipt["snapshot_id"] + ".json")
    path.write_text(json.dumps(receipt))
    private_queue._cycle()
    assert published[0]["transfer"] == {"status": "verified", "verified_at": receipt["verified_at"], "local_path": str(path)}


@pytest.mark.parametrize("mutation", ["pilot-proof", "foreign-job", "shared"], ids=["pilot", "foreign", "shared"])
def test_queue_transfer_rejects_unrelated_or_shared_receipts(resources, connection, monkeypatch, mutation):
    plan = private_queue.make_plan()
    receipt = _queue_receipt(plan)
    directory = private_queue.LOCAL_ROOT / "results/queue-receipts"
    directory.mkdir(parents=True)
    if mutation == "pilot-proof":
        receipt["checkpoint_validation"] = "cpu_weights_only"
    elif mutation == "foreign-job":
        receipt["job_id"] = "a" * 64
    path = directory / (receipt["job_id"] + "-" + receipt["snapshot_id"] + ".json")
    path.write_text(json.dumps(receipt))
    if mutation == "shared":
        _mark_shared(monkeypatch, path)
    with pytest.raises(ValueError):
        private_queue._queue_transfer(plan)


def _tool_fixture(resources, tmp_path, monkeypatch):
    project = tmp_path / "controller"
    source = project / "scripts/queue-cleanup-v2.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# separately qualified cleanup adapter\n")
    monkeypatch.setattr(private_queue, "PROJECT", project)
    private_queue.LOCAL_ROOT.mkdir()
    receipt = {"version": 1, "adapter_sha256": _sha(source.read_bytes()),
               "base_sha256": _sha(resources["queue_cleanup.py"].read_bytes())}
    approval = private_queue.LOCAL_ROOT / "cleanup-tool-v2.json"
    approval.write_text(json.dumps(receipt))
    return private_queue.make_plan(), source, receipt, approval


def test_cleanup_adapter_requires_separate_receipt_before_any_network(connection, resources, monkeypatch):
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: pytest.fail("No tool approval exists"))
    assert private_queue._cleanup_tool(connection, private_queue.make_plan()) == private_queue.REMOTE_ROOT + "/queue_cleanup.py"


@pytest.mark.parametrize("mutation", ["base", "adapter", "version", "shared-receipt", "shared-source"],
    ids=["base", "adapter", "version", "shared-receipt", "shared-source"])
def test_cleanup_adapter_rejects_changed_or_shared_approvals_before_upload(resources, connection, tmp_path, monkeypatch, mutation):
    plan, source, receipt, approval = _tool_fixture(resources, tmp_path, monkeypatch)
    if mutation in {"base", "adapter"}:
        receipt[mutation + "_sha256"] = "a" * 64
    elif mutation == "version":
        receipt["version"] = True
    elif mutation == "shared-receipt":
        _mark_shared(monkeypatch, approval)
    else:
        _mark_shared(monkeypatch, source)
    approval.write_text(json.dumps(receipt))
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: pytest.fail("Invalid approval must not reach the server"))
    with pytest.raises(ValueError):
        private_queue._cleanup_tool(connection, plan)


def _tool_remote(monkeypatch, receipt, *, existing=None, base=None):
    files = {"queue_cleanup.py": base or receipt["base_sha256"]}
    name = "queue-cleanup-v2-" + receipt["adapter_sha256"] + ".py"
    if existing is not None:
        files[name] = existing
    calls = []
    def remote(_config, code, *args, **_):
        calls.append((code, args))
        if code == private_queue.INSPECT:
            return {"exists": True, "files": dict(files)}
        files[name] = receipt["adapter_sha256"]
        return {"verified": True}
    monkeypatch.setattr(private_queue, "_remote", remote)
    return name, files, calls


def test_cleanup_adapter_stages_immutable_name_and_rechecks_base_and_adapter(resources, connection, tmp_path, monkeypatch):
    plan, source, receipt, _ = _tool_fixture(resources, tmp_path, monkeypatch)
    name, _, calls = _tool_remote(monkeypatch, receipt)
    uploaded = []
    monkeypatch.setattr(private_queue.subprocess, "run", lambda command, **options: uploaded.append(command) or SimpleNamespace(returncode=0))
    assert private_queue._cleanup_tool(connection, plan) == private_queue.REMOTE_ROOT + "/" + name
    assert len(uploaded) == 1 and uploaded[0][0] == "scp" and uploaded[0][-2] == str(source)
    assert "StrictHostKeyChecking=yes" in uploaded[0]
    assert len(calls) == 3 and calls[0][0] == calls[-1][0] == private_queue.INSPECT
    assert calls[1][1][1:] == (private_queue.REMOTE_ROOT + "/" + name,
        private_queue.REMOTE_ROOT + "/queue_cleanup.py", receipt["adapter_sha256"], receipt["base_sha256"])
    assert json.loads(calls[-1][1][1]) == ["queue_cleanup.py", name]


def test_matching_cleanup_adapter_is_reused_without_rewriting(resources, connection, tmp_path, monkeypatch):
    plan, _, receipt, _ = _tool_fixture(resources, tmp_path, monkeypatch)
    name, _, calls = _tool_remote(monkeypatch, receipt, existing=receipt["adapter_sha256"])
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: pytest.fail("Qualified adapter must not be uploaded again"))
    assert private_queue._cleanup_tool(connection, plan).endswith(name)
    assert len(calls) == 2 and all(code == private_queue.INSPECT for code, _ in calls)


@pytest.mark.parametrize("mismatch", ["base", "adapter"], ids=["base", "adapter"])
def test_remote_cleanup_mismatch_is_not_overwritten(resources, connection, tmp_path, monkeypatch, mismatch):
    plan, _, receipt, _ = _tool_fixture(resources, tmp_path, monkeypatch)
    _tool_remote(monkeypatch, receipt, base="b" * 64 if mismatch == "base" else None,
        existing="b" * 64 if mismatch == "adapter" else None)
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: pytest.fail("Existing mismatched resources must remain untouched"))
    with pytest.raises(ValueError):
        private_queue._cleanup_tool(connection, plan)


@pytest.mark.parametrize("mismatch", ["queue_cleanup.py", "adapter"], ids=["base", "adapter"])
def test_cleanup_tool_readback_must_match_both_approved_hashes(resources, connection, tmp_path, monkeypatch, mismatch):
    plan, _, receipt, _ = _tool_fixture(resources, tmp_path, monkeypatch)
    name = "queue-cleanup-v2-" + receipt["adapter_sha256"] + ".py"
    inspections = []
    def remote(_config, code, *args, **_):
        assert code == private_queue.INSPECT
        inspections.append(args)
        files = {"queue_cleanup.py": receipt["base_sha256"], name: receipt["adapter_sha256"]}
        if len(inspections) == 2:
            files[name if mismatch == "adapter" else mismatch] = "c" * 64
        return {"exists": True, "files": files}
    monkeypatch.setattr(private_queue, "_remote", remote)
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: pytest.fail("Matching existing tool needs no upload"))
    with pytest.raises(ValueError, match="readback"):
        private_queue._cleanup_tool(connection, plan)


def test_cleanup_adapter_publish_cannot_replace_existing_destination(resources, connection, tmp_path, monkeypatch):
    plan, _, receipt, _ = _tool_fixture(resources, tmp_path, monkeypatch)
    _, _, calls = _tool_remote(monkeypatch, receipt)
    monkeypatch.setattr(private_queue.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    private_queue._cleanup_tool(connection, plan)
    code = calls[1][0]
    staged, target, base = [tmp_path / name for name in ("incoming.py", "immutable.py", "queue_cleanup.py")]
    staged.write_bytes(b"new adapter")
    target.write_bytes(b"preserve existing adapter")
    base.write_bytes(b"pinned base")
    # The remote publisher runs on Linux; Windows rejects fsync on a read-only descriptor.
    monkeypatch.setattr(os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(sys, "argv", ["-c", str(staged), str(target), str(base),
        _sha(staged.read_bytes()), _sha(base.read_bytes())])
    with pytest.raises(FileExistsError):
        exec(code, {})
    assert target.read_bytes() == b"preserve existing adapter" and staged.exists()


def test_cleanup_guard_failure_preserves_verified_import_and_reports_blocked(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    content, export = _export_fixture(plan)
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir(parents=True)
    (downloads / export["archive_name"]).write_bytes(content)
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    calls = []
    def verified(*_a, **_k):
        calls.append("strict-local-import")
        return {"queue_receipt": {"verified": True}, "receipt": {"verified": True}}
    def tool(*_):
        assert calls == ["strict-local-import"]
        calls.append("tool-approval")
        return private_queue.REMOTE_ROOT + "/approved-v2.py"
    def cleanup(*_a, **_k):
        assert calls == ["strict-local-import", "tool-approval"]
        calls.append("guard-refusal")
        raise RuntimeError("source manifest exceeds the old guard")
    monkeypatch.setattr("adp.queue_results.import_queue_result", verified)
    monkeypatch.setattr(private_queue, "_cleanup_tool", tool)
    monkeypatch.setattr(private_queue, "_remote", cleanup)
    imported = private_queue.sync(connection, plan)["imported"]
    assert imported[0]["cleanup"] == "blocked" and "source manifest" in imported[0]["cleanup_message"]
    assert calls == ["strict-local-import", "tool-approval", "guard-refusal"]
    assert (downloads / export["archive_name"]).read_bytes() == content
    assert json.loads((private_queue.LOCAL_ROOT / ("cleanup-" + export["job_id"] + ".json")).read_bytes())["state"] == "blocked"


def test_verified_adapter_is_invoked_only_after_durable_import(resources, connection, monkeypatch):
    plan = private_queue.make_plan()
    content, export = _export_fixture(plan)
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir(parents=True)
    (downloads / export["archive_name"]).write_bytes(content)
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [export]})
    events = []
    def imported(*_a, **_k):
        events.append("durable-import")
        return {"queue_receipt": {"verified": True}, "receipt": {"verified": True}}
    tool_path = private_queue.REMOTE_ROOT + "/queue-cleanup-v2-" + "d" * 64 + ".py"
    def tool(*_):
        assert events == ["durable-import"]
        events.append("verified-tool")
        return tool_path
    def execute(_config, code, *args, stdin=None, **_):
        assert events == ["durable-import", "verified-tool"]
        assert args == (tool_path, private_queue.REMOTE_ROOT, export["job_id"])
        assert json.loads(stdin)["queue_receipt"]["verified"] is True and "'--execute'" in code
        events.append("cleanup")
        return {"state": "completed"}
    monkeypatch.setattr("adp.queue_results.import_queue_result", imported)
    monkeypatch.setattr(private_queue, "_cleanup_tool", tool)
    monkeypatch.setattr(private_queue, "_remote", execute)
    assert private_queue.sync(connection, plan)["imported"][0]["cleanup"] == "completed"
    assert events == ["durable-import", "verified-tool", "cleanup"]


def test_blocked_cleanup_does_not_hide_live_worker_or_verified_transfer(resources, connection, monkeypatch):
    plan, observed, published = _cycle_fixture(resources, connection, monkeypatch)
    receipt = _queue_receipt(plan)
    directory = private_queue.LOCAL_ROOT / "results/queue-receipts"
    directory.mkdir(parents=True)
    (directory / (receipt["job_id"] + "-" + receipt["snapshot_id"] + ".json")).write_text(json.dumps(receipt))
    monkeypatch.setattr(private_queue, "sync", lambda *_: {"observed": observed, "imported": [
        {"job_id": receipt["job_id"], "snapshot_id": receipt["snapshot_id"], "cleanup": "blocked", "cleanup_message": "Guard refused unexpected cache bytes."}]})
    result = private_queue._cycle()
    assert result["state"] == published[0]["status"] == "running"
    assert published[0]["transfer"]["status"] == "verified"
    assert published[0]["cleanup"]["status"] == "blocked" and "Guard refused" in published[0]["cleanup"]["message"]


def _cached_delivery_fixture(resources):
    from adp.remote_transfer import _json_bytes
    plan = private_queue.make_plan()
    specification = plan["host_plans"]["private"][0]
    job_id = spec_hash(specification)
    files = {"last.pt": b"previously strictly loaded final model", "best.pt": b"previously strictly loaded best model",
             "job.json": b"{}", "config.json": b"{}", "result.json": b'{"state":"completed"}'}
    index = {name: {"size_bytes": len(data), "sha256": _sha(data)} for name, data in files.items()}
    core = {"format_version": 1, "run_id": job_id, "provenance": {"queue_job_spec": specification,
            "payload_sha256": private_queue.PAYLOAD_SHA256, "job_protocol_sha256": "e" * 64}, "files": index}
    snapshot_id = _sha(_json_bytes(core))
    directory = private_queue.LOCAL_ROOT / "results" / snapshot_id
    directory.mkdir(parents=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    manifest = _json_bytes({**core, "snapshot_id": snapshot_id})
    (directory / "snapshot.json").write_bytes(manifest)
    archive = b"authenticated, already imported ZIP bytes"
    item = {"job_id": job_id, "snapshot_id": snapshot_id, "archive_name": snapshot_id + ".zip",
            "sha256": _sha(archive), "bytes": len(archive)}
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir()
    (downloads / item["archive_name"]).write_bytes(archive)
    receipt = {"format_version": 1, "verified": True, "checkpoint_validation": "cpu_weights_only",
               "snapshot_id": snapshot_id, "manifest_sha256": _sha(manifest), "files": index,
               "verified_at": "2026-10-20T07:25:00+00:00", "local_directory": str(directory)}
    queue_receipt = {**_queue_receipt(plan), "snapshot_id": snapshot_id, "archive_sha256": item["sha256"],
        "manifest_sha256": _sha(manifest), "files": index, "local_directory": str(directory), "allocated_device_count": 1}
    receipt_path = directory.parent / "receipts" / (snapshot_id + ".json")
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps(receipt))
    queue_path = directory.parent / "queue-receipts" / (job_id + "-" + snapshot_id + ".json")
    queue_path.parent.mkdir()
    queue_path.write_text(json.dumps(queue_receipt))
    envelope = {"queue_receipt": queue_receipt, "import_receipt": receipt, "export": item}
    journal = {"version": 1, "state": "completed", "context": {"job_id": job_id, "snapshot_id": snapshot_id,
        "envelope_sha256": spec_hash(envelope), "job_protocol_sha256": "e" * 64,
        "status_sha256": spec_hash(json.loads(files["result.json"]))}, "files": []}
    journal["receipt_sha256"] = spec_hash(journal)
    journal_path = private_queue.LOCAL_ROOT / ("cleanup-" + job_id + ".json")
    journal_path.write_text(json.dumps(journal))
    return plan, item, directory, queue_path, receipt_path, journal_path, envelope


def test_unchanged_strict_results_skip_cpu_reloading_and_preserve_cleanup_envelope(resources, connection, monkeypatch):
    plan, item, _, queue_path, receipt_path, _, envelope = _cached_delivery_fixture(resources)
    previous_receipts = queue_path.read_bytes(), receipt_path.read_bytes()
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [item]})
    monkeypatch.setattr("adp.queue_results.import_queue_result", lambda *_a, **_k: pytest.fail("Unchanged strict proof must not load Torch or re-extract the payload"))
    monkeypatch.setattr(private_queue, "_cleanup_tool", lambda *_: "approved cleanup tool")
    calls = []
    def guard(_config, code, *args, stdin=None, **_):
        assert json.loads(stdin) == envelope
        calls.append("idempotent remote guard")
        return json.loads((private_queue.LOCAL_ROOT / ("cleanup-" + item["job_id"] + ".json")).read_bytes())
    monkeypatch.setattr(private_queue, "_remote", guard)
    assert private_queue.sync(connection, plan)["imported"][0]["cleanup"] == "completed"
    assert calls == ["idempotent remote guard"]
    assert (queue_path.read_bytes(), receipt_path.read_bytes()) == previous_receipts


@pytest.mark.parametrize("mutation", ["weights", "missing", "queue-proof", "import-proof", "journal"],
    ids=["weights", "missing", "queue-proof", "import-proof", "journal"])
def test_cached_result_corruption_blocks_before_cpu_repeat_or_cleanup(resources, connection, monkeypatch, mutation):
    plan, item, directory, queue_path, receipt_path, journal_path, _ = _cached_delivery_fixture(resources)
    if mutation == "weights":
        (directory / "last.pt").write_bytes(b"changed weights")
    elif mutation == "missing":
        (directory / "best.pt").unlink()
    else:
        path = {"queue-proof": queue_path, "import-proof": receipt_path, "journal": journal_path}[mutation]
        value = json.loads(path.read_bytes())
        value["receipt_sha256" if mutation == "journal" else "verified"] = "bad" if mutation == "journal" else False
        path.write_text(json.dumps(value))
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [item]})
    monkeypatch.setattr("adp.queue_results.import_queue_result", lambda *_a, **_k: pytest.fail("Corrupt proof must remain available for inspection"))
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: pytest.fail("Remote copies must remain protected"))
    with pytest.raises((ValueError, OSError)):
        private_queue.sync(connection, plan)


def test_pending_cleanup_reuses_same_verified_receipts_for_safe_resume(resources, connection, monkeypatch):
    plan, item, _, _, receipt_path, journal_path, envelope = _cached_delivery_fixture(resources)
    journal_path.write_text(json.dumps({"state": "blocked", "message": "Interrupted cleanup; remote journal is resumable."}))
    original_receipt = receipt_path.read_bytes()
    monkeypatch.setattr(private_queue, "status", lambda _: {"exports": [item]})
    monkeypatch.setattr("adp.queue_results.import_queue_result", lambda *_a, **_k: pytest.fail("A second import would change the pinned envelope timestamp"))
    monkeypatch.setattr(private_queue, "_cleanup_tool", lambda *_: "approved cleanup tool")
    def resume(_config, code, *args, stdin=None, **_):
        assert json.loads(stdin) == envelope
        return {"state": "blocked", "message": "Remote guard remains authoritative."}
    monkeypatch.setattr(private_queue, "_remote", resume)
    assert private_queue.sync(connection, plan)["imported"][0]["cleanup"] == "blocked"
    assert receipt_path.read_bytes() == original_receipt


def _terminal_fixture(resources, monkeypatch):
    plan = private_queue.make_plan()
    specifications = plan["host_plans"]["private"]
    downloads = private_queue.LOCAL_ROOT / "downloads"
    downloads.mkdir(parents=True)
    exports = []
    delivered = {}
    for index, specification in enumerate(specifications):
        job_id, snapshot_id = spec_hash(specification), _sha(str(index).encode())
        content = ("verified archive " + str(index)).encode()
        item = {"job_id": job_id, "snapshot_id": snapshot_id, "archive_name": snapshot_id + ".zip",
                "sha256": _sha(content), "bytes": len(content)}
        exports.append(item)
        (downloads / item["archive_name"]).write_bytes(content)
        (private_queue.LOCAL_ROOT / ("cleanup-" + job_id + ".json")).write_text(json.dumps({"state": "completed"}))
        delivered[job_id] = {"queue_receipt": {"job_id": job_id, "verified": True}, "receipt": {"snapshot_id": snapshot_id, "verified": True}}
    def previous(item, specification):
        assert spec_hash(specification) == item["job_id"]
        return delivered[item["job_id"]]
    monkeypatch.setattr(private_queue, "_previous_delivery", previous)
    observed = {"state": "completed", "completed_jobs": 12, "process_alive": False, "worker_lock_held": False, "exports": exports}
    return plan, observed


def test_terminal_envelope_binds_all_twelve_canonical_proofs_and_cleanup_journal_hashes(resources, monkeypatch):
    plan, observed = _terminal_fixture(resources, monkeypatch)
    envelope = private_queue._retirement_envelope(plan, observed)
    assert envelope["version"] == 1 and envelope["approved_plan_sha256"] == spec_hash(plan)
    assert envelope["files"] == plan["files"] and envelope["payload_sha256"] == private_queue.PAYLOAD_SHA256
    assert [item["export"] for item in envelope["deliveries"]] == observed["exports"]
    assert all(item["cleanup_receipt_sha256"] == _sha((private_queue.LOCAL_ROOT /
        ("cleanup-" + item["export"]["job_id"] + ".json")).read_bytes()) for item in envelope["deliveries"])


@pytest.mark.parametrize("mutation", ["worker", "lease", "incomplete", "duplicate", "archive", "cleanup"],
    ids=["worker", "lease", "incomplete", "duplicate", "archive", "cleanup"])
def test_terminal_envelope_refuses_incomplete_changed_or_unowned_evidence(resources, monkeypatch, mutation):
    plan, observed = _terminal_fixture(resources, monkeypatch)
    if mutation == "worker":
        observed["process_alive"] = True
    elif mutation == "lease":
        observed["worker_lock_held"] = True
    elif mutation == "incomplete":
        observed["completed_jobs"] = 11
    elif mutation == "duplicate":
        observed["exports"][-1] = observed["exports"][0]
    elif mutation == "archive":
        (private_queue.LOCAL_ROOT / "downloads" / observed["exports"][0]["archive_name"]).write_bytes(b"changed")
    else:
        (private_queue.LOCAL_ROOT / ("cleanup-" + observed["exports"][0]["job_id"] + ".json")).write_text('{"state":"in_progress"}')
    with pytest.raises(ValueError):
        private_queue._retirement_envelope(plan, observed)


def _retirement_summary(envelope, state="completed"):
    return {"version": 1, "tool": "queue-retire-v1", "state": state,
            "envelope_sha256": spec_hash(envelope), "receipt_sha256": "a" * 64,
            "journal_file_sha256": "b" * 64, "target_count": 1101, "remaining_files": 0,
            "bytes_retired": 890250827}


def test_explicit_retirement_persists_intent_before_execute_then_reuses_inspected_completion(resources, connection, monkeypatch):
    plan, observed = _terminal_fixture(resources, monkeypatch)
    envelope = private_queue._retirement_envelope(plan, observed)
    monkeypatch.setattr(private_queue, "_retirement_tool", lambda *_: "separately approved immutable tool")
    calls = []
    def execute(_config, tool, action, supplied=None):
        calls.append(action)
        intent = json.loads((private_queue.LOCAL_ROOT / "retirement-envelope-v1.json").read_bytes())
        assert intent == envelope
        if action == "--execute":
            assert supplied == envelope
            return _retirement_summary(envelope)
        return {"version": 1, "tool": "queue-retire-v1", "state": "not_started"} if len(calls) == 1 else _retirement_summary(envelope)
    monkeypatch.setattr(private_queue, "_retirement_call", execute)
    result = private_queue.retire_completed(connection, plan, observed)
    assert result["retirement"]["state"] == "completed" and len(result["imported"]) == 12
    assert calls == ["--inspect", "--execute", "--inspect"]
    assert json.loads((private_queue.LOCAL_ROOT / "retirement-receipt-v1.json").read_bytes()) == _retirement_summary(envelope)
    calls.clear()
    monkeypatch.setattr(private_queue, "_retirement_call", lambda *_a, **_k: calls.append("--inspect") or _retirement_summary(envelope))
    private_queue.retire_completed(connection, plan, observed)
    assert calls == ["--inspect"]


def test_authorized_interrupted_retirement_resumes_exact_original_envelope(resources, connection, monkeypatch):
    plan, observed = _terminal_fixture(resources, monkeypatch)
    envelope = private_queue._retirement_envelope(plan, observed)
    (private_queue.LOCAL_ROOT / "retirement-envelope-v1.json").write_text(json.dumps(envelope))
    monkeypatch.setattr(private_queue, "_retirement_tool", lambda *_: "approved tool")
    calls = []
    def resume(_config, tool, action, supplied=None):
        calls.append(action)
        if action == "--execute":
            assert supplied == envelope
        return _retirement_summary(envelope, "in_progress" if len(calls) == 1 else "completed")
    monkeypatch.setattr(private_queue, "_retirement_call", resume)
    assert private_queue.retire_completed(connection, plan, observed)["retirement"]["state"] == "completed"
    assert calls == ["--inspect", "--execute", "--inspect"]


@pytest.mark.parametrize("mutation", ["envelope", "remaining", "count", "version"], ids=["envelope", "remaining", "count", "version"])
def test_changed_terminal_inspection_never_claims_verified_retirement(resources, connection, monkeypatch, mutation):
    plan, observed = _terminal_fixture(resources, monkeypatch)
    envelope = private_queue._retirement_envelope(plan, observed)
    monkeypatch.setattr(private_queue, "_retirement_tool", lambda *_: "approved tool")
    summary = _retirement_summary(envelope)
    summary.update({"envelope": {"envelope_sha256": "c" * 64}, "remaining": {"remaining_files": 1},
                    "count": {"target_count": 1}, "version": {"version": True}}[mutation])
    monkeypatch.setattr(private_queue, "_retirement_call", lambda *_a, **_k: summary)
    with pytest.raises(ValueError):
        private_queue.retire_completed(connection, plan, observed)
    assert not (private_queue.LOCAL_ROOT / "retirement-receipt-v1.json").exists()


def test_saved_retirement_intent_bypasses_old_snapshot_dependent_cleanup(resources, connection, monkeypatch):
    plan, observed = _terminal_fixture(resources, monkeypatch)
    (private_queue.LOCAL_ROOT / "retirement-envelope-v1.json").write_text('{"intent":"already approved"}')
    monkeypatch.setattr(private_queue, "status", lambda _: observed)
    expected = {"observed": observed, "imported": [], "retirement": {"state": "completed"}}
    monkeypatch.setattr(private_queue, "retire_completed", lambda *_: expected)
    monkeypatch.setattr(private_queue, "_cleanup_tool", lambda *_: pytest.fail("Old snapshots are explicitly retired"))
    assert private_queue.sync(connection, plan) == expected


def _retirement_tool_fixture(resources, tmp_path, monkeypatch):
    plan, _, cleanup_approval, _ = _tool_fixture(resources, tmp_path, monkeypatch)
    source = private_queue.PROJECT / "scripts/queue-retire-v1.py"
    source.write_bytes(b"# separately reviewed terminal retirement tool\n")
    approval = {**cleanup_approval, "retirement_sha256": _sha(source.read_bytes())}
    path = private_queue.LOCAL_ROOT / "retirement-tool-v1.json"
    path.write_text(json.dumps(approval))
    adapter_name = "queue-cleanup-v2-" + approval["adapter_sha256"] + ".py"
    monkeypatch.setattr(private_queue, "_cleanup_tool", lambda *_: private_queue.REMOTE_ROOT + "/" + adapter_name)
    return plan, source, approval, path, adapter_name


def test_retirement_tool_has_separate_pins_and_immutable_remote_name(resources, connection, tmp_path, monkeypatch):
    plan, source, approval, _, adapter_name = _retirement_tool_fixture(resources, tmp_path, monkeypatch)
    name = "queue-retire-v1-" + approval["retirement_sha256"] + ".py"
    files = {"queue_cleanup.py": approval["base_sha256"], adapter_name: approval["adapter_sha256"]}
    uploads = []
    def inspect(_config, code, *args, **_):
        assert code == private_queue.INSPECT
        return {"exists": True, "files": dict(files)}
    def upload(_config, staged_source, staged_name, digest, base_hash):
        assert staged_source == source and staged_name == name and digest == approval["retirement_sha256"]
        assert base_hash == approval["base_sha256"]
        uploads.append(name)
        files[name] = digest
    monkeypatch.setattr(private_queue, "_remote", inspect)
    monkeypatch.setattr(private_queue, "_upload_tool", upload)
    assert private_queue._retirement_tool(connection, plan) == private_queue.REMOTE_ROOT + "/" + name
    assert uploads == [name]
    private_queue._retirement_tool(connection, plan)
    assert uploads == [name]


@pytest.mark.parametrize("mutation", ["source", "base", "version"], ids=["source", "base", "version"])
def test_unapproved_retirement_source_or_dependencies_cannot_be_uploaded(resources, connection, tmp_path, monkeypatch, mutation):
    plan, source, approval, path, _ = _retirement_tool_fixture(resources, tmp_path, monkeypatch)
    if mutation == "source":
        source.write_bytes(b"changed")
    else:
        approval["base_sha256" if mutation == "base" else "version"] = "d" * 64 if mutation == "base" else True
        path.write_text(json.dumps(approval))
    monkeypatch.setattr(private_queue, "_remote", lambda *_a, **_k: pytest.fail("Unapproved tool must not reach the host"))
    with pytest.raises(ValueError):
        private_queue._retirement_tool(connection, plan)

