"""Portable cloud code is pinned before extraction/import; data stays separate."""
import hashlib
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build-cloud-queue.py"
spec = importlib.util.spec_from_file_location("build_cloud_queue", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    members = {"initial-config.json": b'{"seed":42}', "source-manifest.json": b'{"splits":{}}',
        "cache/manifest.json": b'{"splits":{"train":[],"validation":[]}}', "cache/preprocessing.json": b'{"version":1}'}
    members.update({"source/adp/" + name: ("# frozen " + name).encode() for name in builder.FROZEN_SOURCE})
    members["package-files.json"] = json.dumps({name: {"sha256": _sha(data), "size": len(data)} for name, data in members.items()}).encode()
    payload = tmp_path / "input.bin"
    _zip(payload, members)
    digest = _sha(payload.read_bytes())
    monkeypatch.setattr(builder, "PAYLOAD_SHA256", digest)
    project = tmp_path / "project"
    sources = {"scripts/compute-queue-worker.py": b"# worker\n", "src/adp/queue_cleanup.py": b"# cleanup\n",
        "artifacts/cloud-pilot-cell.py": b"# helper\n", "src/adp/remote_transfer.py": b"# CPU controller\n"}
    queue = (SCRIPT.parents[1] / "src/adp/compute_queue.py").read_bytes()
    sources["src/adp/compute_queue.py"] = queue.replace(b"f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa", digest.encode())
    for name, content in sources.items():
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    result = builder.build(payload, tmp_path / "bundle", project=project)
    return payload, project, result


def test_exact_small_code_bundle_is_deterministic_and_notebooks_are_unexecuted_valid_python(bundle):
    payload, project, result = bundle
    output = Path(result["path"]).parent
    assert result["bytes"] < 2 * 1024**2
    with zipfile.ZipFile(result["path"]) as archive:
        assert set(archive.namelist()) == builder.MEMBERS | {"bundle-manifest.json"}
        assert not any(name.endswith((".pt", ".npz")) or name == "inputs.zip" for name in archive.namelist())
        assert archive.read("control/adp/model.py") == b"# frozen model.py"
        assert json.loads(archive.read("bundle-manifest.json")) == result["manifest"]
        for name, indexed in result["manifest"]["files"].items():
            assert indexed == {"sha256": _sha(archive.read(name)), "bytes": len(archive.read(name))}
    assert builder.build(payload, output, project=project) == result
    assert (output / "interaction-lab-queue-code.bin").read_bytes() == Path(result["path"]).read_bytes()
    for host in ("colab", "kaggle"):
        notebook = json.loads((output / (host + "-queue.ipynb")).read_bytes())
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == []
                compile("".join(cell["source"]), host, "exec")
        source = (output / (host + "-queue-cell.py")).read_text(encoding="utf-8")
        compile(source, host, "exec")
        compile((output / (host + "-queue-launch-prepared.py")).read_text(encoding="utf-8"), host + "-prepared", "exec")
        assert result["sha256"] in source and builder.PAYLOAD_SHA256 in source
        assert "interaction-lab-" + host + "-queue-v1" in source
    assert "interaction-lab-queue-code.bin" in (output / "kaggle-queue-cell.py").read_text()
    assert "server-pilot-v1-inputs.bin" in (output / "kaggle-queue-cell.py").read_text()


def test_existing_pinned_bundle_and_wrong_original_payload_are_preserved(bundle):
    payload, project, result = bundle
    original = Path(result["path"]).read_bytes()
    (project / "scripts/compute-queue-worker.py").write_bytes(b"# changed worker\n")
    with pytest.raises(ValueError, match="Existing pinned bundle"):
        builder.build(payload, Path(result["path"]).parent, project=project)
    assert Path(result["path"]).read_bytes() == original
    payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="pinned payload"):
        builder.build(payload, Path(result["path"]).parent / "another", project=project)


def _prepare(monkeypatch):
    namespace = {}
    exec(builder.PREPARATION_SOURCE, namespace)
    monkeypatch.setattr(namespace["shutil"], "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    return namespace


def test_notebook_prepares_fresh_atomic_queue_with_exact_payload_and_protocol(bundle, monkeypatch, tmp_path):
    payload, _, result = bundle
    namespace = _prepare(monkeypatch)
    root = tmp_path / "cloud"
    created = namespace["prepare_cloud_queue"](result["path"], payload, root, result["sha256"], result["manifest"], "colab")
    assert created == root and (root / "inputs.zip").read_bytes() == payload.read_bytes()
    resources = json.loads((root / "resources.json").read_bytes())
    assert resources["payload_sha256"] == builder.PAYLOAD_SHA256
    assert set(resources["files"]) == {"worker.py", "compute_queue.py", "queue_cleanup.py", "pilot-helper.py", "inputs.zip"}
    spec = importlib.util.spec_from_file_location("verified_cloud_queue", root / "compute_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = {"version": 1, "host_seconds": 21600, "jobs": [module.job_spec("colab", c, s) for c in module.CONFIGURATIONS for s in module.SEEDS]}
    assert resources["plan_sha256"] == module.spec_hash(plan)
    assert not list(tmp_path.glob(".cloud-*"))
    with pytest.raises(RuntimeError, match="already exists"):
        namespace["prepare_cloud_queue"](result["path"], payload, root, result["sha256"], result["manifest"], "colab")
    assert (root / "inputs.zip").read_bytes() == payload.read_bytes()


@pytest.mark.parametrize("corruption", ["zip-hash", "member-hash", "traversal", "symlink", "duplicate"])
def test_notebook_rejects_untrusted_code_before_any_extraction_or_import(bundle, monkeypatch, tmp_path, corruption):
    payload, _, result = bundle
    archive = tmp_path / "untrusted.zip"
    archive.write_bytes(Path(result["path"]).read_bytes())
    if corruption == "member-hash":
        with zipfile.ZipFile(archive) as zipped:
            members = {name: zipped.read(name) for name in zipped.namelist()}
        members["worker.py"] = b"raise RuntimeError('must never execute')"
        _zip(archive, members)
    elif corruption != "zip-hash":
        with zipfile.ZipFile(archive, "a") as zipped:
            if corruption == "symlink":
                # Keep an otherwise allowlisted name to exercise file-type validation.
                original = {name: zipped.read(name) for name in zipped.namelist()}
            else:
                with pytest.warns(UserWarning) if corruption == "duplicate" else nullcontext():
                    zipped.writestr("worker.py" if corruption == "duplicate" else "../outside.py", b"unapproved")
        if corruption == "symlink":
            with zipfile.ZipFile(archive, "w") as rebuilt:
                for name, data in original.items():
                    entry = zipfile.ZipInfo(name)
                    entry.create_system = 3
                    entry.external_attr = ((stat.S_IFLNK if name == "worker.py" else stat.S_IFREG) | 0o644) << 16
                    rebuilt.writestr(entry, data)
    namespace = _prepare(monkeypatch)
    root = tmp_path / "rejected"
    with pytest.raises(ValueError):
        namespace["prepare_cloud_queue"](archive, payload, root,
            "f" * 64 if corruption == "zip-hash" else _sha(archive.read_bytes()), result["manifest"], "kaggle")
    assert not root.exists() and not list(tmp_path.glob(".rejected-*"))


def test_notebook_storage_guard_runs_before_copy_and_does_not_delete_existing_data(bundle, monkeypatch, tmp_path):
    payload, _, result = bundle
    namespace = _prepare(monkeypatch)
    monkeypatch.setattr(namespace["shutil"], "disk_usage", lambda _: SimpleNamespace(free=8 * 1024**3))
    root = tmp_path / "too-full"
    with pytest.raises(RuntimeError, match="8 GiB"):
        namespace["prepare_cloud_queue"](result["path"], payload, root, result["sha256"], result["manifest"], "colab")
    assert not root.exists() and payload.is_file() and Path(result["path"]).is_file()


def test_notebook_worker_preserves_qualified_venv_python_path(monkeypatch, tmp_path):
    namespace = _prepare(monkeypatch)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"qualified executable")
    python.chmod(0o755)
    checked = []
    unlinked = namespace["unlinked"]
    def check_parent(path):
        checked.append(path)
        return unlinked(path)
    namespace["unlinked"] = check_parent
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == python)
    command = namespace["worker_command"](tmp_path / "queue", python, "colab")
    assert command[0] == str(python)
    assert checked == [python.parent]
    assert command[command.index("--host") + 1] == "colab"


def test_notebook_worker_accepts_final_venv_symlink_without_resolving_it(monkeypatch, tmp_path):
    namespace = _prepare(monkeypatch)
    target = tmp_path / "base-python"
    target.write_bytes(b"qualified executable")
    target.chmod(0o755)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    try:
        python.symlink_to(target)
    except OSError as error:
        pytest.skip("Native symlink creation is unavailable: " + str(error))
    command = namespace["worker_command"](tmp_path / "queue", python, "colab")
    assert command[0] == str(python) and command[0] != str(target)


@pytest.mark.parametrize("invalid", ["missing", "directory", "junction", "linked-parent", "non-executable"])
def test_notebook_worker_rejects_invalid_qualified_python(monkeypatch, tmp_path, invalid):
    namespace = _prepare(monkeypatch)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    if invalid == "directory":
        python.mkdir()
    elif invalid != "missing":
        python.write_bytes(b"qualified executable")
        python.chmod(0o755)
    if invalid == "junction":
        monkeypatch.setattr(Path, "is_junction", lambda path: path == python, raising=False)
    elif invalid == "linked-parent":
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == python.parent)
    elif invalid == "non-executable":
        monkeypatch.setattr(os, "access", lambda *_: False)
    with pytest.raises((RuntimeError, ValueError)):
        namespace["worker_command"](tmp_path / "queue", python, "colab")


@pytest.mark.parametrize("corruption", [None, "worker", "payload", "resources", "lease", "controller"])
def test_prepared_launch_rechecks_pinned_bytes_and_never_resets_existing_owner(bundle, monkeypatch, tmp_path, corruption):
    payload, _, result = bundle
    namespace = _prepare(monkeypatch)
    root = tmp_path / "prepared"
    namespace["prepare_cloud_queue"](result["path"], payload, root, result["sha256"], result["manifest"], "colab")
    if corruption == "worker":
        (root / "worker.py").write_text("raise RuntimeError('must never execute')")
    elif corruption == "payload":
        (root / "inputs.zip").write_bytes(b"corrupt")
    elif corruption == "resources":
        (root / "resources.json").write_text("{}")
    elif corruption == "lease":
        (root / ".worker-lease.json").write_text('{"preserve":"lease"}')
    elif corruption == "controller":
        (root / "controller").mkdir()
    if corruption is None:
        assert namespace["verify_prepared_queue"](root, result["manifest"], "colab") == root
    else:
        with pytest.raises((ValueError, RuntimeError)):
            namespace["verify_prepared_queue"](root, result["manifest"], "colab")
        if corruption == "lease":
            assert (root / ".worker-lease.json").read_text() == '{"preserve":"lease"}'


def test_foreground_progress_prints_initial_periodic_and_final_saved_epochs(monkeypatch, tmp_path, capsys):
    namespace = _prepare(monkeypatch)
    identifier = "a" * 64
    (tmp_path / "worker-status.json").write_text(json.dumps({"state":"running", "current_job_id":identifier, "completed_jobs":2, "planned_jobs":12}))
    run = tmp_path / "jobs" / identifier / "run"
    run.mkdir(parents=True)
    (run / "history.json").write_text('{"epochs":[{}, {}, {}]}')
    waits = []
    class Process:
        pid = 123
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise namespace["subprocess"].TimeoutExpired("worker", timeout)
            return 0
    monkeypatch.setattr(namespace["subprocess"], "Popen", lambda command, env: Process())
    namespace["run_foreground"](["python", "worker.py"], tmp_path, {})
    updates = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert waits == [60, 60] and len(updates) == 3
    assert updates[0]["worker_pid"] == 123 and updates[-1]["worker_exit_code"] == 0
    assert all(update["saved_epochs"] == 3 and update["job_epochs"] == 20 for update in updates)


def test_foreground_interrupt_signals_owned_worker_and_preserves_lease(monkeypatch, tmp_path):
    namespace = _prepare(monkeypatch)
    lease = tmp_path / ".worker-lease.json"
    lease.write_text('{"preserve":"lease"}')
    signals, waits = [], []
    class Process:
        pid = 123
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise KeyboardInterrupt
            return 0
        def poll(self):
            return None
        def send_signal(self, value):
            signals.append(value)
    monkeypatch.setattr(namespace["subprocess"], "Popen", lambda command, env: Process())
    with pytest.raises(KeyboardInterrupt):
        namespace["run_foreground"](["python", "worker.py"], tmp_path, {})
    assert waits == [60, 30] and signals == [namespace["signal"].SIGINT]
    assert lease.read_text() == '{"preserve":"lease"}'


def test_progress_rejects_oversized_json_and_opaque_id_traversal(monkeypatch, tmp_path):
    namespace = _prepare(monkeypatch)
    status = tmp_path / "worker-status.json"
    status.write_bytes(b" " * (1024**2 + 1))
    assert namespace["queue_progress"](tmp_path)["state"] == "progress_unavailable"
    status.write_text('{"state":"running","current_job_id":"../outside"}')
    assert "saved_epochs" not in namespace["queue_progress"](tmp_path)

