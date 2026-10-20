"""Shared processed training cache survives until every consumer and report finishes."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest
import torch
from filelock import FileLock, Timeout

from adp.compute_cleanup import _hash
from adp.data import read_json, write_json
from adp.model import ForecastModel
from adp.protocol import create_protocol
from adp.study_cleanup import (JOURNAL_NAME, PLAN_NAME, archive_and_cleanup, execute_study_cleanup,
                               main, plan_study_cleanup)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def finished(tmp_path, monkeypatch):
    import adp.study_cleanup as cleanup
    root, direction = tmp_path / "study-v1", tmp_path / "direction-study-v1"
    root.mkdir()
    direction.mkdir()
    monkeypatch.setattr(cleanup, "STUDY_ROOT", root)
    monkeypatch.setattr(cleanup, "DIRECTION_ROOT", direction)
    monkeypatch.setattr(cleanup, "MODEL_ARCHIVE_ROOT", tmp_path / "model-archive" / "study-v1")
    monkeypatch.setattr(cleanup.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    ids = [str(uuid.UUID(int=number)) for number in range(1, 7)]
    listing = {"source": "https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/train/", "complete": True, "ids": ids[:2]}
    write_json(root / "listings" / "train.json", listing)
    manifest = {"version": 2, "source": "Argoverse 2 Motion Forecasting",
        "splits": {"train": ids[:2], "validation": [ids[2]], "calibration": [ids[3]], "evaluation": [ids[4]]},
        "listing_sha256": {"train": _sha(root / "listings" / "train.json")}}
    write_json(root / "raw" / "manifest.json", manifest)
    write_json(root / "cache" / "manifest.json", manifest)
    write_json(root / "cache" / "preprocessing.json", {"version": 1})
    for split, members in manifest["splits"].items():
        for sid in members:
            path = root / "cache" / split / f"{sid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"derived {split} scenario {sid}".encode())
    (root / "raw" / "original-map.json").write_bytes(b"preserve raw maps and trajectories")
    create_protocol(root / "cache", root / "protocol.json")
    config = {"model": {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1}, "seed": 42}
    selected = root / "runs" / "full-42" / "best.pt"
    selected.parent.mkdir(parents=True)
    torch.save({"format_version": 2, "config": config, "completed_epochs": 1, "partial_epoch": False,
        "manifest_hash": _sha(root / "cache" / "manifest.json"), "preprocessing_hash": _sha(root / "cache" / "preprocessing.json"),
        "model_state_dict": ForecastModel(config["model"]).state_dict()}, selected)
    copy = tmp_path / "durable-model-archive" / "selected.pt"
    copy.parent.mkdir()
    shutil.copyfile(selected, copy)
    write_json(root / "frozen_selection.json", {"checkpoint": str(selected), "checkpoint_sha256": _sha(selected),
        "deployment_run": "full-42", "all_seed_runs": ["full-42", "full-43", "full-44"]})
    write_json(root / "promotion.json", {"deployment_checkpoint_sha256": _sha(selected), "promotion_gate_passed": False})
    write_json(root / "pipeline_status.json", {"status": "completed", "phase": "finished"})
    direction_protocol = {"reference_root": str(root), "cache_dir": str(root / "cache"),
        "manifest_sha256": _sha(root / "cache" / "manifest.json"), "preprocessing_sha256": _sha(root / "cache" / "preprocessing.json"),
        "train_ids": ids[:2], "selection_ids": [ids[2]]}
    write_json(direction / "protocol.json", direction_protocol)
    write_json(direction / "experiment.json", {"status": "completed", "protocol_sha256": _sha(direction / "protocol.json")})
    write_json(direction / "comparison.json", {"protocol_sha256": _sha(direction / "protocol.json")})
    (direction / "REPORT.md").write_text("Completed CPU comparison; inconclusive hypothesis")
    reports = {}
    for name, kind in (("reference-analysis", "reference"), ("direction-analysis", "direction-study")):
        folder = direction / name
        folder.mkdir()
        files = {}
        for file in ("records.json.gz", "summary.json", "summary.csv", "failures.png"):
            (folder / file).write_bytes(b"completed recorded report")
            files[file] = {"bytes": (folder / file).stat().st_size, "sha256": _sha(folder / file)}
        identity = name.replace("-", "")
        write_json(folder / "analysis.json", {"id": identity, "provenance": {"kind": kind,
            "manifest_sha256": _sha(root / "cache" / "manifest.json"), "inference_device": "cpu", "labels_used": "selection_only"}, "files": files})
        reports[name] = {"id": identity, "output_dir": str(folder)}
    write_json(direction / "followup_status.json", {"status": "completed", "phase": "finished", "direction_result": {"status": "completed"},
        "reference_analysis": reports["reference-analysis"], "direction_analysis": reports["direction-analysis"]})
    proof = {"version": 1, "verified": True, "protocol_sha256": _sha(root / "protocol.json"),
        "dataset_manifest_sha256": _sha(root / "cache" / "manifest.json"), "preprocessing_sha256": _sha(root / "cache" / "preprocessing.json"),
        "frozen_selection_sha256": _sha(root / "frozen_selection.json"), "checkpoint_sha256": _sha(selected),
        "config_sha256": _hash(config), "checkpoint_copy": str(copy)}
    return root, direction, proof, copy


def test_plan_and_execution_retire_only_train_npzs_and_resume_idempotently(finished):
    root, direction, proof, copy = finished
    preserved = {path: path.read_bytes() for path in root.rglob("*") if path.is_file() and path.parent != root / "cache" / "train"}
    plan = plan_study_cleanup(root, direction, proof)
    assert plan["state"] == "ready" and plan["files_to_remove"] == 2
    assert not (root / PLAN_NAME).exists() and not (root / JOURNAL_NAME).exists()
    assert execute_study_cleanup(root, direction, proof)["state"] == "completed"
    assert not (root / "cache" / "train").exists()
    assert all(path.read_bytes() == content for path, content in preserved.items())
    assert copy.is_file()
    assert execute_study_cleanup(root, direction, proof)["state"] == "completed"
    assert plan_study_cleanup(root, direction, proof)["bytes_to_free"] == 0


@pytest.mark.parametrize("gate", ["original-active", "direction-pending", "lock", "missing-proof", "wrong-proof", "wrong-copy", "copy-inside-study", "incomplete-reports", "wrong-root"])
def test_gates_block_before_hashing_any_training_inputs(finished, monkeypatch, gate):
    root, direction, proof, _ = finished
    import adp.study_cleanup as cleanup
    original_sha = cleanup._sha
    def guarded_sha(path):
        if Path(path).suffix == ".npz":
            pytest.fail("Blocked cleanup must not scan the 40,000 training files")
        return original_sha(path)
    monkeypatch.setattr(cleanup, "_sha", guarded_sha)
    owner = None
    if gate == "original-active":
        write_json(root / "pipeline_status.json", {"status": "running", "phase": "train"})
    elif gate == "direction-pending":
        write_json(direction / "followup_status.json", {"status": "waiting", "phase": "waiting_for_reference"})
    elif gate == "lock":
        owner = FileLock(str(direction / "direction.lock"), timeout=0)
        owner.acquire()
    elif gate == "missing-proof":
        proof = None
    elif gate == "wrong-proof":
        proof = {**proof, "protocol_sha256": "f" * 64}
    elif gate == "wrong-copy":
        Path(proof["checkpoint_copy"]).write_bytes(b"corrupt model copy")
    elif gate == "copy-inside-study":
        proof = {**proof, "checkpoint_copy": str(root / "runs" / "full-42" / "best.pt")}
    elif gate == "incomplete-reports":
        (direction / "direction-analysis" / "summary.csv").unlink()
    else:
        root = root.parent
    try:
        result = execute_study_cleanup(root, direction, proof)
    finally:
        if owner is not None:
            owner.release()
    assert result["state"] == "blocked" and result["reason"]


def test_foreign_npz_and_changed_plan_bytes_block_cleanup(finished):
    root, direction, proof, _ = finished
    foreign = root / "cache" / "train" / "foreign.npz"
    foreign.write_bytes(b"preserve unregistered data")
    assert plan_study_cleanup(root, direction, proof)["state"] == "blocked"
    assert foreign.is_file()


def test_interruption_after_unlink_resumes_only_authorized_intent(finished, monkeypatch):
    root, direction, proof, _ = finished
    original = Path.unlink
    count = 0
    def interrupted(path, *args, **kwargs):
        nonlocal count
        if path.suffix == ".npz":
            count += 1
            if count == 2:
                raise OSError("power interruption")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", interrupted)
        with pytest.raises(OSError, match="interruption"):
            execute_study_cleanup(root, direction, proof)
    assert len(list((root / "cache" / "train").glob("*.npz"))) == 1
    assert execute_study_cleanup(root, direction, proof)["state"] == "completed"
    records = [json.loads(line) for line in (root / JOURNAL_NAME).read_text().splitlines()]
    assert [entry["kind"] for entry in records] == ["header", "intent", "done", "completed"]


def test_low_space_blocks_before_audit_or_deletion(finished, monkeypatch):
    root, direction, proof, _ = finished
    monkeypatch.setattr("adp.study_cleanup.shutil.disk_usage", lambda _: SimpleNamespace(free=8 * 1024**3))
    result = execute_study_cleanup(root, direction, proof)
    assert result["state"] == "blocked" and "8 GiB" in result["reason"]
    assert not (root / PLAN_NAME).exists()
    assert len(list((root / "cache" / "train").glob("*.npz"))) == 2


def test_cli_without_proof_is_read_only_blocked(finished, capsys):
    root, direction, _, _ = finished
    main(["--root", str(root), "--direction-root", str(direction), "--plan"])
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "blocked" and "receipt" in result["reason"]
    assert not (root / PLAN_NAME).exists()


def test_write_ahead_intent_and_plan_are_published_once_across_groups(finished, monkeypatch):
    root, direction, proof, _ = finished
    import adp.study_cleanup as cleanup
    monkeypatch.setattr(cleanup, "GROUP_SIZE", 1)
    write, append = cleanup._write, cleanup._append
    writes, events = [], []
    def recorded_write(path, value):
        writes.append(path.name)
        return write(path, value)
    def recorded_append(path, value):
        events.append(value["kind"])
        return append(path, value)
    monkeypatch.setattr(cleanup, "_write", recorded_write)
    monkeypatch.setattr(cleanup, "_append", recorded_append)
    assert execute_study_cleanup(root, direction, proof)["state"] == "completed"
    assert writes == [PLAN_NAME]
    assert events == ["header", "intent", "done", "intent", "done", "completed"]


def test_missing_file_without_durable_intent_is_never_authorized(finished, monkeypatch):
    root, direction, proof, _ = finished
    import adp.study_cleanup as cleanup
    append = cleanup._append
    def fail_before_intent(path, value):
        if value["kind"] == "intent":
            raise OSError("interruption before intent became durable")
        return append(path, value)
    with monkeypatch.context() as scoped:
        scoped.setattr(cleanup, "_append", fail_before_intent)
        with pytest.raises(OSError):
            execute_study_cleanup(root, direction, proof)
    next((root / "cache" / "train").glob("*.npz")).unlink()
    result = execute_study_cleanup(root, direction, proof)
    assert result["state"] == "blocked" and "intent" in result["reason"]
    assert len(list((root / "cache" / "train").glob("*.npz"))) == 1


def test_checkpoint_proof_requires_strict_cpu_model_loading(finished):
    root, direction, proof, copy = finished
    selected = root / "runs" / "full-42" / "best.pt"
    saved = torch.load(selected, map_location="cpu", weights_only=True)
    saved["model_state_dict"]["history_embedding.weight"] = torch.zeros(16, 7)
    torch.save(saved, selected)
    shutil.copyfile(selected, copy)
    frozen = read_json(root / "frozen_selection.json")
    frozen["checkpoint_sha256"] = _sha(selected)
    write_json(root / "frozen_selection.json", frozen)
    proof = {**proof, "checkpoint_sha256": _sha(selected), "frozen_selection_sha256": _sha(root / "frozen_selection.json")}
    result = plan_study_cleanup(root, direction, proof)
    assert result["state"] == "blocked" and "strict CPU" in result["reason"]


def test_plan_cannot_be_rewritten_to_escape_the_train_directory(finished, monkeypatch):
    root, direction, proof, _ = finished
    import adp.study_cleanup as cleanup
    append = cleanup._append
    def stop_after_header(path, value):
        if value["kind"] == "intent":
            raise OSError("stop before deleting")
        return append(path, value)
    with monkeypatch.context() as scoped:
        scoped.setattr(cleanup, "_append", stop_after_header)
        with pytest.raises(OSError):
            execute_study_cleanup(root, direction, proof)
    plan = read_json(root / PLAN_NAME)
    plan["files"][0]["name"] = "../calibration/foreign.npz"
    plan.pop("plan_sha256")
    plan["plan_sha256"] = _hash(plan)
    write_json(root / PLAN_NAME, plan)
    assert execute_study_cleanup(root, direction, proof)["state"] == "blocked"
    assert len(list((root / "cache" / "train").glob("*.npz"))) == 2


def test_released_zero_byte_lock_paths_allow_planning_and_cleanup(finished):
    root, direction, proof, _ = finished
    paths = [root / "pipeline.lock", root / "runner.lock", direction / "followup.lock", direction / "direction.lock"]
    for path in paths:
        with FileLock(str(path), timeout=0):
            pass
        path.touch(exist_ok=True)
    assert all(path.stat().st_size == 0 for path in paths)
    assert plan_study_cleanup(root, direction, proof)["state"] == "ready"
    assert execute_study_cleanup(root, direction, proof)["state"] == "completed"


def test_execution_holds_all_four_study_locks_during_every_unlink(finished, monkeypatch):
    root, direction, proof, _ = finished
    paths = [root / "pipeline.lock", root / "runner.lock", direction / "followup.lock", direction / "direction.lock"]
    original = Path.unlink
    checked = []
    def protected_unlink(path, *args, **kwargs):
        if path.suffix == ".npz":
            for lock in paths:
                with pytest.raises(Timeout):
                    FileLock(str(lock), timeout=0).acquire()
            checked.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", protected_unlink)
    assert execute_study_cleanup(root, direction, proof)["state"] == "completed"
    assert len(checked) == 2


def test_unknown_dot_lock_remains_conservatively_blocking(finished):
    root, direction, proof, _ = finished
    unknown = direction / ".other-owner.lock"
    unknown.touch()
    result = plan_study_cleanup(root, direction, proof)
    assert result["state"] == "blocked" and "Unknown" in result["reason"]
    assert unknown.is_file()


@pytest.mark.parametrize("gate", ["original-active", "direction-active", "held-lock"])
def test_automatic_archive_does_nothing_while_any_study_is_active(finished, monkeypatch, gate):
    root, direction, _, _ = finished
    import adp.study_cleanup as cleanup
    monkeypatch.setattr(cleanup, "_checkpoint", lambda _: pytest.fail("Active studies must not trigger CPU qualification"))
    monkeypatch.setattr(cleanup, "_sha", lambda _: pytest.fail("Active studies must not trigger model or dataset hashing"))
    owner = None
    if gate == "original-active":
        write_json(root / "pipeline_status.json", {"status": "running", "phase": "train"})
    elif gate == "direction-active":
        write_json(direction / "followup_status.json", {"status": "running", "phase": "direction_analysis"})
    else:
        owner = FileLock(str(root / "runner.lock"), timeout=0)
        owner.acquire()
    try:
        result = archive_and_cleanup(root, direction)
    finally:
        if owner:
            owner.release()
    assert result["state"] == "blocked"
    assert not cleanup.MODEL_ARCHIVE_ROOT.exists()
    assert not (root / PLAN_NAME).exists()


def test_completed_studies_automatically_archive_prove_and_clean_idempotently(finished, monkeypatch):
    root, direction, _, _ = finished
    import adp.study_cleanup as cleanup
    write = cleanup._write_new
    copy_checks = []
    def exclusively_owned_write(path, content):
        for lock in (root / "pipeline.lock", root / "runner.lock", direction / "followup.lock", direction / "direction.lock"):
            with pytest.raises(Timeout):
                FileLock(str(lock), timeout=0).acquire()
        copy_checks.append(path.name)
        return write(path, content)
    monkeypatch.setattr(cleanup, "_write_new", exclusively_owned_write)
    result = archive_and_cleanup(root, direction)
    assert result["state"] == "completed"
    copy = Path(result["archived_checkpoint"])
    proof_path = Path(result["proof_path"])
    proof_bytes = proof_path.read_bytes()
    assert _sha(copy) == read_json(root / "frozen_selection.json")["checkpoint_sha256"]
    proof = read_json(proof_path)
    assert proof["verified"] is True and proof["checkpoint_copy"] == str(copy)
    assert not (root / "cache" / "train").exists()
    assert (root / "runs" / "full-42" / "best.pt").is_file()
    assert (root / "cache" / "calibration").is_dir() and (root / "cache" / "evaluation").is_dir()
    assert archive_and_cleanup(root, direction)["state"] == "completed"
    assert proof_path.read_bytes() == proof_bytes
    assert copy_checks == ["selected.pt", "cleanup-proof.json"]


def test_corrupt_existing_automatic_archive_is_never_overwritten(finished):
    root, direction, _, _ = finished
    import adp.study_cleanup as cleanup
    digest = read_json(root / "frozen_selection.json")["checkpoint_sha256"]
    destination = cleanup.MODEL_ARCHIVE_ROOT / digest
    destination.mkdir(parents=True)
    (destination / "selected.pt").write_bytes(b"corrupt preexisting archive; preserve for review")
    result = archive_and_cleanup(root, direction)
    assert result["state"] == "blocked" and "archive" in result["reason"]
    assert (destination / "selected.pt").read_bytes() == b"corrupt preexisting archive; preserve for review"
    assert len(list((root / "cache" / "train").glob("*.npz"))) == 2


@pytest.mark.parametrize("gate", ["low-space", "traversal", "unfinished-report"])
def test_automatic_archive_space_paths_and_reports_gate_before_publication(finished, monkeypatch, gate):
    root, direction, _, _ = finished
    import adp.study_cleanup as cleanup
    if gate == "low-space":
        monkeypatch.setattr(cleanup.shutil, "disk_usage", lambda _: SimpleNamespace(free=8 * 1024**3))
    elif gate == "traversal":
        root = root / ".." / "study-v1"
    else:
        (direction / "direction-analysis" / "summary.csv").unlink()
    result = archive_and_cleanup(root, direction)
    assert result["state"] == "blocked"
    assert not cleanup.MODEL_ARCHIVE_ROOT.exists()


def test_automatic_archive_cli_requires_explicit_execute(finished, capsys):
    root, direction, _, _ = finished
    with pytest.raises(SystemExit) as error:
        main(["--root", str(root), "--direction-root", str(direction), "--archive-completed", "--plan"])
    assert error.value.code == 2
    main(["--root", str(root), "--direction-root", str(direction), "--archive-completed", "--execute"])
    assert json.loads(capsys.readouterr().out)["state"] == "completed"


def test_status_change_during_final_lock_acquisition_blocks_before_model_work(finished, monkeypatch):
    root, direction, _, _ = finished
    import adp.study_cleanup as cleanup
    class ChangingOwner:
        def __init__(self, path, timeout):
            self.path = Path(path)
            self.owner = FileLock(path, timeout=timeout)
        def __enter__(self):
            self.owner.acquire()
            if self.path == direction / "direction.lock":
                write_json(direction / "followup_status.json", {"status": "running", "phase": "direction_analysis"})
            return self.owner
        def __exit__(self, *_):
            self.owner.release()
    monkeypatch.setattr(cleanup, "FileLock", ChangingOwner)
    monkeypatch.setattr(cleanup, "_checkpoint", lambda _: pytest.fail("Changed study status must prevent CPU qualification"))
    monkeypatch.setattr(cleanup, "_sha", lambda _: pytest.fail("Changed study status must prevent model/dataset hashing"))
    result = archive_and_cleanup(root, direction)
    assert result["state"] == "blocked" and "not completed" in result["reason"]
    assert not cleanup.MODEL_ARCHIVE_ROOT.exists()
    for path in (root / "pipeline.lock", root / "runner.lock", direction / "followup.lock", direction / "direction.lock"):
        with FileLock(str(path), timeout=0):
            pass
