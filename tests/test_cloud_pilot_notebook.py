"""The uploaded pilot bootstrap remains runnable without a full project checkout."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "train_pilot_cloud.ipynb"


def pilot_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    code = "".join(cell["source"])
    parsed = ast.parse(code)
    assignment = next(n for n in parsed.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "SCRIPT_SOURCE" for t in n.targets))
    return ast.literal_eval(assignment.value), cell, code


def pilot_helpers():
    source, _, _ = pilot_source()
    scope = {"__name__": "cloud_pilot_test"}
    exec(compile(source, "cloud-pilot-cell.py", "exec"), scope)
    return scope


def bundle(path, extras=None, corrupt_index=False, splits=None):
    members = {"cache/manifest.json": json.dumps({"splits": splits or {"train": ["a"], "validation": ["b"]}}).encode(),
               "cache/preprocessing.json": b'{"version":1}',
               "cache/train/a.npz": b"train fixture", "cache/validation/b.npz": b"selection fixture",
               "initial-config.json": b'{"seed":42}', "source-manifest.json": b'{"splits":{}}',
               "source/adp/__init__.py": b'"""fixture"""', "source/adp/model.py": b"# trusted source fixture"}
    members.update(extras or {})
    index = {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members.items()}
    if corrupt_index:
        index["cache/train/a.npz"]["sha256"] = "0" * 64
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("package-files.json", json.dumps(index))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_standalone_script_and_notebook_compile_without_gpu_import():
    source, cell, notebook_code = pilot_source()
    compile(notebook_code, str(NOTEBOOK), "exec")
    compile(source, "cloud-pilot-cell.py", "exec")
    assert cell["execution_count"] is None and not cell["outputs"]
    scope = pilot_helpers()
    assert "torch" not in scope
    assert "--payload" in source and "--root" in source
    assert "MAXIMUM_COMPUTE_SECONDS = 1800.0" in source
    assert "epoch" in source and "run_job(root, device='cuda'" in source
    assert "calibrate(" not in source and "'evaluation'" not in source
    assert scope["EXPECTED_PAYLOAD_SHA256"] == "f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa"
    assert scope["EXPECTED_PACKAGE_INDEX_SHA256"] == "3770e3f2c6df73ce4fb652a6c52a2841772f25feb70a3480bb035e788f107282"
    assert scope["EXPECTED_INDEXED_MEMBER_COUNT"] == 1076
    assert "--extracted-input" in source
    script_hash = next(node for node in ast.parse(notebook_code).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXPECTED_SCRIPT_SHA256" for target in node.targets))
    assert ast.literal_eval(script_hash.value) == hashlib.sha256(source.encode("utf-8")).hexdigest()


def test_verified_atomic_import_and_resume_with_unicode(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20 * 1024**3})())
    payload, target = tmp_path / "entrée.zip", tmp_path / "résultats"
    digest = bundle(payload)
    assert helpers["verify_and_stage"](payload, target, digest) == target
    assert payload.is_file() and (target / "cache/train/a.npz").read_bytes() == b"train fixture"
    assert not list(tmp_path.glob("*.staging-*"))
    assert helpers["verify_and_stage"](payload, target, digest) == target
    (target / "source/adp/model.py").write_text("# changed")
    with pytest.raises(ValueError, match="Existing job inputs differ"):
        helpers["verify_and_stage"](payload, target, digest)


@pytest.mark.parametrize("unsafe", ["../escape.py", "/tmp/escape.py", "source\\adp\\evil.py",
                                    "source/adp/nested/evil.py", "cache/evaluation/locked.npz"])
def test_rejects_unsafe_members_before_extraction(tmp_path, unsafe):
    helpers = pilot_helpers()
    payload, target = tmp_path / "payload.zip", tmp_path / "job"
    digest = bundle(payload, {unsafe: b"unsafe fixture"})
    with pytest.raises(ValueError, match="Unsafe|unexpected|checksum index"):
        helpers["verify_and_stage"](payload, target, digest)
    assert not target.exists()


def test_rejects_archive_or_member_checksum_mismatch_before_extraction(tmp_path):
    helpers = pilot_helpers()
    payload, target = tmp_path / "payload.zip", tmp_path / "job"
    digest = bundle(payload, corrupt_index=True)
    with pytest.raises(ValueError, match="payload SHA-256"):
        helpers["verify_and_stage"](payload, target, "0" * 64)
    with pytest.raises(ValueError, match="member checksum"):
        helpers["verify_and_stage"](payload, target, digest)
    assert not target.exists()


def test_respects_reserve_and_rejects_extra_label_partition(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    payload, target = tmp_path / "payload.zip", tmp_path / "job"
    digest = bundle(payload)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 8 * 1024**3})())
    with pytest.raises(RuntimeError, match="8 GiB"):
        helpers["verify_and_stage"](payload, target, digest)
    assert not target.exists()
    digest = bundle(payload, splits={"train": ["a"], "validation": ["b"], "calibration": []})
    with pytest.raises(ValueError, match="train and selection only"):
        helpers["verify_and_stage"](payload, target, digest)


def test_consistent_export_preserves_active_run_and_requires_idle_owner(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20 * 1024**3})())
    root = tmp_path / "job"
    (root / "run").mkdir(parents=True)
    (root / "run/last.pt").write_bytes(b"trusted checkpoint fixture")
    (root / "job.json").write_text('{"purpose":"fixture"}')
    (root / ".remote-job.lock").write_text("owned")
    with pytest.raises(RuntimeError, match="owner is active"):
        helpers["export_pilot"](root)
    (root / ".remote-job.lock").unlink()
    target, receipt_path = helpers["export_pilot"](root)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["archive_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        for name, info in receipt["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
    assert (root / "run/last.pt").read_bytes() == b"trusted checkpoint fixture"
    assert not list(target.parent.glob("*.partial"))


def extracted_fixture(tmp_path):
    payload, source = tmp_path / "payload.zip", tmp_path / "Kaggle – extracted"
    bundle(payload)
    with zipfile.ZipFile(payload) as archive:
        archive.extractall(source)
    index = source / "package-files.json"
    return source, hashlib.sha256(index.read_bytes()).hexdigest(), len(json.loads(index.read_bytes()))


def test_index_pinned_extracted_inputs_stage_atomically_and_preserve_resume_files(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20*1024**3})())
    source, digest, count = extracted_fixture(tmp_path)
    (source / "cloud-pilot-cell.py").write_text("# ignored sibling runner; never copied or imported")
    target = tmp_path / "研究 – runtime"
    assert helpers["verify_extracted_and_stage"](source, target, digest, count) == target
    assert not (target / "cloud-pilot-cell.py").exists()
    assert (target / "cache/train/a.npz").read_bytes() == b"train fixture"
    (target / "run").mkdir()
    (target / "run/last.pt").write_bytes(b"active checkpoint must remain")
    assert helpers["verify_extracted_and_stage"](source, target, digest, count) == target
    assert (target / "run/last.pt").read_bytes() == b"active checkpoint must remain"
    (target / "cache/train/a.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing job inputs differ"):
        helpers["verify_extracted_and_stage"](source, target, digest, count)
    assert not list(tmp_path.glob("*.staging-*"))


@pytest.mark.parametrize("mutation", ["index_hash", "member_hash", "missing", "extra", "runner_size", "count"])
def test_extracted_input_integrity_and_exact_membership_fail_before_copy(tmp_path, mutation):
    helpers = pilot_helpers()
    source, digest, count = extracted_fixture(tmp_path)
    target = tmp_path / "job"
    if mutation == "index_hash":
        digest = "0"*64
    elif mutation == "member_hash":
        (source / "cache/train/a.npz").write_bytes(b"changed")
    elif mutation == "missing":
        (source / "cache/train/a.npz").unlink()
    elif mutation == "extra":
        (source / "other-runner.py").write_text("# unindexed sibling")
    elif mutation == "runner_size":
        (source / "cloud-pilot-cell.py").write_bytes(b"#"*(128*1024+1))
    else:
        count += 1
    with pytest.raises(ValueError):
        helpers["verify_extracted_and_stage"](source, target, digest, count)
    assert not target.exists()
    assert (source / "package-files.json").exists()
    assert not list(tmp_path.glob("*.staging-*"))


@pytest.mark.parametrize("unsafe", ["../escape.py", "cache/evaluation/locked.npz", "source\\adp\\evil.py", "/tmp/escape.py"])
def test_extracted_index_paths_are_bounded_even_with_a_matching_index_digest(tmp_path, unsafe):
    helpers = pilot_helpers()
    source, _, _ = extracted_fixture(tmp_path)
    path = source / "package-files.json"
    index = json.loads(path.read_bytes())
    index[unsafe] = {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Unsafe|unexpected"):
        helpers["verify_extracted_and_stage"](source, tmp_path / "job", hashlib.sha256(path.read_bytes()).hexdigest(), len(index))
    assert not (tmp_path / "job").exists()


def test_extracted_staging_rechecks_copy_and_removes_only_its_partial_output(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    source, digest, count = extracted_fixture(tmp_path)
    target = tmp_path / "job"
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20*1024**3})())
    original_hash = helpers["sha256_file"]
    def changing_source(path):
        value = original_hash(path)
        if Path(path) == source / "source/adp/model.py":
            (source / "cache/train/a.npz").write_bytes(b"changed during staging")
        return value
    helpers["sha256_file"] = changing_source
    with pytest.raises(ValueError, match="changed during atomic staging"):
        helpers["verify_extracted_and_stage"](source, target, digest, count)
    assert not target.exists() and not list(tmp_path.glob("*.staging-*"))
    assert (source / "source/adp/model.py").is_file()


def test_extracted_input_reserve_and_source_destination_separation(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    source, digest, count = extracted_fixture(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 8*1024**3})())
    with pytest.raises(RuntimeError, match="8 GiB"):
        helpers["verify_extracted_and_stage"](source, tmp_path / "job", digest, count)
    with pytest.raises(ValueError, match="separate"):
        helpers["verify_extracted_and_stage"](source, source / "nested", digest, count)
    assert not (source / "nested").exists()


def test_extracted_input_export_receipt_keeps_original_zip_hash_and_adds_precise_evidence(tmp_path, monkeypatch):
    helpers = pilot_helpers()
    evidence = {"kind": "verified_indexed_extracted_members", "package_index_sha256": helpers["EXPECTED_PACKAGE_INDEX_SHA256"],
                "verified_member_count": 1076}
    helpers["INPUT_VERIFICATION"] = evidence
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"free": 20*1024**3})())
    root = tmp_path / "job"
    (root / "run").mkdir(parents=True)
    (root / "run/last.pt").write_bytes(b"trusted checkpoint fixture")
    archive_path, receipt_path = helpers["export_pilot"](root)
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["payload_sha256"] == helpers["EXPECTED_PAYLOAD_SHA256"]
    assert receipt["input_verification"] == evidence
    with zipfile.ZipFile(archive_path) as archive:
        internal = json.loads(archive.read("export-manifest.json"))
    assert internal == {key: value for key, value in receipt.items() if key != "archive_sha256"}
