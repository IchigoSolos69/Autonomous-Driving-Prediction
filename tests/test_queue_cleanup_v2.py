"""A versioned adapter fixes only the approved source-manifest size mismatch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]
PAYLOAD = Path("D:/InteractionLab/remote-compute/server-pilot-v1-inputs.zip")
SOURCE_SHA256 = "b898565a6550e013476a6b31eabeaed7eaa3993f577ae599f6d6dbe3decde088"


@pytest.fixture
def adapter(tmp_path):
    source = PROJECT / "scripts/queue-cleanup-v2.py"
    spec = importlib.util.spec_from_file_location("cleanup_v2_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / source.name)
    shutil.copy2(PROJECT / "src/adp/queue_cleanup.py", tmp_path / "queue_cleanup.py")
    return module


def test_canonical_source_manifest_is_accepted_without_widening_other_json(adapter, tmp_path):
    if not PAYLOAD.exists():
        pytest.skip("Canonical approved payload is not present on this machine")
    with zipfile.ZipFile(PAYLOAD) as archive:
        data = archive.read("source-manifest.json")
    assert len(data) == 2534744 and hashlib.sha256(data).hexdigest() == SOURCE_SHA256
    path = tmp_path / "source-manifest.json"
    path.write_bytes(data)
    base = adapter.load_base()
    assert base.MAX_JSON == 2 * 1024**2
    assert base._sha(path, base.MAX_JSON) == SOURCE_SHA256
    assert base._read(path) == json.loads(data)
    with pytest.raises(ValueError):
        base._json_bytes(data)
    other = tmp_path / "status.json"
    other.write_bytes(data)
    with pytest.raises(ValueError):
        base._read(other)
    with pytest.raises(ValueError):
        base._sha(other, base.MAX_JSON)


def test_synthetic_large_manifest_covers_portable_regression(adapter, tmp_path):
    data = b'{"padding":"' + b"x" * (2534744 - len(b'{"padding":""}')) + b'"}'
    assert len(data) == 2534744
    path = tmp_path / "source-manifest.json"
    path.write_bytes(data)
    base = adapter.load_base()
    assert base._sha(path, base.MAX_JSON) == hashlib.sha256(data).hexdigest()
    assert len(base._read(path)["padding"]) > base.MAX_JSON


def test_source_manifest_larger_than_eight_mib_is_rejected(adapter, tmp_path):
    path = tmp_path / "source-manifest.json"
    path.write_bytes(b" " * (adapter.MAX_SOURCE_MANIFEST + 1))
    base = adapter.load_base()
    with pytest.raises(ValueError):
        base._sha(path, base.MAX_JSON)
    with pytest.raises(ValueError):
        base._read(path)


def test_other_metadata_keeps_two_mib_limit_without_canonical_payload(adapter, tmp_path):
    path = tmp_path / "status.json"
    path.write_bytes(b" " * (2 * 1024**2 + 1))
    base = adapter.load_base()
    with pytest.raises(ValueError):
        base._read(path)
    with pytest.raises(ValueError):
        base._sha(path, base.MAX_JSON)


def test_shared_manifest_is_rejected(adapter, tmp_path):
    path = tmp_path / "source-manifest.json"
    path.write_bytes(b'{"splits":{}}')
    alias = tmp_path / "manifest-alias.json"
    try:
        alias.hardlink_to(path)
    except OSError:
        pytest.skip("Hard links are unavailable")
    base = adapter.load_base()
    with pytest.raises(ValueError):
        base._read(path)
    with pytest.raises(ValueError):
        base._sha(path, base.MAX_JSON)


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'])
def test_manifest_parser_keeps_duplicate_finite_and_object_checks(adapter, tmp_path, data):
    path = tmp_path / "source-manifest.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        adapter.load_base()._read(path)


def test_changed_base_is_rejected_before_execution(adapter, tmp_path):
    path = tmp_path / "queue_cleanup.py"
    path.write_bytes(path.read_bytes() + b"\nraise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.load_base()


def test_cli_retains_fixed_private_root_restriction(adapter):
    with pytest.raises(ValueError, match="fixed private"):
        adapter.load_base().main(["--root", "/unapproved", "--job-id", "a" * 64, "--execute"])
