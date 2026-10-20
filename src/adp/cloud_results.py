"""Verify and atomically import a standalone cloud-pilot result ZIP locally."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile

from adp.compute_cleanup import _hash, _links, _safe, _sha
from adp.remote_transfer import (MAX_FILE_BYTES, MAX_JSON_BYTES, MAX_TOTAL_BYTES,
    _checkpoint, _identifier, _json, _json_bytes, _space, _sync_directory, _write_new)


PURPOSE = "standalone cloud infrastructure pilot, not frozen-study evidence"
FILES = {"job.json", "config.json", "compute-ledger.json", "status.json", "environment.json",
    "qualification.json", "package-files.json", "initial-config.json", "cache/manifest.json",
    "cache/preprocessing.json", "run/last.pt", "run/best.pt", "run/history.json"}


def _headers(archive, *, payload=False):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(entries) > (2000 if payload else len(FILES) + 1) or len({name.casefold() for name in names}) != len(names):
        raise ValueError("Duplicate or excessive cloud archive members")
    if sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES:
        raise ValueError("Cloud archive exceeds its expanded size limit")
    for entry in entries:
        name, relative = entry.filename, PurePosixPath(entry.filename)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or relative.as_posix() != name or entry.is_dir() or mode not in (0, stat.S_IFREG)):
            raise ValueError("Unsafe cloud archive path or nonregular file")
        if payload:
            parts = relative.parts
            allowed = name in {"package-files.json", "initial-config.json", "source-manifest.json",
                               "cache/manifest.json", "cache/preprocessing.json"}
            allowed |= len(parts) == 3 and parts[:2] == ("source", "adp") and name.endswith(".py")
            allowed |= len(parts) == 3 and parts[0] == "cache" and parts[1] in {"train", "validation"} and name.endswith(".npz")
        else:
            allowed = name in FILES | {"export-manifest.json"}
        bound = MAX_FILE_BYTES if name.endswith(".pt") else MAX_JSON_BYTES
        if not allowed or not 0 < entry.file_size <= bound:
            raise ValueError("Unexpected cloud archive member or excessive file size")
    return set(names)


def _payload(path, expected_hash):
    path = _safe(path)
    if path.stat().st_size > 300 * 1024**2 or _sha(path) != _identifier(expected_hash):
        raise ValueError("Original input ZIP differs from its independently pinned payload SHA-256")
    with zipfile.ZipFile(path) as archive:
        names = _headers(archive, payload=True)
        required = {"package-files.json", "initial-config.json", "source-manifest.json", "cache/manifest.json", "cache/preprocessing.json"}
        if not required <= names:
            raise ValueError("Pinned input ZIP is missing its provenance metadata")
        metadata = {name: archive.read(name) for name in required}
        index = _json(metadata["package-files.json"])
        if set(index) != names - {"package-files.json"}:
            raise ValueError("Pinned input ZIP checksum index differs from its members")
        for name, entry in index.items():
            if (not isinstance(entry, dict) or set(entry) != {"sha256", "size"}
                    or entry["size"] != archive.getinfo(name).file_size):
                raise ValueError("Invalid pinned input file index")
            _identifier(entry["sha256"])
            if name in metadata and hashlib.sha256(metadata[name]).hexdigest() != entry["sha256"]:
                raise ValueError("Pinned input metadata differs from its checksum index")
    if _sha(path) != expected_hash:
        raise ValueError("Original payload changed during verification")
    return metadata, index


def _provenance(data, payload, input_index):
    for name, content in data.items():
        if name.endswith(".json"):
            _json(content)
    for name in ("package-files.json", "initial-config.json", "cache/manifest.json", "cache/preprocessing.json"):
        if data[name] != payload[name]:
            raise ValueError(f"Cloud result differs from the pinned original input: {name}")
    job, config = _json(data["job.json"]), _json(data["config.json"])
    protocol_hash = _identifier(job.get("protocol_sha256"))
    if (job.get("version") != 1 or _hash({key: value for key, value in job.items() if key != "protocol_sha256"}) != protocol_hash
            or job.get("config_sha256") != _hash(config)
            or job.get("purpose") != "development qualification; not frozen-study benchmark evidence"
            or job.get("epochs") != 1):
        raise ValueError("Cloud job protocol or configuration is inconsistent")
    expected_config = _json(payload["initial-config.json"])
    expected_config.setdefault("training", {}).update(batch_size=1, accumulation_steps=32, amp=False,
        num_workers=0, checkpoint_seconds=300, minimum_free_gib=8)
    source_files = {name.removeprefix("source/adp/"): entry["sha256"] for name, entry in input_index.items() if name.startswith("source/adp/")}
    cache_files = {name.removeprefix("cache/"): entry["sha256"] for name, entry in input_index.items() if name.startswith("cache/")}
    if (config != expected_config or job.get("source_files") != source_files or job.get("cache_files") != cache_files
            or job.get("source_manifest_sha256") != input_index["source-manifest.json"]["sha256"]):
        raise ValueError("Cloud job source, configuration or cache differs from the pinned input ZIP")
    manifest, original = _json(data["cache/manifest.json"]), _json(payload["source-manifest.json"])
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("Cloud result must contain only training and selection scenario membership")
    all_ids = []
    for split, ids in splits.items():
        if (not isinstance(ids, list) or not ids or any(not isinstance(sid, str) or str(uuid.UUID(sid)) != sid for sid in ids)
                or not set(ids) <= set(original["splits"][split])
                or job.get("split_counts", {}).get(split) != len(ids) or job.get("split_sha256", {}).get(split) != _hash(ids)):
            raise ValueError("Cloud split membership differs from its frozen development protocol")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Cloud training and selection scenarios overlap")
    status, ledger, qualification = (_json(data[name]) for name in ("status.json", "compute-ledger.json", "qualification.json"))
    if (status.get("state") != "completed" or status.get("protocol_sha256") != protocol_hash
            or status.get("result", {}).get("completed_epochs") != job.get("epochs")
            or ledger.get("protocol_sha256") != protocol_hash
            or qualification.get("protocol_sha256") != protocol_hash or qualification.get("status") != "passed"
            or qualification.get("cpu_gpu_agreement") != "passed"):
        raise ValueError("Cloud training, ledger or hardware qualification is incomplete or inconsistent")
    charged = ledger.get("charged_seconds")
    if isinstance(charged, bool) or not isinstance(charged, (int, float)) or not math.isfinite(charged) or charged < 0:
        raise ValueError("Cloud ledger compute charge is invalid")
    slices = ledger.get("slices")
    if not isinstance(slices, list) or not slices or any(
            not isinstance(entry, dict) or isinstance(entry.get("charged_seconds"), bool)
            or not isinstance(entry.get("charged_seconds"), (int, float))
            or not math.isfinite(entry["charged_seconds"]) or entry["charged_seconds"] < 0 for entry in slices):
        raise ValueError("Cloud ledger slice charges are invalid")
    if not math.isclose(sum(entry["charged_seconds"] for entry in slices), charged, abs_tol=1e-6):
        raise ValueError("Cloud ledger total differs from its recorded qualification/training slices")
    checkpoints = {name: _checkpoint(data[f"run/{name}"]) for name in ("last.pt", "best.pt")}
    for saved in checkpoints.values():
        if (saved["config"] != config or saved["manifest_hash"] != cache_files["manifest.json"]
                or saved["preprocessing_hash"] != cache_files["preprocessing.json"]):
            raise ValueError("Cloud checkpoint differs from its frozen training provenance")
    if checkpoints["last.pt"]["completed_epochs"] != job.get("epochs") or checkpoints["last.pt"]["partial_epoch"]:
        raise ValueError("Cloud last.pt does not contain the completed final epoch")
    return {"job_protocol_sha256": protocol_hash, "split_counts": {key: len(value) for key, value in splits.items()},
            "charged_seconds": float(charged), "checkpoint_metadata": checkpoints}


def import_cloud_results(archive, receipt, output_root, *, payload_archive, expected_payload_sha256):
    """Keep all downloaded audit data; publish only independently verified CPU-loadable results."""
    archive = _safe(archive)
    if not archive.is_file() or archive.stat().st_size > MAX_TOTAL_BYTES + MAX_JSON_BYTES:
        raise ValueError("Cloud results ZIP is missing or exceeds the compressed size limit")
    if isinstance(receipt, (str, Path)):
        receipt_path = _safe(receipt)
        if receipt_path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Cloud result receipt is too large")
        receipt = _json(receipt_path.read_bytes())
    required_receipt = {"version", "purpose", "payload_sha256", "files", "archive_sha256"}
    if (not isinstance(receipt, dict) or not required_receipt <= set(receipt)
            or set(receipt) - required_receipt - {"input_verification"}
            or receipt.get("version") != 1 or receipt.get("purpose") != PURPOSE
            or receipt.get("payload_sha256") != _identifier(expected_payload_sha256)):
        raise ValueError("Cloud receipt does not identify the expected standalone pilot payload")
    archive_hash = _identifier(receipt.get("archive_sha256"))
    if _sha(archive) != archive_hash:
        raise ValueError("Cloud ZIP archive SHA-256 differs from the downloaded receipt")
    payload, input_index = _payload(payload_archive, expected_payload_sha256)
    if "input_verification" in receipt:
        verification = receipt["input_verification"]
        if (not isinstance(verification, dict)
                or set(verification) != {"kind", "package_index_sha256", "verified_member_count"}
                or verification["kind"] not in {"verified_indexed_extracted_members", "verified_payload_zip"}
                or verification["package_index_sha256"] != hashlib.sha256(payload["package-files.json"]).hexdigest()
                or verification["verified_member_count"] != len(input_index)):
            raise ValueError("Cloud extracted-input verification differs from the independently pinned input index")
    with zipfile.ZipFile(archive) as zipped:
        if _headers(zipped) != FILES | {"export-manifest.json"}:
            raise ValueError("Cloud result ZIP is missing required audit files or checkpoints")
        internal = _json(zipped.read("export-manifest.json"))
        if internal != {key: value for key, value in receipt.items() if key != "archive_sha256"} or set(internal.get("files", {})) != FILES:
            raise ValueError("Cloud external receipt and internal export manifest differ")
        data = {}
        for name, entry in internal["files"].items():
            if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
                raise ValueError("Cloud export checksum index is invalid")
            content = zipped.read(name)
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Cloud exported member SHA-256 or size differs: {name}")
            data[name] = content
        data["export-manifest.json"] = zipped.read("export-manifest.json")
    provenance = _provenance(data, payload, input_index)
    if _sha(archive) != archive_hash:
        raise ValueError("Cloud archive changed during validation")
    root = _safe(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = _safe(root / archive_hash, root)
    local_receipt = {"version": 1, "verified": True, "archive_sha256": archive_hash,
        "payload_sha256": expected_payload_sha256, "checkpoint_validation": "strict_cpu_observed_only",
        "verified_at": datetime.now(timezone.utc).isoformat(), **provenance}
    if "input_verification" in receipt:
        local_receipt["input_verification"] = receipt["input_verification"]
    data["download-receipt.json"] = _json_bytes(receipt)
    if destination.exists():
        actual = set()
        for path in destination.rglob("*"):
            path = _safe(path, destination)
            relative = path.relative_to(destination).as_posix()
            if path.is_file():
                actual.add(relative)
            elif not path.is_dir() or relative not in {"cache", "run"}:
                raise ValueError("Existing local cloud result contains unexpected directories")
        if actual != set(data) | {"import-receipt.json"}:
            raise ValueError("Existing local cloud result contains unexpected or missing files")
        for name, content in data.items():
            path = _safe(destination / name, destination)
            if _sha(path) != hashlib.sha256(content).hexdigest():
                raise ValueError("Existing local cloud result bytes changed")
        installed = _json((destination / "import-receipt.json").read_bytes())
        if {key: value for key, value in installed.items() if key != "verified_at"} != {key: value for key, value in local_receipt.items() if key != "verified_at"}:
            raise ValueError("Existing local cloud import receipt differs")
        return {"path": str(destination), "receipt": installed, "already_imported": True}
    data["import-receipt.json"] = _json_bytes(local_receipt)
    _space(root, sum(len(content) for content in data.values()) + MAX_JSON_BYTES)
    stage = Path(tempfile.mkdtemp(prefix=".cloud-import-", dir=root))
    try:
        for name, content in data.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_new(target, content)
            if _sha(target) != hashlib.sha256(content).hexdigest():
                raise OSError("Local cloud result failed verification after writing")
        for directory in (stage / "run", stage / "cache", stage):
            _sync_directory(directory)
        stage.rename(destination)
        _sync_directory(root)
        return {"path": str(destination), "receipt": local_receipt, "already_imported": False}
    finally:
        if stage.exists() and not _links(stage) and stage.resolve().parent == root and stage.name.startswith(".cloud-import-"):
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--payload-archive", type=Path, required=True)
    parser.add_argument("--expected-payload-sha256", required=True)
    args = parser.parse_args(argv)
    result = import_cloud_results(args.archive, args.receipt, args.output_root,
        payload_archive=args.payload_archive, expected_payload_sha256=args.expected_payload_sha256)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
