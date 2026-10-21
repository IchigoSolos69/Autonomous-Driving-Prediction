"""Import completed finite-queue results only after independent provenance checks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import stat
import tempfile
import uuid
import zipfile

from adp.cloud_results import _payload
from adp.compute_cleanup import _hash, _links, _safe, _write
from adp.compute_queue import (CONFIGURATIONS, HOST_SECONDS, PAYLOAD_SHA256, SEEDS,
    job_spec as approved_job_spec, spec_hash, validate_spec)
from adp.remote_transfer import (FILES, MAX_FILE_BYTES, MAX_JSON_BYTES, MAX_TOTAL_BYTES,
    _identifier, _json, _space, _write_new, import_snapshot, validate_snapshot)


def expected_configuration(payload_metadata, specification):
    config = _json(payload_metadata["initial-config.json"])
    config["seed"] = specification["seed"]
    config.setdefault("model", {}).update(use_graph=specification["configuration"] != "no_graph",
        use_map=specification["configuration"] != "no_map",
        shared_world_ce_weight=.1 if specification["configuration"] == "world_ce" else 0.)
    config.setdefault("training", {}).update(batch_size=1, accumulation_steps=32, amp=False,
        num_workers=0, checkpoint_seconds=300, minimum_free_gib=8)
    return config


def queue_protocol_hash(host):
    """Identity of the complete approved six-GPU-hour host plan."""
    return spec_hash({"version": 1, "host_seconds": HOST_SECONDS,
        "jobs": [approved_job_spec(host, configuration, seed)
                 for configuration in CONFIGURATIONS for seed in SEEDS]})


def _verify(directory, loaded, specification, payload, index, *, protocol_hash=None):
    metadata = loaded["manifest"]
    provenance = metadata["provenance"]
    identifier = spec_hash(specification)
    device_count = 2 if specification["host"] == "kaggle" else 1
    if (provenance.get("queue_job_spec") != specification or provenance.get("queue_job_id") != identifier
            or provenance.get("payload_sha256") != PAYLOAD_SHA256 or metadata.get("run_id") != identifier
            or provenance.get("queue_protocol_sha256") != (protocol_hash or queue_protocol_hash(specification["host"]))
            or type(provenance.get("allocated_device_count")) is not int
            or provenance["allocated_device_count"] != device_count):
        raise ValueError("Queue result differs from its approved immutable job specification")
    config, job, status = (_json((directory / name).read_bytes()) for name in ("config.json", "job.json", "result.json"))
    expected = expected_configuration(payload, specification)
    source = {name.removeprefix("source/adp/"): entry["sha256"] for name, entry in index.items() if name.startswith("source/adp/")}
    cache = {name.removeprefix("cache/"): entry["sha256"] for name, entry in index.items() if name.startswith("cache/")}
    protocol = _identifier(job.get("protocol_sha256"))
    if (type(job.get("version")) is not int or job["version"] != 1
            or job.get("purpose") != "development qualification; not frozen-study benchmark evidence"
            or config != expected or job.get("config_sha256") != _hash(config)
            or _hash({key: value for key, value in job.items() if key != "protocol_sha256"}) != protocol
            or job.get("epochs") != specification["epochs"]
            or job.get("maximum_compute_seconds") != specification["maximum_seconds"] / device_count
            or job.get("source_files") != source or job.get("cache_files") != cache
            or job.get("source_manifest_sha256") != index["source-manifest.json"]["sha256"]):
        raise ValueError("Queue model, source or data differs from the independently pinned payload")
    expected_provenance = {"job_protocol_sha256": protocol, "manifest_sha256": cache["manifest.json"],
                          "preprocessing_sha256": cache["preprocessing.json"], "config_sha256": _hash(config)}
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Queue snapshot provenance differs from its job protocol")
    manifest, original = (_json(payload[name]) for name in ("cache/manifest.json", "source-manifest.json"))
    if set(manifest.get("splits", {})) != {"train", "validation"}:
        raise ValueError("Queue input must be independent training and selection only")
    all_ids = []
    for split, count in (("train", specification["train_count"]), ("validation", specification["selection_count"])):
        ids = manifest["splits"][split]
        if (not isinstance(ids, list) or len(ids) != count
                or any(not isinstance(sid, str) or str(uuid.UUID(sid)) != sid for sid in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= set(original.get("splits", {}).get(split, []))
                or job.get("split_counts", {}).get(split) != count
                or job.get("split_sha256", {}).get(split) != _hash(ids)):
            raise ValueError("Queue split counts or identities differ from the approved recipe")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Queue training and selection scenarios overlap")
    if (status.get("state") != "completed" or status.get("protocol_sha256") != protocol
            or not isinstance(status.get("result"), dict)
            or status["result"].get("completed_epochs") != specification["epochs"]):
        raise ValueError("Partial or incomplete queue results cannot be acknowledged as completed")
    checkpoints = loaded["checkpoint_metadata"]
    if not {"last.pt", "best.pt"} <= set(checkpoints):
        raise ValueError("Queue result requires both final and best checkpoints")
    for saved in checkpoints.values():
        if (saved["config"] != config or saved["manifest_hash"] != cache["manifest.json"]
                or saved["preprocessing_hash"] != cache["preprocessing.json"]
                or type(saved["completed_epochs"]) is not int or not 1 <= saved["completed_epochs"] <= specification["epochs"]
                or type(saved["global_step"]) is not int or saved["global_step"] < 1
                or saved["partial_epoch"] is not False):
            raise ValueError("Queue checkpoint configuration or dataset provenance differs")
    if checkpoints["last.pt"]["completed_epochs"] != specification["epochs"] or checkpoints["last.pt"]["partial_epoch"]:
        raise ValueError("Queue final checkpoint does not complete its fixed epoch schedule")


def import_queue_result(archive, output_root, *, payload_archive, job_spec, expected_archive_sha256):
    """Preserve ZIP/source data; durable verified receipt precedes remote acknowledgement."""
    return _import_result(archive, output_root, payload_archive=payload_archive,
        specification=validate_spec(job_spec), expected_archive_sha256=expected_archive_sha256,
        protocol_hash=queue_protocol_hash(job_spec["host"]))


def import_distributed_result(archive, output_root, *, payload_archive, job_spec, expected_archive_sha256):
    from adp.distributed_plan import validate_spec as validate_distributed, protocol_hash
    specification = validate_distributed(job_spec)
    return _import_result(archive, output_root, payload_archive=payload_archive,
        specification=specification, expected_archive_sha256=expected_archive_sha256,
        protocol_hash=protocol_hash(specification["host"]))


def _import_result(archive, output_root, *, payload_archive, specification, expected_archive_sha256, protocol_hash):
    archive, root = _safe(archive), _safe(output_root)
    digest = _identifier(expected_archive_sha256)
    if not archive.is_file() or archive.stat().st_size > MAX_TOTAL_BYTES + MAX_JSON_BYTES:
        raise ValueError("Queue ZIP is missing or too large")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Queue ZIP differs from the authenticated export hash")
    payload, index = _payload(payload_archive, PAYLOAD_SHA256)
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        names = [entry.filename for entry in entries]
        if (len(entries) > len(FILES) + 1 or len({name.casefold() for name in names}) != len(names)
                or sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES + MAX_JSON_BYTES
                or not {"snapshot.json", "last.pt", "best.pt", "job.json", "config.json", "result.json"} <= set(names)):
            raise ValueError("Invalid or incomplete queue ZIP members")
        for entry in entries:
            limit = MAX_JSON_BYTES if entry.filename.endswith(".json") else MAX_FILE_BYTES
            if (entry.filename not in FILES | {"snapshot.json"} or entry.is_dir()
                    or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)
                    or not 0 < entry.file_size <= limit):
                raise ValueError("Unsafe or excessive queue ZIP member")
        root.mkdir(parents=True, exist_ok=True)
        _space(root, 2 * sum(entry.file_size for entry in entries) + MAX_JSON_BYTES)
        stage = Path(tempfile.mkdtemp(prefix=".queue-receive-", dir=root))
        try:
            for entry in entries:
                _write_new(stage / entry.filename, zipped.read(entry))
            loaded = validate_snapshot(stage)
            _verify(stage, loaded, specification, payload, index, protocol_hash=protocol_hash)
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Queue ZIP changed during verification")
            installed = import_snapshot(stage, root)
            receipt = {"version": 1, "verified": True, "job_id": spec_hash(specification),
                "job_spec": specification, "snapshot_id": installed["snapshot_id"],
                "archive_sha256": digest, "payload_sha256": PAYLOAD_SHA256,
                "queue_protocol_sha256": protocol_hash,
                "allocated_device_count": loaded["manifest"]["provenance"]["allocated_device_count"],
                "manifest_sha256": loaded["manifest_sha256"], "files": loaded["manifest"]["files"],
                "checkpoint_validation": "strict_cpu_observed_only_completed",
                "verified_at": datetime.now(timezone.utc).isoformat(), "local_directory": installed["path"]}
            receipts = _safe(root / "queue-receipts", root)
            receipts.mkdir(exist_ok=True)
            receipt_path = _safe(receipts / (receipt["job_id"] + "-" + receipt["snapshot_id"] + ".json"), receipts)
            if receipt_path.exists():
                previous = _json(receipt_path.read_bytes())
                if {key: value for key, value in previous.items() if key != "verified_at"} != {key: value for key, value in receipt.items() if key != "verified_at"}:
                    raise ValueError("Existing queue receipt is corrupt; it was preserved")
                receipt = previous
            else:
                _write(receipt_path, receipt)
            return {**installed, "queue_receipt": receipt, "queue_receipt_path": str(receipt_path)}
        finally:
            if stage.exists() and not _links(stage) and stage.resolve().parent == root and stage.name.startswith(".queue-receive-"):
                shutil.rmtree(stage)
