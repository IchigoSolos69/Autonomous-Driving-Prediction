"""Resume one existing study using its frozen provenance, without editing its source or protocol."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import sys


@contextmanager
def _original_modules(project_root):
    source = project_root / "src"
    package = source / "adp"
    if not (package / "__init__.py").is_file():
        raise ValueError("--project-root must contain the original src/adp package")
    for name, module in tuple(sys.modules.items()):
        if name == "adp" or name.startswith("adp."):
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(package):
                raise RuntimeError("adp is already loaded from another workspace; use a fresh Python process")
    sys.path.insert(0, str(source))
    try:
        modules = [importlib.import_module("adp." + name) for name in ("data", "protocol", "research")]
        for module in modules:
            expected = package / (module.__name__.rsplit(".", 1)[1] + ".py")
            if Path(module.__file__).resolve() != expected:
                raise RuntimeError("The resume helper did not load the explicitly selected original source")
        yield modules
    finally:
        sys.path.remove(str(source))


def _validated_protocol(root, data, protocol, original_create):
    path, cache = root / "protocol.json", root / "cache"
    if not path.is_file():
        raise ValueError("An existing frozen protocol.json is required; this helper cannot create a new benchmark")
    digest, frozen = data.manifest_hash(path), data.read_json(path)
    snapshots = frozen.get("provenance_manifests")
    if frozen.get("version") != 1 or not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("The frozen protocol lacks versioned provenance snapshots")
    cache_hash = data.manifest_hash(cache / "manifest.json")
    if frozen.get("dataset_manifest_sha256") != cache_hash or cache_hash not in snapshots:
        raise ValueError("The cache manifest no longer matches the frozen protocol")
    manifest = data.read_json(cache / "manifest.json")
    if snapshots[cache_hash] != manifest or data.manifest_hash(root / "raw" / "manifest.json") != cache_hash:
        raise ValueError("Raw/cache manifests differ from the exact frozen dataset snapshot")
    if frozen.get("split_hashes") != {split: protocol.ids_hash(ids) for split, ids in manifest["splits"].items()}:
        raise ValueError("Frozen split hashes differ from the cache manifest")
    paths = []
    for expected, value in snapshots.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or not isinstance(value, dict) or "splits" not in value:
            raise ValueError("Invalid frozen provenance snapshot")
        snapshot = cache / "manifest.json" if expected == cache_hash else root / "provenance" / f"{expected}.json"
        if not snapshot.is_file() or data.manifest_hash(snapshot) != expected or data.read_json(snapshot) != value:
            raise ValueError(f"Frozen provenance snapshot is missing or changed: {expected}")
        if expected != cache_hash:
            paths.append(snapshot)
    # The original implementation performs the complete equality check, including its policy.
    result = original_create(cache, path, paths)
    if data.manifest_hash(path) != digest:
        raise ValueError("The existing benchmark protocol changed during validation")
    return frozen, digest, paths, result


def _additional(paths, recorded_hashes, root, data, fingerprint):
    rows = []
    for path in sorted(map(Path, paths), key=str):
        digest = data.manifest_hash(path)
        if digest not in recorded_hashes:
            name = str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root) else str(path.resolve())
            rows.append({"path": name, "sha256": digest})
    return {"count": len(rows), "sha256": fingerprint(rows), "files": rows}


def resume_frozen_study(study_root, project_root, config=None, reference_checkpoint=None, device="cuda", workers=8):
    root, project = Path(study_root).resolve(), Path(project_root).resolve()
    config = Path(config).resolve() if config else project / "config.json"
    reference = Path(reference_checkpoint).resolve() if reference_checkpoint else project / "artifacts" / "experiment" / "best.pt"
    if not root.is_dir() or not config.is_file() or not reference.is_file():
        raise ValueError("Study root, original configuration and reference checkpoint must already exist")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    with _original_modules(project) as (data, protocol, research):
        original_create = protocol.create_protocol
        frozen, protocol_hash, snapshots, _ = _validated_protocol(root, data, protocol, original_create)
        source = {path.name: data.manifest_hash(path) for path in sorted((project / "src" / "adp").glob("*.py"))}
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{os.getpid()}.json"
        receipt_path = root / "resume-receipts" / name
        receipt = {"version": 1, "status": "running", "purpose": "Resume existing frozen provenance only",
            "study_root": str(root), "project_root": str(project), "helper_sha256": data.manifest_hash(__file__),
            "original_source_sha256": research.config_fingerprint(source), "original_source_files": source,
            "protocol_sha256": protocol_hash, "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
            "recorded_snapshot_sha256": sorted(frozen["provenance_manifests"]),
            "config_sha256": data.manifest_hash(config), "reference_checkpoint_sha256": data.manifest_hash(reference),
            "device": device, "workers": workers, "protocol_calls": []}
        receipt["additional_at_start"] = _additional((root / "provenance").glob("*.json"),
            frozen["provenance_manifests"], root, data, research.config_fingerprint)
        research.ensure_space(root, additional_bytes=1024**2)
        data.write_json(receipt_path, receipt)
        target = (root / "protocol.json").resolve()
        def frozen_create(cache_dir, output, provenance_manifests=()):
            if Path(output).resolve() != target:
                return original_create(cache_dir, output, provenance_manifests)
            if Path(cache_dir).resolve() != (root / "cache").resolve() or data.manifest_hash(target) != protocol_hash:
                raise ValueError("Frozen resume received different cache or protocol inputs")
            _, current_hash, _, result = _validated_protocol(root, data, protocol, original_create)
            if current_hash != protocol_hash:
                raise ValueError("The original protocol changed while resuming")
            extra = _additional(provenance_manifests, frozen["provenance_manifests"], root, data, research.config_fingerprint)
            receipt["protocol_calls"].append({"verified_snapshot_count": len(snapshots) + 1, "ignored_additional": extra})
            data.write_json(receipt_path, receipt)
            print(f"Frozen resume: ignored {extra['count']} additional manifests; list SHA256 {extra['sha256']}", flush=True)
            return result
        protocol.create_protocol = frozen_create
        print(f"Frozen resume receipt: {receipt_path}", flush=True)
        try:
            result = research.run_pipeline(root, project, data.read_json(config), reference, device, workers)
            receipt.update(status="completed", promotion_gate_passed=result.get("promotion_gate_passed"))
            return {"status": "completed", "receipt": str(receipt_path), "protocol_sha256": protocol_hash,
                    "promotion_gate_passed": result.get("promotion_gate_passed")}
        except BaseException as error:
            receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "stopped",
                           error=str(error) or type(error).__name__)
            raise
        finally:
            protocol.create_protocol = original_create
            receipt["protocol_sha256_after"] = data.manifest_hash(target) if target.exists() else None
            after = {path.name: data.manifest_hash(path) for path in sorted((project / "src" / "adp").glob("*.py"))}
            receipt["original_source_sha256_after"] = research.config_fingerprint(after)
            receipt["protocol_unchanged"] = receipt["protocol_sha256_after"] == protocol_hash
            receipt["original_source_unchanged"] = after == source
            data.write_json(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        result = resume_frozen_study(args.study_root, args.project_root, args.config,
                                    args.reference_checkpoint, args.device, args.workers)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Frozen resume: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
