"""A single CLI shared by native Windows and free GPU notebooks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _evaluate(args, stress=False):
    import torch
    from adp.data import CachedDataset, collate_scenes, manifest_hash, preprocessing_hash, read_json
    from adp.metrics import evaluate_predictions, summarize_records
    from adp.model import ForecastModel, constant_velocity, model_inputs
    from adp.reporting import write_report
    from adp.stress import StressEligibilityError, apply_stress, validate_stress_spec
    from adp.training import load_checkpoint, resolve_device
    stress_spec = None
    if stress:
        kind, severity, seed = validate_stress_spec({"kind": args.kind, "severity": args.severity, "seed": args.seed})
        stress_spec = {"kind": kind, "severity": severity, "seed": seed}
    device = resolve_device(args.device)
    dataset = CachedDataset(args.cache_dir, args.split)
    calibration = read_json(args.calibration) if args.calibration else None
    network, saved = None, None
    protocol_path = getattr(args, "protocol", None)
    if args.checkpoint:
        saved = load_checkpoint(args.checkpoint)
        if not protocol_path and saved["manifest_hash"] != manifest_hash(Path(args.cache_dir)/"manifest.json"):
            raise ValueError("Checkpoint/cache manifests differ")
        if "preprocessing_hash" in saved and saved["preprocessing_hash"] != preprocessing_hash(args.cache_dir):
            raise ValueError("Checkpoint/cache preprocessing differs")
        network = ForecastModel(saved["config"].get("model", saved["config"]))
        network.load_state_dict(saved["model_state_dict"])
        network.to(device).eval()
        if calibration and calibration.get("checkpoint_sha256") != hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest():
            raise ValueError("Calibration/checkpoint identifiers differ")
    elif calibration:
        raise ValueError("Calibration requires its matching trained checkpoint")
    protocol_evidence = None
    if protocol_path:
        from adp.protocol import validate_benchmark
        protocol_evidence = validate_benchmark(saved, args.cache_dir, protocol_path, args.split, calibration)
    cv_scale = float(getattr(args, "cv_scale", 1.0))
    if not .15 <= cv_scale <= 20:
        raise ValueError("Constant-velocity scale must be in [0.15,20] metres")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    total = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    records, clean_records, skipped = [], [], []
    effective_calibration = None if stress else calibration
    for index in range(total):
        if getattr(args, "budget_check", None):
            args.budget_check()
        scene = dataset[index]
        clean_scene = scene
        if stress:
            try:
                scene = apply_stress(scene, stress_spec)
            except StressEligibilityError as error:
                skipped.append({"scenario_id": scene["id"], "reason": str(error)})
                continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_scenes([scene]).items()}
        with torch.inference_mode():
            outputs = network(**model_inputs(batch)) if network else constant_velocity(model_inputs(batch))
            if network is None:
                outputs["scales"] = torch.full_like(outputs["means"], cv_scale)
        scene_records = evaluate_predictions(outputs, batch, effective_calibration)
        for record in scene_records:
            record["city"] = scene.get("metadata", {}).get("city")
        records.extend(scene_records)
        if stress:
            clean_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_scenes([clean_scene]).items()}
            with torch.inference_mode():
                clean_outputs = network(**model_inputs(clean_batch)) if network else constant_velocity(model_inputs(clean_batch))
                if network is None:
                    clean_outputs["scales"] = torch.full_like(clean_outputs["means"], cv_scale)
            clean_records.extend(evaluate_predictions(clean_outputs, clean_batch))
        if (index+1) % 10 == 0:
            print(f"Evaluated {index+1}/{total} scenarios", flush=True)
    report = summarize_records(records, bootstrap=args.bootstrap, seed=args.seed, calibration=effective_calibration)
    report["risk"]["raw_source"] = "trained_risk_head" if network else "constant_velocity_heuristic"
    report.update({"split": args.split, "model": "trained" if network else "constant_velocity",
                   "manifest_hash": dataset.manifest_hash, "calibration_applied": effective_calibration is not None,
                   "benchmark_provenance": protocol_evidence,
                   "benchmark_protocol_sha256": manifest_hash(protocol_path) if protocol_path else None,
                   "calibration_sha256": manifest_hash(args.calibration) if args.calibration else None,
                   "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
                   "constant_velocity_scale_m": cv_scale if network is None else None,
                   "available_scenarios": len(dataset), "scenario_limit": args.limit,
                   "scenario_selection": "entire split" if args.limit is None else "first N scenarios in immutable manifest order",
                   "protocol": "retained complete scored dynamic actors; joint modes; miss threshold 2m; six modes",
                   "risk_definition": "minimum centre distance <3m to another retained dynamic actor over6s",
                   "footprints_m": {"vehicle": [4.5, 1.8], "pedestrian": [.6, .6], "motorcyclist": [2, .8], "cyclist": [1.8, .6], "bus": [12, 2.5]},
                   "stress": {**stress_spec, "other_agents_reactive": False} if stress else None,
                   "skipped": skipped})
    if stress:
        from adp.protocol import paired_comparison
        report["calibration_applicability"] = "unknown_under_stress; raw probabilities used"
        report["matched_clean"] = summarize_records(clean_records, bootstrap=0, seed=args.seed)
        report["paired_change"] = paired_comparison(records, clean_records, max(1, args.bootstrap), args.seed) if records else None
        from adp.data import write_json
        write_json(Path(args.output_dir) / "clean_records.json", clean_records)
    write_report(args.output_dir, report, records)
    return report


def main():
    parser = argparse.ArgumentParser(description="AV2 joint forecasting and scenario inspection")
    sub = parser.add_subparsers(dest="command", required=True)
    from adp.desktop import add_arguments
    add_arguments(sub.add_parser("desktop", help="Open the native Windows application"))
    doctor = sub.add_parser("doctor", help="Inspect runtime and perform a real GPU forward/backward structural check")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--config", default="config.json")
    download = sub.add_parser("download", help="Download an immutable real AV2 scenario subset")
    download.add_argument("--raw-dir", default="data/raw")
    download.add_argument("--train-count", type=int, default=5000)
    download.add_argument("--val-count", type=int, default=2000)
    download.add_argument("--seed", type=int, default=42)
    download.add_argument("--workers", type=int, default=4)
    prep = sub.add_parser("preprocess")
    prep.add_argument("--raw-dir", default="data/raw")
    prep.add_argument("--cache-dir", default="data/cache")
    prep.add_argument("--manifest", default=None)
    prep.add_argument("--max-actors", type=int, default=32)
    prep.add_argument("--max-polylines", type=int, default=128)
    prep.add_argument("--map-radius", type=float, default=100)
    prep.add_argument("--workers", type=int, default=4)
    prep.add_argument("--minimum-free-gib", type=float, default=0)
    train = sub.add_parser("train")
    train.add_argument("--config", default="config.json")
    train.add_argument("--cache-dir", default="data/cache")
    train.add_argument("--output-dir", default="artifacts/run")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume")
    train.add_argument("--overfit-batches", type=int)
    train.add_argument("--max-runtime-seconds", type=float)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--cache-dir", default="data/cache")
    calibrate.add_argument("--output-dir", default="artifacts/run")
    calibrate.add_argument("--device", default="auto")
    for name in ("evaluate", "stress-test"):
        evaluate = sub.add_parser(name)
        evaluate.add_argument("--checkpoint")
        evaluate.add_argument("--calibration")
        evaluate.add_argument("--cache-dir", default="data/cache")
        evaluate.add_argument("--output-dir", default=f"artifacts/{name}")
        evaluate.add_argument("--split", choices=["validation", "calibration", "evaluation"], default="evaluation")
        evaluate.add_argument("--device", default="auto")
        evaluate.add_argument("--bootstrap", type=int, default=200)
        evaluate.add_argument("--seed", type=int, default=42)
        evaluate.add_argument("--limit", type=int, help="Bound a smoke/stress run; reports explicitly record the truncated population")
        evaluate.add_argument("--protocol", help="Frozen benchmark protocol allowing explicitly proven external checkpoints")
        evaluate.add_argument("--cv-scale", type=float, default=1.0)
        if name == "stress-test":
            evaluate.add_argument("--kind", required=True, choices=["position_noise", "dropout", "braking", "cut_in"])
            evaluate.add_argument("--severity", type=float, default=.35)
    study = sub.add_parser("study", help="Prepare, pilot, acquire, train, or finalize the bounded production study")
    study.add_argument("phase", choices=["prepare", "pilot", "acquire", "train", "finalize", "pipeline"])
    study.add_argument("--root", type=Path, default=Path("D:/InteractionLab/study-v1"))
    study.add_argument("--project-root", type=Path, default=Path.cwd())
    study.add_argument("--config", default="config.json")
    study.add_argument("--device", default="cuda")
    study.add_argument("--workers", type=int, default=4)
    study.add_argument("--reference-checkpoint", default="artifacts/experiment/best.pt")
    bundle = sub.add_parser("export-model", help="Export a bounded, versioned inference-only model bundle")
    bundle.add_argument("--checkpoint", required=True)
    bundle.add_argument("--cache-dir", default="data/cache")
    bundle.add_argument("--output-dir", required=True)
    bundle.add_argument("--calibration")
    bundle.add_argument("--evaluation")
    bundle.add_argument("--model-card", default="MODEL_CARD.md")
    benchmark = sub.add_parser("benchmark-protocol", help="Freeze split and external-model provenance")
    benchmark.add_argument("--cache-dir", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--provenance-manifest", action="append", default=[])
    compare = sub.add_parser("compare", help="Paired scenario bootstrap on existing evaluation records")
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--reference", action="append", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=42)
    serve = sub.add_parser("serve")
    serve.add_argument("--cache-dir", default="data/cache")
    serve.add_argument("--checkpoint")
    serve.add_argument("--calibration")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        if args.command == "desktop":
            from adp.desktop import run
            run(args)
            return
        elif args.command == "doctor":
            from adp.data import read_json
            from adp.training import environment_report, gpu_smoke_test
            result = {"environment": environment_report(), "smoke": gpu_smoke_test(args.device, read_json(args.config))}
        elif args.command == "download":
            from adp.data import download
            manifest = download(args.raw_dir, args.train_count, args.val_count, args.seed, args.workers)
            result = {"manifest": str(Path(args.raw_dir)/"manifest.json"),
                      "splits": {name: len(ids) for name, ids in manifest["splits"].items()},
                      "sampling": manifest["sampling"]}
        elif args.command == "preprocess":
            from adp.data import preprocess
            if args.manifest and Path(args.manifest).resolve() != (Path(args.raw_dir)/"manifest.json").resolve():
                raise ValueError("--manifest must match --raw-dir/manifest.json")
            result = preprocess(args.raw_dir, args.cache_dir, workers=args.workers, minimum_free_gib=args.minimum_free_gib, max_actors=args.max_actors, max_polylines=args.max_polylines, map_radius=args.map_radius)
        elif args.command == "train":
            from adp.data import read_json, write_json
            from adp.training import train
            from adp.reporting import plot_training
            result = train(read_json(args.config), args.cache_dir, args.output_dir, args.epochs, args.device, args.resume, args.overfit_batches, args.max_runtime_seconds)
            write_json(Path(args.output_dir)/"run.json", result)
            if result["history"]:
                plot_training(args.output_dir)
        elif args.command == "calibrate":
            from adp.training import calibrate
            result = calibrate(args.checkpoint, args.cache_dir, args.output_dir, args.device)
        elif args.command in ("evaluate", "stress-test"):
            result = _evaluate(args, args.command == "stress-test")
        elif args.command == "export-model":
            from adp.bundles import export_model
            metadata = export_model(args.checkpoint, args.cache_dir, args.output_dir, args.calibration, args.model_card, args.evaluation)
            result = {"bundle_id": metadata["bundle_id"], "output_dir": args.output_dir,
                      "evaluation_supplied": metadata.get("evaluation") is not None}
        elif args.command == "benchmark-protocol":
            from adp.protocol import create_protocol
            protocol = create_protocol(args.cache_dir, args.output, args.provenance_manifest)
            result = {"output": args.output, "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
                      "provenance_manifest_count": len(protocol["provenance_manifests"])}
        elif args.command == "compare":
            from adp.data import read_json, write_json
            from adp.protocol import paired_comparison
            candidate = read_json(args.candidate)
            comparisons = {str(path): paired_comparison(candidate, read_json(path), args.bootstrap, args.seed) for path in args.reference}
            result = {"comparisons": comparisons, "promotion_gate_passed": all(v["jointFDE_gate_passed"] for v in comparisons.values())}
            write_json(args.output, result)
        elif args.command == "study":
            from adp.data import preprocess, read_json
            from adp.research import acquire, prepare_study, profile_pilot, run_study, ensure_space
            if args.phase == "prepare":
                manifest = prepare_study(args.root, args.project_root)
                result = {"manifest": str(args.root / "raw" / "manifest.json"),
                          "splits": {name: len(ids) for name, ids in manifest["splits"].items()},
                          "previously_inspected_validation_excluded": len(manifest["excluded_validation_ids"])}
            elif args.phase == "pilot":
                acquire(args.root / "pilot" / "raw", args.workers)
                preprocess(args.root / "pilot" / "raw", args.root / "pilot" / "cache", workers=args.workers, minimum_free_gib=8)
                result = profile_pilot(args.root, read_json(args.config), args.device)
            elif args.phase == "acquire":
                pilot = read_json(args.root / "pilot.json")
                from adp.research import remaining_storage
                ensure_space(args.root, additional_bytes=remaining_storage(args.root, pilot["projected_raw_cache_bytes"]))
                acquire(args.root / "raw", args.workers)
                preprocess(args.root / "raw", args.root / "cache", workers=args.workers, minimum_free_gib=8)
                from adp.protocol import create_protocol
                result = create_protocol(args.root / "cache", args.root / "protocol.json", list((args.root / "provenance").glob("*.json")))
            elif args.phase == "train":
                result = run_study(args.root, read_json(args.config), args.device)
            elif args.phase == "pipeline":
                from adp.research import run_pipeline
                result = run_pipeline(args.root, args.project_root, read_json(args.config), args.reference_checkpoint, args.device, args.workers)
            else:
                from adp.research import finalize_study
                result = finalize_study(args.root, args.reference_checkpoint, args.device)
        else:
            import uvicorn
            from adp.api import create_app
            uvicorn.run(create_app(args.cache_dir, args.checkpoint, args.calibration, args.device), host=args.host, port=args.port)
            return
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        parser.exit(1, f"{args.command}: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
