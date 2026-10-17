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
    from adp.stress import apply_stress
    from adp.training import load_checkpoint, resolve_device
    device = resolve_device(args.device)
    dataset = CachedDataset(args.cache_dir, args.split)
    calibration = read_json(args.calibration) if args.calibration else None
    network = None
    if args.checkpoint:
        saved = load_checkpoint(args.checkpoint)
        if saved["manifest_hash"] != manifest_hash(Path(args.cache_dir)/"manifest.json"):
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
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    total = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    records, skipped = [], []
    for index in range(total):
        scene = dataset[index]
        if stress:
            try:
                scene = apply_stress(scene, {"kind": args.kind, "severity": args.severity, "seed": args.seed})
            except ValueError as error:
                skipped.append({"scenario_id": scene["id"], "reason": str(error)})
                continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_scenes([scene]).items()}
        with torch.inference_mode():
            outputs = network(**model_inputs(batch)) if network else constant_velocity(model_inputs(batch))
        records.extend(evaluate_predictions(outputs, batch, calibration))
        if (index+1) % 10 == 0:
            print(f"Evaluated {index+1}/{total} scenarios", flush=True)
    report = summarize_records(records, bootstrap=args.bootstrap, seed=args.seed)
    report.update({"split": args.split, "model": "trained" if network else "constant_velocity",
                   "manifest_hash": dataset.manifest_hash, "calibration_applied": calibration is not None,
                   "available_scenarios": len(dataset), "scenario_limit": args.limit,
                   "scenario_selection": "entire split" if args.limit is None else "first N scenarios in immutable manifest order",
                   "protocol": "retained complete scored dynamic actors; joint modes; miss threshold 2m; six modes",
                   "risk_definition": "minimum centre distance <3m to another retained dynamic actor over6s",
                   "footprints_m": {"vehicle": [4.5, 1.8], "pedestrian": [.6, .6], "motorcyclist": [2, .8], "cyclist": [1.8, .6], "bus": [12, 2.5]},
                   "stress": {"kind": args.kind, "severity": args.severity, "other_agents_reactive": False} if stress else None,
                   "skipped": skipped})
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
    train = sub.add_parser("train")
    train.add_argument("--config", default="config.json")
    train.add_argument("--cache-dir", default="data/cache")
    train.add_argument("--output-dir", default="artifacts/run")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume")
    train.add_argument("--overfit-batches", type=int)
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
        if name == "stress-test":
            evaluate.add_argument("--kind", required=True, choices=["position_noise", "dropout", "braking", "cut_in"])
            evaluate.add_argument("--severity", type=float, default=.35)
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
            result = preprocess(args.raw_dir, args.cache_dir, workers=args.workers, max_actors=args.max_actors, max_polylines=args.max_polylines, map_radius=args.map_radius)
        elif args.command == "train":
            from adp.data import read_json, write_json
            from adp.training import train
            from adp.reporting import plot_training
            result = train(read_json(args.config), args.cache_dir, args.output_dir, args.epochs, args.device, args.resume, args.overfit_batches)
            write_json(Path(args.output_dir)/"run.json", result)
            plot_training(args.output_dir)
        elif args.command == "calibrate":
            from adp.training import calibrate
            result = calibrate(args.checkpoint, args.cache_dir, args.output_dir, args.device)
        elif args.command in ("evaluate", "stress-test"):
            result = _evaluate(args, args.command == "stress-test")
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
