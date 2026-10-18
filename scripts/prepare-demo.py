"""Copy a bounded, deterministic real-data demonstration without changing its source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


ATTRIBUTION = (
    "Argoverse 2 Motion Forecasting dataset, Wilson et al.\n"
    "© 2021 Argo AI, LLC\n"
    "https://www.argoverse.org/av2.html\n"
    "CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
    "Terms: https://www.argoverse.org/about.html#terms-of-use\n"
    "Processed focal-local trajectories and map geometry. Noncommercial research demonstration only.\n"
)


def prepare_demo(cache_dir, output_dir, limit=24):
    cache_dir, output_dir = Path(cache_dir).resolve(), Path(output_dir).resolve()
    if not 1 <= limit <= 100:
        raise ValueError("Demonstrations support 1–100 scenarios")
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(not isinstance(scenario_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", scenario_id)
           for ids in manifest["splits"].values() for scenario_id in ids):
        raise ValueError("Unsafe demonstration scenario identifier")
    candidates = [(split, scenario_id) for split in ("evaluation", "validation", "calibration", "train")
                  for scenario_id in manifest["splits"].get(split, [])
                  if (cache_dir / split / f"{scenario_id}.npz").is_file()]
    chosen = candidates[:limit]
    if not chosen:
        raise ValueError("No processed real scenarios are available for the demonstration")
    identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    demo_manifest = {"version": 1, "source": "Argoverse 2 Motion Forecasting", "license": "CC-BY-NC-SA-4.0",
                     "purpose": "noncommercial research demonstration; not a benchmark sample",
                     "source_manifest_sha256": identity,
                     "splits": {split: [sid for selected, sid in chosen if selected == split]
                                for split in ("train", "validation", "calibration", "evaluation")}}
    if output_dir.exists():
        if json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == demo_manifest and all((output_dir / split / f"{sid}.npz").is_file() for split, sid in chosen):
            (output_dir / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
            return {"cache_dir": str(output_dir), "scenario_count": len(chosen), "source_manifest_sha256": identity}
        raise FileExistsError("Existing demonstration differs; choose a new output directory")
    if sum((cache_dir / split / f"{sid}.npz").stat().st_size for split, sid in chosen) > 64 * 1024**2:
        raise ValueError("Demonstration exceeds the 64 MiB cache budget")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".demo-", dir=output_dir.parent))
    try:
        catalog = json.loads((cache_dir / "catalog.json").read_text(encoding="utf-8"))
        selected_ids = {sid for _, sid in chosen}
        for split, scenario_id in chosen:
            if (cache_dir / split / f"{scenario_id}.npz").is_symlink():
                raise ValueError("Demonstration scenes must not be symbolic links")
            (stage / split).mkdir(exist_ok=True)
            shutil.copyfile(cache_dir / split / f"{scenario_id}.npz", stage / split / f"{scenario_id}.npz")
        shutil.copyfile(cache_dir / "preprocessing.json", stage / "preprocessing.json")
        (stage / "manifest.json").write_text(json.dumps(demo_manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "catalog.json").write_text(json.dumps({"scenarios": [item for item in catalog["scenarios"] if item["id"] in selected_ids], "total": len(chosen)}, indent=2) + "\n", encoding="utf-8")
        (stage / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"cache_dir": str(output_dir), "scenario_count": len(chosen), "source_manifest_sha256": identity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(prepare_demo(args.cache_dir, args.output_dir, args.limit), indent=2))
