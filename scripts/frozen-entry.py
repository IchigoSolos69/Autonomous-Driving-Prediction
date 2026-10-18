"""Frozen GUI entry point; no developer environment or shell process is needed."""
import multiprocessing


def self_test():
    import argparse
    import json
    from pathlib import Path
    import torch
    from adp.desktop import LocalBackend
    from adp.model import ForecastModel, model_inputs
    from adp.training import _smoke_batch
    from fastapi import FastAPI
    from urllib.request import Request, urlopen

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", "--verify-installation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    import pandas, pyarrow, scipy, shapely, webview
    import adp.desktop
    resources = Path(adp.desktop.__file__).parent / "resources"
    if not (resources / "frontend" / "index.html").is_file():
        raise RuntimeError("Frozen frontend resources are missing")
    torch.set_num_threads(2)
    selected = torch.device(args.device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this installed application")
    with torch.inference_mode():
        result = ForecastModel().to(selected).eval()(**model_inputs(_smoke_batch(selected)))
    assert result["means"].shape == (1, 6, 2, 60, 2)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    backend = LocalBackend()
    trained_demo = False
    try:
        origin = backend.start(lambda: app)
        assert json.loads(urlopen(origin + "/health", timeout=10).read())["ok"]
    finally:
        backend.close()
    if (resources / "demo" / "bundle" / "bundle.json").is_file() and (resources / "demo" / "cache" / "manifest.json").is_file():
        from adp.api import create_app
        from adp.bundles import load_bundle
        bundle = load_bundle(resources / "demo" / "bundle")
        backend = LocalBackend()
        try:
            origin = backend.start(lambda: create_app(cache_dir=resources / "demo" / "cache",
                                   checkpoint=bundle["paths"]["checkpoint"], calibration=bundle["paths"]["calibration"],
                                   device=args.device, frontend_dir=resources / "frontend", bundle_metadata={**bundle["metadata"], "demo": True}))
            health = json.loads(urlopen(origin + "/api/health", timeout=10).read())
            assert health["checkpoint_loaded"]
            scenes = json.loads(urlopen(origin + "/api/scenarios", timeout=10).read())["scenarios"]
            assert scenes
            request = Request(origin + "/api/predict", data=json.dumps({"scenario_id": scenes[0]["id"], "model": "trained"}).encode(), headers={"Content-Type": "application/json"})
            prediction = json.loads(urlopen(request, timeout=30).read())
            assert len(prediction["trajectories"]) == 6 and abs(sum(prediction["probabilities"]) - 1) < 1e-5
            trained_demo = True
        finally:
            backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "torch": torch.__version__, "frozen_resources": True,
                                      "device": args.device,
                                      "model_forward": True, "owned_http_lifecycle": True,
                                      "trained_real_demo_inference": trained_demo}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import sys
    if "--self-test" in sys.argv or "--verify-installation" in sys.argv:
        self_test()
    else:
        from adp.desktop import main
        main()
