# Autonomous Driving Prediction

A Windows desktop research application for joint motion forecasting, proximity-risk calibration, and controlled robustness experiments on **real Argoverse 2 Motion Forecasting scenarios**.

The model combines a temporal Transformer, interaction graph, vector-map attention, and six shared probabilistic scene futures. A separate head estimates whether another retained dynamic actor will come within 3 metres during the following six seconds. The native application window displays the same observed inputs and stress interventions used for inference.

The completed local experiment, measured results, restart command and limitations are recorded in [VALIDATION.md](VALIDATION.md). The tested Windows dependency snapshot is `requirements-windows.lock.txt`; install it with the official CUDA wheel index specified in its header when reproducing this exact environment.

## Quick start on Windows

Requires Python 3.12, Node 22.12+ or 24 for building the interface, the [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/), and several GB for CUDA PyTorch. Installation stays inside `.venv`; setup includes the desktop dependencies. The setup script can discover the bundled Codex Python runtime, or accept `-Python C:\path\to\python.exe`.

```powershell
.\scripts\setup.ps1 -WithAv2
.\.venv\Scripts\python.exe -m adp doctor --device cuda

# Start with 28 real scenes. The subset manifest is immutable.
.\.venv\Scripts\python.exe -m adp download --raw-dir data/smoke/raw --train-count 16 --val-count 12
.\.venv\Scripts\python.exe -m adp preprocess --raw-dir data/smoke/raw --cache-dir data/smoke/cache

cd frontend
npm.cmd ci
npm.cmd run build
cd ..
.\scripts\build-app.ps1
.\.venv\Scripts\python.exe -m adp desktop --cache-dir data/smoke/cache --baseline
```

This opens a native Windows application window. The smoke command explicitly selects constant velocity with an **uncalibrated heuristic** risk score. It never substitutes synthetic scenes or fabricated trained predictions.

For the full local experiment already present in this workspace, double-click **`Interaction Lab.exe`** in the project root. It starts the application without a browser or terminal. The launcher uses this project's `.venv`, built `frontend/dist`, data and model files; it is not a standalone distribution of Python, CUDA or the dataset. Keep it with the project and rebuild it with `scripts/build-app.ps1` after moving the project if necessary.

Without explicit checkpoint options, `adp desktop` uses `artifacts/experiment/best.pt`, then `artifacts/run/best.pt`, and loads the accompanying `calibration.json` when present. If no weights exist, the application labels the constant-velocity baseline. `--baseline` forces that baseline; explicit `--checkpoint` and `--calibration` select a particular trained run. Paths resolve relative to `--project-dir` (the source project by default), rather than the shortcut's working directory.

The application starts its own loopback service on an available port and closes it with the window. Startup errors appear in the application; diagnostic logs are in `artifacts/desktop/app.log`. Close the application before rebuilding its executable. Rebuild the interface after frontend changes, then restart the application.

The HTTP service remains available for API use and development: run `.\.venv\Scripts\python.exe -m adp serve`, then visit `http://127.0.0.1:8000` or `/docs`. For frontend development, run `npm.cmd run dev` inside `frontend`; Vite proxies `/api` to that service on port 8000.

## Train, calibrate, evaluate

```powershell
# A structural learning check on a tiny real batch; do not interpret as a benchmark.
.\.venv\Scripts\python.exe -m adp train --cache-dir data/smoke/cache --output-dir artifacts/overfit --overfit-batches 1 --epochs 20

.\.venv\Scripts\python.exe -m adp train --cache-dir data/smoke/cache --output-dir artifacts/smoke --epochs 20
.\.venv\Scripts\python.exe -m adp calibrate --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --output-dir artifacts/smoke
.\.venv\Scripts\python.exe -m adp evaluate --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --calibration artifacts/smoke/calibration.json --output-dir artifacts/smoke/evaluation
.\.venv\Scripts\python.exe -m adp evaluate --cache-dir data/smoke/cache --output-dir artifacts/smoke/baseline
.\.venv\Scripts\python.exe -m adp desktop --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --calibration artifacts/smoke/calibration.json
```

The default experiment downloads **5,000 train + 2,000 validation scenes** into `data/raw`, then partitions official validation into 500 model-selection, 500 calibration, and 1,000 evaluation scenarios. Use a separate directory from smoke data:

```powershell
.\.venv\Scripts\python.exe -m adp download
.\.venv\Scripts\python.exe -m adp preprocess
.\.venv\Scripts\python.exe -m adp train --epochs 20
.\.venv\Scripts\python.exe -m adp calibrate --checkpoint artifacts/run/best.pt
.\.venv\Scripts\python.exe -m adp evaluate --checkpoint artifacts/run/best.pt --calibration artifacts/run/calibration.json
.\.venv\Scripts\python.exe -m adp desktop --checkpoint artifacts/run/best.pt --calibration artifacts/run/calibration.json
```

Check free storage before scaling. Raw data, processed geometry, checkpoints, and dependencies all consume disk. Keep datasets and artifacts outside Git. Downloads retry and resume existing files. Seeded UUID-prefix block sampling avoids listing the complete dataset for small runs; its exact method and membership are recorded in the manifest. It is **not a uniform full-dataset benchmark sample**. Changing subset sizes requires a new raw/cache directory.

Model and training configuration is in `config.json`. Data settings use the download/preprocess CLI flags: `--seed`, `--max-actors`, `--max-polylines`, and `--map-radius`. `use_graph=false` removes interaction layers; `use_map=false` removes map attention and off-drivable supervision. Default microbatch is 4 with 8 accumulation steps. Lower batch size if memory is constrained. Checkpoints include scheduler, optimizer, scaler, RNG state, configuration, and manifest hash; incompatible resumes are rejected.

Preprocessing uses four CPU workers by default (`--workers 1` disables multiprocessing). Its `audit.json` records complete-group risk supervision and class counts. Checkpoints and calibrators also record the preprocessing hash. The drivable-area loss applies only inside the cached map tile; uncovered locations are unknown. Risk heads without eligible supervision are hidden by the viewer.

```powershell
.\.venv\Scripts\python.exe -m adp train --resume artifacts/run/last.pt --epochs 40
```

Five-minute checkpoints occur at optimizer boundaries. Resuming an incomplete epoch repeats that epoch from the beginning; this preserves progress but is not bit-identical sample replay. Training throughput and peak CUDA memory are measured by the runtime.

## Free cloud training

Open `notebooks/train_colab.ipynb` in Colab. Set the project location and dataset archive, mount Drive, extract data onto runtime-local disk, and invoke the same CLI. Checkpoints persist to Drive and load locally on CPU or GPU. The notebook checks the actual GPU/runtime before selecting dependencies; it does not blindly impose the laptop CUDA wheel on every GPU. Free GPU access is availability-dependent. No paid service or cloud account is configured automatically.

## Research protocol

- Inputs are 50 observed steps (5 seconds); predictions cover 60 steps (6 seconds), at 10 Hz. Focal/AV priority, nearest-actor selection, local coordinate transforms, and map selection use only observed states.
- Default limits are 32 actors and 128 resampled map polylines. Reports include source/retained coverage and incomplete-future exclusions. Partial histories remain masked.
- Mode index `k` is shared across every actor, with one scene-level probability vector. Independent actor modes are never silently paired to estimate interaction risk.
- The objective combines masked joint Gaussian-mixture likelihood, best-mode trajectory loss, binary risk supervision, and a motor-vehicle-only soft off-drivable penalty. Forecast collision hypotheses are retained.
- Reports contain minADE/minFDE, 2 m miss rate, joint-world errors, mixture NLL, Brier-FDE, mode reliability, risk Brier/log loss/AUPRC, reliability bins, and oriented-footprint overlap proxies. Best-mode Brier-FDE uses the FDE-best mode rather than minimizing penalized error. Bootstrap intervals resample whole scenarios.
- Mode calibration targets the nearest joint-FDE world over complete scored dynamic actors. Event calibration uses a separate held-out split at natural retained-scene prevalence. Calibration is tied to checkpoint and manifest hashes.
- Risk labels require complete futures for the retained dynamic group. Incomplete labels are excluded and counted. Fewer than 50 positive events in calibration/evaluation means **insufficient calibration evidence**, regardless of fitted parameters.
- The proximity event includes benign close interactions. Footprint dimensions are fixed class approximations, not measured boxes; overlap rates are not observed crash rates. See `MODEL_CARD.md`.

## Stress experiments

```powershell
.\.venv\Scripts\python.exe -m adp stress-test --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --kind position_noise --severity 0.35 --output-dir artifacts/stress/noise
.\.venv\Scripts\python.exe -m adp stress-test --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --kind braking --severity 0.7 --output-dir artifacts/stress/braking
```

Severity is in `[0,1]`: position sigma up to 1 m, observation dropout up to 50%, braking up to 6 m/s². Zero severity is an exact no-op. Observation corruptions preserve original future labels. Braking/cut-ins replace one actor's final observed second and six-second reference future; other actors replay recorded motion. Cut-ins require an observed adjacent lane with sufficient geometry and fail with an eligibility explanation when unsupported. These are **nonreactive counterfactual replays**, not closed-loop safety tests. Future-only edits never influence inference.

Use `--limit 32` for a bounded stress/evaluation check; reports explicitly record the truncated population. Omit it for the complete split. Connected lane segments are joined only when their endpoints and forward continuation are unambiguous.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .pytest-tmp
cd frontend
npm.cmd test
npm.cmd run build
```

Tests cover geometry, leakage, graph isolation, padding, joint-mode metrics, risk labels, calibration, checkpoint portability, stress transforms, and API error handling. `artifacts/*/metrics.json`, `records.json`, `evaluation.png`, `history.json`, and `training.png` contain generated evidence. Small smoke runs validate mechanics; they do not establish forecast quality or research novelty.

## Interfaces

The desktop application embeds the React/Canvas interface in WebView2 and manages the local FastAPI service. Training, preprocessing, calibration and evaluation remain available through the same CLI for local and Colab runs.

`GET /api/health`, `GET /api/scenarios`, `GET /api/scenarios/{id}`, and `POST /api/predict` serve the application. Predictions use local metres and return `[world, actor, 60, xy]` means/scales, shared probabilities, raw/calibrated per-actor risk, and explicit model/calibration identifiers. Scenario reference futures are returned for inspection but are excluded by the `model_inputs` whitelist.

## Data and attribution

[Argoverse 2](https://www.argoverse.org/av2.html) data is licensed CC BY-NC-SA 4.0; its [API](https://github.com/argoverse/av2-api) is MIT. The ingestion path reads AV2 parquet/JSON directly to avoid making the broader API dependency tree mandatory; `setup.ps1 -WithAv2` installs the official API for interoperability checks. Cite Wilson et al., *Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting* when publishing results. No driving-safety guarantee is made.
