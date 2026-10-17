# Local validation — 17 October 2026

This is an implemented research prototype with a preliminary three-epoch experiment, not a validated driving-safety system.

## What ran

- Downloaded and processed 7,000 real AV2 scenarios: 5,000 training, 500 model-selection, 500 calibration, and 1,000 held-out evaluation scenes. The selected IDs and sampling method are in `data/raw/manifest.json`.
- Verified ingestion against the official AV2 Python API.
- Passed a model forward/backward check on an RTX 5050 Laptop GPU using Python 3.12.14, PyTorch 2.11.0+cu128, and CUDA 12.8.
- Trained the full subset for three epochs in 266.1 seconds. Measured training throughput was 64.6 examples/s; peak PyTorch allocated memory was 132,032,512 bytes (allocator memory, excluding driver and other allocations).
- Ran 300 tiny-batch epochs on four real training scenes. minFDE on those training scenes fell from the constant-velocity value of 4.12 m to 2.14 m. This verifies learning and overfitting mechanics, not generalization.
- Resumed a GPU-trained checkpoint on CPU and completed another training epoch.
- Passed 41 Python tests, six frontend tests, the TypeScript/production build, and dependency consistency checks. The test client emits an upstream Starlette/httpx deprecation warning.
- Exercised live trained inference, baseline fallback, shared modes, reference overlays, stress comparisons, actor selection, keyboard scrubbing, wide layout and mobile drawers in the original browser interface. The 390 px mobile viewport had no horizontal overflow. These checks predate the native desktop host.

## Held-out results

Same 1,000 scenarios and retained scored actors; six modes; six-second horizon. These descriptive results do not establish statistical significance or benchmark competitiveness. Full records and scenario-bootstrap intervals are in `artifacts/experiment/evaluation/` and `artifacts/baseline/`.

| Metric | Constant velocity | Learned model |
|---|---:|---:|
| minADE, m | 2.169 | 2.092 |
| minFDE, m | 5.510 | 5.115 |
| Miss rate | 43.21% | 39.76% |
| Joint minFDE, m | 7.455 | 7.135 |
| Focal minADE, m | 4.539 | 4.327 |
| Focal minFDE, m | 11.724 | 10.905 |
| Normalized mixture NLL | 9.777 | 1.389 |
| Predicted scene footprint-overlap proxy | 70.20% | 86.88% |

Displacement error improved modestly, while the footprint-overlap proxy worsened. The model therefore does **not** demonstrate improved interaction safety. Footprints are assumed class sizes, and this metric measures overlap in predicted mean trajectories, not actual crash probability. The baseline's six identical worlds also make its mode-classification confidence a degenerate comparison.

Temperature scaling changed held-out nearest-joint-FDE world ECE from 0.0845 to 0.0682. This measures mode confidence, not physical collision risk.

## Risk evidence is insufficient

The conservative event-label rule requires complete futures for the entire retained dynamic group. It excluded 994 of 1,000 evaluation scenes. Only 95 actor labels, including 14 positive labels, remained. Calibration had just 21 eligible actor labels and two positives. Training had 209 eligible actor labels, including eight positives.

Risk Brier score was 0.1391 before and 0.1413 after the fitted adjustment. These small, selected populations cannot establish calibrated near-conflict prediction. The viewer displays raw scores and an explicit insufficient-evidence warning. Do not interpret the fitted adjustment or its output as established calibration.

## Controlled stress checks

Severity 0.35, seed 42, first 32 held-out scenarios in immutable manifest order. The clean column uses **exactly the same supported scenarios** as its stressed counterpart. Reports and matched scenario IDs are in `artifacts/experiment/stress/`.

| Intervention | Evaluated / attempted | Matched clean minFDE, m | Stressed minFDE, m |
|---|---:|---:|---:|
| Position noise | 32 / 32 | 7.121 | 33.746 |
| History dropout | 32 / 32 | 7.121 | 6.667 |
| Braking | 32 / 32 | 7.121 | 6.946 |
| Adjacent-lane cut-in | 5 / 32 | 7.699 | 6.986 |

Unsupported cut-ins are explicitly excluded with reasons. Position noise recomputes velocities by finite differences, amplifying positional error; the observed degradation is specific to this protocol. Easier scripted futures or smoothing from dropout can reduce error on this small sample. None of these results establishes closed-loop robustness. Risk calibration evidence was unavailable in these stress subsets.

## Open the local experiment

Double-click `Interaction Lab.exe` in the project root. The desktop launcher opens a native WebView2 application window and automatically selects `artifacts/experiment/best.pt` and its accompanying calibration when present. To rebuild the small launcher, run `scripts/build-app.ps1` after setup and the frontend production build.

The equivalent explicit command is:

```powershell
.\.venv\Scripts\python.exe -m adp desktop --checkpoint artifacts/experiment/best.pt --calibration artifacts/experiment/calibration.json
```

No browser or separately started server is required. The application owns a loopback service on an available port and shuts it down with the window. Logs are written to `artifacts/desktop/app.log`. `--baseline` explicitly selects constant velocity; absent weights also produce a labelled baseline. If the experiment checkpoint is absent, automatic selection next checks `artifacts/run/best.pt`.

The executable is a workspace launcher: the existing `.venv`, WebView2 Runtime, built `frontend/dist`, processed scenarios and model artifacts remain required. It does not bundle the CUDA runtime or dataset. Generated datasets, checkpoints, plots and reports are intentionally ignored by Git. The current workspace contains them; a fresh checkout must regenerate or obtain them.

The original HTTP entry point remains available for API/development work with `python -m adp serve`; desktop operation does not require opening its URL.

Continue training with:

```powershell
.\.venv\Scripts\python.exe -m adp train --output-dir artifacts/experiment --resume artifacts/experiment/last.pt --epochs 20
```

Then refit calibration and rerun evaluation for the new checkpoint. Existing calibration is bound to its original checkpoint hash.

## Verification limits

Built and launched `Interaction Lab.exe` from a different working directory on Windows with the installed WebView2 Runtime. The native window loaded all 7,000 scenarios and the trained CUDA checkpoint (`e35b33a65989ee47d4026140adcca76ddd28c8d717c940f9a6a6be368cbc3d59`). A request through its own local service returned six joint forecast modes in 8.08 ms. The visible window showed the trained forecast, checkpoint, probabilities and calibration-evidence warning.

A separate native instance loaded the 28-scene smoke cache with `--baseline`: no checkpoint or calibration loaded, with the constant-velocity and uncalibrated-heuristic labels visible. Closing that test window with Alt+F4 produced Uvicorn's completed-shutdown log; its listener and Python processes exited. The trained window remained open. Automated desktop checks cover arbitrary working directories, paths with spaces, missing assets/weights, a legacy-renderer installation message, escaped startup errors, foreign Origin/Host rejection, startup failure, closing during initialization, and port release. Missing WebView2 was simulated; the installed runtime was used for actual native-window testing.

The desktop launcher reuses the existing React/Canvas interface inside a native WebView2 window. The earlier detailed browser interaction checks still apply to that shared interface; they were not all repeated manually in the desktop host. The executable remains a workspace launcher, rather than a standalone installer bundling Python/CUDA and data.

The Colab notebook was syntax-checked and includes runtime checks, shared CLI training/resume, calibration, evaluation and Drive persistence. It was not executed against a cloud account. Full-dataset training, diffusion, reactive simulation and public hosting remain outside this delivery. See `MODEL_CARD.md` for population, geometry and uncertainty limitations.
