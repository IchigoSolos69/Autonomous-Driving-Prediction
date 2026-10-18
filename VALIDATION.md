# Local validation

## Production-beta implementation — 18 October 2026

The beta implementation and the expanded model study are separate deliverables. The reference model below remains the deployed model until fresh held-out evidence passes the promotion gate. No stronger-model result has been measured yet.

Python 3.12 with PyTorch **2.13.0+cu130 / CUDA 13.0** passed actual RTX 5050 Laptop GPU forward/backward, optimizer/RNG continuation, and original-checkpoint CPU/GPU compatibility checks. Exact runtime locks are under `requirements/`. The qualified runtime is separate from the original 2.11 development environment. Evidence is in `artifacts/production/runtime-cu130.json` and `artifacts/packaging/qualification-cu130-gpu.json`.

The final production-runtime source suite passed **103 tests plus four subtests** in 28.81 seconds, and the frontend passed **18 tests** and its production build. This includes restoration of the scoped Windows sleep state and inference export of format-2 training checkpoints. Source checks exercise resumable committed batches, scenario provenance, truncated-report rejection, storage/GPU-budget guards, imports/cancellation, native request tokens, calibration evidence, corrupt bundles, installer staging and rollback. One existing Starlette/httpx test-client deprecation warning remains. Windows CI is configured but has not been run remotely.

The exact release dependency set and base PyTorch 2.13.0 were checked against the advisory database with no known vulnerabilities reported at scan time; see `artifacts/production/dependency-audit.json`. This supersedes the old runtime's advisory status for the new releases only.

The immutable expanded manifest contains 40,000 training and 12,000 fresh validation scenarios, with 2,000 previously inspected validation IDs excluded. A **1,000-scenario, training-only pilot** completed. The final configuration-bound profile measured **43.59 examples/s**, 178,998,784 bytes peak PyTorch allocated and 201,326,592 bytes reserved memory. Its 465,818,342 raw/cache bytes project to **22.56 GiB** for the selected study. Pilot eligibility was 44 actor labels, including five positives, across eight eligible scenes. No selection, calibration or evaluation scenes were used for pilot training.

The earlier pilot projected the 20-epoch study beyond 40 GPU-hours, so the uniform schedule was frozen at **15 epochs** before screening. Throughput is a projection, not a completed study. Acquisition pauses when projected space would breach the 8 GiB reserve; calibration/evaluation share the 40-hour GPU ledger. The pipeline prevents idle Windows system sleep only while running and restores the prior thread state afterward; it does not change power-plan settings or keep the display on.

Final CPU and NVIDIA installers both passed dedicated-folder install, bundled-runtime trained-demo inference, owned-service shutdown, corrected attribution and uninstall/data-preservation checks. Their common source hash is `9301a5edaeed0d8e6e7b3c13846e8f9d49f1c27650e46e42f6607865b13dcfef`; exact hashes/timings are in `artifacts/packaging/final/validation.json` and the `D:\InteractionLab\release-*` receipts. CPU verification took 4.88 seconds / 394 MB peak process working set; CUDA verification took 17.07 seconds / 1.63 GB. These are verification durations, not native UI startup or VRAM. These checks do not establish clean-machine acceptance: a clean Windows VM without Python/Node, physical CPU-only hardware, native-dialog interaction, offline restart, and the full installed upgrade workflow remain unverified. Native UI-control approval timed out; no successful dialog interaction is inferred from headless tests.

Generated-only build cleanup left 32.21 GiB free on D:. The resumable expanded-study pipeline was launched at 23:37 IST on 18 October and reached its acquisition phase. Live phase, process and log paths are under `D:\InteractionLab\study-v1\pipeline_status.json` and `pipeline-process.json`. This records a running acquisition, not completed expanded training or evaluation.

A real pilot AV2 parquet/map pair was imported into a separate Unicode-path test cache and then forecast through the trained CUDA service. Source SHA-256 hashes were unchanged, the import appeared in the scenario API, and its six-world response retained experimental risk labels. This exercised import/service integration rather than the native folder dialog; `artifacts/production/real-import.json` records the check.

The same imported scenario passed trained/constant-velocity/position-noise comparison and all four service export formats: prediction JSON, comparison CSV, PNG and diagnostics ZIP. The PNG was visually checked. Output hashes are in `artifacts/production/real-exports.json`; save destinations were explicit test paths, so this does not establish native save-dialog acceptance. The bundled demo's copyright/terms attribution was also corrected and its initial/reused-cache regression passed.

Frozen editions bundle Python, dependencies, frontend, reference weights and real demonstration scenes. The project-root executable described below remains the older workspace launcher. New writable settings, logs, imported datasets and models use `%LOCALAPPDATA%\Interaction Lab`. See [production instructions](PRODUCTION.md) and [packaging instructions](scripts/PACKAGING.md) for current entry points and acceptance limits.

## Reference prototype — 17 October 2026

This is an implemented research prototype with a preliminary three-epoch experiment, not a validated driving-safety system.

## What ran

- Downloaded and processed 7,000 real AV2 scenarios: 5,000 training, 500 model-selection, 500 calibration, and 1,000 held-out evaluation scenes. The selected IDs and sampling method are in `data/raw/manifest.json`.
- Verified ingestion against the official AV2 Python API.
- Passed a model forward/backward check on an RTX 5050 Laptop GPU using Python 3.12.14, PyTorch 2.11.0+cu128, and CUDA 12.8.
- Trained the full subset for three epochs in 266.1 seconds. Measured training throughput was 64.6 examples/s; peak PyTorch allocated memory was 132,032,512 bytes (allocator memory, excluding driver and other allocations).
- Ran 300 tiny-batch epochs on four real training scenes. minFDE on those training scenes fell from the constant-velocity value of 4.12 m to 2.14 m. This verifies learning and overfitting mechanics, not generalization.
- Resumed a GPU-trained checkpoint on CPU and completed another training epoch.
- Passed 55 Python tests (plus four subtests), eight frontend tests, the TypeScript/production build, and dependency consistency checks. The test client emits an upstream Starlette/httpx deprecation warning.
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

Severity 0.35, seed 42, first 32 held-out scenarios in immutable manifest order. The clean column uses **exactly the same supported scenarios** as its stressed counterpart. The post-audit rerun, reports and matched scenario IDs are in `artifacts/audit-fixes/stress/`; the original reports remain in `artifacts/experiment/stress/`.

| Intervention | Evaluated / attempted | Matched clean minFDE, m | Stressed minFDE, m |
|---|---:|---:|---:|
| Position noise | 32 / 32 | 7.121 | 33.746 |
| History dropout | 32 / 32 | 7.121 | 6.667 |
| Braking | 32 / 32 | 7.121 | 6.946 |
| Adjacent-lane cut-in | 18 / 32 | 8.673 | 7.428 |

Automatic cut-ins now try all eligible moving actors until one has suitable lane geometry. This increases support from the earlier 5/32 to 18/32; explicitly selected actors still fail with an explanation when unsupported. Remaining exclusions record reasons. Position noise recomputes velocities by finite differences, amplifying positional error; the observed degradation is specific to this protocol. Easier scripted futures or smoothing from dropout can reduce error on this small sample. None of these results establishes closed-loop robustness. Risk calibration evidence was unavailable in these stress subsets.

## Audit fixes and repeat checks

All seven audit findings are fixed. Resume refuses occupied destinations from another run; matching preprocessing settings preserve their original file bytes; reports carry both calibration and evaluation positive-event counts; invalid stress settings fail before data loading or report writes; automatic cut-ins search eligible actors; failed health refresh clears the connection indicator; unused data configuration was removed in favor of existing CLI flags. Legacy checkpoint configurations remain compatible when only that unused block differs.

Repeated preprocessing over all 7,000 production scenes with explicit `--map-radius 100` wrote zero new scenes and preserved preprocessing hash `b24050f4790a24a4b6666804d80d9a986a395d93f34f3154b47286011ac266bf`. Downloaded a fresh four-scene public S3 test subset, preprocessed it and loaded its training scenario through the official AV2 API.

Resumed the original run for one additional full epoch over 5,000 training and 500 validation scenarios in `artifacts/audit-fixes/resumed-training/`. It completed in 110.4 seconds at 58.1 training examples/s, with peak allocated CUDA memory 132,029,440 bytes. A separate four-scene smoke run trained on CUDA and continued on CPU; its CPU-saved checkpoint loaded on both devices with maximum trajectory difference 0.00000763 m. These are continuation checks. The application continues to load the original three-epoch checkpoint.

Refit calibration on its reserved 500-scene split into `artifacts/audit-fixes/recalibration/` and reran the original checkpoint on all 1,000 held-out scenarios into `artifacts/audit-fixes/evaluation/`. Displacement results reproduce the table above. Risk reports now explicitly show two calibration positives and 14 evaluation positives, with insufficient evidence retained. Original weights and report files are preserved.

Built the Python wheel in an isolated build environment and rebuilt the Windows launcher. Native-window checks exercised trained loading, recorded-future overlays, playback through frame 109, and selecting/running the constant-velocity baseline before restoring the trained model. Corrected automatic cut-in requests also passed through the desktop-owned service. Component tests cover failed refresh and a late prediction completing after a scenario switch. Machine-readable repeat-check results are in `artifacts/audit-fixes/verification.json`.

## Dependency advisory checks

`npm audit` reported zero known advisories. `pip check` passes. Upgraded environment setuptools from 78.1.0 to 81.0.0, which satisfies the installed PyTorch requirement `setuptools<82` and removes the earlier download-path traversal advisory. Isolated project builds require setuptools 83 or later. The scanner is installed separately under ignored `artifacts/validation-tools/`.

Two distinct upstream advisories remain recorded: [setuptools Unicode exclusion matching on macOS](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c), and the [PyTorch scripted-class crash](https://github.com/pytorch/pytorch/issues/149623). The former concerns source-distribution publishing on macOS; this delivery is Windows and uses patched isolated build tooling. The application does not accept TorchScript source and its model uses eager PyTorch; the installed 2.11 CUDA runtime still receives the latter version-based advisory. A move to the advisory database's patched PyTorch 2.13 runtime has not been tested. These are documented residual findings, not a clean Python vulnerability scan.

Full scanner output is in `artifacts/python-dependency-audit.json` and `artifacts/audit-fixes/torch-dependency-audit.json`. The environment scan skips the editable project and cannot resolve the CUDA version suffix, so the base PyTorch 2.11.0 release was checked separately. No live Colab session or clean Windows-machine installation was performed; notebook code cells were parsed and the current Windows environment was exercised.

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
