# Interaction Lab — Autonomous Driving Prediction

A Windows desktop research application for forecasting how vehicles, pedestrians and cyclists move together. Interaction Lab combines a temporal Transformer, interaction graph and vector-map encoder to predict **six shared probabilistic futures**, inspect real Argoverse 2 scenarios and study forecasting failures.

The application also estimates a **proximity-based near-conflict proxy**: another retained dynamic actor coming within 3 metres of an actor's centre during the next six seconds. This can include benign interactions. Forecasts, risk scores and footprint overlaps are research outputs, not driving-safety guarantees.

**Documentation updated: 25 October 2026.** Measurements below are dated evidence, not live training or host status. The latest whole-repository audit reviewed here is from **21 October 2026**; it remains open for installed-application qualification.

## Project status and source locations

This repository currently has two implementation locations. Select the correct source before running commands; the reference executable and older installers do not contain every later feature.

| Component | Location | Availability and status |
|---|---|---|
| Reference application, forecasting model and research CLI | `src/adp/`, `frontend/`, `scripts/` | Tracked source in this checkout; the setup examples below use it. |
| Failures workspace, Training screen, lane-direction study and distributed compute tooling | `artifacts/failure-analysis-direction/` | Later isolated implementation in this local workspace. `artifacts/` is Git-ignored and **absent from a fresh clone**. See [extended workflows](#extended-workflows). |
| Developer desktop launcher | `Interaction Lab.exe` | Local generated launcher; requires the adjacent source, `.venv`, built frontend, data and weights. It is not an independent installation. |
| CPU and NVIDIA beta installers | Local release directories described in [PRODUCTION.md](PRODUCTION.md) | Built 18 October with reference weights. Historical artifacts, not a public or fully qualified release of the later implementation. |
| Data, checkpoints, reports and audit evidence | `data/`, `artifacts/`, configured research storage | Generated/local-only; not supplied by cloning the repository. |

The original 5,000-scenario experiment has measured results. The expanded study is **incomplete**, calibrated risk is **not validated**, and no stronger checkpoint has been established by the evidence documented here. The 21 October audit records **20 confirmed findings: four P1 and sixteen P2**. Its report-only work did not apply product fixes. Historical `artifacts/audit-fixes/` receipts are from 17 October and do not close those later findings.

## Contents

- [Capabilities and architecture](#capabilities-and-architecture)
- [Requirements](#requirements)
- [Windows source setup](#windows-source-setup)
- [Quick start with real scenarios](#quick-start-with-real-scenarios)
- [Using the desktop application](#using-the-desktop-application)
- [Data selection and preprocessing](#data-selection-and-preprocessing)
- [Training, checkpoints and reproducibility](#training-checkpoints-and-reproducibility)
- [Evaluation, calibration and risk](#evaluation-calibration-and-risk)
- [Stress experiments](#stress-experiments)
- [Measured results and limitations](#measured-results-and-limitations)
- [Expanded model study](#expanded-model-study)
- [Extended workflows](#extended-workflows)
- [Standalone monitoring and distributed compute](#standalone-monitoring-and-distributed-compute)
- [HTTP and native interfaces](#http-and-native-interfaces)
- [Testing, audit and release qualification](#testing-audit-and-release-qualification)
- [Packaging and installation](#packaging-and-installation)
- [Repository guide and troubleshooting](#repository-guide-and-troubleshooting)
- [Scope, attribution and licensing](#scope-attribution-and-licensing)

## Capabilities and architecture

- Native Windows window using **pywebview/WebView2**, with an application-owned loopback FastAPI service.
- **React, TypeScript, Vite and Canvas 2D** scenario inspection, playback, map rendering and forecast comparison.
- **Python 3.12 and PyTorch**, with pure-PyTorch graph operations; no PyG or custom graph CUDA extension is required.
- Real AV2 parquet/map ingestion, masked histories, observed-only actor/map selection, coordinate normalization and background imports.
- Multimodal scene forecasts, constant-velocity baseline, graph/map ablations, calibration, stress tests and reproducible exports.
- Portable inference bundles and separate CPU/NVIDIA packaging workflows.
- Later isolated implementation: recorded failure review, controlled lane-direction experiments, training progress and bounded multi-host development queues.

```text
AV2 parquet + map JSON → immutable scenario manifest → preprocessing/cache
                                                        │
                    observed histories + masks + vector map
                                                        │
              temporal Transformer → interaction graph + map attention
                                                        │
               six shared worlds + Gaussian scales + proximity head
                         │                              │
                  desktop inspection             offline evaluation
```

The default network uses two temporal Transformer layers, width 128 and four attention heads, followed by two interaction layers using relative position, velocity, heading and actor type within 50 m. A shared polyline MLP and actor-to-map attention provide map context. Six scene queries decode constant-velocity extrapolations plus learned trajectory residuals, bounded diagonal Gaussian standard deviations **0.15–20 m**, and one six-element scene probability vector.

World index `k` describes the same hypothesized scene across all retained actors within a prediction. World indices from different models are independent. The risk head uses interaction features and decoded clearance summaries. An explicit `model_inputs` whitelist excludes reference futures and training labels from model inference.

## Requirements

| Use | Requirements |
|---|---|
| Source development | Windows 11 x64, Python **3.12**, Node **24** recommended, npm and WebView2. The frontend also declares `^22.12.0` and `>=26.0.0` Node support. |
| CPU inference/training | CPU PyTorch environment; smaller batches may be needed on limited-memory machines. |
| NVIDIA edition | Compatible NVIDIA GPU/driver and a successfully qualified CUDA runtime. The local tested GPU is an RTX 5050 Laptop GPU with 8 GB. |
| Installed beta | Bundled Python, frontend and dependencies; Python and Node need not be installed separately. WebView2 is a prerequisite. |
| Remote work | Provider account/access, network and host-specific qualification. Free GPU availability and runtime persistence are variable. |

Use separate CPU and NVIDIA environments. The release locks pin **PyTorch 2.13.0+cpu** and **2.13.0+cu130**, respectively, with exact common runtime dependencies. See the [official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/) and the checked-in [runtime locks](requirements/). A CUDA version printed by `nvidia-smi` does not establish that the selected PyTorch build works on that GPU.

The ingestion path reads AV2 parquet/JSON directly. The official `av2==0.3.6` API is an optional interoperability dependency, rather than a requirement for every inference installation. Install optional tooling in a separate environment or reverify the lock afterward.

Allow space for dependencies, raw data, caches, checkpoint temporary files and export copies. Operational staging targets **10 GiB free**, dynamically checking projected copies. Frozen study/worker protocols retain their original **8 GiB** settings. This is not a universal disk quota enforced by every command: set preprocessing's reserve explicitly, and see the audit's remaining storage gaps.

## Windows source setup

Run these PowerShell commands from the repository root. The examples use `$Python` for the environment selected here. Do not install both editions into the same environment.

### CPU environment

```powershell
$EnvironmentRoot = Join-Path $PWD 'artifacts\environments\cpu'
py -3.12 -m venv $EnvironmentRoot
$Python = Join-Path $EnvironmentRoot 'Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements/runtime-cpu.lock.txt
& $Python -m pip install --no-deps --no-build-isolation -e .
& $Python scripts/verify-runtime-lock.py --edition cpu
& $Python scripts/qualify-runtime.py --expected-torch 2.13.0+cpu --device cpu --output artifacts/qualification/cpu.json
```

### NVIDIA environment

Use this alternative when qualifying the supported local NVIDIA hardware:

```powershell
$EnvironmentRoot = Join-Path $PWD 'artifacts\environments\nvidia'
py -3.12 -m venv $EnvironmentRoot
$Python = Join-Path $EnvironmentRoot 'Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements/runtime-nvidia.lock.txt
& $Python -m pip install --no-deps --no-build-isolation -e .
& $Python scripts/verify-runtime-lock.py --edition nvidia
& $Python scripts/qualify-runtime.py --expected-torch 2.13.0+cu130 --device cuda --output artifacts/qualification/nvidia.json
```

The qualification script performs synthetic forward/backward, finite-gradient and optimizer/RNG continuation checks without changing research checkpoints. Add `--checkpoint PATH` to verify an existing checkpoint's CPU/device agreement. These checks do not establish forecasting accuracy. `adp doctor --device cuda` provides an additional CUDA kernel/model forward-backward check; the current `doctor` implementation requires CUDA and cannot qualify a CPU-only installation.

Environments above live under Git-ignored `artifacts/`. Launch them directly with `& $Python -m adp desktop`; the older workspace EXE specifically expects `.venv`, not either of these environments. If the Windows `py` launcher is unavailable, replace `py -3.12` with your Python 3.12 executable.

The older `scripts/setup.ps1` and `requirements-windows.lock.txt` remain for reference reproduction. That script can reuse system packages and falls back to PyTorch 2.11 CPU/CUDA 12.8 wheels; it is **not equivalent to the isolated 2.13 release setup above**. Do not treat the legacy lock as a newly qualified deployment environment.

## Quick start with real scenarios

This creates a separate 28-scenario smoke workspace: 16 training and 12 official-validation scenarios. It checks the end-to-end mechanics; it is too small to establish accuracy or risk calibration.

```powershell
& $Python -m adp download --raw-dir data/smoke/raw --train-count 16 --val-count 12 --seed 42
& $Python -m adp preprocess --raw-dir data/smoke/raw --cache-dir data/smoke/cache --minimum-free-gib 10

Push-Location frontend
npm.cmd ci
npm.cmd run build
Pop-Location

& $Python -m adp desktop --cache-dir data/smoke/cache --baseline
```

The native window starts paused and explicitly labels **constant velocity** and its uncalibrated heuristic risk. Missing weights never become fabricated trained predictions.

For a trained run, provide its checkpoint and matching optional calibrator:

```powershell
& $Python -m adp desktop --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --calibration artifacts/smoke/calibration.json --device cpu
```

Without explicit model arguments, the reference desktop looks for `artifacts/experiment/best.pt`, then `artifacts/run/best.pt`, and adjacent calibration. Without weights it offers the labeled baseline. `--baseline` conflicts with explicit checkpoint, calibration or model-bundle selection. Cache/checkpoint/calibration paths resolve against `--project-dir`; absolute paths are safest when launching from another directory.

For API development, run `& $Python -m adp serve --cache-dir data/smoke/cache`, then open `http://127.0.0.1:8000/docs`. Run `npm.cmd run dev` inside `frontend` for Vite development; `/api` proxies to the loopback service on port 8000. The desktop instead owns a dynamically allocated service port and closes its service with the window.

## Using the desktop application

The scenario library sits on the left, the road canvas in the centre, the actor/risk inspector on the right and the timeline below. Select an actor, choose a shared forecast world, toggle separately identified reference futures, scrub/play, pan/zoom and compare baseline or stress predictions. Colour is reinforced by labels and line styles: teal focal actor, blue surrounding actors and amber/red proximity annotations on a graphite road canvas.

Keyboard controls include **Space** for playback, **Left/Right** for stepping, **F** to fit the scene, **+ / =** to zoom in and **−** to zoom out when the canvas has focus, **Escape** to close drawers and **Tab** navigation. Visible focus, reduced-motion support and responsive drawers are included. Playback begins paused.

**Import AV2 folder** opens a native directory chooser. The background job validates parquet/map pairs, exposes progress and per-scenario errors, and supports cancellation. Completed caches publish atomically; source folders and benchmark caches are preserved. Model selection and data-location settings also use native dialogs and take effect after restart.

Export prediction JSON, comparison/report CSV, PNG figures or an explicit diagnostics archive through the native save dialog. The last eight prediction records are held in a bounded cache; export a record before it is evicted. Diagnostics may contain local paths and logs, so inspect an archive before sharing it.

User settings and rotating logs normally live under `%LOCALAPPDATA%\Interaction Lab`:

```text
settings.json
logs/app.log
data/imports/
data/models/
data/analyses/     # extended implementation only
```

`--data-root` or the persisted data location relocates imported datasets/models/analyses; settings and logs remain in LocalAppData. Original project data and generated research results are separate from these imported resources.

## Data selection and preprocessing

Inputs follow the [AV2 forecasting specification](https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/constants.py): **50 observed steps and 60 future steps at 10 Hz**. Coordinates are translated and rotated around the focal actor at the observation cutoff, with metadata for conversion back to world coordinates. Missing states retain masks.

Defaults retain the focal actor, AV when available and nearest observed actors, up to **32 actors**. Observed geometry selects up to **128 map polylines**, resampled to 20 points, within a 100 m map radius. Report retained scored-target coverage and excluded/incomplete futures alongside accuracy. A small retained scene is not the full traffic environment.

| Protocol | Official training | Selection | Calibration | Evaluation |
|---|---:|---:|---:|---:|
| Reference/default subset | 5,000 | 500 | 500 | 1,000 |
| Expanded study | 40,000 | 2,000 | 5,000 | 5,000 locked |
| Approved distributed development package | 1,000 | 64 | None | None |

All partitions split by scenario. The CLI calls the selection partition **`validation`**. Default small-subset downloading uses deterministic seeded UUID-prefix blocks, then recorded ordering; it is **not uniform sampling over the complete official listing**. The expanded protocol instead orders complete official listings deterministically with seed 42 and excludes previously inspected validation IDs, including smoke/audit subsets.

For a new reference-sized experiment, use directories separate from the smoke workspace:

```powershell
& $Python -m adp download --raw-dir data/reference/raw --train-count 5000 --val-count 2000 --seed 42
& $Python -m adp preprocess --raw-dir data/reference/raw --cache-dir data/reference/cache --workers 4 --minimum-free-gib 10
```

Downloads use public scenario-level S3 access and can resume existing files. Manifests record IDs, sampling method and partitions; changing sample counts requires a new directory. Preprocessing records schema/configuration hashes and `audit.json` coverage/eligibility counts. Use `--workers 1` to disable preprocessing multiprocessing. An explicit `--manifest` must be the chosen raw directory's `manifest.json`.

Compatible imported inputs do not inherit the training manifest's calibration claims. Preserve manifests, preprocessing settings, exclusion history and model provenance when moving caches or checkpoints. Never mix calibration/locked-evaluation labels into training, development queues or failure-driven model tuning.

## Training, checkpoints and reproducibility

The following commands use the **new smoke workspace**, not an existing frozen study or locked holdout:

```powershell
# Tiny real-batch overfit: a learning/mechanics check only.
& $Python -m adp train --cache-dir data/smoke/cache --output-dir artifacts/overfit --overfit-batches 1 --epochs 20

& $Python -m adp train --cache-dir data/smoke/cache --output-dir artifacts/smoke --epochs 20
& $Python -m adp train --cache-dir data/smoke/cache --output-dir artifacts/smoke --resume artifacts/smoke/last.pt --epochs 30
```

[config.json](config.json) defaults to seed 42, microbatch **4**, **8** gradient-accumulation steps (effective batch 32), learning rate `3e-4`, weight decay `0.01`, zero loader workers, mixed precision when supported and 300-second checkpoint intervals. Lower microbatch size for memory pressure and adjust accumulation deliberately. Graph/map ablations use `model.use_graph=false` or `model.use_map=false`; disabling map context also disables off-drivable supervision.

The objective combines normalized masked scene-mixture likelihood, best-mode trajectory auxiliary loss (weight 0.1), eligible binary risk loss (0.2), and a soft motor-vehicle off-drivable penalty (0.01). The penalty uses cached distance fields only inside known map tiles. Potentially conflicting forecast hypotheses are preserved. A screened study variant adds a 0.1 shared-world classification loss targeting the detached world with lowest eligible joint final error.

`last.pt` is resumable state; `best.pt` is the selected checkpoint. Checkpoints save model, optimizer, scheduler, scaler, RNGs, configuration, split-manifest and preprocessing identities. Format 2 additionally preserves sampler order, independent loader RNGs, committed sample cursor and objective totals. Mid-epoch saves occur at optimizer boundaries, about every five minutes; epoch boundaries are also saved. Committed batches are not replayed on compatible resume.

Exact CPU continuation is tested. Cross-device floating-point execution need not be bit-identical. Legacy epoch-boundary training checkpoints remain supported; legacy partial epochs and inference-only bundles cannot resume training. Keep ledgers and source/protocol metadata with study checkpoints; a checkpoint alone does not prove remaining compute allowance.

Generated run evidence includes `history.json`, `run.json`, training curves, throughput and peak PyTorch CUDA allocation. Allocator peaks exclude GPU driver/context memory. Generic `train` is not the bounded study controller: use the prescribed controller for a metered protocol rather than assuming all CLI training inherits a host cap.

## Evaluation, calibration and risk

Freeze the checkpoint before calibration. Fit only on the calibration partition; keep model selection and final evaluation separate.

```powershell
& $Python -m adp calibrate --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --output-dir artifacts/smoke
& $Python -m adp evaluate --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --calibration artifacts/smoke/calibration.json --output-dir artifacts/smoke/evaluation
& $Python -m adp evaluate --cache-dir data/smoke/cache --cv-scale 1.0 --output-dir artifacts/smoke/baseline
& $Python -m adp compare --candidate artifacts/smoke/evaluation/records.json --reference artifacts/smoke/baseline/records.json --output artifacts/smoke/comparison.json --bootstrap 1000
```

Smoke reports verify mechanics and will generally have insufficient risk evidence. For a different checkpoint population, `benchmark-protocol` records fixed evaluation membership, preprocessing compatibility and external-model training/calibration provenance. `--protocol` permits explicitly proven compatible references rather than bypassing disjointness checks. Do not point exploratory commands at a locked study holdout; study finalization controls that evaluation.

Primary displacement metrics use retained scored dynamic actors with **complete, finite 60-step futures**. Per-actor minADE/minFDE can choose different worlds for different actors; joint-world errors choose one world across the eligible scene. Reports include focal and joint errors, miss rate at 2 m, normalized mixture NLL, Brier-FDE/mode reliability, risk Brier score, binary log loss, AUPRC, reliability plots, coverage and exclusions. Brier-FDE uses the FDE-best world before applying its probability penalty. Bootstrap uncertainty resamples whole scenarios, not independent actors.

Mode temperature scaling calibrates confidence in the nearest joint-FDE world. Monotone logistic/Platt calibration adjusts proximity logits. Neither establishes physical crash probability or Gaussian coverage. Calibrators bind to checkpoint and population provenance. Under imports or stress, applicability is unknown and calibrated-risk claims are withheld.

### Risk and footprint definitions

The proximity event is centre distance **strictly below 3 m** to another retained dynamic actor at any sampled future step. Eligible labels require at least two retained dynamic actors and complete finite futures for **every retained dynamic actor**. One incomplete retained future excludes that group's event supervision/evaluation; exclusion counts must remain visible.

If **either calibration or evaluation has fewer than 50 positive risk events**, there is no validated calibrated-risk score. Matching fitted parameters alone are insufficient. A raw learned/heuristic score, an unavailable risk label and a validated calibrated probability are different states.

Footprint overlap is reported separately using fixed approximate dimensions; AV2 motion tracks contain no measured vehicle extents:

| Actor class | Length × width, metres |
|---|---:|
| Vehicle | 4.5 × 1.8 |
| Bus | 12 × 2.5 |
| Pedestrian | 0.6 × 0.6 |
| Cyclist | 1.8 × 0.6 |
| Motorcyclist | 2.0 × 0.8 |

Heading follows forecast displacement, falling back to observed heading when stationary. Matched overlap diagnostics separate overlaps already present at the observation cutoff from newly appearing overlaps. These mean-path geometric proxies do not measure crashes, causal danger or reactive driving outcomes. See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.

## Stress experiments

```powershell
& $Python -m adp stress-test --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --kind position_noise --severity 0.35 --output-dir artifacts/stress/noise
& $Python -m adp stress-test --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --kind braking --severity 0.7 --output-dir artifacts/stress/braking
```

Keep two suites separate:

| Suite | Perturbation | Reference labels and interpretation |
|---|---|---|
| Observation corruption | Seeded `position_noise` or history `dropout` | Original future labels remain unchanged. |
| Behaviour counterfactual | Scripted `braking` or adjacent-lane `cut_in` beginning in the final observed second | Edited actor continues into a replacement reference future; other actors replay recorded motion. **Nonreactive counterfactual**, not closed-loop simulation. |

Severity is in `[0,1]`: position-noise sigma up to 1 m, history dropout up to 50% and braking up to 6 m/s². Zero severity is an exact no-op. Cut-ins require eligible adjacent-lane geometry and otherwise return/expose an exclusion reason. Lane connections require unambiguous forward continuation.

Inference and visualization receive the same observed perturbation. Future-only edits never change model inputs. Stress reports use matched clean comparisons, record supported/excluded cases and withhold clean-population calibration claims. `--limit 32` bounds a check and labels the truncated manifest-order population; omit it for the complete permitted split.

## Measured results and limitations

### Reference experiment — 17 October 2026

The three-epoch model trained on 5,000 scenarios and was evaluated on the reference 1,000-scenario holdout. This is the nonuniform subset described above, not an official full-dataset benchmark. Full evidence and stress tables are in [VALIDATION.md](VALIDATION.md).

| Held-out metric | Constant velocity | Learned reference |
|---|---:|---:|
| minADE, m | 2.169 | 2.092 |
| minFDE, m | 5.510 | 5.115 |
| Miss rate at 2 m | 43.21% | 39.76% |
| Joint minFDE, m | 7.455 | 7.135 |
| Focal minFDE, m | 11.724 | 10.905 |
| Predicted scene footprint-overlap proxy | 70.20% | 86.88% |

Displacement improved modestly while predicted overlap **worsened**. This does not demonstrate safer interactions or benchmark competitiveness. Constant velocity produces six identical worlds; its mode-confidence comparison is degenerate. The historical likelihood comparison used fixed baseline versus learned Gaussian scales, so it does not isolate trajectory quality; the expanded protocol also fits a global baseline scale on calibration data.

Training took 266.1 seconds at a measured 64.6 examples/s, with 132,032,512 bytes peak PyTorch allocation. Mode ECE changed from 0.0845 to 0.0682 after temperature scaling. Risk evidence was insufficient: calibration had **2 positives / 21 eligible labels**; evaluation had **14 / 95**, with **994 of 1,000 scenes excluded**. Risk Brier worsened from 0.1391 to 0.1413 after fitting. No validated risk or safety improvement follows.

Historical matched stress checks also showed substantial sensitivity to position corruption; finite-difference velocity reconstruction can amplify position noise. Lower errors on scripted futures or dropped histories are not evidence of safer closed-loop behaviour.

### Local service measurements — 18 October 2026

Twenty warmed requests on a 27-actor scene used Windows 11, an i7-14650HX, approximately 24 GB RAM and an RTX 5050 Laptop GPU (8 GB), with two CPU threads. These are development-machine measurements, not minimum supported hardware or native-window startup timings.

| Measurement | CPU runtime | NVIDIA runtime |
|---|---:|---:|
| Scientific imports to HTTP-ready service | 4.45 s | 17.08 s |
| Median server forecast computation | 20.93 ms | 10.83 ms |
| 95th-percentile server computation | 38.57 ms | 13.56 ms |
| Median complete loopback JSON request | 180.82 ms | 187.46 ms |
| Peak process RAM, excluding WebView2 | 357.8 MiB | 1,517.4 MiB |
| Peak PyTorch CUDA allocated / reserved | — | 74.1 / 92.0 MiB |

Serialization accounts for part of the difference between forecast time and complete requests. CUDA allocator numbers exclude driver/context memory. [PRODUCTION.md](PRODUCTION.md) describes the measurement harness and historical installer checks; those checks do not replace clean-machine acceptance.

## Expanded model study

The intended model-first study uses the 40,000/12,000 full-listing selection, a fixed training-only 1,000-scenario pilot and an original **40 GPU-hour allowance**. Four seed-42 configurations are screened: full model, no graph, no map and full model plus 0.1 shared-world classification loss. The winning configuration is repeated with seeds 43 and 44.

The pilot's updated configuration-bound profile measured 43.59 examples/s and projected 22.56 GiB of research data. A 20-epoch projection exceeded the allowance, so a uniform **15 epochs** was frozen before the six study runs. Pilot memory/eligibility measurements are implementation evidence, not a stronger-model result.

The protocol specifies selection by joint minFDE, then focal minFDE, then earlier epoch; the audit identifies a missing cross-run epoch tie-break implementation. Deployment selection must freeze before calibration. Intended promotion requires 1,000 paired scenario-bootstrap replicates and a 95% interval below zero for joint minFDE change against both the old model and constant velocity, reporting all three seeds and secondary outcomes. There is no automatic promotion justified by the current evidence.

**Dated status, 21 October:** four of six runs complete, **66/90 saved epochs**, and **45.44275 charged hours against a 40-hour ceiling**. This establishes an accounting discrepancy; it does not establish 45.44 hours of active GPU computation or a particular sleep event as the cause. The original study and direction follow-up are incomplete. Do not reset the ledger, increase the cap or blindly rerun `study pipeline` to clear this condition.

`study prepare|pilot|acquire|train|finalize|pipeline` exposes the controller phases. Existing frozen work uses its provenance-checked `resume-frozen-study.py` workflow, unchanged source/configuration and original ledger, subject to the remaining audit/recovery gates. Preserve frozen protocols and artifacts while repairing separately versioned code. The launch-time state in older production notes is historical; current execution requires fresh evidence.

## Extended workflows

These commands require the **local isolated implementation**, not root `src/adp`. Its detailed guides are local-only: `artifacts/failure-analysis-direction/FAILURE_ANALYSIS.md`, `STUDY_CLEANUP.md` and `docs/`. A fresh clone needs a reviewed/versioned copy of that implementation before these features are available.

Choose its source explicitly to avoid importing the root editable package. This example assumes the existing qualified local runtime and previously generated reference resources:

```powershell
$ReferenceRoot = (Get-Location).Path
$AppSource = Join-Path $ReferenceRoot 'artifacts\failure-analysis-direction'
$ResearchPython = 'D:\InteractionLab\runtime-cu130\Scripts\python.exe'
$PreviousAdpPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $AppSource 'src'

# Build the extended interface, not the root interface.
Push-Location (Join-Path $AppSource 'frontend')
npm.cmd ci
npm.cmd run build
Pop-Location

& $ResearchPython -m adp desktop --project-dir $AppSource --cache-dir "$ReferenceRoot\data\cache" --checkpoint "$ReferenceRoot\artifacts\experiment\best.pt" --device cpu --training
```

`D:\InteractionLab` is this machine's configured research location, not a portable default installation. Verify that the source, runtime, cache and checkpoint paths exist. Keep this source override for the extended commands below; restore both the working directory and prior `PYTHONPATH` before returning to root-source commands, as shown after the monitor section.

### Failure-review workspace

The **Failures** tab reuses the road canvas, paused playback, actor inspector, reference-future toggle and prediction exports. It displays recorded predictions labelled as development-set evidence. Shared worlds remain aligned within each model; world controls between models are independent. Recorded cases replay offline without executing model weights or applying new live stresses.

Reference-report generation requires the original pipeline to report `completed`. The currently documented incomplete study cannot yet run this command successfully; previously recorded analysis folders can still be imported and replayed.

```powershell
& $ResearchPython -m adp analyze-failures --study-root 'D:\InteractionLab\study-v1' --output-dir 'D:\InteractionLab\analyses\reference-v1'
```

Freeze slice definitions before comparisons:

| Filter | Definition |
|---|---|
| City, actor type, role | Recorded city; retained actor class; focal or other actor. |
| Observed speed | `<1`, `[1,5)`, `[5,15)`, `>=15` m/s. |
| Observed turning | Heading change over the final observed second: left `>15°`, right `<−15°`, otherwise straight; insufficient history is unknown. |
| Nearby density | Other retained dynamic actors within 20 m: `0`, `1–3`, `4+`. |
| History coverage | Below 80% versus at least 80%. |

Filters intersect on the same actor. Every slice reports actor and unique-scenario counts; fewer than 50 scenarios is flagged as a small sample. Primary statistics follow complete-future benchmark eligibility; inspector errors over available steps are separately labelled.

Analysis folders contain provenance, full selection-set statistics, JSON/CSV summaries, plots and at most **64 compressed recorded replays within 64 MiB**. Files include `analysis.json`, `records.json.gz`, `summary.json`, `summary.csv` and `failures.png`; no executable model weights are included. Native analysis selection validates and atomically installs the folder. HTTP access uses opaque IDs.

Reference previews rank focal minFDE. Experiment previews combine the 32 largest joint-error regressions and 32 largest directional-model focal errors, deduplicate and fill deterministically. A historical CPU qualification processed 500 reference-selection scenes, producing 64 cases/36.19 MiB; that is workflow evidence, not a lane-direction accuracy result.

### Lane-direction controlled experiment

`direction-study --reference-root PATH --root PATH` and `direction-followup` implement a separate **12 GPU-hour** experiment, gated on completion of the original study. Both variants use eight map-input channels: existing six features plus two zeros for control, or signed unit lane tangents for directional. Tangents derive inside the model from ordered map points, using centred differences, one-sided endpoints and zeros for degenerate points, crosswalks and padding. Legacy six-input checkpoints remain supported without changing cache/inference shapes.

The best map-enabled seed-42 configuration supplies the recipe. Control/directional runs train from scratch for paired seeds **42, 43 and 44**, with matched initialization and batch order. A training-only pilot profiles both variants; the slower throughput, 35% overhead and 1.5-hour reserve determine one frozen epoch count before six runs begin. Five epochs is a projection, not a completed schedule/result.

Compare **final fixed-epoch checkpoints**, not independently chosen best epochs. Final selection scoring/reporting defaults to CPU. The primary outcome is directional-minus-control joint minFDE averaged over paired seeds, with 1,000 scenario-bootstrap draws resampling each scenario across all seeds together. These intervals are development evidence. Interrupted/incomplete six-run work is reported as an incomplete experiment.

No additional acquisition, calibration fitting, locked-holdout inspection, risk-label redesign or automatic weight promotion belongs to this follow-up. Source-order qualification against AV2 successor links and tests of tangents, legacy loading, matched initialization and budget interruption are part of the workflow; no completed accuracy improvement is claimed.

### Training screen

Use the **Training** navigation button or extended desktop `--training` to open saved study/host progress. The view polls every five seconds and shows completed runs/epochs, committed examples, measured throughput, budget limits, source age, transfer verification and cleanup state. Checkpoints normally update every five minutes or at epoch boundaries. Partial progress, calibration, reporting and evaluation are separate from completed-epoch totals; no completion-time estimate is inferred.

The screen reads saved records. It does not itself contact SSH/cloud providers, dispatch jobs or establish live GPU utilization. A connection check, fresh file read, old running observation and independently verified transfer are different evidence.

## Standalone monitoring and distributed compute

The extended implementation includes a **separate local Windows monitor program**. Codex, this chat and the application may close while it runs. Windows must remain awake and connected. It is a per-user sign-in process, not a Windows service, and it still depends on its configured Python runtime and source deployment.

The **Interaction Lab Compute Monitor** desktop shortcut starts/reuses the worker and opens a native status dialog showing PID, last/next check and file locations. Closing that dialog leaves the worker running. The Startup shortcut runs quietly at user sign-in. An exclusive OS lock prevents duplicate monitor owners.

Checks run immediately at startup, then wait **five minutes after the preceding cycle finishes**. This is not a strict five-minute wall-clock guarantee: sleep, network timeouts and long verification/cleanup can delay the next check. Monitor status is under `D:\InteractionLab\remote-compute\monitor\status.json`, with rotating `monitor.log`; aggregate host evidence is `D:\InteractionLab\remote-compute\hosts.json`.

From the configured extended source directory, with `$ResearchPython` and `PYTHONPATH` selecting it, choose the operation you need. Stopping the monitor and removing Startup launch are optional operations:

```powershell
Push-Location -LiteralPath $AppSource

# Install/update this user's Desktop and sign-in Startup shortcuts.
.\scripts\install-compute-monitor.ps1

# Start/reuse the worker and display its native status dialog.
& $ResearchPython scripts/start-compute-monitor.py --show-status

# Read saved monitor status, or request a safe stop after its current cycle.
& $ResearchPython scripts/compute-monitor.py --status
& $ResearchPython scripts/compute-monitor.py --stop

# Remove sign-in launch; preserve monitor data and existing training workers.
.\scripts\install-compute-monitor.ps1 -RemoveStartup
Pop-Location
```

Starting the configured monitor can perform **approved dispatch, transfers and guarded cleanup**. `compute-monitor --once` is also an operational cycle, not a read-only diagnostic or audit shortcut. `--stop` stops the monitor after its cycle; it does not stop training or the application. Shortcut installation supports `-Pythonw` and `-StateDir`; existing arguments require updating after deployment changes.

### Host roles and finite queues

| Host | Role and limitations |
|---|---|
| Local NVIDIA PC | Frozen model study first; direction experiment and extra development queue respect dependencies and separate allowances. |
| Private Ubuntu server | Trusted SSH staging/transfer and finite sequential development jobs. The 4 GB GeForce 940MX needs its own compatible, qualified runtime; do not blindly install the laptop's modern CUDA wheel. |
| Google Colab | Normal interactive free-GPU notebook with user-completed Drive permissions. Saved allocation evidence, immutable snapshots and exports can sync through private Drive. |
| Kaggle | Normal private saved notebook/batch workflow. Read-only status/output access is attempted where authorized; notebook visibility does not imply API output permission. Two allocated T4s are charged even when only one computes. |

The approved development package contains **1,000 training and 64 selection caches plus source/provenance**, without credentials or calibration/locked-evaluation future labels. These jobs are infrastructure/development evidence, not the frozen benchmark. Repeating a seed on another host is hardware replication, not an independent scientific seed.

Historical v1 queues requested four configurations × seeds 42–44, 20 epochs each. Distributed v2 uses four configurations × 10 epochs, with private seeds 45–47, Colab 48–50 and Kaggle 51–53. Host allocations are capped at **6 GPU-hours**, carrying applicable prior charges; Colab's separately approved additional six-hour allocation is distinct from its exhausted original allowance. Original-study and direction-study budgets remain separate.

Workers advance to the next **approved finite job** only after completion and within allowance. Failure, pause, overrun or interruption blocks automatic advancement. Reservations, preparation overhead and all allocated GPUs count according to the pinned protocol; losing a runtime does not reset its ledger. Completed jobs are not retrained. Do not remove leases or ledgers to force recovery.

### Freshness, transfer and cleanup

Colab v2 publishes saved evidence to Drive periodically, but Drive sync can lag and a readable connection-check file does not prove a live GPU. Kaggle API denial was recorded in October; unavailable access must remain explicit rather than fabricated as progress. Running v2 observations expire after ten minutes without fresh source evidence. Connection failures preserve the last observation and its original timestamp.

There is no free-cloud SSH worker, keepalive bypass, guaranteed allocation or unattended runtime reconnection. Notebook permissions and normal interactive controls remain necessary. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html). The monitor can check configured evidence providers; it cannot make an inaccessible cloud session observable.

The transfer contract requires archive/member hashes, approved payload/source/configuration/protocol identities, strict CPU loading/execution of final and best checkpoints, and a durable local receipt before deleting remote copies. Partial exports remain incomplete. The audit found a cached-receipt validation gap, so this intended contract must not be represented as fully hardened.

Operational transfers/staging recheck the **10 GiB reserve**, including archive, extraction and canonical-result copies. Insufficient projected space pauses work; recovered space is recognized later. Preserve pinned 8 GiB settings in frozen protocols. Training-cache retirement waits for both studies and CPU reports, released owners, and an independent strict-CPU-verified selected-model copy with matching proof. Only registered derived training NPZs are eligible; preserve raw sources, selection/calibration/evaluation caches, canonical models, reports, manifests, protocols, ledgers and audit records. Final remote duplicate retirement and permanent cloud deletion retain their approval requirements.

Detailed local-only guides are under the extended `docs/`: `LOCAL_COMPUTE_MONITOR.md`, `DISTRIBUTED_COMPUTE_V2.md`, `CLOUD_COMPUTE.md`, `CLOUD_SAVED_EVIDENCE.md` and `PRIVATE_QUEUE_RETIREMENT.md`. Older three-hour cadence descriptions are historical; they do not override the five-minute-after-cycle implementation.

After finishing extended commands, return to the reference source before using the root CLI, test and packaging examples:

```powershell
Set-Location -LiteralPath $ReferenceRoot
if ($null -eq $PreviousAdpPythonPath) {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
} else {
    $env:PYTHONPATH = $PreviousAdpPythonPath
}
```

## HTTP and native interfaces

The reference service has **10 custom HTTP operations**; the extended service has **14**. Framework `/docs`, `/redoc`, `/openapi.json`, static `/`, `/favicon.svg` and `/assets/*` are additional surfaces.

| Method | Route | Purpose | Source |
|---|---|---|---|
| GET | `/api/health` | Device, checkpoint and readiness information | Both |
| GET | `/api/scenarios` | Available scenario metadata | Both |
| GET | `/api/scenarios/{scenario_id}` | Map, observed tracks and separately identified reference futures | Both |
| POST | `/api/predict` | Baseline/trained forecast with optional validated stress | Both |
| POST | `/api/imports` | Start native-authorized background import | Both |
| GET | `/api/imports/{job_id}` | Import progress and per-scenario errors | Both |
| POST | `/api/imports/{job_id}/cancel` | Cancel import | Both |
| GET | `/api/reports` | Available report data | Both |
| GET | `/api/reports/{name}.png` | `evaluation` or `training` figure | Both |
| POST | `/api/exports` | Native-authorized save/export | Both |
| GET | `/api/training` | Saved study and host progress | Extended |
| GET | `/api/analyses` | Installed recorded analyses | Extended |
| GET | `/api/analyses/{analysis_id}` | Analysis summary and slice filtering | Extended |
| GET | `/api/analyses/{analysis_id}/cases/{scenario_id}` | Recorded replay and prediction records | Extended |

Example prediction body:

```json
{
  "scenario_id": "<available-scenario-id>",
  "model": "trained",
  "stress": {"kind": "position_noise", "severity": 0.35, "seed": 42}
}
```

`model` is `constant_velocity` by default or `trained`; stress is optional and can select an actor. Seeds are unsigned 32-bit values. Responses contain actor IDs, coordinate metadata, means/scales shaped **`[6, actors, 60, 2]`**, six normalized probabilities, raw/available calibrated risk and evidence status, model/calibration provenance, stress metadata and a prediction ID. Reference futures remain separately identified and are excluded from `forward`. Requesting trained prediction without weights returns a conflict rather than silently fabricating it.

The desktop binds loopback, restricts Host/Origin, generates a per-launch session token and uses native dialog grants. Imports, import status/cancel and exports require the desktop `X-Interaction-Token`/bridge context; ordinary `adp serve` denies those operations with 403. Filesystem paths are not arbitrary HTTP inputs. Analysis requests use opaque IDs. `/api/training` reads startup-configured roots and rejects query parameters; analysis detail supports city/type/role/speed/turn/density/history filters.

The extended native bridge has six public methods: `desktop_context()`, `choose_import_directory()`, `choose_analysis_directory()`, `choose_data_directory()`, `choose_model_bundle(archive=False)` and `save_path(kind)`. The reference bridge lacks analysis selection. Export kinds are `prediction_json`, `report_csv`, `report_png` and `diagnostics`. These local controls are not an authorization model for public hosting; exposing the service remotely is outside this beta.

### CLI reference

Run `& $Python -m adp <command> --help` for exact options. CLI names are shared between CPU, GPU and notebook environments.

| Reference command | Purpose / important options |
|---|---|
| `desktop` | Native window; cache/checkpoint/calibration/device, `--baseline`, `--data-root`, `--model-bundle`. |
| `doctor` | CUDA environment/kernel/model forward-backward check; use `--device cuda`. CPU qualification uses `scripts/qualify-runtime.py`. |
| `download` | Immutable subset acquisition, counts/seed/workers. |
| `preprocess` | Masks, coordinates, map/cache generation and explicit storage reserve. |
| `train` | Training, configuration, overfit check, resume and optional runtime bound. |
| `calibrate` | Fit matching calibrators on the calibration partition. |
| `evaluate` | Forecast/coverage/calibration metrics; split, protocol, bootstrap, limit, CV scale. |
| `stress-test` | Matched clean/corrupted or counterfactual reports. |
| `study` | Prepare/pilot/acquire/train/finalize/pipeline controller phases. |
| `benchmark-protocol` | Freeze evaluation IDs and external-model provenance. |
| `compare` | Paired scenario-bootstrap comparison of existing records. |
| `export-model` | Versioned inference-only bundle with schema, hashes, config, model card and optional calibration/evaluation. |
| `serve` | Local FastAPI service; default `127.0.0.1:8000`. |

Extended commands add `analyze-failures`, `direction-study` and `direction-followup`. Extended `desktop` also accepts `--training`, `--study-root` and `--direction-root`; these flags do not exist in the root reference CLI.

For example, export an existing smoke checkpoint without making calibration claims:

```powershell
& $Python -m adp export-model --cache-dir data/smoke/cache --checkpoint artifacts/smoke/best.pt --output-dir artifacts/smoke/inference-bundle
```

Inference bundles are not resumable training checkpoints. Preserve their schema/hash/provenance files together; the presence of a calibrator does not establish applicability to imported or stressed populations.

## Testing, audit and release qualification

Use a separate development/test environment when adding pytest/httpx to a locked runtime. From the selected source tree, with those development dependencies installed:

```powershell
& $Python -m pytest --basetemp .pytest-tmp
Push-Location frontend
npm.cmd ci
npm.cmd test
npm.cmd run build
Pop-Location
```

Tests cover coordinate round trips, missing masks, observed/future separation, graph isolation, map geometry, world alignment, hand-computed metrics, calibration eligibility, scenario bootstrap, checkpoint continuation, stress consistency, API validation and desktop import/export behaviour. Extended tests also cover tangents, recorded analyses, ownership, bundles, transfer receipts and bounded queues. Existing tests are not a claim that all audit findings are fixed.

**Newly executed in the 21 October audit snapshot:** 720 backend tests plus four subtests, 40 frontend tests and a TypeScript/Vite build passed. Bounded NVIDIA forward/backward, optimizer/RNG continuation and checkpoint-portability checks passed. Dependency scans reported no advisories for the then-current runtime/build/npm locks. These are dated results from the reviewed isolated source, not tests rerun for this README edit or proof that every deployment matches it.

The local-only audit folder `artifacts/audit-20261021/` contains `AUDIT.md`, `findings.json/csv`, coverage and endpoint matrices, sanitized evidence, `REPAIR_ROADMAP.md` and Stage B records. It inventories source/deployment differences and reports defects without product-code changes. Main outstanding repairs include:

1. Working-time versus suspended-time accounting and missing-ledger recovery; preserve the historical cap discrepancy.
2. Durable model selection when settings writes fail, and validation of optional evaluation reports that can break rendering.
3. Complete cached transfer-proof validation, source/protocol fingerprints, monitor cadence/recovery and storage handling.
4. Scientific tie-break/diagnostic edge cases, packaging cleanup/notices and remaining error-feedback gaps.

Release qualification remains pending for clean Windows without Python/Node, installed CPU/NVIDIA editions, actual native dialogs, offline restart, upgrades/uninstall, Unicode paths, corrupt/missing resources, repeated lifecycle/orphan-process checks and monitor startup independent of Codex. GPU qualification on the development PC does not qualify other GPUs or a physical CPU-only machine. A blocked provider or VM check is not a pass.

Do not run operational monitor/worker commands against production defaults as audit tests. Use disposable fixtures and cleanup-disabled/mocked providers. Do not inspect locked labels, retrain, fit calibrators or change frozen protocols merely to validate documentation.

## Packaging and installation

There are two distinct build paths:

```powershell
# Developer workspace launcher; requires this source tree and its .venv.
.\scripts\build-app.ps1

# Independent frozen editions, built into separate directories.
.\scripts\build-release.ps1 -Edition cpu -Python $Python -BuildRoot D:\InteractionLab\release-builds\cpu
.\scripts\build-release.ps1 -Edition nvidia -Python $Python -BuildRoot D:\InteractionLab\release-builds\nvidia
```

See [scripts/PACKAGING.md](scripts/PACKAGING.md) for build prerequisites, exact runtime/build locks, NSIS **3.13+**, qualified `-RuntimePython`, `-Python`, `-DemoCache`, `-ModelBundle`, `-Version` and installer options. Release tooling builds the frontend, records source/runtime provenance, runs frozen self-tests and creates checksums. CI definitions under `.github/workflows/` configure Windows tests/builds and release artifacts; configured CI is not evidence of a newly completed release run.

The independent distribution uses **PyInstaller one-directory builds inside per-user NSIS installers**, with Python, dependencies, frontend assets and configured model/demo resources. The CPU edition is normally one setup executable. Large NVIDIA builds require the setup executable **and its matching adjacent payload ZIP**; verify SHA-256 files and keep the pair together. Historical demo packaging is limited to 24 real scenarios/64 MiB.

Install resources are separated from writable user data. The intended upgrade/uninstall design preserves settings, imported data, models and logs; complete clean-machine acceptance and the audit's packaging repairs remain outstanding. WebView2 detection/bootstrap follows [Microsoft's distribution guidance](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution); installing a missing prerequisite can require internet. Normal inference/replay should operate offline once prerequisites and local resources exist.

Local reference artifacts built **18 October 2026** are described in [PRODUCTION.md](PRODUCTION.md): CPU setup under `D:\InteractionLab\release-cpu\dist`, NVIDIA setup/payload under `D:\InteractionLab\release-nvidia\dist`. They are unsigned private beta artifacts, not published downloads, and do not contain every extended feature. Updates are manual and versioned; automatic updates and paid signing are outside this beta.

## Repository guide and troubleshooting

```text
src/adp/                 Reference data, model, metrics, CLI, API and desktop
frontend/                React/TypeScript/Canvas interface and component tests
tests/                   Python regression and research-contract tests
scripts/                 Setup, qualification, launcher and release tooling
requirements/            Exact CPU/NVIDIA runtime and build locks
notebooks/               Portable notebook entry points
.github/workflows/       Windows validation/build/release workflows
config.json              Default model/training configuration
MODEL_CARD.md            Intended use, assumptions and limitations
VALIDATION.md            Historical measured experiments and caveats
PRODUCTION.md            Beta protocol and historical packaging/service evidence
data/                    Local raw/cache resources; not in Git
artifacts/               Generated models, reports and isolated workspaces; not in Git
```

`notebooks/train_colab.ipynb` invokes the same CLI, extracts approved data onto runtime-local storage and persists checkpoints to Drive. Later remote/pilot notebooks belong to the isolated implementation. Account sign-in, Drive permissions and notebook execution remain user/provider-controlled. Moving a portable checkpoint does not transfer training-budget authorization.

| Symptom | Check / recovery |
|---|---|
| Application shows only baseline | Confirm the selected source, checkpoint/bundle path and compatibility. `--baseline` deliberately disables trained inference. |
| Missing/broken native window | Check WebView2 and `%LOCALAPPDATA%\Interaction Lab\logs\app.log`; source use also requires the matching built `frontend/dist`. |
| Training/Failures tab missing | The root application and historical installers lack those screens. Select and build the extended source explicitly. |
| New frontend code is not visible | Rebuild the selected tree's frontend and restart its application. Do not overwrite a frozen running worker's source. |
| CUDA fails or runs out of memory | Qualify the actual GPU/wheel pair, use visible CPU fallback, or reduce microbatch size. Do not assume an older GPU supports the local NVIDIA lock. |
| Resume or calibration rejected | Restore the matching checkpoint, configuration, manifest and preprocessing/provenance records. Do not edit hashes or drop a ledger to bypass the error. |
| Monitor status missing just after launch | Allow the first publication, then inspect `--status`, PID ownership and configured source/runtime paths. |
| Cloud row is unknown/stale | Check source observation time separately from connection time. Drive availability/API sign-in does not prove current training. Use normal provider controls if evidence is unavailable. |
| Transfer remains partial | Preserve source/checkpoints; confirm saved exports and Drive availability. Only independently verified results count as transferred. |
| Storage guard pauses work | Free safe disposable space; allow the next projection check. Preserve raw sources, canonical models and frozen study evidence. |
| Monitor stopped but GPU still active | Monitor shutdown does not stop workers. Use the worker's documented pause/ownership controls. |

For bug reports, include selected source/version, edition/runtime, action, sanitized error and relevant diagnostic evidence. Avoid credentials, account tokens or unreviewed archives. Distinguish a current reproduction from an old host observation.

## Scope, attribution and licensing

The current scope is a free, noncommercial **research desktop beta**. Remaining work prioritizes the audit repairs, trustworthy budget/recovery handling, completed controlled model evidence, provider-observable transfers and installed-application qualification. Neither the original study's completion nor a positive lane-direction result is assumed. Diffusion, full-dataset experiments, CARLA, public hosting, commercial deployment, automatic updates and driving-system integration are outside this delivery.

[Argoverse 2](https://www.argoverse.org/av2.html) data/documentation are provided under **CC BY-NC-SA 4.0**; the [AV2 API](https://github.com/argoverse/av2-api) is **MIT**. Follow the [official AV2 terms](https://www.argoverse.org/about.html#terms-of-use), attribution and redistribution conditions. Cite Wilson et al., *Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting*, when publishing work using the dataset.

Third-party dependencies retain their own licenses. This checkout does **not** currently include a project-level `LICENSE` file; AV2 API licensing does not grant a license to this project's code. Resolve project-code licensing and outstanding release notices before redistribution. No real-world driving-safety claim is made.
