# Interaction Lab research desktop beta

This release keeps the native Windows application and its React/Canvas interface. Frozen editions bundle Python and dependencies; users do not need this repository, Python, or Node. It is a noncommercial research tool under the [AV2 terms](https://www.argoverse.org/about.html#terms-of-use), not a driving system.

## Application and installation

The CPU release is one setup executable. The NVIDIA release is a setup executable **plus its matching adjacent payload ZIP**; keep both files together. Check the supplied SHA-256 files. Installers run per user, detect WebView2, and preserve settings, imported datasets, models and logs during upgrades/uninstall. WebView2 installation can require internet; ordinary inference is local. Artifacts are unsigned private betas.

Local 0.1.0-beta artifacts built on 18 October 2026:

- [CPU installer](D:/InteractionLab/release-cpu/dist/Interaction-Lab-0.1.0-beta-cpu-Setup.exe), approximately 172 MiB.
- [NVIDIA installer](D:/InteractionLab/release-nvidia/dist/Interaction-Lab-0.1.0-beta-nvidia-Setup.exe) and its required [matching payload ZIP](D:/InteractionLab/release-nvidia/dist/Interaction-Lab-0.1.0-beta-nvidia-payload.zip), approximately 1.92 GiB combined.
- Each `dist/SHA256SUMS` and adjacent `.sha256` sidecar verifies the shipped files. [Combined local installation/cleanup receipts](artifacts/packaging/final/validation.json) record source hash `9301a5edaeed0d8e6e7b3c13846e8f9d49f1c27650e46e42f6607865b13dcfef` and exact artifact hashes.

Both editions were installed into dedicated test folders and verified using their own bundled Python/Torch, frontend, 24 real demonstration scenarios and trained reference weights. CPU verification took 4.88 seconds with 394 MB peak process working set; CUDA verification took 17.07 seconds with 1.63 GB. These are whole verification-process measurements, not native UI startup or VRAM. Uninstall removed application files/registration and preserved the test user data plus the existing application log. The test installations were then removed; these paths point to installers, not a claim that an edition is currently registered on this computer.

Use **Import AV2 folder** in the scenario library to choose a directory containing AV2 parquet/map pairs. Import runs in the background and supports cancellation. Invalid scenarios have specific errors; incomplete caches never become visible, and source files are preserved. Data location and model-bundle selection use native dialogs and take effect on restart.

Prediction JSON, comparison CSV, figures and diagnostics are saved through explicit native dialogs. Reference futures remain separate from the observed-only model input. Imported and stressed populations use raw scores with unknown calibration applicability. A fitted risk transformation is not exposed as validated unless both calibration and evaluation have at least 50 positive events and their provenance matches.

See [packaging instructions](scripts/PACKAGING.md) for exact runtime locks, CI, installer construction, and installed-application verification.

## Model-first study

The original three-epoch model remains the reference. No stronger-model claim follows from the new implementation or pilot. The new immutable manifest selects 40,000 official training scenarios and 12,000 fresh official-validation scenarios using SHA-256 ordering over complete official listings with seed 42. The latter split is 2,000 selection, 5,000 calibration and 5,000 locked evaluation scenarios. Previously inspected validation IDs, including smoke/audit datasets, are excluded.

An isolated Python 3.12/PyTorch 2.13 CUDA 13 runtime has passed actual GPU forward/backward, optimizer/RNG continuation and original-checkpoint CPU/GPU portability checks on the RTX 5050 Laptop GPU. The training-only pilot uses 1,000 real scenarios and discards its learned weights. Its report records configuration, throughput, raw/cache size, label eligibility and allocated/reserved CUDA memory.

The first pilot measured approximately 40.5 examples/second and projected a 20-epoch study beyond the ceiling. The uniform schedule was therefore reduced to **15 epochs before screening**. Updated configuration-bound measurements retain this conservative schedule. The study preserves 8 GiB free on D:, including checkpoint temporary files, and pauses rather than reducing sample counts when capacity is insufficient.

Four seed-42 configurations are screened: full reference loss, no graph, no map, and full model with a 0.1 nearest-joint-FDE world-classification auxiliary. The winning configuration is repeated with seeds 43 and 44. Selection uses joint minFDE, then focal minFDE, then earliest epoch. A shared 40-hour GPU ledger meters training, calibration and evaluation; interrupted work conservatively charges its reserved slice. A single optimizer group/validation batch and checkpoint I/O set the deadline granularity.

The deployment checkpoint is frozen before calibration. Final evaluation compares all three seeds, the reference model, and fixed/calibration-fitted-scale constant velocity. Promotion requires paired 1,000-replicate scenario-bootstrap joint-FDE intervals below zero against both references. Cached reports must match the exact request and full scenario membership. Failed/insufficient evidence preserves the reference model; no blind-test retuning occurs automatically.

Matched footprint diagnostics use identical valid actor pairs/timestamps and separate existing-cutoff from new overlaps. These fixed-dimension mean-path proxies do not establish crash probability. Corruption and nonreactive behavior counterfactuals remain separate, with matched clean records and unsupported-case exclusions.

## Run or resume

From this checkout, using the already qualified local runtime:

```powershell
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& 'D:\InteractionLab\runtime-cu130\Scripts\python.exe' -m adp study pipeline --root 'D:\InteractionLab\study-v1'
```

This prepares or resumes manifests, reuses a matching pilot, acquires/preprocesses data, trains the frozen schedule, calibrates/evaluates, and exports an inference bundle only if the promotion gate passes. Repeating the command resumes committed training progress and compatible complete evaluations. `pipeline_status.json`, `study_ledger.json`, per-run `history.json`, `frozen_selection.json`, `evaluation_plan.json`, and `promotion.json` record actual progress. Never interpret a prepared manifest as a finished download or a running study as completed evaluation.

The local pipeline was launched on **18 October 2026 at 23:37 IST**, after generated build cleanup left 32.21 GiB free on D:. Its first observed phase is `acquire`, with eight acquisition/preprocessing workers. [Live phase/status](D:/InteractionLab/study-v1/pipeline_status.json) and [launch/process/log locations](D:/InteractionLab/study-v1/pipeline-process.json) are authoritative for its current state. Training and fresh-holdout evaluation were not complete at launch. The runner prevents idle Windows system sleep for this process and restores the prior thread state on exit; manual sleep and critical battery actions remain under Windows/user control.

Keep the study's training/model source and configuration fixed while it runs. When it finishes, inspect `promotion.json`. A passing result publishes `promoted-model-bundle`, selectable through the application's model settings or usable as `-ModelBundle` for the next manually versioned installer build. The current installers continue to contain the reference model; no automatic application update occurs. A failed promotion gate retains the reference and reports the measured comparisons.

Individual `study prepare|pilot|acquire|train|finalize` phases remain available. `benchmark-protocol`, `compare`, and `export-model` expose the same reproducible interfaces. For Colab, copy the immutable manifests/cache bytes and use a runtime-local `--root`; checkpoints remain portable. Cloud execution has not been claimed or configured automatically.

## Acceptance evidence and remaining checks

Local reference-model service measurements used Windows 11 build 26200, 23.6 GiB reported usable RAM, an Intel Core i7-14650HX and RTX 5050 Laptop GPU (8 GB, NVIDIA driver 617.14), two PyTorch CPU threads, a 27-actor demonstration scenario, and 20 warmed requests. These are descriptive measurements on this development computer while release builds were running; they are not minimum hardware requirements or native-window startup measurements.

| Measurement | CPU edition runtime | NVIDIA edition runtime |
|---|---:|---:|
| Python imports to HTTP-ready service | 4.45 s | 17.08 s |
| Median server forecast computation | 20.93 ms | 10.83 ms |
| 95th-percentile server computation | 38.57 ms | 13.56 ms |
| Median complete loopback JSON request | 180.82 ms | 187.46 ms |
| Peak process working set, excluding WebView2 | 357.8 MiB | 1,517.4 MiB |
| Peak PyTorch CUDA allocated / reserved | — | 74.1 / 92.0 MiB |

The source service used the exact release runtimes and exported reference bundle. Startup timing starts immediately before scientific imports; it excludes executable bootstrap and WebView2 rendering. Request latency includes serialization of all six worlds. CUDA allocator figures exclude driver/context memory; the process working set is host RAM. Raw measurements and the repeatable harness are under `artifacts/production/service-{cpu,cuda}.json` and `measure-service.py` in this workspace. Only the listed NVIDIA hardware has been exercised; physical CPU-only hardware and other GPUs remain unqualified.

Source tests cover coordinate/mask separation, shared worlds, hand-computed metrics, exact resume, budget/storage stops, protocol leakage, truncated-report rejection, import cancellation, native tokens, calibration evidence, bundle corruption and installer rollback. Frontend component tests cover readiness races, imports/cancellation, stale responses, exports, settings and evidence labels.

Actual local runtime and installer checks are recorded under `artifacts/production/` and the release build directories. Installed-artifact verification is distinct from source tests. A clean Windows VM without developer tools, actual native-dialog interaction, offline relaunch and physical CPU-only-machine testing remain acceptance items until explicitly recorded. Windows UI-control approval timed out during this implementation; no successful native dialog test is inferred from headless checks.
