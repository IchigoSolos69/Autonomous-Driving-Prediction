# Failure review and lane-direction development experiment

This milestone adds recorded failure review to the desktop application and a separate, capped experiment. Follow-up models remain experimental. The workflow does not acquire data, fit calibrators, read calibration/evaluation scenario labels, promote weights, or rebuild the reference installers.

## Run from an isolated source checkout

Keep the running production study's source and artifacts unchanged. Use Python 3.12 and the already qualified runtime; set `PYTHONPATH` to the isolated checkout's `src` when it is not installed into that environment. Build its interface with `npm.cmd run build` inside `frontend`.

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$python = 'D:\InteractionLab\runtime-cu130\Scripts\python.exe'

# Build one reference report after the reference study completes.
& $python -m adp analyze-failures --study-root D:\InteractionLab\study-v1 --output-dir D:\InteractionLab\reference-failures-v1

# This command returns waiting_for_reference until the original pipeline finishes.
& $python -m adp direction-study --reference-root D:\InteractionLab\study-v1 --root D:\InteractionLab\direction-study-v1

# One-off dependency workflow: wait, reference report, experiment, experiment report.
& $python -m adp direction-followup --reference-root D:\InteractionLab\study-v1 --root D:\InteractionLab\direction-study-v1
```

`direction-followup` polls at 30-second intervals. Waiting does not consume the experiment's compute ledger. It stops if the reference pipeline fails or the experiment is incomplete. It prevents duplicate runners, keeps Windows awake while active, restores the previous execution state on exit, and records an actionable status. Re-run the same command to continue after resolving a storage interruption. Existing reports are validated against current provenance before reuse.

All generated data stays under the new experiment root. Existing caches are referenced in place. Both report generation and training preserve an 8 GiB free-space reserve. Reports need at most 64 MiB each; checkpoint and score files require additional space. A reserve failure is a safe pause, not successful completion. Do not delete original results to make room automatically.

If the original pipeline has already frozen `protocol.json`, use the isolated helper below to resume it after a storage interruption. The original preparation code otherwise scans newly created test manifests and tries to add them to an immutable benchmark. The helper verifies the exact raw/cache manifests, split hashes, original provenance files and protocol equality, then runs the original pipeline with only its recorded provenance snapshots. Additional test manifests are recorded and ignored; no files are deleted and the original source is not edited. A separate hash receipt is saved under `study-v1/resume-receipts/`.

```powershell
& $python .\scripts\resume-frozen-study.py --study-root D:\InteractionLab\study-v1 --project-root 'C:\Users\Adi\Documents\GitHub\Autonomous Driving Prediction' --device cuda --workers 8
```

## Desktop Failures workflow

Launch the isolated application using `adp desktop` and its built frontend. Select **Failures**, then **Load analysis folder**. The native folder chooser validates and atomically copies a portable report into the application's writable data directory. Cancelling changes nothing. HTTP routes accept opaque IDs rather than source paths:

For this machine, launch from the isolated checkout using the historical compatible model while the expanded study runs:

```powershell
$referenceProject = 'C:\Users\Adi\Documents\GitHub\Autonomous Driving Prediction'
& $python -m adp desktop --project-dir "$PWD" --cache-dir "$referenceProject\data\cache" --checkpoint "$referenceProject\artifacts\experiment\best.pt" --device cpu --data-root D:\InteractionLab\failure-workspace-data
```

This uses the existing reference weights and leaves the training GPU available. It does not promote a new model. Load `D:\InteractionLab\validation-failure-smoke\historical-analysis` to inspect the qualified historical report.

Select **Training** in the header, or add `--training` to the desktop command, for a full-width progress screen. It reads the original and follow-up study records without modifying their source or artifacts. Overall progress counts completed epochs across six runs; the active run's committed partial epoch is shown separately. Checkpoint timestamps, completed-epoch curves, measured throughput, ledger reservations and the 8 GiB storage floor distinguish saved evidence from live activity. No ETA is inferred. Missing and temporarily unreadable records are labeled; the monitor retains the last readable progress while retrying.

Read-only interfaces:

- `GET /api/analyses`
- `GET /api/analyses/{id}` with optional `city`, `actor_type`, `role`, `speed_bin`, `turn_bin`, `density_bin`, `history_bin` filters
- `GET /api/analyses/{id}/cases/{scenario_id}`

The full statistics cover every scenario in the selection manifest. Filters intersect on the same actor. Each slice shows actor and unique-scenario counts; fewer than 50 scenarios is flagged as a small sample. Joint errors describe whole scenes containing matching actors and preserve shared-world alignment.

Frozen observed-only slice definitions:

| Slice | Definition |
| --- | --- |
| Speed | Cutoff speed below 1, 1–below 5, 5–below 15, or at least 15 m/s |
| Turning | Signed heading change from observed step 39 to 49; left above 15°, right below −15°, otherwise straight; missing endpoints are unknown |
| Density | Other retained dynamic actors within 20 m at the cutoff: 0, 1–3, or 4+ |
| History | Below 80% or at least 80% of the 50 observed steps |

Primary actor displacement statistics use the existing scored dynamic actor eligibility and require all 60 future steps. Reports expose incomplete-future exclusions and retained-target coverage. Exported inspector errors over available reference steps are labeled separately and must not be treated as benchmark aggregates.

Recorded cases start paused. Use the existing canvas, playback, scrubbing, actor selection, reference futures and JSON/CSV/PNG exports offline. Select the paired training seed and each model's world independently: a world index is shared across actors within a prediction, but does not identify the same world between models. Live forecasting and stress edits are disabled for recorded cases. Raw risk remains explicitly experimental, with no validated calibrated-risk claim.

## Portable report format

An analysis directory contains `analysis.json`, `records.json.gz`, `summary.json`, `summary.csv`, `failures.png`, and up to 64 compressed recorded cases. It contains provenance and predictions, without executable model weights. File hashes, schema, finite values, mode/actor alignment, bounded expansion and filesystem links are validated before import. Total stored size is capped at 64 MiB; fewer previews may be retained when needed. Statistics still cover the complete selection set.

Reference previews rank by focal minFDE. Experiment previews combine the 32 largest paired-seed mean joint-error regressions and 32 largest mean directional focal errors, deduplicate, then fill deterministically. Predictions are labeled as recorded development results.

## Controlled comparison

The original pipeline must finish with `status=completed` and `phase=finished`. The follow-up chooses the best map-enabled seed-42 screening configuration by the existing joint minFDE, focal minFDE, then earlier-epoch rule.

Both variants use eight map-encoder inputs. Control appends two zeros; directional appends signed unit lane tangents derived inside the model from ordered cached points. Interior points use centered differences, endpoints one-sided differences, and degenerate points/crosswalks/padding use zeros. The legacy six-input encoder remains the default when loading existing checkpoints. Cache and inference-input shapes stay compatible.

Lane order is checked on the existing 1,000 training-only pilot against [AV2 successor topology](https://github.com/argoverse/av2-api/blob/main/src/av2/map/lane_segment.py). The implementation audit of that pilot found 68,888 forward successor links, zero reversed and zero ambiguous; the runner repeats and records this qualification before profiling.

The two variants train from scratch with paired initialization and batch order for seeds 42, 43 and 44. All other model, loss and optimizer settings come from the selected configuration. Both pilots, training and training-time validation count against a separate 12-hour ledger. Crash reservations are retained conservatively. The slower measured pilot throughput, 35% overhead and a 1.5-hour reserve freeze one uniform epoch count before the six runs. The number of epochs is measured rather than assumed to be five.

The comparison uses **final fixed-epoch `last.pt` checkpoints**, not independently selected best epochs. Final selection scoring runs on CPU. The primary difference is directional minus control joint minFDE, averaged across the three seeds. Exactly 1,000 bootstrap replicates resample scenarios together across all paired seeds. Reports include individual seeds, focal errors, miss rate, likelihood, overlap proxies, coverage and learning curves. Intervals are development evidence after configuration selection. An incomplete run produces no completed six-run comparison. Negative or inconclusive results remain valid outcomes.

Important artifacts under the experiment root:

- `protocol.json`, `pilot.json`, `schedule.json`: immutable membership, source/configuration hashes, lane qualification, measured throughput and fixed epochs
- `direction_ledger.json`, `experiment.json`: separate compute use, resumable state and completion evidence
- `runs/{control,directional}-{42,43,44}/`: checkpoints, objective components, learning curves
- `scores/`, `comparison.json`, `REPORT.md`: complete CPU selection scores and paired evidence
- `reference-analysis/`, `direction-analysis/`: portable failure reports from the one-off workflow
- `followup_request.json`, `followup_status.json`: one-off dependency workflow provenance and status

## Qualification limits

Automated tests cover tangent signs/reversal/duplicates/padding, legacy loading, matched initialization, observed-only inference, aggregation, clustered bootstrap, budget interruption, corrupt folders, traversal, Unicode paths, cancelled selection, offline replay, independent worlds and exports. The implementation passed 163 Python tests (plus four subtests), 24 frontend tests and a production frontend build. A memory-backed check used three historical real selection scenarios with 77 retained actors and the existing compatible checkpoint. Visual replay checks covered paused startup, future scrubbing, reference tracks and independent world controls.

The saved-folder integration check processed all 500 historical selection scenarios (11,584 retained actors) on CPU, published 64 cases in a 36.19 MiB portable folder, validated atomic import and full-set filtering, and exported JSON/CSV/PNG comparisons. The measured report/import/export check took 150.93 seconds; this is implementation qualification using historical data, not evidence for the new experiment. Its receipt is under `artifacts/validation/real-failure-report.json`; generated data is at `D:\InteractionLab\validation-failure-smoke`. Native startup, folder-picker cancellation, persisted report loading after restart, paused case replay and clean local-service shutdown were checked with isolated test settings. Automated control of the native picker's path field was unreliable, so backend import and native replay checks are identified separately.

The controlled six-run experiment remains pending until the original study completes; no follow-up accuracy result should be inferred from implementation tests. Refreshed installers and clean-machine acceptance are a separate milestone.
