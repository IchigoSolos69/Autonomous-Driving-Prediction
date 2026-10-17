# Model card — Joint motion and proximity-risk predictor

## Intended use

Offline research, learning, portfolio demonstrations, and inspection of interaction forecasts. The local application does not control a vehicle. It does not establish roadworthiness or real-world safety.

## Architecture

Two temporal Transformer layers (128 channels, four heads), two relative-state graph attention layers, polyline map encoding and cross-attention, six shared scene-mode queries, and a diagonal Gaussian trajectory-mixture decoder. All modes remain aligned across actors. A separate binary head consumes interaction features and forecast clearance summaries.

## Data

Argoverse 2 Motion Forecasting: 5 seconds observed, 6 forecast, 10 Hz. The default subset is 5,000 official training scenarios and 2,000 official validation scenarios partitioned 500/500/1,000 for selection/calibration/evaluation. Exact scenario IDs, seed, and UUID-prefix sampling limitations are recorded in each immutable manifest. The default subset has not been evaluated merely by implementing the pipeline. Consult the generated run reports for actual data sizes and results.

Maximum 32 retained actors and 128 nearby polylines. Actor/type/scored coverage and incomplete-label exclusions accompany metrics. No raw imagery, lidar, measured actor extents, or driver intent are inputs. No synthetic fixture is treated as AV2 evaluation data.

The local delivery includes a three-epoch run on the default subset. `VALIDATION.md` records actual held-out results and their limits: modest displacement improvements, a worse footprint-overlap proxy, and insufficient positive event labels to establish risk calibration. Weights and source data are generated artifacts rather than committed repository contents.

## Risk definition

For a retained dynamic actor: another retained dynamic actor's centre is less than 3 m away at any sampled timestep over the forecast horizon. This is a proximity proxy and includes benign encounters. Complete futures for the retained group are required for event calibration; exclusions change the evaluated population and must be considered when interpreting scores.

The footprint proxy uses oriented rectangles with fixed `(length,width)` in metres: vehicle `(4.5,1.8)`, bus `(12,2.5)`, pedestrian `(0.6,0.6)`, cyclist `(1.8,0.6)`, motorcyclist `(2,0.8)`. Prediction headings follow trajectory displacement, falling back to observed headings when stationary. These are assumptions, not measured collision labels.

Temperature scaling calibrates nearest-joint-FDE mode classification. Monotone logistic scaling calibrates supervised proximity logits on the reserved calibration split. Neither metric establishes physical crash probability. Fewer than 50 positive events is explicitly insufficient evidence. Distribution shift invalidates any assumption that calibration remains unchanged.

## Evaluation and limitations

Reports include actor and joint-world displacement errors, miss rate, mixture likelihood, Brier-FDE, reliability, risk proper scores/AUPRC, overlap proxies, and scenario-bootstrap intervals. The constant-velocity baseline's score is an uncalibrated heuristic. Six worlds cannot represent all combinations of independent actor behavior; diagonal Gaussian scales simplify temporal uncertainty. Context caps omit road users. Maps can be incomplete, and class footprints are coarse.

Observation corruption preserves logged reference futures. Behavioral interventions are scripted, single-actor, nonreactive replays. Unsupported cut-ins are excluded with reasons. These experiments measure controlled robustness, not a reactive planner's performance. Results from tiny overfit/smoke runs are implementation checks and must not be presented as benchmark performance.

## Reproducibility

Store configuration, manifest hash, preprocessing version, seeds, checkpoint/calibrator hashes, environment versions, training history, and evaluation records with every experiment. GPU-to-CPU checkpoint transfer is supported; exact numerical continuation across hardware is not guaranteed. The Colab notebook is supplied for user-run free GPU sessions, not evidence that cloud training has occurred.

## Licensing

AV2 data: CC BY-NC-SA 4.0. AV2 API: MIT. Preserve dataset attribution and terms when sharing derived data or results.
