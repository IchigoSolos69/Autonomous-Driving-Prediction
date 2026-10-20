# Free cloud qualification and development jobs

Use `notebooks/train_remote.ipynb` for a standalone, interactive Colab or Kaggle run. The notebook is separate from `train_colab.ipynb`: it accepts only the existing training and model-selection partitions, never calibrates, never opens the locked holdout and never promotes its checkpoint. It leaves the running Windows study, the frozen 40 GPU-hour study budget and the separate 12 GPU-hour direction experiment unchanged.

For the already prepared **`server-pilot-v1-inputs.zip`** private payload, use `notebooks/train_pilot_cloud.ipynb` instead. It contains the standalone bootstrap, requires no full checkout or project installation and invokes an explicitly configured isolated interpreter, even when the notebook kernel uses another Python version. Upload the original ZIP through the platform's file interface. The pinned archive SHA-256 is `f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa`. Every indexed member is checked before extraction or executing its trusted source. This pilot has one FP32 epoch, microbatch one, accumulation 32 and a separate **1,800-second** qualification/training/selection cap. It exports a ZIP plus receipt for local verification, preserving its active run. The embedded script also accepts `--payload PATH --root PATH` when uploaded and run directly.

## Prepare the input

Give each job a new name and one active owner. Supply this exact source revision, `config.json`, `scripts/qualify-runtime.py` and a private preprocessed cache whose `manifest.json` has exactly `train` and `validation` keys. `validation` means model selection here. Each ID must be unique across both partitions. Copy the corresponding NPZs and `preprocessing.json` only. Preserve the source manifest/configuration hashes as provenance; this development subset has its own manifest identity and cannot resume a checkpoint from another manifest.

Record the SHA-256 of that train/selection-only manifest on Windows:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\InteractionLab\cloud-job\cache\manifest.json'
```

Set `REPO_DIR`, `SOURCE_CACHE`, `WORK_DIR`, `EXPORT_DIR` and `EXPECTED_MANIFEST_SHA256` in the notebook. Paths are explicit trusted runtime configuration, never public API input. Colab examples use `/content`; Kaggle inputs are typically under `/kaggle/input` and writable work under `/kaggle/working`. Keep inputs and outputs private under the AV2 noncommercial research terms. This notebook downloads no additional scenarios.

## Qualify before training

The notebook requires Python 3.12 and PyTorch 2.13.0, retaining a compatible platform-provided CUDA wheel. It fails clearly when the allocated runtime differs or has no GPU. Qualify an official compatible environment separately rather than substituting an untested build. It records the interpreter, libraries, driver, device and capability.

Checks cover a batch of four with 32 actors and 128 polylines, all six shared worlds, finite FP32 gradients, optimizer stepping, Gaussian bounds, probability normalization and a weights-only checkpoint round trip with CPU/GPU prediction agreement. Mixed precision is used only after a successful finite-gradient trial. The existing runtime script checks optimizer/RNG continuation; on resume it also loads this job's trusted checkpoint. One actual training batch measures real data execution separately from synthetic checks. Timing from that batch is qualification evidence, not a sustained throughput or accuracy result.

Runtime-local staging verifies copied bytes and publishes the cache atomically. The importer excludes every file not identified in the permitted partitions, rejects unsafe IDs and symlinks and reserves 8 GiB free storage. It never deletes the source folder. A failed staging directory is retained for inspection and is never treated as a completed cache.

## Train and export

The default development allowance is **two GPU-hours total**, including qualification, training and selection. This is an explicit infrastructure job allowance, not an extension of either frozen experiment. One interactive training cell uses the existing CLI, with a bounded runtime and resumable optimizer-group checkpoints:

```text
python -m adp train --config JOB/config.json --cache-dir JOB/cache --output-dir JOB/run --epochs 2 --device cuda --max-runtime-seconds SECONDS
```

Use `--resume JOB/run/last.pt` only when that run has stopped elsewhere and source, configuration, preprocessing and manifest match. The notebook sets microbatch four, accumulation eight, no data-loader workers and a five-minute checkpoint interval. A slice can overrun by one optimizer group or validation batch plus checkpoint I/O. A terminated runtime conservatively consumes the reserved slice until independently reconciled; do not erase the ledger to acquire additional time.

After the CLI returns, the snapshot cell creates an atomic ZIP and JSON receipt, verifies every included member hash and records the archive hash. Snapshots contain resumable/best checkpoints, configuration, history, qualification, provenance and the separate budget ledger. They contain no dataset NPZs or account credentials. Download both files with the notebook's normal file/output interface, or write them to an explicitly mounted private Drive export directory. An interruption before export can lose local progress; these free runtimes do not provide guaranteed persistence.

The Windows/private-server transfer workflow is separate. **Delete only acknowledged immutable export snapshots after local hash and checkpoint verification. Preserve active `last.pt`, `best.pt`, run/configuration files and source data.** A notebook runtime is never turned into an SSH endpoint or a polling worker. Its export ZIP/receipt format is self-contained; it is not silently assumed to be the private-server snapshot-directory protocol.

## Free platform limits

Colab's free resources and limits vary, and sessions may terminate. Free managed runtimes prohibit remote SSH control and distributed workers; use the normal notebook UI. No keepalive, multiple-account workaround or purchase is part of this workflow. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html).

For Kaggle, inspect the quota shown in your own notebook settings; no fixed weekly allowance or GPU model is promised here. Select an available accelerator and end the session when finished. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks), [efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [NVIDIA's Kaggle setup instructions](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/).

An authenticated account and an actual allocated GPU are needed before cloud qualification can be reported as passed. This repository supplies the workflow; a locally passing notebook helper test does not establish cloud execution, available quota or model quality. No credentials are included, and the notebook cannot sign up, purchase compute or transfer files from a disconnected account.
