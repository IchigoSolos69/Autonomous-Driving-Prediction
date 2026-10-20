# Compute operations

Operational record updated **20 October 2026**. Infrastructure pilots and the new development queue use only the approved **1,000 training and 64 selection scenarios**. They provide no calibration, locked-holdout, safety or model-promotion evidence. The original **40 GPU-hour** study and separate **12 GPU-hour** map-direction experiment retain their existing protocols and budgets.

## Current deployment state

Observed **20 October 2026, 07:11 UTC**: the independent monitor held its OS lock with actual worker PID **33896** (launcher PID 16956), the approved queue hook was enabled, and its next scheduled check was **10:02 UTC**. Desktop and user-sign-in Startup shortcuts are installed; the chat heartbeat remains **paused**. The original GPU study had **40/90 saved epochs**: `full-42` and `no_graph-42` finished 15 each, while `no_map-42` had saved epoch 10. Its unchanged ledger charged **24.40/40 GPU-hours**, including the current reservation. The direction watcher and extra local queue remained waiting for their prerequisites. These are point-in-time observations, not live status.

The private server is `aadi@192.168.29.71`. SSH was restored after its reboot. Its new development queue was directly observed running as PID **108576**, with the first `full` / seed-42 job at **10/20 epochs** at 07:11 UTC; the remaining jobs follow only after successful completion. The GeForce 940MX was observed doing GPU work. Its historical one-epoch pilot remains completed, archived locally and cleaned remotely; absent retired checkpoints must not trigger a rerun. Colab and Kaggle's new queues were still being configured at this observation; their historical verified pilots do not prove that new jobs are running.

The original 16 source files, protocol, configuration, reference checkpoint and resume helper still matched the frozen resume receipt; the seven-file direction guard also matched. At that observation, D: had **9,002,942,464 bytes free (8.38 GiB)** after removal of the checked, disposable upload alias, with the original **8 GiB** reserve. The newer operational monitor/controller policy described below requires 10 GiB and rechecks current capacity. The current 40-hour study may reach its fixed budget before all requested epochs; no extension or promotion is implied.

The `interaction-lab-compute-transfers` chat heartbeat is **paused**. The separate monitor process owns transfers, with an immediate first cycle and a five-minute wait after each completed cycle. It works without this chat or the application open. Its operational preflight now dynamically preserves 10 GiB free, including projected transfer copies; frozen studies and approved workers retain their original 8 GiB protocol settings. See [LOCAL_COMPUTE_MONITOR.md](LOCAL_COMPUTE_MONITOR.md) for Desktop/Startup launch, status, stop, one-cycle operation, dispatch and pause controls.

## Historical qualification and results

These one-epoch pilots are reference artifacts. Throughput and end-to-end duration measure different scopes; they do not establish relative forecast accuracy or sustained full-study performance.

| Host | Verified work | Measured result | Recorded lifecycle |
| --- | --- | --- | --- |
| Private 940MX | One epoch on 1,000 training and 64 selection scenarios; final/best checkpoints verified locally with strict CPU loading. | 17.49 training scenarios/s; 64.75 s training plus selection. | Retired 479,377,903 bytes of owned duplicate cache, input ZIP and captured checkpoints. Preserved source, audit metadata and canonical local results. |
| Colab T4 | Python 3.12.15, PyTorch 2.13.0+cu130; full-shape forward/backward, CPU/GPU agreement and real-data training passed. | 23.97 training scenarios/s; 43.51 s training plus selection; 50.14 s qualification plus training/selection. | Verified local export/receipt; removed 498,842,812 bytes across 1,068 owned runtime files, then disconnected and deleted the runtime through its UI. |
| Kaggle, two T4s allocated | Python 3.12.15, PyTorch 2.13.0+cu130; full-shape forward/backward, CPU/GPU agreement and real-data epoch passed. One GPU performed computation. | 26.23 training scenarios/s; 39.85 s training plus selection; 45.73 s qualification plus training/selection. | Independently verified final/best models before removing 267,214,059 bytes in 1,067 owned runtime files. Stopped the session and deleted both approved private input datasets; retained the private notebook and local models. |

Verified historical result directories:

```text
Private: D:/InteractionLab/remote-compute/results/dc21e6730617431787546f38a90ff4e885a9b0df68507cddb95d7b20f2759f71
Colab:   D:/InteractionLab/remote-compute/cloud-results/colab/66a12c129f59539e77277bb15eccadb4921918b51ac5a048902a10c7612d1c11
Kaggle:  D:/InteractionLab/remote-compute/cloud-results/kaggle/f4802f6272d76d4151de25d1e23aba15fe7ec63cb858a68141aae7a527e9b716
```

Preserve these results, import receipts, qualification records and cleanup journals. Cloud cleanup receipts are under `D:/InteractionLab/remote-compute/cleanup/`: `colab-cleanup-receipt.json`, `kaggle-cleanup-receipt.json` and `kaggle-lifecycle-receipt.json`. The Kaggle working-file receipt hashes to `c8d701d4c9fecca889201b7fd5f3deb6be657bd93482afe09293a6e28365e899`.

The deleted historical Kaggle datasets were `adimaitre/interaction-lab-compute-pilot-20261019` and `adimaitre/interaction-lab-pilot-archive-20261019`. Old SDK examples naming them cannot fetch those deleted inputs. The new queue uses a separately staged, checked upload; historical jobs and inputs must not be treated as live.

The original Windows study was previously resumed from **19/90 completed epochs** and a committed partial cursor of **17,184 examples** after a dead worker was identified. These are historical resume coordinates. The direction watcher was restored to wait for it. Saved study records and the application supply current progress.

## Finite development queue

The new allowance is **six additional allocated GPU-hours per host**: local Windows, private server, Colab and Kaggle. Each finite plan has 12 fresh jobs:

| Configuration | Graph | Map | Shared-world classification auxiliary weight |
| --- | --- | --- | --- |
| `full` | enabled | enabled | 0 |
| `no_graph` | disabled | enabled | 0 |
| `no_map` | enabled | disabled | 0 |
| `world_ce` | enabled | enabled | 0.1 |

Each configuration runs seeds **42, 43 and 44**, from scratch, requesting **20 epochs**. Microbatch is one, accumulation 32, FP32, no data-loader workers and five-minute checkpoint saves. Six shared worlds and existing optimizer/loss settings remain. A repeated seed on another host is hardware replication, not another independent training seed.

The host ledger caps **21,600 allocated GPU-seconds**, at most **1,800 per job**, including qualification, staging, training, selection and export overhead. Local/private/Colab contracts allocate one GPU. Kaggle allocates two T4s and is charged **2 × wall time**, even though one GPU computes; its job wall allowance is at most **900 seconds**, with preparation/export time deducted before training. These caps do not guarantee that 20 epochs complete. Partial, interrupted and over-budget results remain incomplete.

Completed jobs advance sequentially. A failed or paused job stops its worker. A crash preserves the active lease and requires positive process-death proof for recovery; no stale timestamp resets ownership or erases cost. `--stop-after-current` pauses before the next claim without killing the current job; `--resume-queue` requires the exclusive worker lock and a validated pause flag. Stopping the monitor alone leaves existing queue workers intact.

The local extra queue waits for the original and direction studies, CPU reports and verified derived training-cache cleanup. The private namespace is `/home/aadi/interaction-lab/queues/development-v1`, separate from retired `server-pilot-v1`. Shared base inputs remain leased to pending jobs.

Local records:

```text
D:/InteractionLab/remote-compute/queues/development-v1/approved-plan.json
D:/InteractionLab/remote-compute/queues/development-v1/dispatch-status.json
D:/InteractionLab/remote-compute/queues/development-v1/last-sync.json
D:/InteractionLab/remote-compute/queues/development-v1/results/
D:/InteractionLab/remote-compute/queues/development-v1/results/queue-receipts/
D:/InteractionLab/remote-compute/queues/development-v1/local/
```

Payload: `D:/InteractionLab/remote-compute/server-pilot-v1-inputs.zip`, SHA-256 `f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa`. Approved manifests freeze source/helper/resource hashes, exact host plans, configurations, input/preprocessing hashes and ownership. Changed resources block dispatch; existing approved plans are not rewritten.

From the managed workspace, these implemented controller commands read private status or execute one approved dispatch/transfer cycle:

```powershell
$ResearchPython = 'D:\InteractionLab\runtime-cu130\Scripts\python.exe'
$env:PYTHONPATH = 'src'
& $ResearchPython -m adp.private_queue --status
& $ResearchPython -m adp.private_queue `
  --plan 'D:\InteractionLab\remote-compute\queues\development-v1\approved-plan.json'
```

`--approve-plan` persists authorized resources only when the plan is absent or identical. It cannot replace an existing plan with different hashes. The startup monitor passes the default approved plan when present; direct `compute-monitor.py` invocations require `--queue-plan` for dispatch.

## Transfer, verification and cleanup

Three-hour monitor cycles collect completed private queue snapshots and refresh evidence. ZIPs contain opaque snapshot IDs, flat allowlisted files, member/archive hashes and exact queue/job provenance, without dataset NPZs or credentials. `adp.queue_results.import_queue_result` independently checks the original pinned payload, configuration, source/cache/protocol identities, completed 20-epoch status and strict CPU observed-only execution of final/best checkpoints. A durable local import/queue receipt precedes remote cleanup. Partial exports are not accepted as completed queue results.

The local operational controller now applies a **10 GiB reserve** against current capacity and projected incoming copies. Frozen study and approved worker protocols keep their original **8 GiB** reserve. Incoming archives, extracted verification copies and permanent local results may coexist. Insufficient projected capacity pauses transfers and preserves their source. Cleanup accepts only registered copies of completed jobs matched to verified receipts and exact checkpoint/cache bytes. Source, protocols, logs, budget metadata, shared inputs needed by pending jobs, canonical local results and durable cleanup records remain protected.

The historical pilot uses a different receipt/cleanup contract and is already retired. Do not feed an old pilot/cloud receipt to queue cleanup or delete an entire job tree manually. Original local training caches stay protected until both studies and reports finish; the existing proof archives canonical models before retiring registered derived training NPZs. Raw data, validation/calibration/evaluation inputs, manifests, reports and model archives remain.

For the historical **one-epoch** cloud format only:

```powershell
& $ResearchPython -m adp.cloud_results `
  --archive 'PATH\TO\PILOT.zip' --receipt 'PATH\TO\PILOT.receipt.json' `
  --output-root 'D:\InteractionLab\remote-compute\cloud-results\colab' `
  --payload-archive 'D:\InteractionLab\remote-compute\server-pilot-v1-inputs.zip' `
  --expected-payload-sha256 'f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa'
```

The new queue importer instead requires its frozen job specification and authenticated archive hash through the controller. New 20-epoch exports must not use the old one-epoch importer.

## Free cloud operation and desktop status

Cloud queues are finite workloads explicitly started inside normal notebooks. They can advance through the approved jobs while that interactive runtime remains available. Free GPU allocation, quota and runtime persistence vary. Reconnection, uploads, result downloads and final cloud deletion may still require interactive notebook controls; the local monitor cannot guarantee unattended cloud recovery or transfers. It does not allocate or remotely control Colab through API/SSH workers. Colab's free managed-runtime restrictions include remote SSH control and distributed computing workers; see its [official FAQ](https://research.google.com/colaboratory/faq.html).

Kaggle documents [normal notebooks](https://www.kaggle.com/docs/notebooks) and [official API execution without interactive sessions](https://www.kaggle.com/docs/efficient-gpu-usage). A credentialed API adapter could be added separately; none is configured here. Current cloud uploads, execution, downloads, cleanup and session shutdown use notebook/file interfaces. End GPU allocations after their finite workloads rather than leaving idle sessions running.

`C:/Users/Adi/Desktop/Interaction Lab.lnk` opens the workspace application in **Training** using CPU inference. This launcher requires the qualified runtime and built frontend; it is not the independently installed beta release. The page reads `/api/training` every five seconds. Local progress comes from saved checkpoints; remote rows come from `D:/InteractionLab/remote-compute/hosts.json`. The page starts no training and performs no SSH/browser actions. Stale and unknown observations stay explicit; a verified result receipt can remain valid after a host goes offline.
