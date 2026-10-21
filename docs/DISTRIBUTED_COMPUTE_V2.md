# Distributed development allocation v2

This follow-up uses only the already approved 1,000 training and 64 selection caches. It does not migrate or modify the frozen 40-hour study, the 12-hour direction experiment, calibration, locked evaluation, or application deployment weights.

Each remote host receives four configurations (`full`, `no_graph`, `no_map`, `world_ce`), ten epochs per job, and distinct seeds:

| Host | Seeds | Previous GPU-seconds charged | Total cap for this ledger |
|---|---|---:|---:|
| Private server | 45–47 | 13,157.440873906 | 21,600 |
| Colab | 48–50 | 0 (new six-hour allowance explicitly approved) | 21,600 |
| Kaggle | 51–53 | 1,800 (lost first job's full allowance) | 21,600 |

The earlier Colab allowance is conservatively treated as consumed; its completed jobs are never repeated. These fresh, shorter development runs are not a replacement benchmark or evidence for model promotion. Differences in hardware are recorded in every result.

Before each job the worker durably reserves 1,800 allocated GPU-seconds. Kaggle's two allocated GPUs count twice wall time even though one computes. Preparation, GPU qualification, training, validation and export are measured inside that reservation. Cloud setup is charged separately with an additional conservative five-minute allowance per device for admission. An interrupted reservation stays fully charged and blocks automatic restart. A failed, paused or overrun job stops that host; it never silently starts from scratch. Hosts advance after completed jobs until their finite plan or remaining cap is exhausted.

The independent Windows monitor checks every five minutes after its prior cycle, while the PC is awake. Its new hook stages the private allocation when SSH is reachable and no other GPU owner is present, then imports completed archives through payload/protocol/hash checks and strict CPU loading of both checkpoints. Originals and already completed queues remain protected. Free cloud allocation and reconnect remain normal interactive notebook actions; there is no SSH worker, keepalive or automatic cloud reconnect.

Colab saves allocation evidence, immutable checkpoints and completed exports into `MyDrive/InteractionLab/compute-v1/colab/development-v2` every 60 seconds. The local Drive client can lag behind publication. The monitor reads saved observations with their original timestamps and verifies completed model archives independently. This does not prove the cloud GPU is live. A crash between checkpoint creation and the next successful Drive save can still lose that interval.

Kaggle uses a new private notebook, **Interaction Lab distributed development v2**, with a normal **Save & Run All** version. Its original qualification notebook and saved Version 1 remain intact. Output is saved through Kaggle's batch mechanism; a prematurely reclaimed run may still lose unfinished work. The five-minute monitor attempts the official read-only output API and will import matching saved exports if accessible. At initial qualification Kaggle returned `kernels.get` permission denied despite notebook metadata being visible. This is explicitly reported as unavailable, with no invented progress or credential expansion.

The new notebook's first batch (`360513724`) refused the conservative 5 GiB upfront storage projection before any training job. The separately pinned repair (`artifacts/distributed-v2-kaggle-recovery1`) checks a 512 MiB next-operation projection plus the same 10 GiB reserve, with subsequent staging/export checks retained. Version 2 (`360515620`) conservatively carries another 1,800 GPU-seconds for that failed admission in its setup charge. The original controller and already running private/Colab allocations are unchanged. Its completed first ten-epoch job and automatic handoff were observed in the saved-run logs.

All operational staging/transfers preserve 10 GiB free. The user-cleared extra D: capacity is recognized dynamically, without acquiring additional data. New allocation copies remain protected pending a completed and locally verified cleanup qualification; original canonical inputs/models/protocols are never deleted by this worker.

Distributed running observations expire after ten minutes without new source evidence. Private SSH synchronization failures update a separate connection result while preserving the last training observation and its original age. A retry occurs only for an explicit pre-connection timeout; failed remote actions are never replayed automatically. The Windows monitor retries checks, not completed jobs.

The monitor publishes a validated allocation observation before transferring models. Drive can synchronize `allocation.json` ahead of the referenced ZIPs: missing exports stay pending and available later exports are still verified. Transfer errors cannot suppress newer training progress or replace an earlier verified receipt. Partial transfers show the verified/expected archive counts separately from source freshness.

On 20 October 2026, the Kaggle connection was rechecked against the correct private notebook with active read-only credentials. Status, output listing and live logs returned HTTP 403 even though notebook metadata was visible. This is an API permission denial, not evidence that training stopped or credentials expired. A manually recorded notebook observation expires after ten minutes and does not repair unattended access.

Code is pinned in `artifacts/distributed-v2/controller.zip`; the immutable plans and its hash are in `D:/InteractionLab/remote-compute/distributed-v2/approved.json`. Resuming does not rewrite either. Original frozen source files are included unchanged from the approved payload. Do not replace the pinned worker to change a running job.
