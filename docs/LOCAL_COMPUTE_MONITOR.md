# Independent compute monitor

The monitor runs as a separate Windows process. It synchronizes verified private-server results, updates the Training screen and checks guarded cleanup. An approved plan also enables the finite development queue. Codex, this chat and the application do not need to remain open. Windows must remain running, with network access to the private server for transfers.

The `interaction-lab-compute-transfers` chat heartbeat is **paused**. The independent process checks immediately after startup, then waits **five minutes after each completed cycle**. Sleep, network failures and lengthy verification can delay transfers; saved status records actual checks and the next check.

Observed **20 October 2026, 07:11 UTC**: actual monitor PID **33896** held the exclusive lock, its approved queue hook was active, and the next check was **10:02 UTC**. Desktop and user-sign-in Startup shortcuts were installed. The original study had **40/90 saved epochs** and **24.40/40 GPU-hours** charged, including its current reservation; the direction watcher and extra local queue correctly waited. The private worker PID **108576** was running its first new job at **10/20 epochs**. Cloud queue setup was still in progress. These observations expire; `--status`, saved study history and fresh host evidence supply current state.

## Start, inspect and stop

The Desktop and Windows Startup shortcuts use `D:\InteractionLab\runtime-cu130\Scripts\pythonw.exe` and `scripts/start-compute-monitor.py`. The launcher detaches the monitor from its caller and reuses an existing monitor owner. Startup runs at user sign-in; this is a per-user process, not a Windows service.

The Desktop shortcut **Interaction Lab Compute Monitor** adds `--show-status`: double-click it to start or reuse the background worker and open a native status dialog with its PID, latest check, next check, and status/log file locations. Closing that dialog leaves the monitor running. A fresh start may briefly report that its first status has not been published. The Windows Startup shortcut omits the flag and stays quiet at sign-in. These shortcut arguments are applied by `scripts/install-compute-monitor.ps1`; source changes alone do not update an already installed shortcut.

Run console commands from the managed workspace:

```powershell
$ProjectRoot = 'C:\Users\Adi\.codex\worktrees\failure-analysis-direction\Autonomous Driving Prediction'
$ResearchPython = 'D:\InteractionLab\runtime-cu130\Scripts\python.exe'
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'

& $ResearchPython scripts/start-compute-monitor.py
& $ResearchPython scripts/start-compute-monitor.py --show-status
& $ResearchPython scripts/compute-monitor.py --status
& $ResearchPython scripts/compute-monitor.py --stop
```

`--stop` finishes the current monitor cycle safely. It does not stop training workers or the application. The monitor stores process ownership, cycle outcomes and scheduling in `D:/InteractionLab/remote-compute/monitor/status.json`, with a rotating JSON-lines `monitor.log` beside it. An exclusive OS lock prevents duplicate owners.

For one foreground cycle, first stop the background monitor and confirm its status is stopped:

```powershell
& $ResearchPython scripts/compute-monitor.py --once
```

A cycle transfers the historical private pilot using `--no-resume`, refreshes evidence and checks whether the original studies' derived training caches may be retired. It does not restart that retired pilot or modify frozen study budgets or source.

## Approved queue dispatch

The startup launcher includes `D:/InteractionLab/remote-compute/queues/development-v1/approved-plan.json` automatically when it exists. A direct invocation enables dispatch explicitly:

```powershell
& $ResearchPython scripts/compute-monitor.py --once `
  --queue-plan 'D:\InteractionLab\remote-compute\queues\development-v1\approved-plan.json'
```

This performs actual dispatch under the approved immutable plan. Without `--queue-plan`, a direct monitor invocation runs transfers and guarded cleanup only. Resource changes block dispatch instead of replacing approved hashes.

Each host has 12 fresh jobs: `full`, `no_graph`, `no_map` and `world_ce`, each with seeds 42, 43 and 44. Jobs request 20 epochs on the same 1,000 training and 64 selection scenarios. The **21,600 allocated GPU-second** cap per host is separate from the original 40-hour study and 12-hour direction experiment. Each job has at most 1,800 GPU-seconds. Kaggle's two allocated T4s are charged twice wall time, giving at most 900 wall seconds per job before preparation/export overhead. These limits do not guarantee completion.

The local extra queue waits for both studies, their CPU reports and verified training-cache cleanup. The private queue has its own namespace and starts the next approved recipe after a completed job, independently of the five-minute transfer cadence. A failed or paused job stops its worker. An interrupted active lease requires positive process-death proof before recovery; no stale timestamp or monitor restart resets ownership or budget.

## Pause pending jobs

Stopping the monitor leaves a queue worker already running intact. To pause an already staged local worker after its current job:

```powershell
$QueueRoot = 'D:\InteractionLab\remote-compute\queues\development-v1\local'
& $ResearchPython "$QueueRoot\worker.py" `
  --root $QueueRoot --host local --payload "$QueueRoot\inputs.zip" `
  --queue-module "$QueueRoot\compute_queue.py" `
  --pilot-helper "$QueueRoot\pilot-helper.py" `
  --control-source "$ProjectRoot\src" --stop-after-current
```

Use the same arguments with `--resume-queue` to clear the validated pause flag and continue after the exclusive worker lock becomes free. `--status` reads the staged queue without starting training. The private worker supports the same options through the existing trusted SSH connection; its namespace is `/home/aadi/interaction-lab/queues/development-v1`. Do not remove leases or ledgers manually to force a restart.

## Results and free cloud sessions

Completed private exports pass archive/member hashes, approved source/payload/configuration/protocol checks, strict CPU checkpoint execution and a durable local receipt before remote cleanup. The local operational monitor/controller now recheck a **10 GiB reserve**, including projected transfer copies, against actual free space on each cycle. Insufficient projected capacity pauses the operation and preserves its source; recovered capacity is recognized by the next check. Frozen study and approved worker protocols retain their original 8 GiB settings and hashes.

Colab and Kaggle queues are explicitly started inside normal notebook runtimes and advance through their finite jobs while the runtime remains available. Allocation and session persistence remain variable. The independent Windows monitor handles local/private operations; it cannot reconnect a free cloud runtime, guarantee unattended cloud transfers, or bypass interactive deletion confirmation. The local monitor does not allocate a Colab GPU or turn its runtime into an SSH/distributed worker; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Each cycle also checks cloud connection capability. Drive sync and the official read-only Kaggle connection are now installed. The distributed-v2 controller reads Colab allocation evidence through Drive and attempts Kaggle saved-output downloads. Kaggle currently denies status/output access with HTTP 403 despite valid sign-in. `hosts.json` records connection results separately while retaining the last actual training observation, progress and verified receipts. A successful connection check never makes old training evidence fresh. See [the active distributed-v2 workflow](DISTRIBUTED_COMPUTE_V2.md) for current allocations; the v1 queue details above are historical.

The read-only Kaggle adapter invokes only the official [`kaggle kernels status`](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-status) command when a local official CLI and credential are available. This reports the latest saved run; it does not establish the state of an unsaved interactive editor session, GPU use or per-epoch progress. The adapter performs no upload, allocation, execution, output download or deletion. Credentials remain inside the CLI and are never written to application status/logs. Free Colab has no supported live-runtime adapter in this deployment; notebook inspection remains interactive. The optional [saved Drive evidence workflow](CLOUD_SAVED_EVIDENCE.md) supplies checksummed status and immutable completed exports through a locally synced folder. It remains explicitly saved and possibly stale notebook evidence, separate from live runtime status.

The Training screen reads saved evidence and its age; it makes no SSH or cloud API calls. A prepared plan, available SSH connection or passing test does not establish that a new job started. See [COMPUTE_OPERATIONS.md](COMPUTE_OPERATIONS.md) for historical measurements and dated deployment observations.
