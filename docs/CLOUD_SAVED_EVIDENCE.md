# Saved cloud results and status

This optional path saves the already approved finite notebook queue into the user's own Drive. It starts no training, changes no job/budget and performs no cloud cleanup. The status envelope has SHA-256 and fixed payload/protocol provenance checks; it is **not digitally signed**. Access to the user's authenticated Drive folder is the source of authenticity. Local strict CPU result verification remains required before model use or remote deletion.

## Save an existing Colab queue

Paste `artifacts/cloud-queue-v1/colab-queue-save-to-drive.py` into a normal foreground notebook cell. Its `drive.mount('/content/drive')` uses Google's ordinary account permission flow. The user completes that permission themselves. The cell requires a mounted Drive and the existing `/content/interaction-lab-colab-queue-v1/worker-status.json`; if the old runtime disappeared, it refuses to recreate the completed queue.

The cell copies only recorded approved-job archives into:

```text
MyDrive/InteractionLab/compute-v1/colab/development-v1/
  status.json
  exports/<snapshot SHA>/<snapshot SHA>.zip
```

Each ZIP is checked against the source record, fixed training/selection payload, immutable host/job protocol and checkpoint inventory. The versioned directory is published atomically. Existing conflicting bytes cause an error and remain untouched. The mutable bounded `status.json` is replaced atomically only after its referenced archives are published. Source inputs, checkpoints, leases and budget ledgers remain untouched.

## Save during an existing foreground run

The same cell defines `watch_foreground(process, root, destination, host)`. A future **separately approved** foreground training cell can route the process it already owns through that helper:

```python
# `process` is the foreground child already started by the reviewed training cell.
saved = watch_foreground(process, ROOT, DESTINATION, 'colab')
```

The helper saves completed recorded exports and status every 60 seconds while that child is active, then performs one final save and returns. It starts no worker, does not extend runtime life, and does not reconnect a session. Interrupting the cell signals only its owned child, preserves the lease and warns if it has not exited. Unfinished checkpoints are not periodically copied by this version; those jobs remain unfinished and require a later recovery design. Never rerun a completed queue or reset its budget to make this wrapper run.

## Read from the local application

Install/sign in to Google Drive for desktop normally, make the saved queue folder available locally, and configure its existing absolute path:

```powershell
& $ResearchPython scripts/configure-cloud-folder.py --host colab `
  --directory 'G:\My Drive\InteractionLab\compute-v1\colab\development-v1'
```

Use the actual chosen Drive path. The settings command refuses missing folders, relative paths, links and junctions. It writes only `D:/InteractionLab/remote-compute/cloud-folders.json`. It does not install Drive, grant account permissions or copy model weights.

The five-minute independent monitor reads that local folder only. It validates the bounded checksum envelope, opaque IDs, fixed twelve-job protocol, progress totals, 6 GPU-hour ledger, timestamps and archive entries. It reports **saved notebook evidence**, with publication/source observation times separate from the latest local check. It does not establish that a cloud GPU or notebook kernel is currently running. A missing or corrupt/offline source produces `saved_source_unavailable`, retaining earlier training observations, verified models and ledgers.

Archive presence/size are availability evidence only. They do not establish that streamed Drive bytes are pinned for offline use or that checkpoints pass strict CPU loading. The existing `adp.queue_results.import_queue_result` path must independently hash and verify each result against the canonical approved payload before recording a completed local transfer. This adapter never executes cloud code or loads checkpoints.
