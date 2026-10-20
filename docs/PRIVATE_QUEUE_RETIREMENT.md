# Completed private queue retirement

This is a separate, explicit final step for the fixed `development-v1` queue. It does not add jobs, reset its six GPU-hour budget, change its approved plan or alter either frozen study.

The controller first requires all twelve jobs to be complete with no worker owner. It checks every local export ZIP, every canonical result/model file, both strict CPU verification receipts and every completed per-job cleanup journal. Unchanged verified model bytes reuse their original receipts; they do not require repeated PyTorch loading or a new verification timestamp.

The separately reviewed `scripts/queue-retire-v1.py` is approved in:

```text
D:/InteractionLab/remote-compute/queues/development-v1/retirement-tool-v1.json
```

Its exact fields are `version: 1`, `retirement_sha256`, `adapter_sha256` and `base_sha256`. The latter two must match the existing separately approved cleanup adapter and original cleanup resource. The controller hashes the actual sources, uploads only an absent immutable filename containing the tool SHA-256, and verifies the remote dependencies and tool again. Existing mismatched files are preserved and block execution. This receipt is separate from `approved-plan.json`; never rewrite that plan to approve a new cleanup tool.

After review and approval, run from the managed application workspace:

```powershell
$env:PYTHONPATH = 'src'
& 'D:\InteractionLab\runtime-cu130\Scripts\python.exe' -m adp.private_queue --retire-completed
```

The exact 1,101 owned targets are the 1,064 shared pilot NPZ copies, `inputs.zip`, twelve ZIP exports and twenty-four snapshot checkpoint copies. Source code, cache manifests, protocols, snapshot metadata, logs and audit journals remain. Canonical local input data, exports, models and receipts remain.

Before deletion, the controller durably saves `retirement-envelope-v1.json` locally. The remote tool saves a self-hashed full per-file journal at `cleanup/terminal-v1.json`, flushes it, and verifies each file immediately before unlinking it. A completed target that reappears is preserved and reported. An interrupted operation resumes only the identical envelope and target bytes. There is no recursive directory deletion.

Normal monitor cycles recognize the saved retirement intent before calling the older snapshot-dependent cleanup. They recheck the canonical local hashes and the remote terminal journal, resume an interrupted authorized operation if necessary, and save the compact verified result as `retirement-receipt-v1.json`. Missing proofs, changed files, active owners or mismatched tool hashes stop retirement; they never authorize deleting replacement data.
