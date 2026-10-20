# Retiring processed training data after both studies

`adp.study_cleanup` can remove only `D:\InteractionLab\study-v1\cache\train\*.npz`.
It preserves raw scenarios/maps, selection, calibration and evaluation caches,
manifests, source code, checkpoints, model archives, results and failure reports.
It does not run against the currently active studies.

The default is a data-read-only plan:

```powershell
python -m adp.study_cleanup --plan
```

An active original study, pending direction experiment, missing CPU report,
held OS owner lock or missing proof returns `state: blocked` with a reason. Blocking
happens before scanning training NPZ files. Finished status means the real
`pipeline_status.json` and `followup_status.json` both say `completed` and
`phase: finished`; direction `experiment.json`, comparison and both recorded
failure-analysis folders must also be complete and intact.

Planning and execution acquire the real `pipeline.lock`, `runner.lock`,
`followup.lock` and `direction.lock` with a zero timeout. Released, zero-byte
lock paths are accepted; file existence does not mean a process owns the lock.
Execution holds all four locks throughout preflight and removal, so a study
cannot resume between verification and deletion. The helper never manually
unlinks these lock paths. Unknown dot-prefixed owner-lock files remain blocking.
Normal OS lock bookkeeping may create or retain zero-byte lock paths during a plan.

For unattended completion checks, use:

```powershell
python -m adp.study_cleanup --archive-completed --execute
```

While either study is active or an OS study lock is held, this returns blocked
before CPU qualification, model/dataset hashing or archival. After both studies
and their final reports finish, it automatically creates a byte-identical,
strictly CPU-verified selected model copy and durable proof under
`D:\InteractionLab\model-archive\study-v1\<checkpoint-SHA256>\`. It then runs the
same training-cache cleanup while holding all four study locks throughout.
The copy and proof publish atomically, preserve the 8 GiB reserve, and remain
stable for retries. A corrupt existing archive blocks cleanup and is never
overwritten. This does not promote weights or delete any trained models.

After completion, keep a durable, byte-identical copy of the selected
`frozen_selection.json` checkpoint outside both study roots. A version-1 proof
JSON contains:

```json
{
  "version": 1,
  "verified": true,
  "protocol_sha256": "SHA256 of study-v1/protocol.json",
  "dataset_manifest_sha256": "SHA256 of cache/manifest.json",
  "preprocessing_sha256": "SHA256 of cache/preprocessing.json",
  "frozen_selection_sha256": "SHA256 of frozen_selection.json",
  "checkpoint_sha256": "the frozen selected checkpoint SHA256",
  "config_sha256": "adp.compute_cleanup._hash(checkpoint config)",
  "checkpoint_copy": "D:/InteractionLab/model-archive/selected.pt"
}
```

The helper revalidates this proof, both checkpoint byte hashes, architecture,
strict CPU state loading and tiny observed-only inference. A failed promotion
gate does not prevent archival of the selected research checkpoint, and cleanup
does not promote any model.

```powershell
python -m adp.study_cleanup --proof-file D:\InteractionLab\model-archive\study-cleanup-proof.json --plan
python -m adp.study_cleanup --proof-file D:\InteractionLab\model-archive\study-cleanup-proof.json --execute
```

Before deleting, it verifies the exact official training membership, every NPZ
hash, exclusive ownership and all ancestor paths. It publishes one immutable
`processed-train-cleanup-plan.json`, preserving the 8 GiB reserve. A compact
`processed-train-cleanup-progress.jsonl` records durable intent before each
group of at most 64 removals. The complete 40,000-file plan is written once;
progress appends small records. Interrupted runs may repeat the same command;
missing files require an existing write-ahead intent. Inputs changed since
planning block further deletion.

The fixed roots and proof gates are deliberate. Private-server pilot cleanup
uses the separate `adp.compute_cleanup` workflow and its consumer lease ledger.
No source cache is removed on the three-hour checkpoint-transfer timer.
