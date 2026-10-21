"""Build pinned code for the specifically authorized distributed development batch."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from adp.distributed_plan import plan, protocol_hash, PRIOR_SECONDS
from adp.compute_queue import PAYLOAD_SHA256


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    members = {"distributed-worker.py": (PROJECT / "scripts/distributed-worker.py").read_bytes(),
        "worker-base.py": (PROJECT / "scripts/compute-queue-worker.py").read_bytes(),
        "distributed_plan.py": (PROJECT / "src/adp/distributed_plan.py").read_bytes(),
        "pilot-helper.py": (PROJECT / "artifacts/cloud-pilot-cell.py").read_bytes()}
    for name in ("__init__.py", "compute_queue.py", "remote_transfer.py", "model.py"):
        members["control/adp/" + name] = (PROJECT / "src/adp" / name).read_bytes()
    manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}
    path = output / "controller.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3; info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for host in ("private", "colab", "kaggle"):
        source = (PROJECT / "scripts/distributed-notebook.py").read_text(encoding="utf8")
        header = "HOST = " + repr(host) + "\nCODE_SHA = " + repr(digest) + "\nMEMBERS = " + repr(manifest) + "\nCODE_B64 = " + repr(base64.b64encode(path.read_bytes()).decode()) + "\n"
        cell = header + source
        (output / (host + "-batch.py")).write_text(cell, encoding="utf8")
        if host != "private":
            notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["# Interaction Lab distributed development v2\n", "Distinct seeds per host; 10 epochs/job; approved 1,000 training +64 selection only. Frozen studies unchanged.\n", "Previous charges are carried forward. No automatic reconnect or keepalive.\n"]},
                {"cell_type": "code", "metadata": {}, "source": cell.splitlines(keepends=True), "execution_count": None, "outputs": []}]}
            (output / (host + "-batch.ipynb")).write_text(json.dumps(notebook), encoding="utf8")
    (output / "plan.json").write_text(json.dumps({"controller_sha256": digest,
        "payload_sha256": PAYLOAD_SHA256, "hosts": {host: plan(host) for host in PRIOR_SECONDS},
        "protocol_hashes": {host: protocol_hash(host) for host in PRIOR_SECONDS}}, sort_keys=True), encoding="utf8")
    return {"sha256": digest, "bytes": path.stat().st_size}


if __name__ == "__main__":
    print(json.dumps(build(PROJECT / "artifacts/distributed-v2")))
