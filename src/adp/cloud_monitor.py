"""Read-only cloud connection checks; saved Kaggle runs are not live editor sessions."""
from __future__ import annotations

import os
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

KAGGLE_KERNEL = "adimaitre/interaction-lab-gpu-qualification"
MAX_RESPONSE_BYTES = 64 * 1024
TOOLS_KAGGLE = Path("D:/InteractionLab/remote-compute/tools-v1/Scripts/kaggle.exe")


def _kaggle_credentials_present():
    # Inspect presence only. Authentication remains inside the official CLI.
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    directory = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return (any((directory / name).is_file() for name in ("kaggle.json", "access_token"))
            or (Path.home() / ".kaggle/credentials.json").is_file())


def _kaggle_cli():
    sibling = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    for candidate in (TOOLS_KAGGLE, sibling):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("kaggle")


def check_cloud_connections():
    """No uploads, allocations, notebook execution, downloads, deletion or browser access."""
    colab = {"status": "unconfigured", "phase": "connection_required", "scope": "connection_capability",
        "message": "Unattended live Colab status is unavailable through a supported free-runtime API. Open the existing private notebook to inspect it; prior training evidence and verified results are retained."}
    kaggle = {"status": "unconfigured", "phase": "connection_required", "scope": "latest_saved_run_only",
        "message": "Unattended Kaggle checks need a locally configured official Kaggle CLI and API credential. Browser sign-in does not authenticate the local program; prior notebook evidence and verified results are retained."}
    results = {"colab": colab, "kaggle": kaggle}
    try:
        from adp.drive_evidence import configured_evidence
        saved = configured_evidence()
        results.update(saved)
    except (OSError, ValueError, KeyError, TypeError):
        saved = {}
        results["colab"] = {"status":"unavailable", "phase":"saved_source_unavailable", "scope":"saved_notebook_heartbeat",
            "message":"The local saved-notebook configuration is invalid or unavailable. Check the configured Drive folder; prior training evidence and verified models are retained."}
    if "kaggle" in saved:
        return results
    if not _kaggle_credentials_present() or not (client := _kaggle_cli()):
        return results
    try:
        options = {"capture_output":True, "text":True, "encoding":"utf-8", "errors":"replace", "timeout":30,
            "creationflags":subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
        response = subprocess.run([client, "kernels", "status", KAGGLE_KERNEL], **options)
        detail = (response.stdout + response.stderr).casefold()
        denied = "403" in detail or "forbidden" in detail or ("permission" in detail and "denied" in detail)
        if response.returncode and denied and len(detail.encode("utf-8")) <= MAX_RESPONSE_BYTES:
            listed = subprocess.run([client, "kernels", "list", "--mine", "--search",
                "interaction-lab-gpu-qualification", "--page-size", "5", "--format", "json"], **options)
            if listed.returncode or len(listed.stdout.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise ValueError("Expected saved notebook metadata could not be verified")
            entries = json.loads(listed.stdout)
            if (not isinstance(entries, list) or len(entries) > 5
                    or sum(isinstance(entry, dict) and entry.get("ref") == KAGGLE_KERNEL for entry in entries) != 1):
                raise ValueError("Expected private saved notebook was not uniquely returned")
            kaggle.update(status="checked", phase="saved_notebook_visible", scope="saved_notebook_metadata",
                message="Official Kaggle API can read the expected saved notebook. Execution status is unavailable with this permission; notebook visibility does not establish a running editor session, GPU use or training progress.")
            return results
        if response.returncode or len(response.stdout.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Official Kaggle status could not be verified")
        values = set(re.findall(r"\b(queued|running|complete|completed|error|failed|cancelled|canceled)\b", response.stdout.casefold()))
        if len(values) != 1:
            raise ValueError("Official Kaggle status response is unsupported")
        saved = values.pop()
        kaggle.update(status="checked", phase="saved_run_status", saved_run_status=saved,
            message=f"Official Kaggle API: latest saved run is {saved}. This does not report the current interactive editor session, GPU utilization or saved epochs.")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        # Never put API stderr, credential paths or tokens into application records.
        kaggle.update(status="unavailable", phase="connection_unavailable", message="The official Kaggle read-only status check failed or timed out. Check local CLI authentication/network access; existing notebook progress and verified results are retained.")
    return results
