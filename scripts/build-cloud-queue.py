"""Build the small hash-pinned interactive cloud controller; input data stays separate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from adp.cloud_results import _payload
from adp.compute_cleanup import _safe

PAYLOAD_SHA256 = "f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa"
PAYLOAD = Path("D:/InteractionLab/remote-compute/server-pilot-v1-inputs.zip")
OUTPUT = PROJECT / "artifacts/cloud-queue-v1"
FROZEN_SOURCE = {"__init__.py", "calibration.py", "data.py", "metrics.py", "model.py",
                 "remote_jobs.py", "research.py", "training.py"}
MEMBERS = {"worker.py", "compute_queue.py", "queue_cleanup.py", "pilot-helper.py",
           "control/adp/__init__.py", "control/adp/model.py", "control/adp/remote_transfer.py"}
MAX_CODE = 2 * 1024**2


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# Kept self-contained so a notebook can verify every byte before importing any uploaded code.
PREPARATION_SOURCE = '''from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import zipfile

def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def unlinked(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()) for p in (path, *path.parents)):
        raise ValueError('Cloud paths must not contain links or junctions.')
    return path

def prepare_cloud_queue(code_zip, payload, root, expected_zip_sha, expected_manifest, host):
    code_zip, payload, root = map(unlinked, (code_zip, payload, root))
    if host not in ('colab', 'kaggle'):
        raise ValueError('Use an explicitly selected interactive cloud host.')
    if root.exists():
        raise RuntimeError('Queue directory already exists; use the status cell. Never reset an interrupted lease.')
    if not code_zip.is_file() or not 0 < code_zip.stat().st_size <= 2*1024**2 or file_sha(code_zip) != expected_zip_sha:
        raise ValueError('Controller ZIP SHA-256 differs from the pinned reviewed bundle.')
    if not payload.is_file() or not 0 < payload.stat().st_size <= 300*1024**2 or file_sha(payload) != expected_manifest['payload_sha256']:
        raise ValueError('Input bytes differ from the approved train/selection payload. ZIP or .bin aliases are accepted.')
    contents = {}
    with zipfile.ZipFile(code_zip) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(name.casefold() for name in names)) or set(names) != set(expected_manifest['files']) | {'bundle-manifest.json'}:
            raise ValueError('Controller ZIP contains missing, duplicate or unexpected paths.')
        if sum(entry.file_size for entry in entries) > 2*1024**2:
            raise ValueError('Controller ZIP exceeds its expanded size bound.')
        for entry in entries:
            if entry.is_dir() or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG) or not 0 < entry.file_size <= 512*1024:
                raise ValueError('Controller member is linked, nonregular or excessive.')
            data = archive.read(entry)
            if entry.filename == 'bundle-manifest.json':
                if json.loads(data) != expected_manifest:
                    raise ValueError('Controller manifest differs from its independently pinned notebook copy.')
            else:
                indexed = expected_manifest['files'][entry.filename]
                if len(data) != indexed['bytes'] or hashlib.sha256(data).hexdigest() != indexed['sha256']:
                    raise ValueError('Controller member checksum differs: '+entry.filename)
            contents[entry.filename] = data
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free - payload.stat().st_size - 5*1024**3 - 2*1024**2 < 8*1024**3:
        raise RuntimeError('Runtime cannot fit the queue copies while preserving 8 GiB free.')
    stage = Path(tempfile.mkdtemp(prefix='.'+root.name+'-', dir=root.parent))
    try:
        for name, data in contents.items():
            destination = stage/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
        with payload.open('rb') as source, (stage/'inputs.zip').open('xb') as target:
            shutil.copyfileobj(source, target)
            target.flush(); os.fsync(target.fileno())
        if file_sha(stage/'inputs.zip') != expected_manifest['payload_sha256'] or file_sha(code_zip) != expected_zip_sha:
            raise ValueError('Input or controller changed during atomic preparation.')
        # Import only after the complete controller archive has passed both hash checks.
        import importlib.util
        spec = importlib.util.spec_from_file_location('approved_interaction_queue', stage/'compute_queue.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        plan = {'version':1,'host_seconds':21600,'jobs':[module.job_spec(host, c, s) for c in module.CONFIGURATIONS for s in module.SEEDS]}
        resources = {'payload_sha256':expected_manifest['payload_sha256'], 'plan_sha256':module.spec_hash(plan),
            'files':{name:file_sha(stage/name) for name in ('worker.py','compute_queue.py','queue_cleanup.py','pilot-helper.py','inputs.zip')}}
        with (stage/'resources.json').open('x', encoding='utf-8') as stream:
            json.dump(resources, stream, sort_keys=True, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
        stage.rename(root)
    finally:
        if stage.exists() and stage.parent == root.parent and stage.name.startswith('.'+root.name+'-'):
            shutil.rmtree(stage)
    return root

def worker_command(root, python, host):
    python = Path(python).absolute()
    unlinked(python.parent)
    if hasattr(python, 'is_junction') and python.is_junction():
        raise ValueError('Qualified Python must not be a directory junction.')
    # Venv bin/python is normally a symlink; resolving it would bypass the venv.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError('Qualified Python 3.12/PyTorch 2.13.0 environment is missing; restore the qualified environment first.')
    return [str(python), str(root/'worker.py'), '--host', host, '--root', str(root),
        '--payload', str(root/'inputs.zip'), '--queue-module', str(root/'compute_queue.py'),
        '--pilot-helper', str(root/'pilot-helper.py'), '--control-source', str(root/'control')]

def small_json(path):
    path = unlinked(path)
    with path.open('rb') as stream:
        data = stream.read(1024*1024+1)
    if len(data) > 1024*1024:
        raise ValueError('Progress/control JSON exceeds its 1 MiB bound.')
    return json.loads(data)

def verify_prepared_queue(root, expected_manifest, host):
    root = unlinked(root)
    if host not in ('colab', 'kaggle'):
        raise ValueError('Use an explicitly selected interactive cloud host.')
    for name, indexed in expected_manifest['files'].items():
        member = unlinked(root/name)
        if not member.is_file() or member.stat().st_size != indexed['bytes'] or file_sha(member) != indexed['sha256']:
            raise ValueError('Prepared controller checksum differs: '+name)
    payload = unlinked(root/'inputs.zip')
    if not payload.is_file() or not 0 < payload.stat().st_size <= 300*1024**2 or file_sha(payload) != expected_manifest['payload_sha256']:
        raise ValueError('Prepared payload differs from the approved bytes.')
    if any(unlinked(root/name).exists() for name in ('controller', '.worker-lease.json', 'worker-status.json')):
        raise RuntimeError('This queue was already initialized; use read-only status and preserve its lease.')
    # All imported bytes were checked against the independently pinned notebook manifest.
    import importlib.util
    spec = importlib.util.spec_from_file_location('approved_prepared_queue', root/'compute_queue.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    plan = {'version':1,'host_seconds':21600,'jobs':[module.job_spec(host, c, s) for c in module.CONFIGURATIONS for s in module.SEEDS]}
    expected = {'payload_sha256':expected_manifest['payload_sha256'], 'plan_sha256':module.spec_hash(plan),
        'files':{name:file_sha(root/name) for name in ('worker.py','compute_queue.py','queue_cleanup.py','pilot-helper.py','inputs.zip')}}
    if small_json(root/'resources.json') != expected:
        raise ValueError('Prepared resources differ from the approved finite host plan.')
    return root

def queue_progress(root):
    try:
        status = small_json(root/'worker-status.json')
        if not isinstance(status, dict):
            raise ValueError('Worker status must be a JSON object.')
        result = {name:status.get(name) for name in ('state','current_job_id','completed_jobs','planned_jobs')}
        identifier = result['current_job_id']
        if isinstance(identifier, str) and len(identifier) == 64 and all(c in '0123456789abcdef' for c in identifier):
            history = root/'jobs'/identifier/'run/history.json'
            if history.exists():
                result['saved_epochs'] = len(small_json(history)['epochs'])
                result['job_epochs'] = 20
        return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'state':'progress_unavailable', 'detail':str(error)[:200]}

def run_foreground(command, root, environment):
    process = subprocess.Popen(command, env=environment)
    print(json.dumps({'worker_pid':process.pid, **queue_progress(root)}, sort_keys=True), flush=True)
    try:
        while True:
            try:
                code = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                print(json.dumps(queue_progress(root), sort_keys=True), flush=True)
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('Worker has not exited. Preserve this runtime and lease; do not start another owner.', flush=True)
        raise
    print(json.dumps({'worker_exit_code':code, **queue_progress(root)}, sort_keys=True), flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)
'''


def _parameters(host):
    if host == "colab":
        return """HOST = 'colab'
CODE_ZIP = Path('/content/interaction-lab-cloud-code-v1.zip')
PAYLOAD_PATH = Path('/content/server-pilot-v1-inputs.zip')
ROOT = Path('/content/interaction-lab-colab-queue-v1')
QUALIFIED_PYTHON = Path('/content/interaction-lab-cloud-env/bin/python')
"""
    return """import kagglehub
HOST = 'kaggle'
DATASET = 'adimaitre/interaction-lab-development-queue-20261020'
# .bin aliases preserve the original ZIP bytes when Kaggle publishes this private dataset.
CODE_ZIP = Path(kagglehub.dataset_download(DATASET, path='interaction-lab-queue-code.bin'))
PAYLOAD_PATH = Path(kagglehub.dataset_download(DATASET, path='server-pilot-v1-inputs.bin'))
ROOT = Path('/kaggle/working/interaction-lab-kaggle-queue-v1')
QUALIFIED_PYTHON = Path('/kaggle/working/interaction-lab-cloud-env/bin/python')
"""


RUN_SOURCE = """if not QUALIFIED_PYTHON.is_file():
    raise RuntimeError('Restore the previously qualified Python environment before preparing the queue.')
ROOT = prepare_cloud_queue(CODE_ZIP, PAYLOAD_PATH, ROOT, EXPECTED_CODE_SHA256, EXPECTED_MANIFEST, HOST)
command = worker_command(ROOT, QUALIFIED_PYTHON, HOST)
# Interactive process only. No SSH, background keepalive, mounted Drive or automatic restart.
environment = {**os.environ, 'PYTHONPATH':str(ROOT/'control'), 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}
run_foreground(command, ROOT, environment)
"""
LAUNCH_SOURCE = """# Only for a previously prepared queue that has never initialized or claimed a job.
ROOT = verify_prepared_queue(ROOT, EXPECTED_MANIFEST, HOST)
command = worker_command(ROOT, QUALIFIED_PYTHON, HOST)
environment = {**os.environ, 'PYTHONPATH':str(ROOT/'control'), 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}
run_foreground(command, ROOT, environment)
"""
STATUS_SOURCE = """# Read-only one-shot status; an interrupted lease is deliberately not reset.
subprocess.run(worker_command(ROOT, QUALIFIED_PYTHON, HOST)+['--status'],
    env={**os.environ,'PYTHONPATH':str(ROOT/'control')}, check=True)
"""
EXPORT_SOURCE = """status = json.loads((ROOT/'worker-status.json').read_bytes())
for record in status.get('exports', []):
    archive = ROOT/'exports'/record['archive_name']
    if archive.name != record['snapshot_id']+'.zip' or file_sha(archive) != record['sha256']:
        raise ValueError('Export identity/checksum changed; preserve the runtime for diagnosis.')
    print(json.dumps({**record,'download_path':str(archive)}, sort_keys=True))
print('Download these ZIPs locally. Cleanup awaits durable strict CPU verification of both checkpoints; do not delete results here yet.')
"""


def _cell(kind, source):
    value = {"cell_type": kind, "id": _sha(source.encode())[:12], "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(payload=PAYLOAD, output=OUTPUT, *, project=PROJECT):
    payload, output, project = map(_safe, (payload, output, project))
    _, index = _payload(payload, PAYLOAD_SHA256)
    sources = {name.removeprefix("source/adp/") for name in index if name.startswith("source/adp/")}
    if sources != FROZEN_SOURCE:
        raise ValueError("Canonical payload must contain exactly the eight frozen job source files")
    members = {}
    with zipfile.ZipFile(payload) as archive:
        for name in FROZEN_SOURCE:
            relative = "source/adp/" + name
            data = archive.read(relative)
            if len(data) != index[relative]["size"] or _sha(data) != index[relative]["sha256"]:
                raise ValueError("Pinned frozen source checksum differs: " + name)
            if name in {"__init__.py", "model.py"}:
                members["control/adp/" + name] = data
    paths = {"worker.py": project / "scripts/compute-queue-worker.py",
        "compute_queue.py": project / "src/adp/compute_queue.py", "queue_cleanup.py": project / "src/adp/queue_cleanup.py",
        "pilot-helper.py": project / "artifacts/cloud-pilot-cell.py", "control/adp/remote_transfer.py": project / "src/adp/remote_transfer.py"}
    for name, path in paths.items():
        path = _safe(path)
        if not path.is_file() or path.stat().st_nlink != 1 or not 0 < path.stat().st_size <= 512 * 1024:
            raise ValueError("Controller source must be a bounded unshared file")
        members[name] = path.read_bytes()
        compile(members[name], name, "exec")
    if set(members) != MEMBERS:
        raise ValueError("Unexpected cloud controller membership")
    manifest = {"version": 1, "purpose": "development_only", "payload_sha256": PAYLOAD_SHA256,
        "host_gpu_seconds": 21600, "job_count": 12, "epochs_per_job": 20,
        "files": {name: {"sha256": _sha(data), "bytes": len(data)} for name, data in sorted(members.items())}}
    members["bundle-manifest.json"] = _bytes(manifest)
    if sum(map(len, members.values())) > MAX_CODE:
        raise ValueError("Cloud controller exceeds its 2 MiB expanded bound")
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "interaction-lab-cloud-code-v1.zip"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".bundle-", dir=output)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        if temporary.stat().st_size > MAX_CODE:
            raise ValueError("Cloud controller ZIP exceeds 2 MiB")
        data = temporary.read_bytes()
        if archive_path.exists() and archive_path.read_bytes() != data:
            raise ValueError("Existing pinned bundle differs; use a new versioned output folder")
        if not archive_path.exists():
            os.replace(temporary, archive_path)
        digest = _sha(data)
        (output / "interaction-lab-queue-code.bin").write_bytes(data)
        (output / "checksums.sha256").write_text(digest + "  " + archive_path.name + "\n" + digest + "  interaction-lab-queue-code.bin\n", encoding="utf-8")
        (output / "manifest.json").write_bytes(_bytes({**manifest, "archive_sha256": digest, "archive_bytes": len(data)}))
        for host in ("colab", "kaggle"):
            pin = "EXPECTED_CODE_SHA256 = " + repr(digest) + "\nEXPECTED_MANIFEST = " + repr(manifest) + "\n"
            parameters = _parameters(host)
            notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [_cell("markdown", f"# Interaction Lab: {host.title()} development queue\n\n12 fixed jobs, seeds 42/43/44 and four configurations; 20 epochs each over the approved 1,000 training/64 selection pilot. Maximum 6 allocated GPU-hours per host. Kaggle's two allocated T4 GPUs count twice wall time. Development evidence only; frozen studies and holdout labels remain separate.\n\nUse the previously qualified Python 3.12/PyTorch 2.13.0 environment. Upload controller and input bytes separately. Free GPU availability and session lifetime are variable. Start the next cells interactively; do not run a second queue owner.\n"),
                    _cell("code", PREPARATION_SOURCE + "\n" + pin), _cell("code", parameters),
                    _cell("code", RUN_SOURCE), _cell("code", STATUS_SOURCE),
                    _cell("markdown", "## Export and release\n\nDownload each completed ZIP listed below. Local `adp.queue_results.import_queue_result` independently checks the original payload, approved job protocol and both strict CPU-loadable checkpoints. Keep cloud result/cache copies until the local durable receipt is verified. Cleanup and GPU release are a separate action after verification; these cells never delete result folders. An interrupted/incomplete job remains incomplete.\n"),
                    _cell("code", EXPORT_SOURCE)]}
            (output / (host + "-queue.ipynb")).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
            (output / (host + "-queue-cell.py")).write_text(PREPARATION_SOURCE + "\n" + pin + parameters + "\n" + RUN_SOURCE, encoding="utf-8")
            (output / (host + "-queue-launch-prepared.py")).write_text(PREPARATION_SOURCE + "\n" + pin + parameters + "\n" + LAUNCH_SOURCE, encoding="utf-8")
        (output / "README.md").write_text(
            "# Interactive development queues\n\nThe small controller ZIP contains code only. The canonical 232 MB train/selection payload is a separate upload. "
            "The notebooks pin both byte hashes and all controller members before importing code.\n\n"
            "Colab: upload `interaction-lab-cloud-code-v1.zip` and `server-pilot-v1-inputs.zip` to `/content`, then open `colab-queue.ipynb`. "
            "Kaggle: the approved private dataset `adimaitre/interaction-lab-development-queue-20261020` contains the same bytes named "
            "`interaction-lab-queue-code.bin` and `server-pilot-v1-inputs.bin`; open `kaggle-queue.ipynb`. Aliases prevent automatic ZIP extraction.\n\n"
            "Both use the existing isolated `interaction-lab-cloud-env/bin/python`. Each host has one finite 12-job/6 allocated GPU-hour queue, "
            "with no automatic notebook restart, SSH, Drive mounting, or keepalive. Download exports, verify locally through `adp.queue_results`, "
            "and only then perform explicit cloud cleanup/release. `queue_cleanup.py` is carried for reproducibility but its destructive CLI is private-server-only.\n",
            encoding="utf-8")
        return {"path": str(archive_path), "sha256": digest, "bytes": len(data), "manifest": manifest}
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    result = build(args.payload, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, sort_keys=True))


if __name__ == "__main__":
    main()
