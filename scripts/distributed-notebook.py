"""Foreground notebook launcher. Header supplies the pinned host/controller bytes."""
from pathlib import Path
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile

admission_started = time.monotonic()

PAYLOAD_SHA = 'f3ce396b8bbfa20b62edc6cbf7ccb42c22ae5ac191dd5e956ab46595a226c9fa'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

if HOST == 'colab':
    from google.colab import drive
    drive.mount('/content/drive')
    DESTINATION = Path('/content/drive/MyDrive/InteractionLab/compute-v1/colab/development-v2')
    PAYLOAD = DESTINATION / 'inputs.zip'
    ROOT = Path('/content/interaction-lab-distributed-v2')
elif HOST == 'kaggle':
    import kagglehub
    PAYLOAD = Path(kagglehub.dataset_download('adimaitre/interaction-lab-development-queue-20261020', path='server-pilot-v1-inputs.bin'))
    ROOT = Path('/kaggle/working/interaction-lab-distributed-v2')
    # A Save & Run All version preserves this normal notebook output after execution.
    DESTINATION = Path('/kaggle/working/interaction-lab-saved-v2')
else:
    raise RuntimeError('The private host uses its local dispatcher, not a notebook launcher')

DESTINATION.mkdir(parents=True, exist_ok=True)
raw = base64.b64decode(CODE_B64, validate=True)
assert len(raw) < 2*1024**2 and hashlib.sha256(raw).hexdigest() == CODE_SHA
assert PAYLOAD.is_file() and sha(PAYLOAD) == PAYLOAD_SHA, 'Approved input bytes are missing or changed'
assert not ROOT.exists(), 'Allocation already exists. Preserve its ledger and use read-only status; never reset it.'
assert not (DESTINATION/'allocation.json').exists(), 'Durable allocation exists: recovery must preserve the prior ledger'
assert shutil.disk_usage(ROOT.parent).free > 15*1024**3, 'Preserve the 10 GiB runtime reserve'
with tempfile.TemporaryFile() as stream:
    stream.write(raw); stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        assert set(archive.namelist()) == set(MEMBERS) and len(archive.namelist()) == len(MEMBERS)
        assert sum(entry.file_size for entry in archive.infolist()) < 2*1024**2
        contents = {name: archive.read(name) for name in MEMBERS}
        assert all(hashlib.sha256(data).hexdigest() == MEMBERS[name] for name, data in contents.items())
        for name, data in contents.items():
            assert '..' not in Path(name).parts and not Path(name).is_absolute()
            destination = ROOT/name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
shutil.copyfile(PAYLOAD, ROOT/'inputs.zip')
sys_path = str(ROOT/'control')
import sys
sys.path.insert(0, sys_path)
spec = importlib.util.spec_from_file_location('distributed_plan', ROOT/'distributed_plan.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
resources = {'protocol_sha256': module.protocol_hash(HOST),
    'files': MEMBERS}
atomic_json(ROOT/'resources.json', resources)
atomic_json(DESTINATION/'launch.json', {'host': HOST, 'protocol_sha256': module.protocol_hash(HOST),
    'controller_sha256': CODE_SHA, 'prior_charged_seconds': module.PRIOR_SECONDS[HOST],
    'started_at': datetime.now(timezone.utc).isoformat()})

# Official, pinned bootstrap. No system Python changes; setup precedes training.
bootstrap = ROOT/'bootstrap'; bootstrap.mkdir()
uv = bootstrap/'uv'
import io, tarfile, urllib.request
data = urllib.request.urlopen('https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz', timeout=120).read(64*1024**2)
assert hashlib.sha256(data).hexdigest() == 'eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220'
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    entries = [entry for entry in archive.getmembers() if entry.isfile() and Path(entry.name).name == 'uv']
    assert len(entries) == 1
    uv.write_bytes(archive.extractfile(entries[0]).read()); uv.chmod(0o700)
environment = {**os.environ, 'UV_PYTHON_INSTALL_DIR': str(bootstrap/'python'), 'UV_CACHE_DIR': str(bootstrap/'cache'),
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONPATH': sys_path}
python = ROOT/'runtime/bin/python'
subprocess.run([str(uv), 'venv', '--python', '3.12', str(ROOT/'runtime')], env=environment, check=True)
subprocess.run([str(uv), 'pip', 'install', '--python', str(python), 'torch==2.13.0', '--index-url', 'https://download.pytorch.org/whl/cu130'], env=environment, check=True)
subprocess.run([str(uv), 'pip', 'install', '--python', str(python), 'numpy', 'filelock', 'matplotlib'], env=environment, check=True)
state = module.initial_state(HOST)
# Conservatively reserve five minutes/device for allocation before cell execution.
state['setup_charged_seconds'] = (time.monotonic()-admission_started+300) * (2 if HOST == 'kaggle' else 1)
state['charged_seconds'] += state['setup_charged_seconds']
module.validate_state(state, HOST)
atomic_json(ROOT/'allocation.json', state)
atomic_json(DESTINATION/'allocation.json', state)

def immutable_copy(source, destination):
    digest = sha(source)
    if destination.exists():
        assert sha(destination) == digest, 'Existing immutable saved bytes conflict'
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    shutil.copyfile(source, temporary)
    assert sha(temporary) == digest
    os.replace(temporary, destination)

def save_evidence():
    path = ROOT/'allocation.json'
    if not path.is_file():
        return
    before = path.read_bytes()
    state = module.validate_state(json.loads(before), HOST)
    if state['active'] is not None:
        history = ROOT/'jobs'/state['active']/'run/history.json'
        if history.is_file() and history.stat().st_size < 128*1024:
            rows = json.loads(history.read_bytes())['epochs']
            assert len(rows) <= 10 and all(row['epoch']==i for i,row in enumerate(rows,1))
            state['progress'] = {'saved_epochs':len(rows)}
            state['observed_at'] = datetime.fromtimestamp(history.stat().st_mtime,timezone.utc).isoformat()
    for export in state['exports']:
        assert export['archive_name'] == export['snapshot_id']+'.zip'
        source = ROOT/'exports'/export['archive_name']
        assert sha(source) == export['sha256']
        immutable_copy(source, DESTINATION/'exports'/export['archive_name'])
    # Checkpoints are published by atomic replacement in training.py. Hash before
    # and after copying; retain the last sound version if a checkpoint changes.
    references = []
    for attempt in state['attempts']:
        identifier = attempt['job_id']
        for name in ('last.pt', 'best.pt'):
            source = ROOT/'jobs'/identifier/'run'/name
            if not source.is_file():
                continue
            digest = sha(source)
            target = DESTINATION/'checkpoints'/identifier/(digest+'.pt')
            try:
                immutable_copy(source, target)
                if sha(source) != digest:
                    continue
                references.append({'job_id': identifier, 'name': name, 'sha256': digest})
            except (OSError, AssertionError):
                continue
    atomic_json(DESTINATION/'allocation.json', state)
    atomic_json(DESTINATION/'evidence.json', {'version': 2, 'host': HOST, 'controller_sha256': CODE_SHA,
        'protocol_sha256': module.protocol_hash(HOST), 'published_at': datetime.now(timezone.utc).isoformat(),
        'allocation_sha256': hashlib.sha256((json.dumps(state, sort_keys=True, allow_nan=False)+'\n').encode()).hexdigest(),
        'checkpoint_references': references})

process = subprocess.Popen([str(python), '-u', str(ROOT/'distributed-worker.py'), '--root', str(ROOT), '--host', HOST],
    env=environment, start_new_session=True)
print('Bounded distributed worker started:', process.pid, HOST, flush=True)
try:
    while True:
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            save_evidence()
            state = json.loads((ROOT/'allocation.json').read_bytes())
            print(json.dumps({'host': HOST, 'state': state['state'], 'completed_jobs': len(state['completed']),
                'charged_seconds_including_reservation': state['charged_seconds'], 'active': state['active']}), flush=True)
        else:
            save_evidence()
            print('Foreground allocation finished, exit code', code, flush=True)
            if code:
                raise subprocess.CalledProcessError(code, process.args)
            break
except BaseException:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    save_evidence()
    raise
