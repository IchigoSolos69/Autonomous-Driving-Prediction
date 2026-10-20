"""Transport the explicitly approved finite queue to the one private SSH host."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

from filelock import FileLock

from adp.compute_cleanup import _safe, _write
from adp.compute_monitor import _publish_host
from adp.compute_queue import (CONFIGURATIONS, HOSTS, HOST_SECONDS, PAYLOAD_SHA256,
                               SEEDS, job_spec, spec_hash)
from adp.remote_control import _config
from adp.remote_transfer import MAX_TOTAL_BYTES, _identifier

PROJECT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = Path("D:/InteractionLab/remote-compute/queues/development-v1")
REMOTE_ROOT = "/home/aadi/interaction-lab/queues/development-v1"
PAYLOAD = Path("D:/InteractionLab/remote-compute/server-pilot-v1-inputs.zip")
DEFAULT_PLAN = LOCAL_ROOT / "approved-plan.json"
DEFAULT_CONFIG = Path("D:/InteractionLab/remote-compute/private-server.json")
HOSTS_PATH = Path("D:/InteractionLab/remote-compute/hosts.json")
MAX_STATUS = 2 * 1024**2
MINIMUM_FREE_BYTES = 10 * 1024**3


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resource_paths():
    return {"worker.py": PROJECT / "scripts/compute-queue-worker.py",
            "compute_queue.py": PROJECT / "src/adp/compute_queue.py",
            "queue_cleanup.py": PROJECT / "src/adp/queue_cleanup.py",
            "pilot-helper.py": PROJECT / "artifacts/cloud-pilot-cell.py",
            "inputs.zip": PAYLOAD}


def make_plan():
    files = resource_paths()
    for path in files.values():
        _safe(path)
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Queue resources must be existing regular unshared files")
    if _sha(PAYLOAD) != PAYLOAD_SHA256:
        raise ValueError("Queue data differs from the approved original pilot package")
    return {"version": 1, "host_seconds": HOST_SECONDS, "payload_sha256": PAYLOAD_SHA256,
            "files": {name: _sha(path) for name, path in files.items()},
            "host_plans": {host: [job_spec(host, cfg, seed) for cfg in CONFIGURATIONS for seed in SEEDS]
                           for host in HOSTS}}


def approve_plan(path=DEFAULT_PLAN):
    """Persist the user's fixed six-hour authorization; never rewrite an existing plan."""
    path = _safe(path)
    value = make_plan()
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError("Approved queue resources changed; preserve the existing plan and jobs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, value)
    return {"state": "approved", "plan_sha256": spec_hash(value), "path": str(path)}


def _plan(path):
    path = _safe(path)
    if not path.is_file() or path.stat().st_size > MAX_STATUS:
        raise ValueError("An explicit bounded approved-plan.json is required before dispatch")
    value = json.loads(path.read_bytes())
    if value != make_plan():
        raise ValueError("Queue plan or source no longer matches its approved hashes")
    return value


def _ssh_base(config):
    return ["ssh", "-i", config["identityfile"], "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", config["host"]]


def _remote(config, code, *args, stdin=None, timeout=120):
    command = shlex.join([config["remote_python"], "-c", code, *map(str, args)])
    result = subprocess.run([*_ssh_base(config), command], input=stdin, capture_output=True,
                            timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        message = result.stderr.decode("utf8", errors="replace").strip()[-600:]
        raise RuntimeError(f"Private queue SSH operation failed ({result.returncode}): {message}")
    if len(result.stdout) > MAX_STATUS:
        raise ValueError("Private queue response exceeds its metadata bound")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("Private queue returned invalid metadata")
    return value


INSPECT = r'''
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); names=json.loads(sys.argv[2])
assert root==Path('/home/aadi/interaction-lab/queues/development-v1')
assert not any(p.is_symlink() for p in (root,*root.parents))
out={'exists':root.exists(),'files':{}}
for name in names:
 p=root/name
 if p.exists():
  assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1
  with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
  out['files'][name]=digest
print(json.dumps(out))
'''


def stage(config, plan):
    files = resource_paths()
    inspected = _remote(config, INSPECT, REMOTE_ROOT, json.dumps(list(files) + ["resources.json"]))
    if inspected["exists"] and any(name in inspected["files"] and inspected["files"][name] != plan["files"].get(name)
                                  for name in files):
        raise ValueError("Existing private queue code or data differs; no running resources were replaced")
    setup = "from pathlib import Path; import shutil,json,sys; p=Path(sys.argv[1]); assert not any(x.is_symlink() for x in (p,*p.parents)); p.mkdir(parents=True,exist_ok=True); assert shutil.disk_usage(p).free>12*1024**3; print(json.dumps({'ready':True}))"
    _remote(config, setup, REMOTE_ROOT)
    for name, path in files.items():
        if inspected["files"].get(name) == plan["files"][name]:
            continue
        target = REMOTE_ROOT + "/.incoming-" + uuid.uuid4().hex
        options = _ssh_base(config)[1:-1]
        response = subprocess.run(["scp", *options, str(path), config["host"] + ":" + target],
                                  capture_output=True, timeout=900,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if response.returncode:
            raise RuntimeError("Private queue resource upload failed; the input or staged partial was preserved")
        publish = "from pathlib import Path; import hashlib,json,os,sys; p=Path(sys.argv[1]); t=Path(sys.argv[2]); assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1; assert not t.exists(); f=p.open('rb'); assert hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3]; f.close(); os.replace(p,t); print(json.dumps({'verified':True}))"
        _remote(config, publish, target, REMOTE_ROOT + "/" + name, plan["files"][name])
    from adp.queue_results import queue_protocol_hash
    resources = {"files": plan["files"], "payload_sha256": PAYLOAD_SHA256,
                 "plan_sha256": queue_protocol_hash("private"), "approved_plan_sha256": spec_hash(plan)}
    publish_manifest = "from pathlib import Path; import json,os,sys; p=Path(sys.argv[1]); v=json.loads(sys.stdin.buffer.read()); assert not p.is_symlink(); assert not p.exists() or json.loads(p.read_bytes())==v; t=p.with_suffix('.tmp'); f=t.open('w'); json.dump(v,f,sort_keys=True); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(t,p); print(json.dumps({'verified':True}))"
    _remote(config, publish_manifest, REMOTE_ROOT + "/resources.json", stdin=json.dumps(resources).encode())
    return {"state": "staged", "plan_sha256": resources["plan_sha256"]}


def status(config):
    code = r'''
from pathlib import Path
from datetime import datetime,timezone
import json,os,re,sys
from filelock import FileLock,Timeout
def read(path):
 assert not any(x.is_symlink() for x in (path,*path.parents)), 'Linked queue metadata'
 if not path.exists(): return None
 info=path.stat()
 assert path.is_file() and info.st_nlink==1 and info.st_size<2*1024**2, 'Invalid queue metadata file'
 with path.open('rb') as stream: raw=stream.read(2*1024**2+1)
 assert len(raw)<2*1024**2, 'Queue metadata exceeds its bound'
 value=json.loads(raw)
 assert isinstance(value,dict), 'Queue metadata must be an object'
 return value
root=Path(sys.argv[1]); p=root/'worker-status.json'
assert not any(x.is_symlink() for x in (root,*root.parents))
out=read(p) or {'state':'not_started','exports':[]}
data=read(root/'controller/queue.json')
if data is not None:
 expected=json.loads(sys.argv[2])
 assert isinstance(data.get('jobs'),list) and len(data['jobs'])<=48, 'Invalid queue jobs'
 jobs=[j for j in data['jobs'] if j['spec']['host']=='private']
 assert len(jobs)<=12 and len({j['id'] for j in jobs})==len(jobs), 'Invalid private queue jobs'
 active=[]; committed=0
 for job in jobs:
  identifier=job['id']
  assert isinstance(identifier,str) and re.fullmatch('[0-9a-f]{64}',identifier), 'Invalid queue job ID'
  assert identifier in expected and job['spec']==expected[identifier], 'Unapproved private queue job'
  assert job['state'] in {'queued','running','manual_start_required','completed','failed','blocked'}, 'Invalid queue job state'
  epochs=20 if job['state']=='completed' else 0
  history_path=root/'jobs'/identifier/'run/history.json'
  history=read(history_path) if job['state']!='completed' else None
  if history is not None:
   rows=history.get('epochs')
   assert isinstance(rows,list) and len(rows)<=20, 'Invalid committed epoch history'
   assert all(isinstance(row,dict) and type(row.get('epoch')) is int and row['epoch']==i for i,row in enumerate(rows,1)), 'Invalid committed epoch sequence'
   epochs=len(rows)
  committed+=epochs
  if job['state'] in {'running','manual_start_required'}:
   current={'job_id':identifier,'configuration':job['spec']['configuration'],'seed':job['spec']['seed'],
            'completed_epochs':epochs,'planned_epochs':20,
            'progress_updated_at':datetime.fromtimestamp(history_path.stat().st_mtime,timezone.utc).isoformat() if history is not None else None}
   active.append(current)
 assert len(active)<=1, 'Multiple active private jobs'
 out['completed_jobs']=sum(j['state']=='completed' for j in jobs); out['total_jobs']=len(jobs)
 out['charged_seconds']=data['hosts']['private']['charged_seconds']
 out['jobs']=[{'id':j['id'],'state':j['state'],'spec':j['spec']} for j in jobs]
 out['completed_epochs']=committed; out['planned_epochs']=len(jobs)*20
 out['current_run']=active[0] if active else None
out['observed_at']=datetime.now(timezone.utc).isoformat()
out['worker_lock_held']=False
if root.exists():
 lockpath=root/'.worker.lock'; assert not lockpath.is_symlink()
 assert not lockpath.exists() or lockpath.stat().st_nlink==1
 lock=FileLock(str(lockpath),timeout=0)
 try: lock.acquire()
 except Timeout: out['worker_lock_held']=True
 else: lock.release()
pid=out.get('pid',out.get('worker_pid'))
try: os.kill(pid,0); out['process_alive']=True
except (TypeError,ValueError,ProcessLookupError): out['process_alive']=False
except PermissionError: out['process_alive']=True
print(json.dumps(out))
'''
    expected = {spec_hash(job_spec("private", configuration, seed)): job_spec("private", configuration, seed)
                for configuration in CONFIGURATIONS for seed in SEEDS}
    return _remote(config, code, REMOTE_ROOT, json.dumps(expected))


def start(config, plan):
    current = status(config)
    if current.get("worker_lock_held") or current.get("process_alive") or current.get("state") in {"completed", "budget_exhausted", "blocked", "failed"}:
        return current
    if current.get("state") == "running":
        raise RuntimeError("A previous queue worker stopped; retain its lease and resumable checkpoint for verified recovery")
    stage(config, plan)
    code = r'''
from pathlib import Path
import json,os,subprocess,sys
root=Path(sys.argv[1]); control=sys.argv[2]
command=[sys.executable,str(root/'worker.py'),'--root',str(root),'--host','private',
 '--payload',str(root/'inputs.zip'),'--queue-module',str(root/'compute_queue.py'),
 '--pilot-helper',str(root/'pilot-helper.py'),'--control-source',control]
env={**os.environ,'PYTHONPATH':control,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
with (root/'worker.stdout.log').open('ab') as out,(root/'worker.stderr.log').open('ab') as err:
 p=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
value={'state':'starting','pid':p.pid,'exports':[]}
path=root/'worker-status.json'; temp=path.with_suffix('.tmp'); f=temp.open('w'); json.dump(value,f); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(temp,path)
print(json.dumps(value))
'''
    return _remote(config, code, REMOTE_ROOT, config["remote_source"])


def _export(value):
    if not isinstance(value, dict):
        raise ValueError("Invalid queue export metadata")
    identifier = _identifier(value.get("snapshot_id"))
    _identifier(value.get("job_id")); _identifier(value.get("sha256"))
    if value.get("archive_name") != identifier + ".zip" or type(value.get("bytes")) is not int or not 0 < value["bytes"] <= MAX_TOTAL_BYTES:
        raise ValueError("Invalid queue export path or size")
    return value


def _upload_tool(config, source, name, digest, base_hash):
    incoming = REMOTE_ROOT + "/.incoming-cleanup-" + uuid.uuid4().hex
    options = _ssh_base(config)[1:-1]
    response = subprocess.run(["scp", *options, str(source), config["host"] + ":" + incoming],
        capture_output=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if response.returncode:
        raise RuntimeError("Separately approved cleanup tool upload failed; remote job files were preserved")
    publish = r'''
from pathlib import Path
import hashlib,json,os,sys
incoming,target,base=map(Path,sys.argv[1:4]); digest,base_digest=sys.argv[4:6]
assert incoming.parent==target.parent==base.parent
for path in (incoming,target,base):
 assert not any(x.is_symlink() for x in (path,*path.parents))
for path,expected in ((incoming,digest),(base,base_digest)):
 assert path.is_file() and path.stat().st_nlink==1
 with path.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
with incoming.open('rb') as stream: os.fsync(stream.fileno())
os.link(incoming,target)
incoming.unlink()
descriptor=os.open(target.parent,os.O_RDONLY)
try: os.fsync(descriptor)
finally: os.close(descriptor)
print(json.dumps({'verified':True}))
'''
    _remote(config, publish, incoming, REMOTE_ROOT + "/" + name,
            REMOTE_ROOT + "/queue_cleanup.py", digest, base_hash)


def _cleanup_tool(config, plan):
    approval = _safe(LOCAL_ROOT / "cleanup-tool-v2.json")
    if not approval.exists():
        return REMOTE_ROOT + "/queue_cleanup.py"
    if not approval.is_file() or approval.stat().st_nlink != 1 or approval.stat().st_size > 4096:
        raise ValueError("Cleanup tool approval must be bounded unshared metadata")
    with approval.open("rb") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("Cleanup tool approval exceeds its metadata bound")
    receipt = json.loads(raw)
    if (not isinstance(receipt, dict) or set(receipt) != {"version", "adapter_sha256", "base_sha256"}
            or type(receipt["version"]) is not int or receipt["version"] != 1):
        raise ValueError("Invalid separate cleanup tool approval")
    adapter_hash, base_hash = _identifier(receipt["adapter_sha256"]), _identifier(receipt["base_sha256"])
    source = _safe(PROJECT / "scripts/queue-cleanup-v2.py")
    if (base_hash != plan["files"]["queue_cleanup.py"] or not source.is_file()
            or source.stat().st_nlink != 1 or source.stat().st_size > MAX_STATUS or _sha(source) != adapter_hash):
        raise ValueError("Cleanup adapter or pinned base differs from its separate approval")
    name = "queue-cleanup-v2-" + adapter_hash + ".py"
    names = json.dumps(["queue_cleanup.py", name])
    inspected = _remote(config, INSPECT, REMOTE_ROOT, names)
    if not inspected.get("exists") or inspected["files"].get("queue_cleanup.py") != base_hash:
        raise ValueError("Remote cleanup base differs from the running queue's approved resource")
    existing = inspected["files"].get(name)
    if existing is not None and existing != adapter_hash:
        raise ValueError("Existing immutable cleanup adapter differs; it was not overwritten")
    if existing is None:
        _upload_tool(config, source, name, adapter_hash, base_hash)
    verified = _remote(config, INSPECT, REMOTE_ROOT, names)
    if (not verified.get("exists") or verified["files"].get("queue_cleanup.py") != base_hash
            or verified["files"].get(name) != adapter_hash):
        raise ValueError("Remote cleanup tool readback did not match its separate approval")
    return REMOTE_ROOT + "/" + name


def _previous_delivery(item, specification):
    """Reuse strict CPU evidence only while every archived result byte still matches."""
    from adp.queue_cleanup import _read as read, _sha as file_hash
    from adp.queue_results import queue_protocol_hash
    from adp.remote_transfer import _metadata
    root = _safe(LOCAL_ROOT / "results")
    queue_path = _safe(root / "queue-receipts" / (item["job_id"] + "-" + item["snapshot_id"] + ".json"))
    if not queue_path.exists():
        return None
    directory = _safe(root / item["snapshot_id"])
    receipt_path = _safe(root / "receipts" / (item["snapshot_id"] + ".json"))
    queue_receipt, receipt = read(queue_path), read(receipt_path)
    metadata, manifest_hash, _ = _metadata(directory)
    if (type(queue_receipt.get("version")) is not int or queue_receipt["version"] != 1
            or queue_receipt.get("verified") is not True or queue_receipt.get("job_id") != item["job_id"]
            or queue_receipt.get("job_spec") != specification or queue_receipt.get("snapshot_id") != item["snapshot_id"]
            or queue_receipt.get("archive_sha256") != item["sha256"] or queue_receipt.get("payload_sha256") != PAYLOAD_SHA256
            or queue_receipt.get("queue_protocol_sha256") != queue_protocol_hash("private")
            or type(queue_receipt.get("allocated_device_count")) is not int or queue_receipt["allocated_device_count"] != 1
            or queue_receipt.get("checkpoint_validation") != "strict_cpu_observed_only_completed"
            or queue_receipt.get("manifest_sha256") != manifest_hash or queue_receipt.get("files") != metadata["files"]
            or not isinstance(queue_receipt.get("local_directory"), str) or Path(queue_receipt["local_directory"]).resolve() != directory
            or type(receipt.get("format_version")) is not int or receipt["format_version"] != 1
            or receipt.get("verified") is not True or receipt.get("checkpoint_validation") != "cpu_weights_only"
            or receipt.get("snapshot_id") != item["snapshot_id"] or receipt.get("manifest_sha256") != manifest_hash
            or receipt.get("files") != metadata["files"] or not isinstance(receipt.get("local_directory"), str)
            or Path(receipt["local_directory"]).resolve() != directory
            or metadata.get("run_id") != item["job_id"] or metadata["provenance"].get("queue_job_spec") != specification
            or metadata["provenance"].get("payload_sha256") != PAYLOAD_SHA256
            or not {"last.pt", "best.pt", "job.json", "config.json", "result.json"} <= set(metadata["files"])):
        raise ValueError("Existing strict queue receipts differ from the approved archived result")
    for name, entry in metadata["files"].items():
        path = _safe(directory / name)
        if path.stat().st_size != entry["size_bytes"] or file_hash(path, entry["size_bytes"]) != entry["sha256"]:
            raise ValueError("Previously CPU-verified result bytes changed; remote copies were preserved")
    if file_hash(directory / "snapshot.json") != manifest_hash:
        raise ValueError("Previously verified snapshot metadata changed")
    cleanup_path = _safe(LOCAL_ROOT / ("cleanup-" + item["job_id"] + ".json"))
    if cleanup_path.exists():
        journal = read(cleanup_path)
        if journal.get("state") == "completed":
            context = journal.get("context", {})
            envelope = {"queue_receipt": queue_receipt, "import_receipt": receipt, "export": item}
            if (not isinstance(context, dict)
                    or journal.get("receipt_sha256") != spec_hash({key: value for key, value in journal.items() if key != "receipt_sha256"})
                    or context.get("job_id") != item["job_id"] or context.get("snapshot_id") != item["snapshot_id"]
                    or context.get("envelope_sha256") != spec_hash(envelope)
                    or context.get("job_protocol_sha256") != metadata["provenance"].get("job_protocol_sha256")
                    or context.get("status_sha256") != spec_hash(read(directory / "result.json"))):
                raise ValueError("Completed local cleanup journal differs from its durable result envelope")
    return {"snapshot_id": item["snapshot_id"], "path": str(directory), "receipt": receipt,
            "receipt_path": str(receipt_path), "queue_receipt": queue_receipt, "queue_receipt_path": str(queue_path)}


def _retirement_tool(config, plan):
    from adp.queue_cleanup import _read as read
    approval = _safe(LOCAL_ROOT / "retirement-tool-v1.json")
    if not approval.is_file() or approval.stat().st_size > 4096:
        raise ValueError("A separate bounded retirement-tool-v1.json approval is required")
    receipt = read(approval)
    if (set(receipt) != {"version", "retirement_sha256", "adapter_sha256", "base_sha256"}
            or type(receipt.get("version")) is not int or receipt["version"] != 1):
        raise ValueError("Invalid separate terminal retirement tool approval")
    digest = _identifier(receipt["retirement_sha256"])
    adapter_hash, base_hash = _identifier(receipt["adapter_sha256"]), _identifier(receipt["base_sha256"])
    adapter_name = "queue-cleanup-v2-" + adapter_hash + ".py"
    source = _safe(PROJECT / "scripts/queue-retire-v1.py")
    if not source.is_file() or source.stat().st_nlink != 1 or source.stat().st_size > MAX_STATUS or _sha(source) != digest:
        raise ValueError("Terminal retirement source differs from its separate approval")
    if base_hash != plan["files"]["queue_cleanup.py"] or _cleanup_tool(config, plan) != REMOTE_ROOT + "/" + adapter_name:
        raise ValueError("Retirement approval differs from the separately qualified cleanup dependencies")
    name = "queue-retire-v1-" + digest + ".py"
    hashes = {"queue_cleanup.py": base_hash, adapter_name: adapter_hash, name: digest}
    names = json.dumps(list(hashes))
    inspected = _remote(config, INSPECT, REMOTE_ROOT, names)
    if any(inspected.get("files", {}).get(key) != value for key, value in hashes.items() if key != name):
        raise ValueError("Remote retirement dependencies differ from their approved hashes")
    existing = inspected["files"].get(name)
    if existing is not None and existing != digest:
        raise ValueError("Existing immutable retirement tool differs; it was not overwritten")
    if existing is None:
        _upload_tool(config, source, name, digest, base_hash)
    verified = _remote(config, INSPECT, REMOTE_ROOT, names)
    if any(verified.get("files", {}).get(key) != value for key, value in hashes.items()):
        raise ValueError("Remote retirement tool readback differs from its separate approval")
    return REMOTE_ROOT + "/" + name


def _retirement_envelope(plan, observed):
    from adp.queue_cleanup import _read as read, _sha as file_hash
    specifications = plan["host_plans"]["private"]
    exports = observed.get("exports", [])
    if (observed.get("state") != "completed" or observed.get("completed_jobs") != 12
            or observed.get("process_alive") or observed.get("worker_lock_held") or len(exports) != 12):
        raise ValueError("Terminal retirement requires all twelve completed jobs and no live owner")
    exports = {_export(item)["job_id"]: item for item in exports}
    if set(exports) != {spec_hash(spec) for spec in specifications}:
        raise ValueError("Terminal retirement exports differ from the complete approved queue")
    deliveries = []
    for spec in specifications:
        item = exports[spec_hash(spec)]
        archive = _safe(LOCAL_ROOT / "downloads" / item["archive_name"])
        if archive.stat().st_size != item["bytes"] or file_hash(archive) != item["sha256"]:
            raise ValueError("Canonical local export archive changed before terminal retirement")
        delivered = _previous_delivery(item, spec)
        journal_path = _safe(LOCAL_ROOT / ("cleanup-" + item["job_id"] + ".json"))
        if delivered is None or read(journal_path).get("state") != "completed":
            raise ValueError("Terminal retirement requires durable strict result and completed per-job cleanup proofs")
        deliveries.append({"queue_receipt": delivered["queue_receipt"], "import_receipt": delivered["receipt"],
                           "export": item, "cleanup_receipt_sha256": file_hash(journal_path)})
    return {"version": 1, "approved_plan_sha256": spec_hash(plan), "payload_sha256": PAYLOAD_SHA256,
            "files": plan["files"], "deliveries": deliveries}


def _retirement_call(config, tool, action, envelope=None):
    code = "import subprocess,sys; x=subprocess.run([sys.executable,sys.argv[1],'--root',sys.argv[2],sys.argv[3]],input=sys.stdin.buffer.read(),capture_output=True); sys.stdout.buffer.write(x.stdout); sys.stderr.buffer.write(x.stderr); sys.exit(x.returncode)"
    return _remote(config, code, tool, REMOTE_ROOT, action,
                   stdin=json.dumps(envelope).encode() if envelope is not None else b"", timeout=900)


def retire_completed(config, plan, observed):
    """Persist explicit retirement intent before executing or resuming its exact target set."""
    from adp.queue_cleanup import _read as read
    envelope = _retirement_envelope(plan, observed)
    tool = _retirement_tool(config, plan)
    intent = _safe(LOCAL_ROOT / "retirement-envelope-v1.json")
    if intent.exists():
        if read(intent) != envelope:
            raise ValueError("Existing immutable terminal retirement intent changed")
    else:
        _write(intent, envelope)
    inspected = _retirement_call(config, tool, "--inspect")
    if (type(inspected.get("version")) is not int or inspected["version"] != 1 or inspected.get("tool") != "queue-retire-v1"
            or inspected.get("state") not in ("not_started", "in_progress", "completed")):
        raise ValueError("Invalid terminal retirement inspection")
    if inspected["state"] != "not_started" and inspected.get("envelope_sha256") != spec_hash(envelope):
        raise ValueError("Remote terminal retirement belongs to a different local proof envelope")
    if inspected["state"] != "completed":
        _retirement_call(config, tool, "--execute", envelope)
        inspected = _retirement_call(config, tool, "--inspect")
    if (type(inspected.get("version")) is not int or inspected["version"] != 1 or inspected.get("tool") != "queue-retire-v1"
            or inspected.get("state") != "completed" or inspected.get("envelope_sha256") != spec_hash(envelope)
            or type(inspected.get("target_count")) is not int or inspected["target_count"] != 1101
            or type(inspected.get("remaining_files")) is not int or inspected["remaining_files"] != 0
            or type(inspected.get("bytes_retired")) is not int or not 0 <= inspected["bytes_retired"] <= 8 * 1024**3):
        raise ValueError("Terminal retirement completion or absence evidence did not verify")
    _identifier(inspected.get("receipt_sha256")); _identifier(inspected.get("journal_file_sha256"))
    _write(LOCAL_ROOT / "retirement-receipt-v1.json", inspected)
    imported = [{"job_id": item["export"]["job_id"], "snapshot_id": item["export"]["snapshot_id"],
                 "cleanup": "completed", "cleanup_message": None} for item in envelope["deliveries"]]
    return {"observed": observed, "imported": imported, "retirement": inspected}


def sync(config, plan):
    from adp.queue_results import import_queue_result
    observed = status(config)
    exports = observed.get("exports", [])
    if not isinstance(exports, list) or len(exports) > 12:
        raise ValueError("Too many private queue exports")
    if _safe(LOCAL_ROOT / "retirement-envelope-v1.json").exists():
        return retire_completed(config, plan, observed)
    expected = {spec_hash(spec): spec for spec in plan["host_plans"]["private"]}
    downloads = _safe(LOCAL_ROOT / "downloads"); downloads.mkdir(parents=True, exist_ok=True)
    imported = []
    for item in exports:
        item = _export(item)
        if item["job_id"] not in expected:
            raise ValueError("Private export is not an approved queue job")
        archive = _safe(downloads / item["archive_name"], downloads)
        if archive.exists() and (not archive.is_file() or archive.stat().st_nlink != 1):
            raise ValueError("Queue archive must be an unshared regular file")
        if not archive.exists():
            if shutil.disk_usage(downloads).free - 3 * item["bytes"] < MINIMUM_FREE_BYTES:
                raise OSError("Queue transfer paused to preserve the 10 GiB reserve")
            descriptor, temporary_name = tempfile.mkstemp(prefix=".receive-", suffix=".zip", dir=downloads)
            temporary = Path(temporary_name)
            remote_path = REMOTE_ROOT + "/exports/" + item["archive_name"]
            reader = "from pathlib import Path; import sys; p=Path(sys.argv[1]); n=int(sys.argv[2]); assert p.is_file() and not p.is_symlink() and p.stat().st_size==n; sys.stdout.buffer.write(p.read_bytes())"
            command = shlex.join([config["remote_python"], "-c", reader, remote_path, str(item["bytes"])])
            with os.fdopen(descriptor, "wb") as output:
                result = subprocess.run([*_ssh_base(config), command], stdout=output, stderr=subprocess.PIPE, timeout=900,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                output.flush(); os.fsync(output.fileno())
            if result.returncode or temporary.stat().st_size != item["bytes"] or _sha(temporary) != item["sha256"]:
                raise OSError("Queue archive transfer did not verify; remote output was preserved")
            os.replace(temporary, archive)
        if _sha(archive) != item["sha256"]:
            raise ValueError("Local queue archive changed; it was preserved for inspection")
        delivered = _previous_delivery(item, expected[item["job_id"]])
        if delivered is None:
            delivered = import_queue_result(archive, LOCAL_ROOT / "results", payload_archive=PAYLOAD,
                job_spec=expected[item["job_id"]], expected_archive_sha256=item["sha256"])
        envelope = {"queue_receipt": delivered["queue_receipt"], "import_receipt": delivered["receipt"], "export": item}
        cleanup = "import json,subprocess,sys; p=sys.argv[1]; r=sys.argv[2]; j=sys.argv[3]; x=subprocess.run([sys.executable,p,'--root',r,'--job-id',j,'--execute'],input=sys.stdin.buffer.read(),capture_output=True); sys.stdout.buffer.write(x.stdout); sys.stderr.buffer.write(x.stderr); sys.exit(x.returncode)"
        try:
            tool = _cleanup_tool(config, plan)
            cleaned = _remote(config, cleanup, tool, REMOTE_ROOT, item["job_id"],
                              stdin=json.dumps(envelope).encode(), timeout=900)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            cleaned = {"state": "blocked", "message": str(error)[-600:]}
        _write(LOCAL_ROOT / ("cleanup-" + item["job_id"] + ".json"), cleaned)
        imported.append({"job_id": item["job_id"], "snapshot_id": item["snapshot_id"],
                         "cleanup": cleaned.get("state"), "cleanup_message": cleaned.get("message")})
    return {"observed": observed, "imported": imported}


def _queue_transfer(plan):
    transfer = {"status": "waiting", "verified_at": None, "local_path": None}
    directory = _safe(LOCAL_ROOT / "results/queue-receipts")
    if not directory.exists():
        return transfer
    expected = {spec_hash(spec): spec for spec in plan["host_plans"]["private"]}
    from adp.queue_results import queue_protocol_hash
    receipts = []
    for count, path in enumerate(directory.glob("*.json"), 1):
        path = _safe(path, directory)
        if count > 12 or not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size > MAX_STATUS:
            raise ValueError("Invalid bounded private queue receipt")
        with path.open("rb") as stream:
            raw = stream.read(MAX_STATUS + 1)
        if len(raw) > MAX_STATUS:
            raise ValueError("Private queue receipt exceeds its metadata bound")
        receipt = json.loads(raw)
        if not isinstance(receipt, dict):
            raise ValueError("Private queue receipt must be an object")
        job_id, snapshot_id = _identifier(receipt.get("job_id")), _identifier(receipt.get("snapshot_id"))
        if (job_id not in expected or receipt.get("job_spec") != expected[job_id]
                or path.name != job_id + "-" + snapshot_id + ".json" or type(receipt.get("version")) is not int or receipt["version"] != 1
                or receipt.get("verified") is not True or receipt.get("payload_sha256") != PAYLOAD_SHA256
                or receipt.get("queue_protocol_sha256") != queue_protocol_hash("private")
                or receipt.get("checkpoint_validation") != "strict_cpu_observed_only_completed"
                or not isinstance(receipt.get("verified_at"), str)):
            raise ValueError("Private queue receipt differs from the approved strict CPU proof")
        receipts.append((receipt["verified_at"], str(path)))
    if receipts:
        verified_at, local_path = max(receipts)
        transfer = {"status": "verified", "verified_at": verified_at, "local_path": local_path}
    return transfer


def _cycle(config_path=DEFAULT_CONFIG, plan_path=DEFAULT_PLAN):
    plan, config = _plan(plan_path), _config(config_path)
    start(config, plan)
    result = sync(config, plan)
    observed = result["observed"]
    raw_state = observed.get("state", "unknown")
    state = "running" if observed.get("worker_lock_held") or observed.get("process_alive") else raw_state
    if state not in {"running", "completed", "failed", "blocked", "budget_exhausted", "waiting"}:
        state = "unknown"
    done, total = observed.get("completed_jobs", 0), observed.get("total_jobs", 12)
    charged = observed.get("charged_seconds", 0)
    if (type(done) is not int or type(total) is not int or not 0 <= done <= total <= 12
            or type(charged) not in (int, float) or not math.isfinite(charged) or not 0 <= charged <= 43200):
        raise ValueError("Private queue progress or budget evidence is invalid")
    completed = observed.get("completed_epochs", done * 20)
    if type(completed) is not int or not done * 20 <= completed <= total * 20:
        raise ValueError("Private committed epoch progress is invalid")
    message = f"Separate pilot-data queue: {done}/{total} jobs complete; {charged/3600:.2f}/6 allocated GPU-hours charged."
    current = observed.get("current_run")
    if current is not None:
        if (not isinstance(current, dict) or current.get("configuration") not in CONFIGURATIONS
                or type(current.get("seed")) is not int or current["seed"] not in SEEDS
                or current.get("job_id") != spec_hash(job_spec("private", current["configuration"], current["seed"]))
                or type(current.get("completed_epochs")) is not int or not 0 <= current["completed_epochs"] <= 20
                or current.get("planned_epochs") != 20 or completed < done * 20 + current["completed_epochs"]):
            raise ValueError("Private current-job progress is invalid")
        message += f" Current {current['configuration']}, seed {current['seed']}: {current['completed_epochs']}/20 epochs committed."
    changes = {"status": state, "phase": "development queue", "message": message,
        "observed_at": observed["observed_at"], "training": {"completed_epochs": completed, "planned_epochs": total * 20, "current_run": current},
        "transfer": _queue_transfer(plan),
        "cleanup": {"status": "guarded", "message": "Completed job copies are cleared only after a durable, strict CPU-verified local transfer; shared inputs remain leased to pending jobs."}}
    blocked = [item for item in result["imported"] if item.get("cleanup") != "completed"]
    if blocked:
        reason = blocked[-1].get("cleanup_message") or "The remote cleanup guard did not confirm completion."
        changes["cleanup"] = {"status": "blocked", "message": ("Verified local results are preserved; remote job copies remain protected. " + reason)[-600:]}
        message += f" Cleanup is blocked for {len(blocked)} verified job(s); remote copies are retained."
        changes["message"] = message
    if result.get("retirement", {}).get("state") == "completed":
        changes["cleanup"] = {"status": "completed", "message": "Completed per-job training copies, shared cached scenarios, input ZIP and duplicate remote checkpoint exports were retired; source, manifests and audit records remain."}
        message += " Completed shared data and duplicate exports were retired."
        changes["message"] = message
    _publish_host(HOSTS_PATH, "private", changes)
    _write(LOCAL_ROOT / "last-sync.json", result)
    return {"state": state, "message": message}


def local_dispatch(plan):
    """The extra local queue starts only after the existing study cleanup proof passes."""
    spec = importlib.util.spec_from_file_location("interaction_local_queue_worker", resource_paths()["worker.py"])
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    try:
        worker.local_dependencies(PROJECT / "src")
    except (OSError, ValueError, RuntimeError) as error:
        return {"state": "waiting", "message": "Local extra training waits for both frozen studies, CPU reports and verified cache cleanup. " + str(error)[:240]}
    root = _safe(LOCAL_ROOT / "local")
    root.mkdir(exist_ok=True)
    lock = _safe(root / ".worker.lock", root)
    if lock.exists() and (not lock.is_file() or lock.stat().st_nlink != 1):
        raise ValueError("Local queue worker lock is invalid")
    try:
        probe = FileLock(str(lock), timeout=0)
        probe.acquire()
    except __import__("filelock").Timeout:
        return {"state": "running", "message": "The local extra-training queue already has its own worker."}
    else:
        probe.release()
    saved = root / "worker-status.json"
    if saved.exists():
        value = json.loads(_safe(saved, root).read_bytes())
        if value.get("state") in {"completed", "failed", "blocked", "paused", "budget_exhausted", "running"}:
            return {"state": value["state"], "message": "Local queue state was preserved; interrupted leases are never reset automatically."}
    if shutil.disk_usage(root).free - 6 * 1024**3 < MINIMUM_FREE_BYTES:
        return {"state": "blocked", "message": "Local queue is waiting for its data copies and the 8 GiB reserve to fit."}
    for name, source in resource_paths().items():
        target = _safe(root / name, root)
        if target.exists():
            if not target.is_file() or target.stat().st_nlink != 1 or _sha(target) != plan["files"][name]:
                raise ValueError("Existing local queue resources differ; preserving them")
        else:
            temporary = target.with_suffix(target.suffix + ".incoming")
            with source.open("rb") as stream, temporary.open("xb") as output:
                shutil.copyfileobj(stream, output)
                output.flush(); os.fsync(output.fileno())
            if _sha(temporary) != plan["files"][name]:
                raise ValueError("Local resource copy did not verify")
            os.replace(temporary, target)
    from adp.queue_results import queue_protocol_hash
    resources = {"files": plan["files"], "payload_sha256": PAYLOAD_SHA256,
                 "plan_sha256": queue_protocol_hash("local"), "approved_plan_sha256": spec_hash(plan)}
    _write(root / "resources.json", resources)
    python = Path(sys.executable)
    if os.name == "nt":
        python = python.with_name("pythonw.exe")
    args = [str(python), str(root / "worker.py"), "--root", str(root), "--host", "local",
            "--payload", str(root / "inputs.zip"), "--queue-module", str(root / "compute_queue.py"),
            "--pilot-helper", str(root / "pilot-helper.py"), "--control-source", str(PROJECT / "src")]
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
             subprocess.CREATE_BREAKAWAY_FROM_JOB) if os.name == "nt" else 0
    with (root / "worker.stdout.log").open("ab") as output, (root / "worker.stderr.log").open("ab") as error:
        process = subprocess.Popen(args, cwd=root, stdin=subprocess.DEVNULL, stdout=output, stderr=error,
            close_fds=True, creationflags=flags, start_new_session=os.name != "nt",
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src"), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    return {"state": "starting", "message": f"Local extra queue launched as PID {process.pid}; the original study source and artifacts remain separate."}


def cycle(config_path=DEFAULT_CONFIG, plan_path=DEFAULT_PLAN):
    root = _safe(LOCAL_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    path = _safe(root / ".dispatch.lock", root)
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError("Queue dispatcher lock must be an unshared regular file")
    with FileLock(str(path), timeout=0):
        plan = _plan(plan_path)
        results = {}
        try:
            results["private"] = _cycle(config_path, plan_path)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            results["private"] = {"state": "unavailable", "message": str(error)[-600:]}
            _publish_host(HOSTS_PATH, "private", {"status": "unavailable", "phase": "development queue",
                "observed_at": datetime.now(timezone.utc).isoformat(), "message": results["private"]["message"].replace("\n", " ")})
        try:
            results["local"] = local_dispatch(plan)
        except (OSError, ValueError, RuntimeError) as error:
            results["local"] = {"state": "blocked", "message": str(error)[-600:]}
        _write(LOCAL_ROOT / "dispatch-status.json", results)
        return {"state": results["private"]["state"],
                "message": results["private"]["message"] + " " + results["local"]["message"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--approve-plan", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--retire-completed", action="store_true", help="Explicitly retire completed queue copies using a separately approved tool")
    args = parser.parse_args(argv)
    if args.approve_plan:
        value = approve_plan(args.plan)
    elif args.status:
        value = status(_config(args.config))
    elif args.retire_completed:
        root = _safe(LOCAL_ROOT)
        with FileLock(str(_safe(root / ".dispatch.lock", root)), timeout=0):
            config, plan = _config(args.config), _plan(args.plan)
            value = retire_completed(config, plan, status(config))
    else:
        value = cycle(args.config, args.plan)
    print(json.dumps(value, allow_nan=False))


if __name__ == "__main__":
    main()
