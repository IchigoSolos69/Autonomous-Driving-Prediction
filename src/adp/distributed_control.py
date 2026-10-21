"""Monitor the separately authorized finite allocation; never reconnect cloud runtimes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

from adp.compute_cleanup import _safe, _write
from adp.compute_monitor import _publish_host
from adp.distributed_plan import plan, protocol_hash, validate_state, spec_hash
from adp.private_queue import _remote as _remote_once, _ssh_base, DEFAULT_CONFIG, PAYLOAD, HOSTS_PATH
from adp.remote_control import _config
from filelock import FileLock

ROOT = Path('D:/InteractionLab/remote-compute/distributed-v2')
REMOTE = '/home/aadi/interaction-lab/queues/distributed-v2'
PROJECT = Path(__file__).resolve().parents[2]
DRIVE = Path('G:/My Drive/InteractionLab/compute-v1/colab/development-v2')
RESERVE = 10 * 1024**3
KAGGLE_KERNEL = 'adimaitre/interaction-lab-distributed-development-v2'


def _remote(*args, **kwargs):
    try:
        return _remote_once(*args, **kwargs)
    except RuntimeError as error:
        message=str(error)
        # A connection timeout precedes remote execution, so one retry is safe.
        if 'ssh: connect to host ' not in message or 'port 22: Connection timed out' not in message:
            raise
        time.sleep(2)
        return _remote_once(*args, **kwargs)


def poll_kaggle():
    from adp.cloud_monitor import _kaggle_cli, _kaggle_credentials_present
    if not _kaggle_credentials_present() or not (client := _kaggle_cli()):
        return {'state':'connection_required'}
    folder = ROOT/'kaggle/saved'
    folder.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ROOT).free < RESERVE+2*512*1024**2:
        return {'state':'storage_reserve'}
    pattern = r'^interaction-lab-saved-v2-recovery1/(allocation|evidence)\.json$'
    options = {'capture_output':True,'timeout':90,
        'creationflags':subprocess.CREATE_NO_WINDOW if __import__('os').name=='nt' else 0}
    response = subprocess.run([client,'kernels','output',KAGGLE_KERNEL,'-p',str(folder),
        '--file-pattern',pattern,'--quiet'],**options)
    if response.returncode:
        # Never persist API stderr or credential metadata in host evidence.
        _publish_host(HOSTS_PATH,'kaggle',{'connection':{'status':'unavailable',
            'phase':'saved_output_api_unavailable','scope':'saved_notebook_outputs',
            'checked_at':datetime.now(timezone.utc).isoformat(),
            'message':'The official read-only API cannot access the new private saved run outputs. Browser run evidence is retained; no new training progress is inferred.'}})
        return {'state':'saved_output_api_unavailable'}
    path = _safe(folder/'interaction-lab-saved-v2-recovery1/allocation.json')
    if not path.exists(): return {'state':'awaiting_saved_outputs'}
    if path.stat().st_size>128*1024: raise ValueError('Saved allocation exceeds its bound')
    state = validate_state(json.loads(path.read_bytes()),'kaggle')
    for export in state['exports']:
        identifier=export.get('snapshot_id')
        if not isinstance(identifier,str) or len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid output snapshot ID')
        target=folder/'interaction-lab-saved-v2-recovery1/exports'/(identifier+'.zip')
        if target.exists(): continue
        expression=r'^interaction-lab-saved-v2-recovery1/exports/'+identifier+r'\.zip$'
        result=subprocess.run([client,'kernels','output',KAGGLE_KERNEL,'-p',str(folder),
            '--file-pattern',expression,'--quiet'],**options)
        if result.returncode: raise RuntimeError('The official API could not download a saved model export')
    return {'state':'checked'}


def digest(path):
    with _safe(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def approval():
    path = _safe(ROOT / 'approved.json')
    if not path.is_file() or path.stat().st_size > 128*1024:
        raise ValueError('The versioned distributed allocation is not approved')
    value = json.loads(path.read_bytes())
    expected = json.loads((PROJECT/'artifacts/distributed-v2/plan.json').read_bytes())
    if value != expected or digest(PROJECT/'artifacts/distributed-v2/controller.zip') != value['controller_sha256']:
        raise ValueError('Distributed source or approved allocation changed')
    if any(value['hosts'][host] != plan(host) for host in ('private','colab','kaggle')):
        raise ValueError('Distributed recipes changed')
    return value


def inspect_private(config):
    code = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
from filelock import FileLock,Timeout
root=Path(sys.argv[1]); assert str(root)=='/home/aadi/interaction-lab/queues/distributed-v2'
assert not any(p.is_symlink() for p in (root,*root.parents))
out={'exists':root.exists(),'observed_at':datetime.now(timezone.utc).isoformat(),'worker_lock_held':False}
if root.exists():
 p=root/'allocation.json'
 if p.exists():
  assert p.is_file() and p.stat().st_nlink==1 and p.stat().st_size<128*1024
  out['allocation']=json.loads(p.read_bytes())
  active=out['allocation'].get('active')
  if isinstance(active,str) and len(active)==64 and all(c in '0123456789abcdef' for c in active):
   history=root/'jobs'/active/'run/history.json'
   if history.is_file() and not history.is_symlink() and history.stat().st_size<128*1024:
    rows=json.loads(history.read_bytes())['epochs']
    assert len(rows)<=10 and all(row['epoch']==i for i,row in enumerate(rows,1))
    out['allocation']['progress']={'saved_epochs':len(rows)}
 lock=FileLock(str(root/'.distributed.lock'),timeout=0)
 try: lock.acquire()
 except Timeout: out['worker_lock_held']=True
 else: lock.release()
print(json.dumps(out))
'''
    return _remote(config, code, REMOTE, timeout=30)


def start_private(config, approved):
    observed = inspect_private(config)
    if observed.get('allocation') is not None:
        validate_state(observed['allocation'], 'private')
        # A previously started allocation is never reset or relaunched from a timestamp.
        return observed
    if observed['worker_lock_held']:
        return observed
    _remote(config, "import json,subprocess; p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True); assert not p.stdout.strip(), 'Private GPU has another owner; do not overlap training'; print(json.dumps({'gpu_idle':True}))")
    if not observed['exists']:
        _remote(config, "from pathlib import Path; import json,shutil,sys; p=Path(sys.argv[1]); assert not any(x.is_symlink() for x in (p,*p.parents)); assert shutil.disk_usage(p.parent).free>16*1024**3; p.mkdir(); print(json.dumps({'ready':True}))", REMOTE)
    payloads = {'controller.zip': PROJECT/'artifacts/distributed-v2/controller.zip', 'inputs.zip': PAYLOAD}
    for name, source in payloads.items():
        incoming = REMOTE + '/.incoming-' + name
        response = subprocess.run(['scp', *_ssh_base(config)[1:-1], str(source), config['host']+':'+incoming],
            capture_output=True, timeout=900, creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name=='nt' else 0)
        if response.returncode:
            raise RuntimeError('Private staging failed; source and any partial copy were preserved')
        code = "from pathlib import Path; import hashlib,json,os,sys; a,b=map(Path,sys.argv[1:3]); assert not any(p.is_symlink() for p in (a,b,*a.parents)); assert a.is_file() and a.stat().st_nlink==1; f=a.open('rb'); assert hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3]; f.close(); assert not b.exists() or b.read_bytes()==a.read_bytes(); os.replace(a,b); print(json.dumps({'verified':True}))"
        _remote(config, code, incoming, REMOTE+'/'+name, digest(source))
    code = r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path(sys.argv[1]); expected=json.loads(sys.argv[2]); archive=root/'controller.zip'
with archive.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected['controller_sha256']
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist())) and sum(i.file_size for i in z.infolist())<2*1024**2
 for name in z.namelist():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  p=root/name; assert not any(x.is_symlink() for x in (p,*p.parents)); data=z.read(name)
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists(): assert p.read_bytes()==data
  else: p.write_bytes(data)
with zipfile.ZipFile(archive) as z:
 files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in z.namelist()}
resources={'protocol_sha256':expected['protocol_hashes']['private'],'files':files}
path=root/'resources.json'
if path.exists(): assert json.loads(path.read_bytes())==resources
else: path.write_text(json.dumps(resources,sort_keys=True))
env={**os.environ,'PYTHONPATH':str(root/'control'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
with (root/'worker.stdout.log').open('ab') as out,(root/'worker.stderr.log').open('ab') as err:
 p=subprocess.Popen([sys.executable,str(root/'distributed-worker.py'),'--root',str(root),'--host','private'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
print(json.dumps({'starting':True,'pid':p.pid}))
'''
    _remote(config, code, REMOTE, json.dumps(approved))
    return inspect_private(config)


def verified_exports(host, state, folder, config=None, *, pending=None):
    from adp.queue_results import import_distributed_result
    result = ROOT / host / 'results'
    downloads = ROOT / host / 'downloads'; downloads.mkdir(parents=True, exist_ok=True)
    verified = []
    for export in state['exports']:
        identifier, checksum = export.get('snapshot_id'), export.get('sha256')
        if (not isinstance(identifier,str) or len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier)
                or export.get('archive_name') != identifier+'.zip' or not isinstance(checksum,str)
                or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum)
                or type(export.get('bytes')) is not int or not 0<export['bytes']<512*1024**2):
            raise ValueError('Invalid bounded distributed export')
        specification = next((spec for spec in plan(host)['jobs'] if spec_hash(spec)==export['job_id']),None)
        if specification is None:
            raise ValueError('Export references an unassigned job')
        receipt = result/'queue-receipts'/(export['job_id']+'-'+identifier+'.json')
        if receipt.is_file():
            proof = json.loads(receipt.read_bytes())
            if (proof.get('verified') is not True or proof.get('archive_sha256')!=checksum
                    or proof.get('job_spec')!=specification or proof.get('queue_protocol_sha256')!=protocol_hash(host)):
                raise ValueError('Stored distributed receipt differs')
            directory = _safe(Path(proof['local_directory']), result)
            if any(digest(_safe(directory/name,directory))!=entry['sha256'] for name,entry in proof['files'].items()):
                raise ValueError('Canonical verified model changed; preserve remote copies')
            verified.append(str(receipt)); continue
        target = downloads / export['archive_name']
        if not target.exists() and config is None and pending is not None:
            source = _safe(folder/'exports'/export['archive_name'])
            if not source.is_file():
                # Drive can publish the small ledger before its referenced ZIPs.
                pending.append(identifier)
                continue
        if shutil.disk_usage(ROOT).free - 3*export['bytes'] < RESERVE:
            raise RuntimeError('Transfer paused to preserve 10 GiB free')
        if not target.exists():
            temporary = target.with_suffix('.partial')
            if config is not None:
                response = subprocess.run(['scp', *_ssh_base(config)[1:-1], config['host']+':'+REMOTE+'/exports/'+export['archive_name'], str(temporary)],
                    capture_output=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name=='nt' else 0)
                if response.returncode: raise RuntimeError('Private result transfer failed; original preserved')
            else:
                shutil.copyfile(_safe(folder/'exports'/export['archive_name']), temporary)
            if digest(temporary)!=checksum: raise ValueError('Transferred model archive checksum differs')
            temporary.replace(target)
        installed = import_distributed_result(target, result, payload_archive=PAYLOAD,
            job_spec=specification, expected_archive_sha256=checksum)
        verified.append(installed['queue_receipt_path'])
    return verified


def publish(host, state, observed_at, verified, *, live, pending=0):
    count = len(state['completed'])
    active = next((spec for spec in plan(host)['jobs'] if spec_hash(spec)==state['active']),None)
    current = None if active is None else {'configuration':active['configuration'],'seed':active['seed'],
        'completed_epochs':0,'planned_epochs':10,'job_id':state['active']}
    status = 'running' if state['state']=='running' and live else 'waiting' if state['state']=='ready' else state['state']
    if status not in {'running','waiting','paused','completed','failed','blocked','unavailable'}: status='blocked'
    changes = {'status':status, 'phase':'distributed development v2',
        'observed_at':observed_at,
        'message':f"Distinct host allocation: {count}/12 jobs complete; {state['charged_seconds']/3600:.2f}/6 GPU-hours charged, including prior usage and active reservations. " + ('Worker lock observed.' if live else 'Saved notebook evidence; live GPU state is unverified.'),
        'training':{'completed_epochs':count*10+state.get('progress',{}).get('saved_epochs',0),'planned_epochs':120,'current_run':current},
        'cleanup':{'status':'guarded','message':'New allocation copies remain protected until locally verified completion and guarded cleanup.'}}
    if verified is not None:
        changes['transfer'] = {'status':'partial' if verified and pending else 'verified' if verified else 'pending',
            'verified_at':json.loads(Path(verified[-1]).read_bytes())['verified_at'] if verified else None,
            'local_path':verified[-1] if verified else None}
    _publish_host(HOSTS_PATH, host, changes)
    if host == 'colab':
        _publish_host(HOSTS_PATH,host,{'connection':{'status':'checked','phase':'saved_queue_evidence',
            'scope':'saved_notebook_heartbeat','checked_at':datetime.now(timezone.utc).isoformat(),
            'source_observed_at':observed_at,'completed_jobs':count,
            'available_archives':len(verified) if verified is not None else None,
            'expected_archives':len(state['exports']),
            'message':'Validated allocation evidence is readable through Drive. ' +
                (f'{len(verified)}/{len(state["exports"])} completed model exports verified locally. ' if verified is not None else '') +
                (f'{pending} export(s) await Drive sync; progress remains available. ' if pending else '') +
                'Source timestamps and locally verified results are separate; this does not establish live GPU telemetry.'}})


def _cycle():
    approved = approval()
    outcomes = {}
    try:
        config = _config(DEFAULT_CONFIG)
        observed = start_private(config, approved)
        state = observed.get('allocation')
        if state is not None:
            validate_state(state,'private')
            publish('private',state,observed['observed_at'],None,live=observed['worker_lock_held'])
            verified = verified_exports('private',state,None,config)
            publish('private',state,observed['observed_at'],verified,live=observed['worker_lock_held'])
            _publish_host(HOSTS_PATH,'private',{'connection':{'status':'checked','phase':'ssh_connected',
                'scope':'connection_capability','checked_at':observed['observed_at'],
                'message':'SSH lock, allocation ledger and available completed exports were checked.'}})
        outcomes['private'] = {'state':state['state'] if state else 'starting'}
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        outcomes['private'] = {'state':'unavailable','message':str(error)[-300:]}
    try:
        outcomes['kaggle_api'] = poll_kaggle()
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError):
        outcomes['kaggle_api'] = {'state':'saved_output_api_unavailable'}
    for host, folder in (('colab',DRIVE),('kaggle',ROOT/'kaggle/saved/interaction-lab-saved-v2-recovery1')):
        try:
            path = _safe(folder/'allocation.json')
            if not path.is_file():
                outcomes[host] = {'state':'awaiting_notebook'}; continue
            if path.stat().st_size>128*1024: raise ValueError('Saved allocation exceeds its bound')
            state = validate_state(json.loads(path.read_bytes()),host)
            # A delayed or failed model transfer must not hide a valid observation.
            publish(host,state,state.get('observed_at'),None,live=False)
            pending = []
            try:
                verified = verified_exports(host,state,folder,pending=pending)
            except (ValueError,OSError,RuntimeError) as error:
                message = 'Training evidence was read; model transfer needs attention: ' + ' '.join(str(error).split())[-300:]
                _publish_host(HOSTS_PATH,host,{'connection':{'status':'checked','phase':'model_transfer_pending',
                    'scope':'saved_notebook_heartbeat','checked_at':datetime.now(timezone.utc).isoformat(),
                    'source_observed_at':state.get('observed_at'),'message':message}})
                outcomes[host] = {'state':state['state'],'transfer':'pending','message':message}
                continue
            publish(host,state,state.get('observed_at'),verified,live=False,pending=len(pending))
            outcomes[host] = {'state':state['state'],'verified_exports':len(verified),'pending_exports':len(pending)}
        except (ValueError,OSError,RuntimeError) as error:
            outcomes[host] = {'state':'unavailable','message':str(error)[-300:]}
    _write(ROOT/'monitor-status.json',outcomes)
    return outcomes


def cycle():
    ROOT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(_safe(ROOT/'.controller.lock',ROOT)),timeout=0):
        return _cycle()
