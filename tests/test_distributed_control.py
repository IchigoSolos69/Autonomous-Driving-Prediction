import json
from types import SimpleNamespace

import pytest
from adp import distributed_control as control
from adp.distributed_plan import initial_state


def test_permission_error_does_not_publish_fake_training_or_credentials(tmp_path,monkeypatch):
    from adp import cloud_monitor
    monkeypatch.setattr(control,'ROOT',tmp_path)
    monkeypatch.setattr(cloud_monitor,'_kaggle_cli',lambda:'official-kaggle')
    monkeypatch.setattr(cloud_monitor,'_kaggle_credentials_present',lambda:True)
    monkeypatch.setattr(control.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    calls=[]; published=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1,stderr='private-token-value',stdout='')
    monkeypatch.setattr(control.subprocess,'run',run)
    monkeypatch.setattr(control,'_publish_host',lambda *args:published.append(args))
    assert control.poll_kaggle()['state']=='saved_output_api_unavailable'
    assert calls[0][1:3]==['kernels','output']
    assert set(published[0][2])=={'connection'}
    assert 'private-token-value' not in json.dumps(published[0][2])


def test_private_completed_or_started_allocation_is_never_retrained(monkeypatch):
    state=initial_state('private')
    observed={'allocation':state,'worker_lock_held':False}
    monkeypatch.setattr(control,'inspect_private',lambda _:observed)
    monkeypatch.setattr(control,'_remote',lambda *args,**kwargs:pytest.fail('Unexpected remote mutation'))
    assert control.start_private({}, {}) is observed


def test_only_preconnection_timeout_is_retried(monkeypatch):
    calls=[]
    def remote(*args,**kwargs):
        calls.append(args)
        if len(calls)==1:
            raise RuntimeError('ssh: connect to host 192.168.29.71 port 22: Connection timed out')
        return {'ok':True}
    monkeypatch.setattr(control,'_remote_once',remote)
    monkeypatch.setattr(control.time,'sleep',lambda _:None)
    assert control._remote('config','read')['ok']
    assert len(calls)==2
    monkeypatch.setattr(control,'_remote_once',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('remote action failed')))
    with pytest.raises(RuntimeError,match='remote action failed'):
        control._remote('config','mutation')


def test_saved_colab_evidence_replaces_the_old_connection_probe(monkeypatch):
    calls=[]
    monkeypatch.setattr(control,'_publish_host',lambda *args:calls.append(args[2]))
    state=initial_state('colab'); state['state']='running'
    control.publish('colab',state,'2026-10-20T18:00:00+00:00',[],live=False)
    assert calls[-1]['connection']['phase']=='saved_queue_evidence'
    assert calls[-1]['connection']['source_observed_at']=='2026-10-20T18:00:00+00:00'
    assert 'does not establish live GPU' in calls[-1]['connection']['message']


@pytest.mark.parametrize('identifier',['../outside','A'*64,'0'*63])
def test_download_rejects_untrusted_export_names(tmp_path,monkeypatch,identifier):
    monkeypatch.setattr(control,'ROOT',tmp_path)
    state=initial_state('private')
    state['exports']=[{'snapshot_id':identifier,'archive_name':identifier+'.zip','sha256':'0'*64,'bytes':20}]
    with pytest.raises(ValueError,match='Invalid bounded distributed export'):
        control.verified_exports('private',state,None,{})
