from adp import compute_monitor, distributed_control, private_queue


def test_pending_local_studies_do_not_block_remote_allocations(tmp_path,monkeypatch):
    monkeypatch.setattr(distributed_control,'ROOT',tmp_path)
    (tmp_path/'approved.json').write_text('{}')
    monkeypatch.setattr(private_queue,'_plan',lambda _: {})
    def pending(_):
        raise ValueError('Frozen selected checkpoint is not ready')
    monkeypatch.setattr(private_queue,'local_dispatch',pending)
    calls=[]
    def distributed():
        calls.append(True)
        return {host:{'state':'running'} for host in ('private','colab','kaggle')}
    monkeypatch.setattr(distributed_control,'cycle',distributed)
    def monitor(*args,**kwargs):
        result=kwargs['postcheck']()
        assert result['state']=='running'
        assert 'Frozen selected checkpoint is not ready' in result['message']
        return result
    monkeypatch.setattr(compute_monitor,'run_monitor',monitor)
    assert compute_monitor.main(['--once','--queue-plan',str(tmp_path/'plan.json')])==0
    assert calls==[True]
