import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from adp.distributed_plan import initial_state, plan, spec_hash

PROJECT = Path(__file__).resolve().parents[1]


def worker():
    spec = importlib.util.spec_from_file_location('tested_distributed_worker',PROJECT/'scripts/distributed-worker.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_interrupted_reservation_cannot_be_restarted(tmp_path, monkeypatch):
    module=worker()
    root=tmp_path/'allocation'; root.mkdir()
    base_path=PROJECT/'scripts/compute-queue-worker.py'
    (root/'worker-base.py').write_bytes(base_path.read_bytes())
    (root/'distributed_plan.py').write_bytes((PROJECT/'src/adp/distributed_plan.py').read_bytes())
    (root/'inputs.zip').write_bytes(b'approved test bytes')
    module.BASE_SHA=hashlib.sha256(base_path.read_bytes()).hexdigest()
    import adp.distributed_plan as recipe
    (root/'resources.json').write_text(json.dumps({'protocol_sha256':recipe.protocol_hash('colab'),'files':{}}))
    state=initial_state('colab'); identifier=spec_hash(plan('colab')['jobs'][0])
    state['active']=identifier; state['attempts']=[{'job_id':identifier,'state':'reserved','charged_seconds':1800}]
    state['charged_seconds']+=1800
    (root/'allocation.json').write_text(json.dumps(state))
    original_load=module.load
    def load(path,name):
        if name=='distributed_plan':
            return recipe
        base=original_load(path,name)
        base.sha=lambda p: recipe.PAYLOAD_SHA256 if Path(p).name=='inputs.zip' else hashlib.sha256(Path(p).read_bytes()).hexdigest()
        return base
    monkeypatch.setattr(module,'load',load)
    with pytest.raises(RuntimeError,match='conservatively charged'):
        module.run(SimpleNamespace(root=root,host='colab'))
    assert json.loads((root/'allocation.json').read_bytes())==state


def test_pinned_base_worker_cannot_be_replaced(tmp_path):
    (tmp_path/'worker-base.py').write_text('raise RuntimeError("untrusted")')
    with pytest.raises(ValueError,match='Frozen base worker differs'):
        worker().run(SimpleNamespace(root=tmp_path,host='private'))
