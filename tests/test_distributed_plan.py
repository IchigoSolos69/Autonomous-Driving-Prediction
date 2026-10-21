from copy import deepcopy
import pytest

from adp.distributed_plan import (HOST_SEEDS, HOST_SECONDS, PRIOR_SECONDS, initial_state,
    plan, protocol_hash, validate_spec, validate_state)
from adp.compute_queue import job_spec, spec_hash


def test_followup_assigns_distinct_seeds_and_preserves_legacy():
    work = []
    for host in HOST_SEEDS:
        for spec in plan(host)['jobs']:
            assert validate_spec(spec) is spec
            assert spec['epochs'] == 10
            assert spec['seed'] not in (42,43,44)
            work.append((spec['configuration'],spec['seed']))
    assert len(work) == len(set(work)) == 36
    assert job_spec('private','full',42)['epochs'] == 20
    assert len(set(protocol_hash(host) for host in HOST_SEEDS)) == 3


@pytest.mark.parametrize('host', list(HOST_SEEDS))
def test_initial_state_retains_previous_spend(host):
    state = initial_state(host)
    assert validate_state(state,host) is state
    assert state['charged_seconds'] == PRIOR_SECONDS[host]
    assert HOST_SECONDS-state['charged_seconds'] <= 21600


@pytest.mark.parametrize('mutation', [lambda s: s.update(charged_seconds=0),
    lambda s: s.update(protocol_sha256='0'*64),
    lambda s: s.update(setup_charged_seconds=float('nan')),
    lambda s: s.update(completed=['0'*64]),
    lambda s: s.update(active='0'*64)])
def test_bad_ledger_fails_closed(mutation):
    state = initial_state('private'); mutation(state)
    with pytest.raises(ValueError): validate_state(state,'private')


def test_reserved_interruption_remains_fully_charged():
    state = initial_state('kaggle')
    identifier = spec_hash(plan('kaggle')['jobs'][0])
    state['active'] = identifier
    state['attempts'].append({'job_id':identifier,'state':'reserved','charged_seconds':1800})
    state['charged_seconds'] += 1800
    assert validate_state(state,'kaggle')['charged_seconds'] == 3600
    broken = deepcopy(state); broken['attempts'].append(broken['attempts'][0])
    broken['charged_seconds'] += 1800
    with pytest.raises(ValueError): validate_state(broken,'kaggle')


def test_completed_requires_matching_settled_attempt():
    state = initial_state('colab')
    state['completed'] = [spec_hash(plan('colab')['jobs'][0])]
    with pytest.raises(ValueError): validate_state(state,'colab')


def test_followup_cannot_mutate_recipe():
    spec = plan('private')['jobs'][0]; spec['epochs'] = 20
    with pytest.raises(ValueError): validate_spec(spec)
