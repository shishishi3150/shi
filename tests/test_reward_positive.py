"""Nonnegative task rewards must not pay accidents or prefer escaping costs."""
from copy import deepcopy
import math

import numpy as np
import pytest

from usv_env.config import assemble, PRESETS, validate
from usv_env.environment import PursuitEvasionEnv
from usv_env.reward import evaluate, REWARD_COMPONENTS, FAILURE_REASONS


@pytest.fixture
def env():
    value = PursuitEvasionEnv(assemble(device='cpu'))
    yield value
    value.close()


@pytest.mark.parametrize('reason', ['', 'success', *FAILURE_REASONS])
@pytest.mark.parametrize('mask', [[1]*5, [0]*5, [1,0,1,0,1]])
def test_positive_rewards_balance_and_failures_pay_nothing(env, reason, mask):
    env._agent_active_mask = np.asarray(mask, np.float32)
    rewards, terms = evaluate(None, env, reason)
    assert env.config['reward']['scheme'] == 'cooperative_positive_v2'
    parts = np.asarray([terms[name] for name in REWARD_COMPONENTS])
    assert (parts >= 0.).all() and (rewards >= 0.).all()
    np.testing.assert_allclose(rewards, parts.sum(0)+terms['terminal_reward'], atol=1e-6)
    assert (rewards[np.asarray(mask)==0] == 0.).all()
    if reason in FAILURE_REASONS:
        np.testing.assert_array_equal(rewards, np.zeros(5))
        assert not parts.any() and not np.any(terms['terminal_reward'])
    elif reason == 'success':
        assert np.all(rewards[np.asarray(mask)==1] >= env.config['reward']['terminal']['success'])
    else:
        assert rewards.max() <= env.config['reward']['positive']['step_budget']+1e-7


def test_same_prefix_continuation_cannot_lose_return_to_immediate_crash(env):
    reward, _ = evaluate(None, env)
    crash, _ = evaluate(None, env, 'collision')
    gamma = env.config['training']['gamma']
    assert reward.min() > 0.
    for length in (1, 20, 321, 650, 900):
        raw = np.tile(reward, (length, 1)).sum(0)
        discounted = (gamma**np.arange(length))@np.tile(reward, (length, 1))
        assert np.all(raw > crash) and np.all(discounted > crash)
    assert not any(evaluate(None, env)[1]['time_cost'])


def test_raw_episode_budget_bounds_process_capture_and_completion(env):
    cfg = env.config
    positive = cfg['reward']['positive']
    assert cfg['reward']['terminal']['success'] == 2.
    assert env.max_steps*positive['step_budget']+positive['capture_bonus']+cfg['reward']['terminal']['success'] <= positive['episode_budget']


def test_closer_encirclement_and_safe_motion_increase_auv_reward(env, monkeypatch):
    monkeypatch.setattr(env, '_clearances', lambda i: [2.])
    red = env.red_state.pos
    for i, ship in enumerate(env.submarine_states):
        angle = 2*math.pi*i/3
        ship.pos = red+9*np.array([math.cos(angle), math.sin(angle)])
    far, _ = evaluate(None, env)
    for ship in env.submarine_states:
        ship.pos = red+(ship.pos-red)/3
    near, terms = evaluate(None, env)
    assert np.all(near[2:] > far[2:])
    assert terms['auv_hull_reward'][2] > 0.
    # Unsafe clearance with otherwise identical task geometry earns no reward.
    monkeypatch.setattr(env, '_clearances', lambda i: [0.])
    np.testing.assert_array_equal(evaluate(None, env)[0], np.zeros(5))


def test_usv_causal_credit_requires_fresh_positive_measured_effect(env, monkeypatch):
    monkeypatch.setattr(env, '_clearances', lambda i: [2.])
    _, missing = evaluate(None, env)
    assert missing['usv_blocking_reward'] == [0.]*5
    assert missing['usv_contribution_reward'] == [0.]*5
    env.counterfactual_audit = dict(valid=True, step=0, total_effect_m=.1,
        shapley_effect_m=[-.1, .2])
    _, measured = evaluate(None, env)
    assert measured['usv_blocking_reward'][0] > 0.
    assert measured['usv_contribution_reward'][0] == 0.
    assert measured['usv_contribution_reward'][1] > 0.
    env.step_count = env.config['counterfactual']['max_age_steps']+1
    _, stale = evaluate(None, env)
    assert not any(stale['usv_blocking_reward'])
    assert not any(stale['usv_contribution_reward'])
    env.capture_success = True
    env.first_capture_step = env.step_count-1
    _, withdrawing = evaluate(None, env)
    assert not any(withdrawing['usv_blocking_reward'])
    assert not any(withdrawing['usv_contribution_reward'])
    assert withdrawing['usv_withdrawal_reward'][0] > 0.


def test_capture_bonus_occurs_only_at_first_capture_step_and_never_on_failure(env):
    env.capture_success = True
    env.first_capture_step = env.step_count
    first, terms = evaluate(None, env)
    assert terms['capture_bonus'] == [env.config['reward']['positive']['capture_bonus']]*5
    env.step_count += 1
    later, terms = evaluate(None, env)
    assert not any(terms['capture_bonus']) and np.all(first > later)
    assert not evaluate(None, env, 'collision')[0].any()


def test_reward_is_identical_for_all_ablation_methods():
    reference = assemble()['reward']
    for method in PRESETS:
        assert assemble(experiment=method)['reward'] == reference


@pytest.mark.parametrize('change', ['negative_weight', 'sum', 'failure_bonus', 'nonpositive_success', 'episode_budget'])
def test_positive_configuration_rejects_incentive_and_budget_breaks(change):
    cfg = assemble()
    positive = cfg['reward']['positive']
    if change == 'negative_weight': positive['auv_weights']['auv_distance_reward'] = -.1
    if change == 'sum': positive['usv_weights']['usv_approach_reward'] += .1
    if change == 'failure_bonus': cfg['reward']['terminal']['collision'] = 1.
    if change == 'nonpositive_success': cfg['reward']['terminal']['success'] = 0.
    if change == 'episode_budget': positive['episode_budget'] = 1.
    with pytest.raises(ValueError, match='reward'):
        validate(cfg)
