"""Compatibility contracts for historical negative-reward policies."""
import numpy as np
import pytest

from usv_env.config import assemble, validate, PRESETS
from usv_env.environment import PursuitEvasionEnv
from usv_env.reward import (evaluate, snapshot, smooth_distance_quality,
                            REWARD_COMPONENTS, FAILURE_REASONS)


def cost_config(**kwargs):
    cfg = assemble(**kwargs)
    cfg['reward']['scheme'] = 'cooperative_cost_v2'
    cfg['reward']['terminal'] = dict(success=0., **{reason:-10. for reason in FAILURE_REASONS})
    return cfg


def test_far_field_distance_has_useful_monotone_guidance_and_real_capture_band():
    q = smooth_distance_quality(np.array([30., 25., 20., 10., 5., 3., 2., 1.]), 2., 5., 8.)
    assert np.all(np.diff(q[:5]) > 0)
    assert q[0] > .2 and q[1]-q[0] > .04
    assert np.all(q[4:7] == 1.) and q[-1] < 1.


@pytest.mark.parametrize('reason', ['', 'success', *FAILURE_REASONS])
@pytest.mark.parametrize('mask', [[1]*5, [0]*5, [1,0,1,0,1]])
def test_paid_rewards_nonpositive_bounded_balanced_and_inactive_zero(reason, mask):
    env = PursuitEvasionEnv(cost_config(device='cpu'))
    env._agent_active_mask = np.array(mask, np.float32)
    env.last_executed_actions[:] = env.agent_action_limits
    env.counterfactual_audit_new = True
    env.counterfactual_audit = dict(valid=True, shapley_effect_m=[-100.,100.])
    reward, terms = evaluate(snapshot(env), env, reason)
    dense = sum(np.asarray(terms[k]) for k in REWARD_COMPONENTS)
    assert np.all(dense <= 0.) and np.all(dense >= -.1-1e-12)
    assert np.all(reward <= 0.)
    np.testing.assert_allclose(reward, dense + terms['terminal_reward'], atol=1e-6)
    assert np.all(reward[np.asarray(mask)==0] == 0.)
    if reason == 'success':
        np.testing.assert_array_equal(reward, np.zeros(5))
        assert not any(np.any(terms[k]) for k in REWARD_COMPONENTS)
    elif reason:
        np.testing.assert_allclose(terms['terminal_reward'], -10*np.asarray(mask))
    env.close()


def test_failure_cannot_beat_any_finite_successful_continuation_by_escaping_costs():
    cfg = cost_config(device='cpu'); gamma = cfg['training']['gamma']
    a = cfg['reward']['cost']['step_budget']; budget = cfg['reward']['cost']['episode_budget']
    for remaining in (1, 20, 321, 900):
        worst_success = -a*sum(gamma**t for t in range(remaining-1))  # Success step zero.
        for reason in FAILURE_REASONS:
            assert cfg['reward']['terminal'][reason] < worst_success
    assert cfg['env']['max_steps']*a + 10 <= budget == 100.


@pytest.mark.parametrize('change', ['gamma', 'step', 'terminal', 'weights', 'success'])
def test_configuration_rejects_broken_bounds_and_early_failure_escape(change):
    cfg = cost_config(device='cpu')
    if change == 'gamma': cfg['training']['gamma'] = .995
    if change == 'step': cfg['reward']['cost']['step_budget'] = .11
    if change == 'terminal': cfg['reward']['terminal']['timeout'] = -1.
    if change == 'weights': cfg['reward']['cost']['auv_weights']['time_cost'] += .01
    if change == 'success': cfg['reward']['terminal']['success'] = 1.
    with pytest.raises(ValueError, match='reward'):
        validate(cfg)


def test_all_experiments_share_identical_reward_and_budget():
    reference = cost_config(device='cpu')['reward']
    for preset in PRESETS:
        assert cost_config(experiment=preset, device='cpu')['reward'] == reference


def test_success_last_step_does_not_refund_accumulated_cost():
    env = PursuitEvasionEnv(cost_config(device='cpu'))
    previous = snapshot(env)
    paid, _ = evaluate(previous, env)
    success, _ = evaluate(previous, env, 'success')
    assert np.all(paid < 0.)
    np.testing.assert_array_equal(success, np.zeros(5))
    np.testing.assert_array_equal(paid+success, paid)


def test_hold_and_signed_causal_credit_reduce_cost_without_positive_rewards():
    env = PursuitEvasionEnv(cost_config(device='cpu'))
    before = snapshot(env); _, initial = evaluate(before, env)
    env.task_state.hold_steps = 10
    _, hold = evaluate(before, env)
    assert hold['auv_hold_cost'][2] > initial['auv_hold_cost'][2]
    env.counterfactual_audit_new = True
    env.counterfactual_audit = dict(valid=True, shapley_effect_m=[-1., 1.])
    _, causal = evaluate(before, env)
    assert causal['usv_shapley_cost'][0] < initial['usv_shapley_cost'][0]
    assert initial['usv_shapley_cost'][1] < causal['usv_shapley_cost'][1] == 0.


def test_yaw_threshold_at_physical_limit_disables_excess_yaw_cost_safely():
    cfg = cost_config(device='cpu')
    for role in ('usv', 'auv'):
        cfg['reward'][role]['yaw_soft_threshold_rps'] = cfg['ship']['max_yaw_rate']
    validate(cfg)
    env = PursuitEvasionEnv(cfg)
    reward, terms = evaluate(snapshot(env), env)
    assert np.isfinite(reward).all() and terms['yaw_rate'] == [0.]*5
