"""Historical cost policies keep their recorded withdrawal reward semantics."""
import math

import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.reward import evaluate, snapshot, REWARD_COMPONENTS, FAILURE_REASONS


@pytest.fixture
def env():
    cfg = assemble(experiment='Full', device='cpu')
    cfg['reward']['scheme'] = 'cooperative_cost_v2'
    cfg['reward']['terminal'] = dict(success=0., **{reason:-10. for reason in FAILURE_REASONS})
    value = PursuitEvasionEnv(cfg)
    yield value
    value.close()


def paid(env, reason=''):
    return evaluate(snapshot(env), env, reason)


@pytest.mark.parametrize('gap', [-5., 0., .16611671447753906, 5.])
@pytest.mark.parametrize('effect', [-.1, 0., .1])
def test_capture_cannot_worsen_reward_at_the_same_physical_state(env, gap, effect):
    env.counterfactual_gap = gap
    env.counterfactual_audit_new = True
    env.counterfactual_audit = dict(valid=True,
        shapley_effect_m=np.full(env.num_surface_blue, effect).tolist())
    before, original = paid(env)
    env.capture_success = True
    env.counterfactual_gap = 0.
    env.counterfactual_audit_new = False
    env.counterfactual_audit = {}
    after, following = paid(env)
    assert np.all(after[:env.num_surface_blue] >= before[:env.num_surface_blue])
    assert after.mean() >= before.mean()
    np.testing.assert_array_equal(after[env.num_surface_blue:], before[env.num_surface_blue:])
    for name in REWARD_COMPONENTS:
        if name in ('usv_blocking_cost', 'usv_shapley_cost'):
            assert not np.any(following[name])
        else:
            np.testing.assert_array_equal(following[name], original[name], err_msg=name)


def test_pre_capture_intrusion_credit_retains_original_formula(env):
    assert not env.capture_success
    env.counterfactual_gap = .8
    env.counterfactual_audit_new = True
    effects = np.linspace(-.06, .08, env.num_surface_blue)
    env.counterfactual_audit = dict(valid=True, shapley_effect_m=effects.tolist())
    _, terms = paid(env)
    settings = env.config['reward']
    budget = settings['cost']['step_budget']
    blocking = -budget*settings['cost']['usv_weights']['usv_blocking_cost']*(
        1.-max(0., math.tanh(.8/settings['usv']['gap_scale_m'])))
    shapley = -budget*settings['cost']['usv_weights']['usv_shapley_cost']*(
        .5-.5*np.clip(effects/settings['usv']['shapley_scale_m'], -1., 1.))
    np.testing.assert_allclose(terms['usv_blocking_cost'][:env.num_surface_blue], blocking, atol=1.e-15)
    np.testing.assert_allclose(terms['usv_shapley_cost'][:env.num_surface_blue], shapley, atol=1.e-15)


def test_withdrawal_still_guides_approach_support_and_smooth_safe_actions(env, monkeypatch):
    env.capture_success = True
    # Isolate each paid signal; independent physics tests cover safety geometry.
    monkeypatch.setattr(env, '_safety_cost', lambda i: 0.)
    env.blue_states[0].pos = env.red_state.pos+np.array([12., 0.], np.float32)
    far, _ = paid(env)
    env.blue_states[0].pos = env.red_state.pos+np.array([3., 0.], np.float32)
    near, near_terms = paid(env)
    assert near[0] > far[0]
    env.last_executed_actions[0] = env.agent_action_limits[0]
    driven, _ = paid(env)
    assert driven[0] < near[0]
    env.last_executed_actions[:] = 0.
    env.blue_states[0].yaw_rate = env.max_yaw_rate
    turning, _ = paid(env)
    assert turning[0] < near[0]
    env.blue_states[0].yaw_rate = 0.
    baseline, _ = paid(env)
    monkeypatch.setattr(env, '_safety_cost', lambda i: 1.8 if i == 0 else 0.)
    unsafe, _ = paid(env)
    assert unsafe[0] < baseline[0]
    monkeypatch.setattr(env, '_safety_cost', lambda i: 0.)
    for index, state in enumerate(env.submarine_states):
        angle = 2.*math.pi*index/env.num_submarines
        state.pos = env.red_state.pos+3.*np.array([math.cos(angle), math.sin(angle)], np.float32)
    supported, terms = paid(env)
    assert supported[0] > baseline[0]
    assert terms['usv_support_cost'][0] > near_terms['usv_support_cost'][0]
    for rewards in (far, near, driven, turning, unsafe, supported):
        assert np.all(rewards <= 0.) and np.all(rewards >= -.1-1.e-7)


@pytest.mark.parametrize('reason', ['', 'success', *FAILURE_REASONS])
def test_post_capture_bounds_terminal_contract_and_reward_ledger(env, reason):
    env.capture_success = True
    env.last_executed_actions[:] = env.agent_action_limits
    rewards, terms = paid(env, reason)
    dense = sum(np.asarray(terms[name]) for name in REWARD_COMPONENTS)
    assert np.all(dense <= 0.) and np.all(dense >= -.1-1.e-12)
    np.testing.assert_allclose(rewards, dense+terms['terminal_reward'], atol=1.e-6)
    if reason == 'success':
        np.testing.assert_array_equal(rewards, np.zeros(env.num_friendly))
    elif reason:
        np.testing.assert_array_equal(terms['terminal_reward'], np.full(env.num_friendly, -10.))
    else:
        assert np.all(rewards <= 0.)


def test_same_state_capture_relief_is_not_a_repeatable_bonus_or_refund(env):
    env.capture_success = True
    first, original = paid(env)
    repeated, following = paid(env)
    np.testing.assert_array_equal(first, repeated)
    for name in REWARD_COMPONENTS:
        np.testing.assert_array_equal(original[name], following[name])
    assert np.all(first < 0.)
    np.testing.assert_array_equal(first+paid(env, 'success')[0], first)
    assert not np.any(original['terminal_reward'])


def test_earlier_capture_improves_return_only_under_matched_trajectory_assumption(env):
    # This isolates the capture-dependent reward on IDENTICAL physical states
    # and a shared endpoint. It makes no claim about different red trajectories.
    env.counterfactual_gap = 0.
    env.counterfactual_audit_new = False
    env.counterfactual_audit = {}
    pre, _ = paid(env)
    env.capture_success = True
    post, _ = paid(env)
    gamma = env.config['training']['gamma']
    end, early, late = 20, 4, 12
    def outcome(capture):
        return sum(gamma**step*(pre if step<capture else post) for step in range(end-1))
    early_return, late_return = outcome(early), outcome(late)
    assert np.all(early_return[:env.num_surface_blue] > late_return[:env.num_surface_blue])
    np.testing.assert_array_equal(early_return[env.num_surface_blue:], late_return[env.num_surface_blue:])
    assert early_return.mean() > late_return.mean()
