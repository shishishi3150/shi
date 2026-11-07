"""Compute reductions must retain safety decisions and actor gradients."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from matd3.agent import Agent
from scripts.train import make_env
from usv_env.config import assemble
from usv_env.safety_kernel import assess_motion
from usv_env.safety_projection import _Predictor, project_from_state
from usv_env.target_tracking import TargetEstimate, fuse_estimates, predict_estimate


def motion_args(cfg, q, action):
    predictor = _Predictor(cfg)
    return (q, action, *predictor.model_data[0], cfg['env']['dt'],
        predictor.models[0].internal_substeps, cfg['ship']['surface_length_m'],
        cfg['ship']['surface_collision_width_m'], predictor.world, predictor.reefs,
        np.empty((0, 8)), cfg['i1']['horizon_steps'], cfg['i1']['backup_steps'],
        cfg['i1']['stop_speed_mps'], cfg['i1']['stop_yaw_rate_rps'], cfg['i1']['margin_m'])


def test_losing_unsafe_trajectory_can_stop_before_its_full_tail():
    cfg = assemble(experiment='Full', device='cpu')
    args = motion_args(cfg, np.array([.4, 10., .4, 0., 0., math.pi]), np.array([2.2, 0.]))
    full = assess_motion(*args)
    early = assess_motion(*args, clearance_cutoff=math.inf)
    assert full[0] <= early[0] < cfg['i1']['margin_m']
    assert early[2] == 0. and not early[1]
    assert early[0] > full[0]  # Work really stopped; this is not just a new flag.


def test_a_safe_recoverable_trajectory_is_never_pruned_even_with_high_incumbent():
    cfg = assemble(experiment='Full', device='cpu')
    args = motion_args(cfg, np.array([10., 10., .1, 0., 0., 0.]), np.array([0., 0.]))
    expected = assess_motion(*args)
    assert expected[0] >= cfg['i1']['margin_m'] and expected[1]
    assert assess_motion(*args, clearance_cutoff=math.inf) == expected


def test_decision_only_screening_matches_full_diagnostics_without_false_safety():
    cfg = assemble(experiment='Full', device='cpu')
    env = make_env(cfg, 76)
    obs, state = env.reset(seed=76)
    states, observations = np.repeat(state[None], 32, 0), np.repeat(obs[None], 32, 0)
    states[:16, 2] = .4 / max(cfg['env']['world_size'])
    actions = np.random.default_rng(9).uniform(-1, 1, (32, 5, 2)).astype(np.float32)
    actions *= env.agent_action_limits
    expected, full = project_from_state(cfg, states, observations, actions, reject_only=True)
    actual, screened = project_from_state(cfg, states, observations, actions,
        reject_only=True, decision_only=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(screened['safe_mask'], full['safe_mask'])
    assert not screened['safe_mask'].all()
    assert set(screened) == {'safe_mask', 'solver_time_s'}
    with pytest.raises(ValueError, match='reject_only'):
        project_from_state(cfg, states, observations, actions, decision_only=True)
    env.close()


def test_actor_reuses_only_an_unchanged_nominal_reference(monkeypatch):
    torch.set_num_threads(1)
    cfg = assemble(experiment='I1_only', device='cpu')
    cfg['training']['hidden_sizes'] = [16, 16]
    cfg['replay_buffer']['capacity'] = 8
    env = make_env(cfg, 77)
    obs, state = env.reset(seed=77)
    agent = Agent(env, cfg)
    obs, state = agent.tensor(np.stack([obs, obs])), agent.tensor(np.stack([state, state]))
    mask = torch.ones((2, 5))
    with torch.no_grad():
        nominal = agent.joint_actions(obs)
        projected, _ = agent._project_tensor(state, obs, nominal)
    calls = []
    original = agent._project_tensor
    def record(*args, **kwargs):
        calls.append(kwargs.get('receivers'))
        return original(*args, **kwargs)
    monkeypatch.setattr(agent, '_project_tensor', record)
    stale = deepcopy(agent)
    agent._actor_step(0, obs, state, mask, nominal, projected)
    assert calls == []
    # A stale/external nominal reference must still trigger fresh projections.
    calls.clear()
    stale._actor_step(0, obs, state, mask, nominal + .1, projected)
    assert calls == [(0,), (1,)]
    env.close()


def test_single_source_i2_keeps_ci_result_without_matrix_inversions(monkeypatch):
    source = TargetEstimate(True, np.array([3., 8., .2, -.1]), np.diag([.2, .4, .1, .3]),
        1., 1., .8, 2, '2:one')
    now = 1.7
    predicted = predict_estimate(source, now)
    quality = .8 * np.exp(-(now-source.t_obs)/12.)
    covariance = predicted.covariance / quality + np.eye(4)*1e-9
    def forbidden(*args, **kwargs):
        pytest.fail('single-source CI needs no matrix inversion')
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    actual = fuse_estimates([source, source], now)
    np.testing.assert_allclose(actual.state, predicted.state, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=0, atol=1e-12)
    assert actual.contribution_source_ids == (2,)
    assert actual.contribution_measurement_ids == ('2:one',)
    assert actual.contribution_weights == (1.,)
    assert actual.t_obs == 1. and actual.state_time == now
    assert actual.confidence == pytest.approx(quality)
