"""Reward counterexamples; synthetic fixtures are not physical rollout evidence."""
import itertools
import math
from copy import deepcopy

import numpy as np
import pytest

from usv_env.config import assemble, refresh_hashes, validate
from usv_env.environment import PursuitEvasionEnv
from usv_env.reward import (distance_quality, gap_quality, hull_quality,
                            potentials, snapshot, evaluate, REWARD_COMPONENTS)
from usv_env.target_tracking import TargetEstimate


def fixture():
    # PBRS is a supported historical scheme, not the production default.
    cfg = assemble(device='cpu')
    cfg['reward']['scheme'] = 'cooperative_pbrs_v1'
    env = PursuitEvasionEnv(cfg)
    env.red_state.pos = np.array([32., 15.])
    for j, ship in enumerate(env.submarine_states):
        angle = j * 2 * math.pi / 3
        ship.pos = env.red_state.pos + 3.5 * np.array([math.cos(angle), math.sin(angle)])
        i = j + 2
        env.estimators[i].estimate = TargetEstimate(True, np.r_[env.red_state.pos, 0., 0.],
            np.eye(4)*.1, 0., 0., 1., i, f'fixture:{i}')
    return env


@pytest.mark.parametrize('distance', [2., 3., 5.])
def test_distance_band_has_unit_quality(distance):
    assert distance_quality(np.array([distance]), 2., 5., 1.5)[0] == 1.


@pytest.mark.parametrize('distance', [0., 1., 5.1, 8.])
def test_too_close_or_far_loses_quality(distance):
    assert 0 < distance_quality(np.array([distance]), 2., 5., 1.5)[0] < 1.


def test_gap_improves_until_actual_160_degree_gate():
    values = [gap_quality(math.radians(g), math.radians(160)) for g in [300, 220, 180, 160]]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert values[-1] == 1. and values[-2] < 1.


def test_negative_hull_margin_has_continuous_guidance():
    values = [hull_quality(h, .25, 2., 2.) for h in [-2., -1., 0., .25]]
    assert values[0] == 0. and values[-1] == 1.
    assert all(a < b for a, b in zip(values, values[1:]))


def test_team_potential_is_invariant_to_rotation_translation_and_order():
    env = fixture(); before = potentials(env)
    positions = np.array([s.pos for s in env.submarine_states])
    rotation = np.array([[math.cos(.7), -math.sin(.7)], [math.sin(.7), math.cos(.7)]])
    offset = np.array([5., 6.]); red = env.red_state.pos.copy()
    for order in itertools.permutations(range(3)):
        env.red_state.pos = red @ rotation.T + offset
        for j, source in enumerate(order):
            env.submarine_states[j].pos = positions[source] @ rotation.T + offset
            e = env.estimators[2+j].estimate
            env.estimators[2+j].estimate = TargetEstimate(True, np.r_[env.red_state.pos, 0., 0.],
                e.covariance, 0., 0., 1., 2+j, 'fixture')
        after = potentials(env)
        assert after['auv_potential'] == pytest.approx(before['auv_potential'])


def test_laggard_and_incomplete_team_cannot_hide_in_average():
    env = fixture(); before = potentials(env)
    env.submarine_states[0].pos += [20., 0.]
    assert potentials(env)['auv_distance_quality'] < .5 * before['auv_distance_quality']
    env._agent_active_mask[2] = 0
    p = potentials(env)
    assert p['auv_potential'] == p['usv_readiness'] == 0.


def test_stationary_quality_and_boundary_oscillation_cannot_farm_pbrs():
    env = fixture(); start = snapshot(env)
    _, terms = evaluate(start, env)
    assert all(x <= 0 for x in terms['auv_potential_reward'])
    gamma = env.config['training']['gamma']; phi = start['auv_potential']
    values = [phi, phi+.05, phi, phi+.05, phi]
    total = sum(gamma**t * (gamma*b-a) for t, (a,b) in enumerate(zip(values, values[1:])))
    stable = sum(gamma**t * (gamma*phi-phi) for t in range(4))
    assert total == pytest.approx(stable)


def test_hold_uses_task_counter_and_resets_on_gate_loss():
    env = fixture(); previous = snapshot(env)
    env._task_success(safe=True)
    _, terms = evaluate(previous, env)
    assert potentials(env)['auv_hold_quality'] == 1/env.required_hold_steps
    assert min(terms['auv_potential_reward'][2:]) > 0
    env.submarine_states[0].pos += [15., 0.]
    env._task_success(safe=True)
    assert env.task_state.hold_steps == env.hold_steps == 0


@pytest.mark.parametrize('reason', ['success', 'breakthrough', 'timeout', 'collision', 'scenario_anomaly', 'numerical_error'])
def test_terminal_next_potentials_zero_and_terminal_reward_separate(reason):
    env = fixture(); env.counterfactual_gap = 1.
    prev = snapshot(env); reward, terms = evaluate(prev, env, reason)
    assert terms['next_auv_potential'] == terms['next_usv_potential'] == 0.
    np.testing.assert_allclose(terms['auv_potential_reward'][2:], -10*prev['auv_potential'])
    np.testing.assert_allclose(terms['auv_individual_progress'][2:], -.625*prev['auv_individual_quality'])
    np.testing.assert_allclose(terms['usv_potential_reward'][:2], -3.75*prev['usv_potential'])
    expected = sum(np.array(terms[k]) for k in (*REWARD_COMPONENTS, 'terminal_reward'))
    np.testing.assert_allclose(reward, expected, rtol=1e-6)
    assert terms['terminal_reward'] == [env.config['reward']['terminal'][reason]]*5


def test_discounted_pbrs_telescopes_with_terminal_and_external_cut():
    env = fixture(); gamma = env.config['training']['gamma']
    initial = snapshot(env); prev = initial; total = np.zeros(5)
    for t in range(5):
        env.submarine_states[0].pos += [.05, .01]
        _, terms = evaluate(prev, env, 'timeout' if t == 4 else '')
        total += gamma**t * (np.array(terms['auv_potential_reward']) +
            np.array(terms['auv_individual_progress']) + np.array(terms['usv_potential_reward']))
        prev = snapshot(env)
    expected = np.r_[np.full(2, -3.75*initial['usv_potential']),
        -10*initial['auv_potential']-.625*initial['auv_individual_quality']]
    np.testing.assert_allclose(total, expected, atol=1e-10)
    # A collector cut is outside evaluate; the actual endpoint is retained.
    from scripts.train import make_env
    wrapped = make_env(env.config, 42, episode_limit=1)
    before = snapshot(wrapped.env)
    _, _, _, done, info = wrapped.step(np.zeros((5,2), np.float32))
    assert done and info.truncated and not info.terminated
    assert info.reward_terms['next_auv_potential'] == potentials(wrapped.env)['auv_potential']
    assert info.reward_terms['next_auv_potential'] > 0
    wrapped.reset(seed=43)
    assert wrapped.step_count == 0 and wrapped.env.team_first_detection_step is None


def test_discovery_once_no_search_tracking_information_occupancy_rewards():
    env = fixture(); prev = snapshot(env); env.team_first_detection_step = 1
    env._new_cells[:] = 100
    _, terms = evaluate(prev, env)
    assert sum(terms['discovery']) == pytest.approx(.75)
    for key in ('search', 'tracking', 'information', 'observation_geometry', 'encirclement', 'blocking', 'usv_contribution'):
        assert terms[key] == [0.]*5
    _, again = evaluate(snapshot(env), env)
    assert sum(again['discovery']) == 0.


def test_signed_usv_gap_and_zero_gap_readiness_credit():
    env = fixture(); prev = snapshot(env)
    env.blue_states[0].pos = env.red_state.pos + [2., 0.]
    env.hold_steps = 10
    _, zero = evaluate(prev, env)
    assert zero['usv_potential_reward'] == [0.]*5
    env.counterfactual_gap = 1.
    _, positive = evaluate(prev, env)
    assert min(positive['usv_potential_reward'][:2]) > 0
    env.counterfactual_gap = -1.
    _, negative = evaluate(prev, env)
    assert max(negative['usv_potential_reward'][:2]) < 0
    env._agent_active_mask[:2] = 0
    assert potentials(env)['usv_potential'] == 0.


@pytest.mark.parametrize('valid,new', [(False, True), (True, False), (False, False)])
def test_invalid_or_reused_audit_has_no_credit(valid, new):
    env = fixture(); env.counterfactual_audit = {'valid':valid, 'shapley_effect_m':[.3, 0.]}
    env.counterfactual_audit_new = new
    assert evaluate(snapshot(env), env)[1]['usv_shapley_credit'] == [0.]*5


def test_individual_shapley_signed_bounded_and_distinct():
    env = fixture(); env.counterfactual_audit = {'valid':True, 'shapley_effect_m':[1., 0.]}
    env.counterfactual_audit_new = True
    credit = evaluate(snapshot(env), env)[1]['usv_shapley_credit']
    assert 0 < credit[0] <= .15 and credit[1] == 0
    env.counterfactual_audit['shapley_effect_m'] = [-100., 100.]
    credit = evaluate(snapshot(env), env)[1]['usv_shapley_credit']
    assert -.15 <= credit[0] < 0 < credit[1] <= .15


def test_no_usv_no_natural_reef_delay_reward_and_reset_zero():
    env = PursuitEvasionEnv(assemble(device='cpu'))
    assert abs(potentials(env)['usv_potential']) < 1e-12
    env.set_active_agents([0,0,1,1,1]); env.reset(seed=5)
    for _ in range(25):
        _, _, reward, _, info = env.step(np.zeros((5,2), np.float32))
        assert abs(env.counterfactual_gap) < 1e-8
        assert info.reward_terms['usv_potential_reward'] == [0.]*5
        assert np.isfinite(reward).all()


@pytest.mark.parametrize('mask', [[0,0,0,0,0], [1,0,1,1,1], [1,1,0,1,1]])
def test_inactive_never_receives_cost_discovery_or_terminal(mask):
    env = fixture(); env._agent_active_mask = np.array(mask, np.float32)
    prev = snapshot(env); env.team_first_detection_step = 1
    reward, terms = evaluate(prev, env, 'success')
    assert np.isfinite(reward).all()
    assert np.all(reward[np.array(mask)==0] == 0)
    assert sum(terms['discovery']) <= .75+1e-12


def test_tracking_freshness_jumps_are_bounded_without_sensor_changes():
    env = fixture(); fresh = snapshot(env)
    env.step_count = 7  # 1.05 seconds: original estimator becomes stale.
    _, terms = evaluate(fresh, env)
    assert potentials(env)['auv_tracking_quality'] == 0
    assert abs(terms['auv_potential_reward'][2]) <= 2.025


def test_potential_gamma_hash_and_executed_control_cost():
    env = fixture(); prev = snapshot(env)
    cfg = env.config; old = cfg['hashes']['reward']
    cfg['training']['gamma'] = .995; refresh_hashes(cfg)
    assert cfg['hashes']['reward'] != old
    env.last_nominal_actions[:] = env.agent_action_limits
    env.last_executed_actions[:] = 0
    _, terms = evaluate(prev, env)
    assert terms['action_penalty'] == [0.]*5
    assert terms['auv_potential_reward'][2] == pytest.approx(10*(.995-1)*prev['auv_potential'])
    assert np.array_equal(env._observations()[0], env._observations()[0])


def test_historical_scale_success_beats_maximal_credit_timeout():
    env = fixture(); gamma = env.config['training']['gamma']
    # Historical pre-scaling budget only, not a guarantee for current defaults.
    # These are bounds, not physical trajectories; current limits are audited separately.
    credit_bound = .06*sum(gamma**t for t in range(0,900,10))
    assert credit_bound + gamma**899*(-15) < gamma**320*30


def test_invalid_reward_scales_rejected():
    cfg = assemble(device='cpu'); cfg['reward']['potential']['distance_scale_m'] = 0
    with pytest.raises(ValueError, match='reward'):
        validate(cfg)


def test_real_reward_nstep_discount_execution_and_reset_boundaries():
    from scripts.train import make_env
    from matd3.agent import Agent
    cfg = assemble('configs/pilot.yaml', device='cpu')
    cfg['training']['hidden_sizes'] = [16,16]
    cfg['replay_buffer']['capacity'] = 16; refresh_hashes(cfg)
    env = make_env(cfg, 55, episode_limit=3); agent = Agent(env,cfg)
    assert agent.replay.gamma == env.config['training']['gamma']
    obs,state = env.reset(seed=55); actual_rewards=[]; last_endpoint=None
    for t in range(3):
        before=env.learning_context(); action=2*env.agent_action_limits
        following,end,reward,done,info=env.step(action)
        actual_rewards.append(reward)
        agent.add_transition(obs,state,env.last_executed_actions,reward,following,end,info,
            current_context=before,next_context=env.learning_context(),nominal_actions=action)
        obs,state=following,end; last_endpoint=end.copy()
    assert done and info.truncated and len(agent.replay)==3
    expected=sum(cfg['training']['gamma']**t*r for t,r in enumerate(actual_rewards))
    np.testing.assert_allclose(agent.replay.agent_rewards[0],expected,atol=1e-7)
    np.testing.assert_array_equal(agent.replay.actions[0],env.agent_action_limits)
    np.testing.assert_array_equal(agent.replay.nominal_actions[0],2*env.agent_action_limits)
    np.testing.assert_array_equal(agent.replay.next_states[:3],np.tile(last_endpoint,(3,1)))
    assert not agent.replay._pending
    sample=agent.replay._sample_rows(np.array([0]),np.ones(1),np.ones(1))
    assert sample['bootstrap_discounts'][0,0] == pytest.approx(cfg['training']['gamma']**3)
    obs,state=env.reset(seed=56); reset_state=state.copy()
    following,end,reward,_,info=env.step(np.zeros((5,2),np.float32))
    agent.add_transition(obs,state,env.last_executed_actions,reward,following,end,info)
    agent.replay.external_reset()
    assert len(agent.replay)==4 and agent.replay.n_steps[3,0]==1
    np.testing.assert_array_equal(agent.replay.states[3],reset_state)
    assert not np.array_equal(agent.replay.next_states[0],reset_state)
    env.close()


def test_potential_weight_scaling_and_identical_base_full_definitions():
    env=fixture(); before=snapshot(env)
    env.counterfactual_gap=1.
    _,original=evaluate(before,env)
    env.config['reward']['auv']['potential_weight']*=2
    env.config['reward']['usv']['potential_weight']*=2
    _,scaled=evaluate(before,env)
    for key in ('auv_potential_reward','usv_potential_reward'):
        np.testing.assert_allclose(scaled[key],2*np.array(original[key]))
    # Mechanism toggles do not secretly change reward definitions.
    env.config['mechanisms']=assemble(experiment='Full',device='cpu')['mechanisms']
    _,full=evaluate(before,env)
    for key in REWARD_COMPONENTS:
        np.testing.assert_array_equal(full[key],scaled[key])
