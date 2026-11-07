"""Capture is a latched milestone; withdrawal remains a learning trajectory."""
import numpy as np
import pytest
import torch

from matd3.agent import Agent
from scripts.train import make_env
from usv_env.config import assemble, refresh_hashes, validate
from usv_env.environment import PursuitEvasionEnv


def controlled_env(monkeypatch, *, capture_step=2, end_step=650, start_step=0):
    cfg = assemble(experiment='Base', device='cpu')
    cfg['counterfactual']['enabled'] = False
    cfg['task']['post_capture_end_step'] = end_step
    env = PursuitEvasionEnv(cfg)
    env.step_count = start_step
    # Isolate state-machine boundaries; physical capture gates have their own
    # real-dynamics coverage in test_encirclement and the rule audit.
    monkeypatch.setattr(env, '_task_success', lambda safe: safe and env.step_count == capture_step)
    return env


def step(env):
    return env.step(np.full((5, 2), .05, np.float32))


def test_capture_continues_reward_motion_and_bootstrap_without_repeated_confirmation(monkeypatch):
    env = controlled_env(monkeypatch, capture_step=1)
    _, _, reward, done, info = step(env)
    assert not done and not info.terminated and not info.success
    assert info.capture_success and info.capture_just_confirmed
    assert info.first_capture_step == 1 and info.first_capture_time_s == .15
    assert info.phase == 'WITHDRAW' and info.reward_terms['bootstrap_mask'] == 1.
    assert np.all(reward >= 2.) and np.all(reward <= 2.02)
    assert env.red_navigator.mode == 'WITHDRAW'
    np.testing.assert_array_equal(env.red_navigator.withdrawal_goal, [6., 30.])
    shadow = env.shadow_red_state.pos.copy()
    red = env.red_state.pos.copy()
    env.shadow_navigator.recovery_count = 99  # Retired intrusion diagnostic.
    for _ in range(3):
        _, _, reward, done, info = step(env)
        assert not done and info.capture_success and not info.capture_just_confirmed
        assert info.first_capture_step == 1 and np.all(reward >= 0.) and np.all(reward <= .02)
        assert env.counterfactual_gap == 0. and not env.counterfactual_evidence_valid()
        assert not env.counterfactual_audit_new
    assert info.post_capture_steps == 3
    assert sum(e['event'] == 'capture_confirmed' for e in env.events) == 1
    np.testing.assert_array_equal(env.shadow_red_state.pos, shadow)
    assert not np.array_equal(env.red_state.pos, red)
    assert env.learning_context()['phase_id'] == 3
    assert not env.render_frame().show_shadow_red


def test_captured_episode_ends_at_total_step_650_not_650_extra(monkeypatch):
    env = controlled_env(monkeypatch, capture_step=648, start_step=647)
    assert not step(env)[3]
    assert not step(env)[3]
    _, _, reward, done, info = step(env)
    assert done and info.success and info.terminated and not info.truncated
    assert info.step == 650 and info.first_capture_step == 648 and info.post_capture_steps == 2
    assert info.terminal_reason == 'success' and not info.timeout
    assert info.reward_terms['bootstrap_mask'] == 0.
    assert np.all(reward >= 2.) and np.all(reward <= 2.02)
    with pytest.raises(RuntimeError, match='reset required'):
        step(env)


def test_no_capture_keeps_original_900_step_timeout(monkeypatch):
    env = controlled_env(monkeypatch, capture_step=None, start_step=649)
    assert not step(env)[3] and not env.capture_success
    env.step_count = 899
    _, _, reward, done, info = step(env)
    assert done and info.timeout and not info.success and not info.capture_success
    assert info.terminal_reason == 'timeout' and info.first_capture_step is None
    np.testing.assert_array_equal(reward, np.zeros(5))


@pytest.mark.parametrize('capture_step', [650, 701, 900])
def test_late_capture_ends_on_actual_capture_step(monkeypatch, capture_step):
    env = controlled_env(monkeypatch, capture_step=capture_step, start_step=capture_step-1)
    *_, done, info = step(env)
    assert done and info.success and info.capture_success
    assert info.step == capture_step and info.post_capture_steps == 0


def test_post_capture_safety_failure_has_priority_and_retains_milestone(monkeypatch):
    env = controlled_env(monkeypatch, capture_step=648, start_step=647)
    step(env)
    env.step_count = 649
    monkeypatch.setattr(env, '_contacts', lambda old, new: ('collision', '0-1'))
    _, _, reward, done, info = step(env)
    assert done and info.terminal_reason == 'collision' and not info.success
    assert info.capture_success and info.first_capture_step == 648
    assert info.collision and not np.any(reward)


def test_clearing_entry_is_recorded_but_does_not_end_early(monkeypatch):
    env = controlled_env(monkeypatch, capture_step=1)
    step(env)
    env.red_state.pos[:] = [9., 30.]
    env.red_state.heading = np.pi
    env.red_state.speed = .2
    *_, done, info = step(env)
    assert not done and info.withdrawal_cleared_entry
    assert sum(e['event'] == 'withdrawal_cleared_entry' for e in env.events) == 1


def test_reset_clears_capture_withdrawal_and_phase(monkeypatch):
    env = controlled_env(monkeypatch, capture_step=1)
    step(env)
    env.reset(seed=9)
    assert not env.capture_success and env.first_capture_step is None
    assert not env.withdrawal_cleared_entry and not env._episode_done
    assert env.phase == 'SEARCH' and env.step_count == 0
    assert env.red_navigator.mode == 'ADVANCE' and env.red_navigator.withdrawal_goal is None


def test_external_collector_cut_after_capture_still_bootstraps(monkeypatch):
    cfg = assemble(device='cpu')
    env = make_env(cfg, 3, episode_limit=3)
    monkeypatch.setattr(env.env, '_task_success', lambda safe: safe and env.step_count == 1)
    env.reset(seed=3)
    for _ in range(3):
        obs, state, _, done, info = step(env)
    assert done and info.truncated and not info.terminated and not info.success
    assert info.capture_success and info.first_capture_step == 1
    assert info.reward_terms['bootstrap_mask'] == 1.
    np.testing.assert_array_equal(info.terminal_observation, obs)
    np.testing.assert_array_equal(info.terminal_state, state)


def test_historical_missing_setting_preserves_immediate_capture(monkeypatch):
    cfg = assemble(device='cpu')
    del cfg['task']['post_capture_end_step']
    env = PursuitEvasionEnv(cfg)
    monkeypatch.setattr(env, '_task_success', lambda safe: safe)
    *_, done, info = step(env)
    assert done and info.success and env.red_navigator.mode != 'WITHDRAW'


@pytest.mark.parametrize('invalid', [-1, 901, 2.5, True])
def test_invalid_post_capture_end_is_rejected(invalid):
    cfg = assemble(device='cpu')
    cfg['task']['post_capture_end_step'] = invalid
    with pytest.raises(ValueError, match='post_capture_end_step'):
        validate(cfg)


def test_real_learning_and_nstep_replay_continue_across_capture(monkeypatch):
    torch.set_num_threads(1)
    cfg = assemble(experiment='Base', device='cpu')
    cfg['counterfactual']['enabled'] = False
    cfg['task']['post_capture_end_step'] = 6
    cfg['training'].update(batch_size=2, n_step_return=3, hidden_sizes=[16, 16])
    cfg['replay_buffer']['capacity'] = 16
    refresh_hashes(cfg)
    env = make_env(cfg, 3)
    monkeypatch.setattr(env.env, '_task_success', lambda safe: safe and env.step_count == 2)
    agent = Agent(env, cfg)
    obs, state = env.reset(seed=3)
    rewards = []
    for t in range(6):
        current = env.learning_context()
        action = agent.act(obs)
        next_obs, next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        agent.add_transition(obs, state, env.last_executed_actions.copy(), reward,
            next_obs, next_state, info, current_context=current,
            next_context=env.learning_context(), nominal_actions=env.last_nominal_actions.copy())
        obs, state = next_obs, next_state
        assert done == (t == 5)
    assert len(agent.replay) == 6 and agent.global_env_steps == 6
    sample = agent.replay._unpack_phase_activity(agent.replay.phase_ids[:6])[0]
    assert np.all(sample[2:] == 3)  # Actual post-capture actions are resident.
    np.testing.assert_allclose(agent.replay.agent_rewards[0],
        rewards[0]+.99*rewards[1]+.99**2*rewards[2], atol=1e-6)
    assert agent.replay.terminated[0, 0] == 0
    assert agent.replay.terminated[5, 0] == 1
    before = [p.detach().clone() for p in agent.surface_actor.parameters()]
    agent.update()
    metrics = agent.update()
    assert np.isfinite(metrics['usv_actor_loss']) and np.isfinite(metrics['auv_actor_loss'])
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.surface_actor.parameters()))
