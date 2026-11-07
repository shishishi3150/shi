"""Skip unused cost-mode snapshots while retaining historical PBRS inputs."""
from copy import deepcopy
import pickle

import numpy as np

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from usv_env.red_policy import RuleController
from usv_env import reward as reward_module
from usv_env.target_tracking import TargetEstimate


def test_cost_step_does_not_build_unused_previous_snapshot(monkeypatch):
    env = PursuitEvasionEnv(assemble(experiment='Base', device='cpu'))

    def forbidden(*args, **kwargs):
        raise AssertionError('cost reward does not consume the previous snapshot')

    monkeypatch.setattr(reward_module, 'snapshot', forbidden)
    try:
        _, _, reward, done, info = env.step(np.zeros((5, 2), np.float32))
        assert not done and not info.terminal_reason
        assert np.isfinite(reward).all()
    finally:
        env.close()


def test_pbrs_step_keeps_previous_snapshot_and_previous_dependent_reward(monkeypatch):
    cfg = assemble(experiment='Base', device='cpu')
    cfg['reward']['scheme'] = 'cooperative_pbrs_v1'
    env = PursuitEvasionEnv(cfg)
    original = reward_module.snapshot
    snapshots = []

    def record_snapshot(current):
        assert current.step_count == 0
        previous = original(current)
        # This synthetic perturbation verifies that step still passes the
        # historical endpoint into PBRS rather than discarding it.
        previous['auv_potential'] += .125
        snapshots.append(previous)
        return previous

    monkeypatch.setattr(reward_module, 'snapshot', record_snapshot)
    try:
        _, _, actual, _, info = env.step(np.zeros((5, 2), np.float32))
        assert len(snapshots) == 1
        previous = snapshots[0]
        assert previous['step'] == 0 and env.step_count == 1
        expected, terms = reward_module.evaluate(previous, env, info.terminal_reason,
            geometry=env.capture_geometry)
        np.testing.assert_array_equal(actual, expected)
        assert info.reward_terms['auv_potential_reward'] == terms['auv_potential_reward']
        unperturbed = deepcopy(previous)
        unperturbed['auv_potential'] -= .125
        baseline, _ = reward_module.evaluate(unperturbed, env, info.terminal_reason,
            geometry=env.capture_geometry)
        assert np.all(actual[2:] < baseline[2:])
    finally:
        env.close()


def test_snapshot_and_estimator_predict_leave_all_environment_state_unchanged():
    env = PursuitEvasionEnv(assemble(experiment='Full', device='cpu'), seed=5)
    env.step_count = 4
    for i, estimator in enumerate(env.estimators):
        estimator.estimate = TargetEstimate(True, np.array([20., 25., .2, -.1]),
            np.eye(4)*.1, 0., 0., .9, i, f'purity:{i}')
    before = pickle.dumps(env, protocol=5)
    try:
        for estimator in env.estimators:
            current = estimator.estimate
            predicted = estimator.predict(env.step_count*env.dt)
            assert predicted is not current
            assert estimator.estimate is current and current.state_time == 0.
        first = reward_module.snapshot(env)
        second = reward_module.snapshot(env)
        assert first['step'] == second['step'] == 4
        assert pickle.dumps(env, protocol=5) == before
    finally:
        env.close()


class PreviousSnapshotReference(PursuitEvasionEnv):
    """Retain the previous endpoint evaluation even when cost mode skips it."""
    def step(self, actions):
        reward_module.snapshot(self)
        return super().step(actions)


def rule_action(controller, env):
    return controller.actions(env.blue_states, env.submarine_states, None,
        env.protected_center, env.reefs, env.world_size, env.blue_dynamics,
        env.submarine_dynamics, env.step_count*env.dt,
        target_estimates=env.target_estimates, active_mask=env.agent_active_mask())


def test_full_seed5_650_steps_exactly_match_retained_snapshot_reference():
    cfg = assemble('configs/fast.yaml', experiment='Full', seed=5, device='cpu')
    actual = HistoryEnv(PursuitEvasionEnv(cfg, seed=5), cfg['revision']['history_length'])
    reference = HistoryEnv(PreviousSnapshotReference(cfg, seed=5), cfg['revision']['history_length'])
    controllers = [RuleController(), RuleController()]
    try:
        actual_reset, actual_state = actual.reset(seed=5)
        reference_reset, reference_state = reference.reset(seed=5)
        np.testing.assert_array_equal(actual_reset, reference_reset)
        np.testing.assert_array_equal(actual_state, reference_state)
        for step in range(1, 651):
            actions = [rule_action(controller, env)
                for controller, env in zip(controllers, (actual, reference))]
            np.testing.assert_array_equal(*actions)
            obs, state, reward, done, info = actual.step(actions[0])
            old_obs, old_state, old_reward, old_done, old_info = reference.step(actions[1])
            np.testing.assert_array_equal(obs, old_obs)
            np.testing.assert_array_equal(state, old_state)
            np.testing.assert_array_equal(reward, old_reward)
            np.testing.assert_array_equal(actual.last_executed_actions, reference.last_executed_actions)
            assert pickle.dumps(actual.friendly_states()+[actual.red_state, actual.shadow_red_state]) == \
                pickle.dumps(reference.friendly_states()+[reference.red_state, reference.shadow_red_state])
            for field in ('step', 'success', 'phase', 'terminated', 'truncated', 'timeout',
                    'terminal_reason', 'hold_steps', 'capture_success', 'capture_just_confirmed',
                    'first_capture_step', 'first_capture_time_s', 'post_capture_steps',
                    'withdrawal_cleared_entry', 'stage_success_gates', 'support_gates', 'reward_terms'):
                assert getattr(info, field) == getattr(old_info, field), (step, field)
            assert actual.events == reference.events
            assert done == old_done == (step == 650)
        assert info.success and info.capture_success and info.first_capture_step == 321
        assert info.withdrawal_cleared_entry and info.terminal_reason == 'success'
    finally:
        actual.close()
        reference.close()
