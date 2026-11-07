"""Capture requires a real AUV enclosure, independently of USV delay."""
import itertools
import math

import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.target_tracking import TargetEstimate
from usv_env.task import encirclement_geometry


def capture_fixture():
    env = PursuitEvasionEnv(assemble(device="cpu"))
    env.red_state.pos = np.array([32., 15.], np.float32)
    for state, angle, radius in zip(env.submarine_states,
            np.deg2rad([0., 110., 235.]), [3.5, 4., 4.5]):
        state.pos = env.red_state.pos + radius * np.array([math.cos(angle), math.sin(angle)])
    for index in range(2, 5):
        env.estimators[index].estimate = TargetEstimate(True,
            np.r_[env.red_state.pos, 0., 0.], np.eye(4) * .1, 0., 0., 1., index, f"fixture:{index}")
    return env


def test_enclosure_is_invariant_to_vessel_order_rotation_and_translation():
    positions = np.array([[3., 0.], [-1., 4.], [-2., -3.]])
    expected = encirclement_geometry(np.zeros(2), positions)
    assert expected['hull_margin_m'] > .25
    assert expected['maximum_angle_gap_rad'] < math.pi
    angle = .83
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    offset = np.array([8., 11.])
    for order in itertools.permutations(range(3)):
        actual = encirclement_geometry(offset, positions[list(order)] @ rotation.T + offset)
        assert actual['hull_margin_m'] == pytest.approx(expected['hull_margin_m'])
        assert actual['maximum_angle_gap_rad'] == pytest.approx(expected['maximum_angle_gap_rad'])


@pytest.mark.parametrize('positions', [
    [[2., 1.], [3., 2.], [4., -1.]],
    [[-3., 0.], [0., 0.], [3., 0.]],
    [[3., 0.], [3., 0.], [-3., 0.]],
    [[3., 0.], [-3., 0.]],
])
def test_one_sided_collinear_duplicate_or_two_auv_geometry_is_not_enclosure(positions):
    geometry = encirclement_geometry(np.zeros(2), positions)
    assert geometry['hull_margin_m'] <= 0.


def test_auv_capture_succeeds_without_usv_engagement_or_counterfactual_credit():
    env = capture_fixture()
    assert not env.counterfactual_evidence_valid()
    assert all(np.linalg.norm(state.pos - env.red_state.pos) > 4. for state in env.blue_states)
    for _ in range(env.required_hold_steps - 1):
        assert not env._task_success(safe=True)
    assert env._task_success(safe=True)
    assert not env.task_state.support_gates['engagement']
    assert not env.task_state.support_gates['current_state_counterfactual']


@pytest.mark.parametrize('failure', ['too_far', 'too_close', 'one_side', 'inactive', 'stale', 'unsafe'])
def test_capture_rejects_invalid_geometry_information_or_safety_and_resets_hold(failure):
    env = capture_fixture()
    for _ in range(env.required_hold_steps - 1):
        env._task_success(safe=True)
    if failure == 'too_far':
        state = env.submarine_states[0]
        state.pos = env.red_state.pos + np.array([6., 0.])
    elif failure == 'too_close':
        env.submarine_states[0].pos = env.red_state.pos + np.array([.5, 0.])
    elif failure == 'one_side':
        for state, offset in zip(env.submarine_states, [[2., -1.], [3., 0.], [2., 1.]]):
            state.pos = env.red_state.pos + offset
    elif failure == 'inactive':
        env._agent_active_mask[2] = 0
    elif failure == 'stale':
        env.step_count = 20
    assert not env._task_success(safe=failure != 'unsafe')
    assert env.hold_steps == 0


def test_usv_blocking_and_three_tracks_alone_cannot_finish_capture():
    env = capture_fixture()
    for state, offset in zip(env.submarine_states, [[2., -1.], [3., 0.], [2., 1.]]):
        state.pos = env.red_state.pos + offset
    env.counterfactual_gap = 10.
    env.progress_deficit = 1.
    env.counterfactual_audit = dict(step=0, valid=True, total_effect_m=1.)
    env.blue_states[0].pos = env.red_state.pos + np.array([2., 0.])
    for _ in range(60):
        assert not env._task_success(safe=True)


def test_capture_progress_rewards_closing_an_enclosure_without_point_references():
    from usv_env.reward import snapshot, evaluate
    env = capture_fixture()
    enclosed = [state.pos.copy() for state in env.submarine_states]
    for state in env.submarine_states:
        state.pos = env.red_state.pos + 2. * (state.pos - env.red_state.pos)
    before = snapshot(env)
    far_reward, _ = evaluate(before, env)
    for state, position in zip(env.submarine_states, enclosed):
        state.pos = position
    env._task_success(safe=True)
    enclosed_reward, terms = evaluate(before, env)
    assert np.all(enclosed_reward[2:] > far_reward[2:])
    assert np.all(enclosed_reward >= 0.)
    _, stationary = evaluate(snapshot(env), env)
    assert stationary['auv_potential_reward'] == [0.] * 5
    for state, offset in zip(env.submarine_states, [[2., -1.], [3., 0.], [2., 1.]]):
        state.pos = env.red_state.pos + offset
    _, one_sided = evaluate(snapshot(env), env)
    assert one_sided['observation_geometry'] == [0.] * 5
    assert one_sided['auv_hull_quality'][2] < 1.


def test_real_dynamics_episode_finishes_after_stable_auv_capture(monkeypatch):
    env = capture_fixture()
    # Historical immediate-capture mode remains explicit and supported.
    env.post_capture_end_step = 0
    env.red_state.speed = .1
    for state in env.submarine_states:
        offset = state.pos - env.red_state.pos
        state.pos = env.red_state.pos + 3.5 * offset / np.linalg.norm(offset)

    def ideal_local_measurements():
        # Explicit measurement fixture; production sensors are unchanged.
        now = env.step_count * env.dt
        for index in range(2, 5):
            env.estimators[index].estimate = TargetEstimate(True,
                np.r_[env.red_state.pos, env.red_state.velocity], np.eye(4) * .1,
                now, now, 1., index, f"fixture:{index}:{env.step_count}")

    monkeypatch.setattr(env, '_sense_and_communicate', ideal_local_measurements)
    for step in range(env.required_hold_steps):
        _, _, _, done, info = env.step(np.zeros((5, 2), np.float32))
        assert done == (step == env.required_hold_steps - 1)
    assert info.success and info.terminated and not info.timeout
    assert info.terminal_reason == 'success' and info.phase == 'SUCCESS'
    assert not info.surface_intervention_success
    assert set(info.stage_success_gates) == {'auv_participation', 'capture_distance',
        'encirclement', 'observation_support', 'route_available', 'safety'}


def test_old_usv_blocking_task_config_is_not_silently_reinterpreted():
    config = assemble(device='cpu')
    del config['task']['objective']
    with pytest.raises(ValueError, match='task objective differs'):
        PursuitEvasionEnv(config)
