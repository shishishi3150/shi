"""An intruder may bypass a screen and withdraw after confirmed capture."""
from copy import deepcopy
import math

import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.red_policy import RedNavigator
from usv_env.types import ShipState


def test_frontal_apf_equilibrium_gets_a_bypass_direction_when_enabled():
    cfg = assemble(device="cpu")
    red = ShipState(np.array([15., 10.]), 0., 0., 0., "surface")
    influence = 5. - cfg['ship']['surface_length_m']
    distance = 5. - influence / math.sqrt(1.5)
    blue = ShipState(red.pos + [distance, 0.], 0., math.pi, 0., "surface")
    nav = RedNavigator(dict(cfg['red_policy'], bypass_gain=.7))
    reference = nav.reference(red, [49., 10.], [], [blue], [55., 55.], cfg['ship'])
    assert abs(reference[0]) > .01
    assert nav.mode == 'BYPASS' and nav.bypass_side in (-1, 1)
    side = nav.bypass_side
    for offset in [-.02, .02, -.01]:
        blue.pos[1] = 10. + offset
        nav.reference(red, [49., 10.], [], [blue], [55., 55.], cfg['ship'])
        assert nav.bypass_side == side


def test_bypass_is_not_triggered_by_time_or_an_absent_usv():
    cfg = assemble(device="cpu")
    red = ShipState(np.array([10., 10.]), .4, 0., 0., "surface")
    nav = RedNavigator(dict(cfg['red_policy'], bypass_gain=.7))
    for _ in range(50):
        nav.reference(red, [49., 10.], [], [], [55., 55.], cfg['ship'])
        assert nav.mode == 'ADVANCE'


def test_withdrawal_goal_is_latched_and_replaces_breakthrough_attraction():
    cfg = assemble(device="cpu")
    nav = RedNavigator(cfg['red_policy'])
    red = ShipState(np.array([24., 24.]), .2, 0., 0., "surface")
    assert nav.request_withdrawal([6., 30.], reason='capture_confirmed')
    assert not nav.request_withdrawal([49., 27.], reason='other')
    reference = nav.reference(red, [49., 27.], [], [], [55., 55.], cfg['ship'])
    assert nav.mode == 'WITHDRAW' and reference[1] < 0.
    assert abs(reference[0]) > .01
    np.testing.assert_array_equal(nav.withdrawal_goal, [6., 30.])


def completed_environment():
    env = PursuitEvasionEnv(assemble(device='cpu'), seed=5)
    env.red_state = ShipState(np.array([24., 24.], np.float32), .15, 0., 0., 'surface')
    for state, angle in zip(env.submarine_states, np.deg2rad([0., 120., 240.])):
        state.pos = env.red_state.pos + 3.5 * np.array([math.cos(angle), math.sin(angle)])
    env.terminal_reason = 'success'
    env._episode_done = True
    env.step_count = 300
    env.hold_steps = env.required_hold_steps
    return env


def test_withdrawal_requires_success_and_never_reopens_the_training_episode():
    from usv_env.withdrawal import WithdrawalReplay
    env = PursuitEvasionEnv(assemble(device='cpu'))
    with pytest.raises(ValueError, match='success'):
        WithdrawalReplay(env)
    env = completed_environment()
    states = deepcopy(env.friendly_states() + [env.red_state])
    navigation = deepcopy(env.red_navigator.__dict__)
    replay = WithdrawalReplay(env)
    for _ in range(20):
        replay.step()
    assert replay.elapsed_seconds == pytest.approx(3.)
    assert not replay.verified and not replay.done
    assert env.step_count == 300 and env.terminal_reason == 'success'
    assert env._episode_done and env.hold_steps == env.required_hold_steps
    assert env.red_navigator.__dict__ == navigation
    for before, after in zip(states, env.friendly_states() + [env.red_state]):
        np.testing.assert_array_equal(before.pos, after.pos)
        np.testing.assert_array_equal(before.body_velocity, after.body_velocity)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((5, 2), np.float32))


def test_exit_is_verified_only_after_whole_hull_crosses_entry_line_outward():
    from usv_env.withdrawal import WithdrawalReplay
    replay = WithdrawalReplay(completed_environment())
    replay.red_state.heading = math.pi
    replay.red_state.speed = .3
    replay.red_state.pos = np.array([replay.exit_x - .1, 27.], np.float32)
    assert not replay.has_cleared_entry()
    replay.red_state.pos[0] = replay.exit_x - 1.
    assert replay.has_cleared_entry()
    replay.red_state.heading = 0.
    assert not replay.has_cleared_entry()


def test_blue_continues_local_pursuit_and_refreshes_sensing_after_capture():
    from usv_env.red_policy import RuleController
    from usv_env.withdrawal import WithdrawalReplay
    env = completed_environment()
    controller = RuleController()
    expected = controller.actions(env.blue_states, env.submarine_states, None,
        env.protected_center, env.reefs, env.world_size, env.blue_dynamics,
        env.submarine_dynamics, env.step_count * env.dt,
        target_estimates=env.target_estimates, active_mask=env.agent_active_mask())
    replay = WithdrawalReplay(env)
    replay.step()
    np.testing.assert_array_equal(replay.last_blue_actions, expected)
    for _ in range(19):
        replay.step()
    assert replay.env.step_count == env.step_count + 20
    assert any(e.valid for e in replay.env.local_target_estimates)
    assert not any(e.valid for e in env.local_target_estimates)
    assert replay.trace_row()['blue_behavior'] == 'CONTINUE_PURSUIT'
    assert replay.env is not env


@pytest.mark.parametrize('failure', ['contact', 'breakthrough', 'timeout'])
def test_departure_failures_cannot_be_reported_as_verified_withdrawal(failure):
    from usv_env.withdrawal import WithdrawalReplay
    env = completed_environment()
    replay = WithdrawalReplay(env, max_seconds=env.dt)
    if failure == 'contact':
        replay.red_state.pos = np.asarray(env.reefs[0]['center'], dtype=np.float32)
    elif failure == 'breakthrough':
        replay.red_state.pos = env.protected_center.copy()
    replay.step()
    assert replay.done and not replay.verified
    assert replay.status == 'withdrawal_' + failure
    assert replay.render_frame().phase == 'WITHDRAWAL_FAILED'
    assert env.terminal_reason == 'success' and env._episode_done
    with pytest.raises(RuntimeError, match='already ended'):
        replay.step()
