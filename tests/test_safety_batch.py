"""Batch I1 must preserve scalar safety decisions and receiver boundaries."""
import math
import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from usv_env.observation import build_actor_observation, build_critic_state
from usv_env.safety_projection import project_from_state
from usv_env.target_tracking import TargetEstimate
from usv_env.safety_batch import _receiver_rotations
from usv_env.safety_batch import project_batch
from usv_env.safety_projection import _Predictor


def fixture(cfg):
    env = PursuitEvasionEnv(cfg, 410)
    history = HistoryEnv(env, 1)
    rng = np.random.default_rng(420)
    observations, states = [], []
    for row in range(32):
        env.reset(seed=410+row)
        for i, ship in enumerate(env.friendly_states()):
            ship.heading = rng.uniform(-math.pi, math.pi)
            ship.speed = rng.uniform(-.1, .4)
            ship.sway_speed = rng.uniform(-.3, .3)
            ship.yaw_rate = rng.uniform(-.6, .6)
            if row % 4 == 0:
                ship.pos = np.array([1.2+i*.05, 10.+i*.2], np.float32)
            elif row % 4 == 1:
                ship.pos = np.asarray(cfg['reefs'][0]['center'], np.float32)+i*.05
        if row % 7 == 0:
            env.set_active_agents([0, 1, 1, 1, 1])
        env.communication.cache.clear()
        if row % 3:
            env.communication.step(env.friendly_states(), 0)
        env.target_estimates = [TargetEstimate.unknown() for _ in range(env.num_friendly)]
        if row % 2:
            env.target_estimates[0] = TargetEstimate(True,
                np.array([14., 11., .18, .04]), np.eye(4)*.1, 0., 0., .9, 2, '2:0')
        raw = np.stack([build_actor_observation(env, i) for i in range(env.num_friendly)])
        history.frames.clear()
        observations.append(history._push(raw))
        states.append(build_critic_state(env))
    return np.stack(states), np.stack(observations)


def test_compiled_receiver_rotation_matches_double_precision_python():
    rows = np.random.default_rng(514).uniform(-1., 1., (1280, 9)).astype(np.float32)
    rows[:4, 6:8] = [[0., 0.], [-0., -0.], [1., 0.], [-1., -0.]]
    heading = np.array([math.atan2(a, b) for a, b in rows[:, 6:8]])
    expected = heading, np.array([math.cos(x) for x in heading]), np.array([math.sin(x) for x in heading])
    for actual, original in zip(_receiver_rotations(rows), expected):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize('reject', [False, True])
@pytest.mark.parametrize('receivers', [None, (0,), (3,)])
@pytest.mark.parametrize('separated', [False, True])
def test_batch_matches_scalar_near_reefs_boundaries_and_missing_packets(reject, receivers, separated):
    cfg = assemble(experiment='Full', device='cpu')
    cfg['communication'].update(delay_steps=[0, 0], loss_rate=0., target_loss_range=[0., 0.])
    if not separated:
        cfg['depth']['auv_depth_m'] = .6
    states, observations = fixture(cfg)
    rng = np.random.default_rng(11)
    actions = rng.uniform(-3., 3., (32, 5, 2)).astype(np.float32)
    actual, detail = project_from_state(cfg, states, observations, actions,
        reject_only=reject, receivers=receivers)
    expected, rows = [], []
    for state, obs, action in zip(states, observations, actions):
        value, info = project_from_state(cfg, state, obs, action,
            reject_only=reject, receivers=receivers)
        expected.append(value); rows.append(info)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    for key in rows[0].keys()-{'solver_time_s'}:
        np.testing.assert_allclose(detail[key], np.concatenate([row[key] for row in rows]),
            rtol=0, atol=2e-6, err_msg=key)
    # The selected receiver's output cannot use other centralized truth rows.
    if receivers == (0,):
        changed = states.copy()
        changed[:, 9:52] = 1234.
        after, _ = project_from_state(cfg, changed, observations, actions,
            reject_only=reject, receivers=receivers)
        np.testing.assert_array_equal(actual, after)


def test_empty_selected_receivers_and_all_inactive_batch():
    cfg = assemble(experiment='Full', device='cpu')
    states, observations = fixture(cfg)
    frames = observations.reshape(32, 5, -1, observations.shape[-1])[:, :, -1]
    actions = np.ones((32, 5, 2), np.float32)
    empty, detail = project_batch(_Predictor(cfg), cfg, states, frames, actions, set())
    np.testing.assert_array_equal(empty, actions)
    assert detail['safe_mask'].all() and not detail['intervention'].any()
    inactive = states.copy()
    inactive[:, np.arange(5)*9] = 0.
    stopped, detail = project_batch(_Predictor(cfg), cfg, inactive, frames, actions, set(range(5)))
    np.testing.assert_array_equal(stopped, np.zeros_like(actions))
    assert detail['safe_mask'].all() and not detail['infeasible'].any()


def test_projection_cache_preserves_exact_repeats_and_detects_action_changes():
    cfg = assemble(experiment='Full', device='cpu')
    states, observations = fixture(cfg)
    frames = observations.reshape(32, 5, -1, observations.shape[-1])[:, :, -1]
    actions = np.zeros((32, 5, 2), np.float32)
    predictor = _Predictor(cfg)
    original, detail = project_batch(predictor, cfg, states, frames, actions, set(range(5)))
    misses, hits = predictor._batch_cache_misses, predictor._batch_cache_hits
    repeated, repeated_detail = project_batch(predictor, cfg, states, frames, actions, set(range(5)))
    np.testing.assert_array_equal(repeated, original)
    for key in detail:
        np.testing.assert_array_equal(repeated_detail[key], detail[key])
    assert predictor._batch_cache_misses == misses and predictor._batch_cache_hits > hits
    changed = actions.copy()
    # Inactive commands deliberately do not create projection jobs.
    row, receiver = np.argwhere(states[:, np.arange(5)*9] != 0)[0]
    changed[row, receiver, 0] = .1
    project_batch(predictor, cfg, states, frames, changed, set(range(5)))
    assert predictor._batch_cache_misses == misses+1
    repeated[:] = 123.
    pristine, _ = project_batch(predictor, cfg, states, frames, actions, set(range(5)))
    np.testing.assert_array_equal(pristine, original)
