"""A cheap sufficient certificate must never accept a reference-unsafe path."""
import math

import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.safety_kernel import (
    _full_motion_traces, _radial_motion_enclosure, assess_motion, assess_nominal_motion,
)
from usv_env.safety_projection import _Predictor, _dimensions


@pytest.mark.parametrize("role", [0, 1])
def test_radial_certificate_only_accepts_reference_safe_complete_paths(role):
    cfg = assemble(experiment="Full", device="cpu")
    predictor = _Predictor(cfg)
    data = predictor.model_data[role]
    dt = cfg["env"]["dt"]
    steps = predictor.models[role].internal_substeps
    length, width = _dimensions(cfg, role == 0)
    rng = np.random.default_rng(704 + role)
    accepted = 0
    for case in range(80):
        q = np.array([*rng.uniform(3, 52, 2), rng.uniform(0, data[-1][1]),
                      rng.uniform(-.25, .25), rng.uniform(-.4, .4),
                      rng.uniform(-math.pi, math.pi)])
        action = rng.uniform(-1, 1, 2) * data[-1][4:6]
        obstacles = np.array([[*rng.uniform(2, 53, 2), *rng.uniform(-.4, .4, 2),
                               .63, .1, 0., 0.]])
        traces, recovered, _ = _full_motion_traces(
            q, action, *data, dt, steps, 20, 200, .02, .02)
        lower = _radial_motion_enclosure(q, traces, data[-1], dt, steps,
            length, width, predictor.world, predictor.reefs, obstacles, .15)
        reference = assess_motion(q, action, *data, dt, steps, length, width,
            predictor.world, predictor.reefs, obstacles, 20, 200, .02, .02, .15)
        if recovered and lower >= .15:
            accepted += 1
            assert reference[1] and reference[0] >= .15
            assert lower <= reference[0] + 1.e-7
    assert accepted > 5


def test_radial_certificate_rejects_float32_peer_propagation_and_crossing_peer():
    cfg = assemble(experiment="Full", device="cpu")
    predictor = _Predictor(cfg)
    q = np.array([10., 45., .1, 0., 0., 0.])
    data = predictor.model_data[0]
    steps = predictor.models[0].internal_substeps
    dt = cfg["env"]["dt"]
    length, width = _dimensions(cfg, True)
    traces, _, _ = _full_motion_traces(q, np.zeros(2), *data, dt, steps, 20, 200, .02, .02)
    for flags in [(1., 0.), (0., 1.)]:
        peers = np.array([[30., 45., 0., 0., .63, .1, *flags]])
        value = _radial_motion_enclosure(q, traces, data[-1], dt, steps,
            length, width, predictor.world, predictor.reefs, peers, .15)
        assert value == -np.inf
    duration = len(traces) * dt
    peers = np.array([[10., 43., 0., 4. / duration, .63, .1, 0., 0.]])
    value = _radial_motion_enclosure(q, traces, data[-1], dt, steps,
        length, width, predictor.world, predictor.reefs, peers, .15)
    assert value < .15


def test_positive_reward_scale_allows_informative_early_history():
    from matd3.history_enhancement import JointHistory
    from matd3.replay_buffer import ReplayBuffer

    cfg = assemble(experiment="Full", device="cpu")
    n = 5
    state = np.zeros(cfg["observation_schema"]["critic_dim"], np.float32)
    obs = np.zeros((n, cfg["observation_schema"]["frame_dim"]), np.float32)
    replay = ReplayBuffer(8, n, obs.shape[-1], len(state), 2)
    context = dict(phase_id=0, information_valid=np.ones(n),
        information_age=np.zeros(n), information_variance=np.ones(n),
        safety_margin=np.ones(n), quality=.35)
    replay.add(obs, state, np.zeros((n, 2)), np.full(n, .04), obs, state,
        terminated=False, truncated=False, timeout=False,
        current_context=context, next_context=context)
    query, _ = replay._prepare_context(context)
    history = JointHistory(cfg, 704)
    found = history.retrieve(replay, state[None], query[None], [[0]], np.ones((1, n)))
    assert found["valid"].sum() == 1
    assert found["quality"][0, 0] >= cfg["i3"]["minimum_quality"]
    # Unknown/low-information history and negative outcomes do not gain a pass.
    replay.current_context[0, -2] = .05
    found = history.retrieve(replay, state[None], query[None], [[0]], np.ones((1, n)))
    assert not found["valid"].any()
    replay.current_context[0, -2] = .35
    replay.agent_rewards[0] = -.04
    found = history.retrieve(replay, state[None], query[None], [[0]], np.ones((1, n)))
    assert not found["valid"].any()


@pytest.mark.parametrize("role", [0, 1])
def test_fast_boolean_screen_matches_full_recovery_and_geometry_decision(role):
    cfg = assemble(experiment="Full", device="cpu")
    predictor = _Predictor(cfg)
    data = predictor.model_data[role]
    dt = cfg["env"]["dt"]
    substeps = predictor.models[role].internal_substeps
    length, width = _dimensions(cfg, role == 0)
    rng = np.random.default_rng(805+role)
    for index in range(64):
        q = np.array([*rng.uniform(.1, 54.9, 2), rng.uniform(0, data[-1][1]),
                      rng.uniform(-.6, .6), rng.uniform(-.675, .675),
                      rng.uniform(-math.pi, math.pi)])
        action = rng.uniform(-1, 1, 2)*data[-1][4:6]
        peers = np.array([[*rng.uniform(0, 55, 2), *rng.uniform(-.4, .4, 2),
                           .63, .1, float(index % 3 == 0), float(index % 3 == 0)]])
        backup = 200 if index % 2 else 0
        args = (q, action, *data, dt, substeps, length, width,
            predictor.world, predictor.reefs, peers, 20, backup, .02, .02, .15)
        original = assess_motion(*args, np.inf)
        actual = assess_nominal_motion(*args, True)
        assert (actual[0] >= .15 and actual[1]) == (original[0] >= .15 and original[1])
