"""Regression checks for protected-zone entry along actual integration traces."""
from copy import deepcopy

import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.types import ShipState


def _state(position):
    return ShipState(np.asarray(position, dtype=float), 0., 0., 0., 'surface')


def test_center_entry_detects_an_inside_substep_between_outside_endpoints():
    from usv_env.safety import center_path_enters_circle
    old, new = _state([-2., 0.]), _state([2., 0.])
    new.integration_trace = [(old.pos.copy(), 0.), (np.array([0., 0.]), 0.), (new.pos.copy(), 0.)]
    assert np.linalg.norm(old.pos) > 1. and np.linalg.norm(new.pos) > 1.
    assert center_path_enters_circle(old, new, np.zeros(2), 1.)


def test_center_entry_uses_trace_instead_of_the_endpoint_chord():
    from usv_env.safety import center_path_enters_circle
    old, new = _state([-2., 0.]), _state([2., 0.])
    new.integration_trace = [(old.pos.copy(), 0.), (np.array([-2., 2.]), 0.),
                             (np.array([2., 2.]), 0.), (new.pos.copy(), 0.)]
    assert not center_path_enters_circle(old, new, np.zeros(2), 1.)


@pytest.mark.parametrize('y,expected', [(0., True), (1., True), (1.01, False)])
def test_center_entry_fallback_and_no_hull_inflation(y, expected):
    from usv_env.safety import center_path_enters_circle
    # Only the center's path matters; a nearby surface hull is not the entry criterion.
    assert center_path_enters_circle(_state([-2., y]), _state([2., y]), np.zeros(2), 1.) == expected


@pytest.mark.parametrize('contact,expected', [('', 'breakthrough'), ('collision', 'collision'),
    ('reef_collision', 'reef_collision'), ('out_of_bounds', 'out_of_bounds')])
def test_environment_swept_breakthrough_preserves_accident_and_breakthrough_priority(monkeypatch, contact, expected):
    env = PursuitEvasionEnv(assemble(device='cpu'))
    env.red_state.pos = np.array([44., 27.], np.float32)

    def scripted_step(state, action, wave=None):
        result = deepcopy(state)
        if state is env.red_state:
            result.pos = np.array([54., 27.], np.float32)
            result.integration_trace = [(state.pos.copy(), state.heading),
                (env.protected_center.copy(), state.heading), (result.pos.copy(), state.heading)]
        else:
            result.integration_trace = [(state.pos.copy(), state.heading), (state.pos.copy(), state.heading)]
        return result

    monkeypatch.setattr(env.red_dynamics, 'step', scripted_step)
    monkeypatch.setattr(env, '_contacts', lambda *args: (contact, 'fixture' if contact else ''))
    monkeypatch.setattr(env, '_task_success', lambda safe: True)
    _, _, _, done, info = env.step(np.zeros((5, 2), np.float32))
    assert done and info.terminal_reason == expected
    assert info.landing_success and not info.success
