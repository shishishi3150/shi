"""Escort ideas must guide pursuit and interception without prescribed slots."""
import math

import numpy as np
import pytest

from copy import deepcopy

from usv_env.config import assemble, validate
from usv_env.environment import PursuitEvasionEnv
from usv_env.reward import evaluate, pursuit_motion_quality


@pytest.fixture
def scene(monkeypatch):
    env = PursuitEvasionEnv(assemble(device='cpu'))
    # Isolate task credit from hull orientation or island clearance changes.
    monkeypatch.setattr(env, '_clearances', lambda index: [2.])
    env.red_state.pos = np.array([25., 40.], np.float32)
    env.red_state.speed = env.red_state.sway_speed = 0.
    env.red_state.heading = 0.
    env.last_executed_actions[:] = 0.
    env.reefs = []
    yield env
    env.close()


def place_auvs(env, radius):
    for j,ship in enumerate(env.submarine_states):
        angle = math.pi+2*math.pi*j/3
        ship.pos = env.red_state.pos+radius*np.array([math.cos(angle),math.sin(angle)])
        ship.speed = ship.sway_speed = ship.yaw_rate = 0.
        ship.heading = 0.


def test_pursuit_earns_more_than_parking_or_fleeing_at_identical_positions(scene):
    place_auvs(scene, 12.)
    ship = scene.submarine_states[0]
    ship.speed = .15
    pursuing = evaluate(None,scene)[0][2]
    ship.speed = 0.
    parking = evaluate(None,scene)[0][2]
    ship.speed = .15; ship.heading = math.pi
    fleeing = evaluate(None,scene)[0][2]
    assert pursuing > parking > fleeing


def test_moving_encirclement_prefers_matching_target_over_stopping(scene):
    place_auvs(scene, 3.)
    scene.red_state.speed = .3
    ship = scene.submarine_states[0]
    ship.speed = .3
    matching = evaluate(None,scene)[0][2]
    ship.speed = 0.
    stopping = evaluate(None,scene)[0][2]
    ship.speed = .3; ship.heading = math.pi
    opposite = evaluate(None,scene)[0][2]
    assert matching > stopping > opposite


def test_usv_gets_earlier_credit_for_blocking_goal_side_than_following_behind(scene):
    scene.red_state.pos = np.array([25.,27.],np.float32)
    scene.blue_states[0].pos = scene.red_state.pos+np.array([3.,0.])
    ahead = evaluate(None,scene)[0][0]
    scene.blue_states[0].pos = scene.red_state.pos-np.array([3.,0.])
    behind = evaluate(None,scene)[0][0]
    assert ahead > behind


def test_velocity_matching_does_not_force_heading_toward_target_center(scene):
    place_auvs(scene, 3.)
    scene.red_state.heading = math.pi/2; scene.red_state.speed = .3
    ship = scene.submarine_states[0]
    ship.heading = math.pi/2; ship.speed = .3
    following = evaluate(None,scene)[0][2]
    ship.heading = 0.
    charging_center = evaluate(None,scene)[0][2]
    assert following > charging_center


def test_direct_pursuit_credit_is_suppressed_by_reef_but_not_reef_behind_annulus(scene):
    place_auvs(scene, 12.)
    ship = scene.submarine_states[0]
    ship.speed = .15
    scene.reefs = [dict(center=[17.,40.], radius=1.)]
    _, blocked = evaluate(None,scene)
    assert blocked['auv_velocity_reward'][2] == blocked['auv_heading_reward'][2] == 0.
    scene.reefs = [dict(center=[24.,40.], radius=1.)]
    _, beyond_capture_band = evaluate(None,scene)
    assert beyond_capture_band['auv_velocity_reward'][2] > 0.
    assert beyond_capture_band['auv_heading_reward'][2] > 0.


def test_no_heading_credit_when_stationary_target_and_auv_in_valid_distance_band(scene):
    place_auvs(scene, 3.)
    motion = pursuit_motion_quality(scene,scene.submarine_states[0])
    assert motion['heading'] == 0.
    assert motion['velocity'] == 1.
    np.testing.assert_array_equal(motion['reference_velocity'],[0.,0.])


def test_capture_annulus_motion_reference_stays_inside_physical_speed_cap(scene):
    place_auvs(scene, 40.)
    scene.red_state.speed = .4
    for heading in np.linspace(-math.pi,math.pi,17):
        scene.red_state.heading = heading
        motion = pursuit_motion_quality(scene,scene.submarine_states[0])
        assert np.linalg.norm(motion['reference_velocity']) <= .5+1e-12
        assert 0. <= motion['velocity'] <= 1. and 0. <= motion['heading'] <= 1.


def test_motion_credit_remains_tied_to_capture_distance(scene):
    scene.red_state.speed = .2
    place_auvs(scene, 3.)
    ship = scene.submarine_states[0]; ship.speed = .2
    _, near = evaluate(None,scene)
    place_auvs(scene, 20.)
    ship.speed = .4
    _, far = evaluate(None,scene)
    # Matching a useful velocity far away must not pay the same as matching
    # it near a valid enclosure. This prevents an extra far-field reward floor.
    for key in ('auv_velocity_reward','auv_heading_reward'):
        assert far[key][2] < .5*near[key][2]


def test_withdrawal_retires_goal_side_credit_without_exceeding_process_budget(scene):
    scene.capture_success = True; scene.first_capture_step = -1
    for ship in scene.blue_states:
        ship.pos = scene.red_state.pos+np.array([3.,0.])
    reward, terms = evaluate(None,scene)
    assert not any(terms['usv_interception_position_reward'])
    assert not any(terms['usv_blocking_reward']) and not any(terms['usv_contribution_reward'])
    assert terms['usv_withdrawal_reward'][0] > 0.
    assert reward.max() <= scene.config['reward']['positive']['step_budget']+1e-8


def test_rotating_scene_preserves_reward_without_absolute_phase_or_vertex(scene):
    place_auvs(scene, 9.)
    for j,ship in enumerate(scene.friendly_states()+[scene.red_state]):
        ship.speed = .03*j; ship.heading = .31*j
    original, _ = evaluate(None,scene)
    angle = 1.27; c,s = math.cos(angle),math.sin(angle)
    rotation = np.array([[c,-s],[s,c]])
    for ship in scene.friendly_states()+[scene.red_state]:
        ship.pos = rotation@ship.pos
        ship.heading += angle
    scene.protected_center = rotation@scene.protected_center
    rotated, _ = evaluate(None,scene)
    np.testing.assert_allclose(rotated, original, atol=1e-8)


@pytest.mark.parametrize('key', ['closing_speed_mps','closing_scale_m','velocity_error_scale_mps',
    'heading_speed_floor_mps','motion_clearance_m','interception_lateral_scale_m'])
def test_motion_design_rejects_nonpositive_scales(key):
    cfg = assemble(device='cpu'); cfg['reward']['positive'][key] = 0.
    with pytest.raises(ValueError, match='reward'):
        validate(cfg)


def test_legacy_positive_configuration_remains_valid_and_pays_no_motion_aliases(scene):
    cfg = deepcopy(scene.config)
    cfg['reward']['scheme'] = 'cooperative_positive_v1'
    cfg['reward']['positive']['auv_weights'] = dict(auv_individual_reward=.3,auv_distance_reward=.2,
        auv_gap_reward=.15,auv_hull_reward=.15,auv_tracking_reward=.1,auv_hold_reward=.1)
    cfg['reward']['positive']['usv_weights'] = dict(usv_approach_reward=.35,usv_blocking_reward=.25,
        usv_support_reward=.2,usv_contribution_reward=.2)
    validate(cfg); scene.config = cfg
    _, terms = evaluate(None,scene)
    assert not any(terms['auv_velocity_reward']) and not any(terms['auv_heading_reward'])
    assert not any(terms['usv_interception_position_reward'])
