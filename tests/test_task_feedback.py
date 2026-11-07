"""Task feedback has no navigation-reference or per-vessel search farming path."""
from copy import deepcopy
import numpy as np

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.reward import evaluate, snapshot


def environment():
    return PursuitEvasionEnv(assemble(device='cpu'))


def test_five_vessels_and_reference_free_environment():
    env = environment()
    assert env.num_friendly == 5 and env.num_surface_blue == 2
    for name in ('reference_positions', 'reference_velocities', '_update_references', '_safe_target'):
        assert not hasattr(env, name)
    _, terms = evaluate(snapshot(env), env)
    assert not {'position', 'progress', 'coverage', 'coordination'} & set(terms)


def test_search_cells_are_shared_across_auvs_and_cannot_repeat():
    env = environment()
    env._team_search_cells.clear()
    for ship in env.submarine_states:
        ship.pos = np.array([12., 28.], np.float32)
    env._update_search_coverage()
    new_count = env.team_new_search_cells
    assert new_count > 0 and np.isclose(sum(env._new_cells), new_count)
    env._update_search_coverage()
    assert env.team_new_search_cells == 0 and np.all(env._new_cells == 0)


def test_first_discovery_is_one_team_event():
    env = environment()
    before = snapshot(env)
    env.team_first_detection_step = 1
    env.first_detection_steps[0] = 1
    _, terms = evaluate(before, env)
    assert sum(terms['discovery']) == 0  # No positive one-off payout in cost v2.
    after = snapshot(env)
    env.first_detection_steps[1] = 2
    _, terms = evaluate(after, env)
    assert sum(terms['discovery']) == 0


def test_current_state_counterfactual_does_not_mutate_online_physics_or_navigation():
    env = environment()
    states = deepcopy(env.friendly_states() + [env.red_state])
    navigator = deepcopy(env.red_navigator.__dict__)
    clamps = deepcopy(env.red_dynamics.clamp_counts)
    audit = env.audit_counterfactual(np.zeros((5, 2), np.float32), force=True)
    assert set(audit['branches']) == {'all', 'none', 'without_0', 'without_1'}
    assert env.red_dynamics.clamp_counts == clamps
    assert env.red_navigator._progress == navigator['_progress']
    for old, new in zip(states, env.friendly_states() + [env.red_state]):
        np.testing.assert_array_equal(old.pos, new.pos)
        np.testing.assert_array_equal(old.body_velocity, new.body_velocity)
    assert np.isclose(sum(audit['shapley_effect_m']), audit['total_effect_m'])


def test_no_counterfactual_effect_without_nearby_usvs():
    env = environment()
    audit = env.audit_counterfactual(np.zeros((5, 2), np.float32), force=True)
    assert audit['valid']
    assert abs(audit['total_effect_m']) < 1e-8
    assert np.allclose(audit['removal_effect_m'], 0.)


def test_missing_or_stale_counterfactual_cannot_give_blocking_credit():
    env = environment()
    previous = snapshot(env)
    env.counterfactual_gap = 3.
    env.blue_states[0].pos = env.red_state.pos + np.array([2., 0.])
    _, terms = evaluate(previous, env)
    assert terms['blocking'] == [0.] * 5


def test_nearby_auv_cannot_satisfy_usv_blocking_engagement():
    env=environment(); previous=snapshot(env)
    env.red_state.pos=np.array([30.,20.],np.float32)
    env.blue_states[0].pos=np.array([20.,20.],np.float32)
    env.blue_states[1].pos=np.array([20.,25.],np.float32)
    env.submarine_states[0].pos=np.array([30.,21.],np.float32)
    env.counterfactual_gap=1.; env.route_available=True
    env.counterfactual_audit={'step':env.step_count,'valid':True,'total_effect_m':.1}
    env.counterfactual_audit_new=False
    _,terms=evaluate(previous,env)
    assert terms['blocking'] == [0.] * 5


def test_counterfactual_rejects_a_blue_open_loop_trajectory_through_a_reef():
    env = environment()
    # Start safely south of reef A, then hold forward thrust for three seconds.
    ship=env.blue_states[0]
    reef = next(r for r in env.reefs if r['name'] == 'A')
    ship.pos=np.asarray(reef['center'],np.float32)+np.array([0.,-5.],np.float32)
    ship.heading=np.pi/2; ship.speed=.4
    action=np.zeros((5,2),np.float32); action[0,0]=env.surface_action_limit[0]
    audit=env.audit_counterfactual(action,force=True)
    assert not audit['valid']
    assert audit['branches']['all']['blue_contacts']
    assert any('reef_A' in item for item in audit['branches']['all']['blue_contacts'])
    assert not audit['branches']['none']['blue_contacts']


def test_counterfactual_rejects_a_blue_open_loop_boundary_escape():
    env=environment(); ship=env.blue_states[0]
    ship.pos=np.array([54.,45.],np.float32); ship.heading=0.; ship.speed=.4
    action=np.zeros((5,2),np.float32); action[0,0]=env.surface_action_limit[0]
    audit=env.audit_counterfactual(action,force=True)
    assert not audit['valid']
    assert any('boundary' in item for item in audit['branches']['all']['blue_contacts'])


def test_counterfactual_rejects_predicted_usv_usv_contact():
    env=environment()
    for ship,position,heading in zip(env.blue_states,([10.,20.],[13.,20.]),(0.,np.pi)):
        ship.pos=np.array(position,np.float32); ship.heading=heading; ship.speed=.4
    audit=env.audit_counterfactual(np.zeros((5,2),np.float32),force=True)
    assert not audit['valid']
    assert 'USV_0:USV_1' in audit['branches']['all']['blue_contacts']
    assert not audit['branches']['without_0']['blue_contacts']
    assert not audit['branches']['without_1']['blue_contacts']
