"""Regressions for retained physical models and continuous hull safety."""
from copy import deepcopy
import numpy as np
import pytest
from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.geometry import capsule_segment, capsule_circle_clearance, capsules_overlap
from usv_env.types import ShipState
from usv_env.safety import path_contact


def test_retained_mass_and_force_units():
    e=PursuitEvasionEnv(assemble())
    assert e.blue_dynamics.parameters['mass']==23.8
    assert e.submarine_dynamics.parameters['m11']==16.5
    assert e.surface_action_limit.tolist()==pytest.approx([2.2,1.5])
    assert e.submarine_action_limit.tolist()==pytest.approx([3.035,1.5])
    np.testing.assert_allclose(e.blue_dynamics.mass_matrix[0,0],25.8)


@pytest.mark.parametrize('role',['surface','submarine'])
def test_nominal_speed_comes_from_force_drag_not_direct_velocity_action(role):
    e=PursuitEvasionEnv(assemble())
    dyn=e.blue_dynamics if role=='surface' else e.submarine_dynamics
    ship=ShipState(np.array([10.,10.]),0.,0.,0.,role,role=='surface')
    moved=dyn.step(ship,np.array([dyn.max_surge_force,0.]))
    assert 0<moved.speed<.05 and 0<moved.pos[0]-ship.pos[0]<.01
    assert ship.speed==0 and ship.pos[0]==10
    assert moved.sway_speed==pytest.approx(0.)


@pytest.mark.parametrize('role',['surface','submarine'])
def test_force_clamps_and_numerical_trace_are_retained(role):
    e=PursuitEvasionEnv(assemble()); dyn=e.blue_dynamics if role=='surface' else e.submarine_dynamics
    ship=ShipState(np.array([10.,10.]),.2,0.,0.,role)
    huge=dyn.step(ship,np.array([1e6,1e6]))
    limited=dyn.step(ship,np.array([dyn.max_surge_force,dyn.max_yaw_moment]))
    np.testing.assert_array_equal(huge.pos,limited.pos)
    np.testing.assert_array_equal(huge.body_velocity,limited.body_velocity)
    assert len(huge.integration_trace)==dyn.internal_substeps+1
    assert abs(huge.yaw_rate)<=dyn.max_yaw_rate and huge.speed<=dyn.max_speed


def test_swept_capsule_detects_contact_between_clear_endpoints():
    e=PursuitEvasionEnv(assemble())
    left=ShipState(np.array([10.,10.]),0.,0.,0.,'surface')
    right=deepcopy(left); right.pos=np.array([14.,10.])
    center=np.array([12.,10.]); radius=.3
    assert capsule_circle_clearance(e._capsule(left),center,radius)>0
    assert capsule_circle_clearance(e._capsule(right),center,radius)>0
    assert path_contact(e,left,right,obstacle=(center,radius))


def test_long_hull_orientation_affects_collision():
    horizontal=capsule_segment(np.array([0.,0.]),0.,1.255,.29)
    vertical=capsule_segment(np.array([0.,0.]),np.pi/2,1.255,.29)
    center=np.array([.6,0.])
    assert capsule_circle_clearance(horizontal,center,.1)<0
    assert capsule_circle_clearance(vertical,center,.1)>0


def test_boundary_sweep_detects_leaving_world():
    e=PursuitEvasionEnv(assemble())
    a=ShipState(np.array([1.,10.]),0.,0.,0.,'surface')
    b=deepcopy(a); b.pos=np.array([-.1,10.])
    assert path_contact(e,a,b,obstacle='boundary')
