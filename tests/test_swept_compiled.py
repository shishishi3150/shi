"""Compiled contacts retain the independent interpolated capsule oracle."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

from usv_env.types import ShipState
from usv_env.safety import path_contact, swept_contact
from usv_env.geometry import capsule_segment, capsule_circle_clearance, wrap_angle
from usv_env.reward import capsule_clearance, boundary_clearance
from usv_env import safety_kernel


def reference():
    return SimpleNamespace(path_contact=object_geometry_contact)


def object_geometry_contact(env, old, new, other_old=None, other_new=None, obstacle='boundary'):
    """Python capsule geometry oracle, independent of the compiled kernel.

    Keep this reference in tests, so cleaning validation artifacts cannot
    silently remove the numerical contact regression checks.
    """
    def points(before, after):
        return getattr(after, 'integration_trace', [(before.pos, before.heading), (after.pos, after.heading)])
    def pose(path, time):
        value = time*(len(path)-1)
        index = min(int(value), len(path)-2)
        fraction = value-index
        p0,h0 = path[index]; p1,h1 = path[index+1]
        return (1-fraction)*np.asarray(p0)+fraction*np.asarray(p1), h0+fraction*wrap_angle(h1-h0)
    def dimensions(ship):
        return ((env.submarine_length, env.submarine_collision_width) if ship.unit_type == 'submarine'
            else (env.surface_length, env.surface_collision_width))
    a = points(old, new)
    b = points(other_old, other_new) if other_new is not None else None
    lengths = [len(a)] + ([len(b)] if b else [])
    times = sorted({t for length in lengths for t in np.linspace(0., 1., length)})
    length, width = dimensions(new)
    def clearance(time):
        position, heading = pose(a, time)
        capsule = capsule_segment(position, heading, length, width)
        if b:
            peer_position, peer_heading = pose(b, time)
            peer = capsule_segment(peer_position, peer_heading, *dimensions(other_new))
            return capsule_clearance(capsule, peer)
        if isinstance(obstacle, tuple): return capsule_circle_clearance(capsule, *obstacle)
        return boundary_clearance(capsule, env.world_size)
    for left,right in zip(times[:-1], times[1:]):
        p0,h0 = pose(a, left); p1,h1 = pose(a, right)
        bound = np.linalg.norm(p1-p0)+.5*length*abs(wrap_angle(h1-h0))
        if b:
            p0,h0 = pose(b, left); p1,h1 = pose(b, right)
            bound += np.linalg.norm(p1-p0)+.5*dimensions(other_new)[0]*abs(wrap_angle(h1-h0))
        if swept_contact(lambda t:clearance(left+t*(right-left)), bound): return True
    return False


def vessel(points, headings, kind='surface'):
    old = ShipState(np.array(points[0], np.float32), 0., float(headings[0]), 0., kind)
    new = ShipState(np.array(points[-1], np.float32), 0., float(headings[-1]), 0., kind)
    new.integration_trace = [(np.array(p, np.float64), float(h)) for p,h in zip(points,headings)]
    return old, new


def test_compiled_continuous_kernel_is_available():
    assert callable(getattr(safety_kernel, 'path_contact_kernel', None))


@pytest.mark.parametrize('kind', ['boundary', 'reef', 'peer'])
def test_contacts_match_original_for_curved_mixed_substep_paths(kind):
    oracle = reference()
    env = SimpleNamespace(world_size=np.array([55.,55.]), surface_length=1.4,
        surface_collision_width=.5, submarine_length=1.2, submarine_collision_width=.4)
    rng = np.random.default_rng(781)
    for row in range(90):
        count = (2, 6, 9)[row%3]
        center = np.array([1.,10.]) if kind=='boundary' else np.array([20.,20.])
        points = center+rng.uniform(-1.,1.,2)+np.cumsum(rng.uniform(-.4,.4,(count,2)),axis=0)
        headings = rng.uniform(-math.pi,math.pi)+np.cumsum(rng.uniform(-.3,.3,count))
        old,new = vessel(points,headings)
        kwargs = dict(obstacle='boundary')
        if kind == 'reef':
            kwargs = dict(obstacle=(center+rng.uniform(-1.5,1.5,2), float(rng.uniform(.1,1.))))
        elif kind == 'peer':
            peer_count = (6, 2, 4)[row%3]
            peer = center+rng.uniform(-1.,1.,2)+np.cumsum(rng.uniform(-.4,.4,(peer_count,2)),axis=0)
            kwargs = dict(zip(('other_old','other_new'), vessel(peer,
                rng.uniform(-math.pi,math.pi)+np.cumsum(rng.uniform(-.3,.3,peer_count)))))
        assert path_contact(env,old,new,**kwargs) == oracle.path_contact(env,old,new,**kwargs), (kind,row)


@pytest.mark.parametrize('offset,expected', [(0.,True), (1.e-10,True), (1.e-6,False)])
def test_grazing_boundary_retains_one_e_minus_eight_tolerance(offset, expected):
    env = SimpleNamespace(world_size=np.array([55.,55.]), surface_length=1.4,
        surface_collision_width=.5, submarine_length=1.2, submarine_collision_width=.4)
    old,new = vessel([[2.,.25+offset], [3.,.25+offset]], [0.,0.])
    assert path_contact(env,old,new) == expected
    assert path_contact(env,old,new) == reference().path_contact(env,old,new)
