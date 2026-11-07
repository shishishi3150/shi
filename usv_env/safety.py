"""Continuous capsule checks on the integrator's piecewise substep path.

Within a numerical substep translation/heading are interpolated. A Lipschitz
bound certifies empty intervals; uncertain intervals are bisected to 1e-8 m.
The physical integration values themselves are never modified.
"""
import numpy as np
from .geometry import capsule_segment, capsule_circle_clearance, point_segment_distance, wrap_angle
from .reward import capsule_clearance, boundary_clearance


def swept_contact(clearance, motion_bound, tolerance=1e-8):
    left,right=clearance(0.),clearance(1.)
    if min(left,right)<=0: return True
    def search(a,b,fa,fb):
        bound=motion_bound*(b-a)
        if (fa+fb-bound)/2>tolerance: return False
        mid=(a+b)/2; fm=clearance(mid)
        if fm<=tolerance: return True
        if bound<=tolerance: return False
        return search(a,mid,fa,fm) or search(mid,b,fm,fb)
    return search(0.,1.,left,right)


def trace(state,old):
    return getattr(state,"integration_trace",[(np.asarray(old.pos),old.heading),(np.asarray(state.pos),state.heading)])


def center_path_enters_circle(old, new, center, radius):
    """Check the center's piecewise integration path against an uninflated disk.

    A curved recorded path must be checked segment by segment: its endpoint
    chord may cross a disk even when the actual numerical trajectory avoids it.
    """
    points = trace(new, old)
    return any(point_segment_distance(center, left[0], right[0]) <= radius
               for left, right in zip(points[:-1], points[1:]))


def pose(points,t):
    value=t*(len(points)-1)
    i=min(int(value),len(points)-2); f=value-i
    p0,h0=points[i]; p1,h1=points[i+1]
    return (1-f)*np.asarray(p0)+f*np.asarray(p1),float(h0+f*wrap_angle(h1-h0))


def path_contact(env,old,new,other_old=None,other_new=None,obstacle="boundary"):
    from .safety_kernel import path_contact_kernel
    a=trace(new,old)
    b=trace(other_new,other_old) if other_new is not None else None
    times=np.asarray(sorted(set(np.linspace(0,1,len(a)).tolist()+(np.linspace(0,1,len(b)).tolist() if b else []))))
    def dimensions(ship):
        return ((env.submarine_length,env.submarine_collision_width) if ship.unit_type=="submarine"
                else (env.surface_length,env.surface_collision_width))
    length,width=dimensions(new)
    length_b,width_b=dimensions(other_new) if b else (0.,0.)
    packed_a=np.asarray([[*p,h] for p,h in a],np.float64)
    packed_b=np.asarray([[*p,h] for p,h in b],np.float64) if b else np.empty((0,3))
    circle=np.array([*obstacle[0],obstacle[1]],np.float64) if isinstance(obstacle,tuple) else np.zeros(3)
    return bool(path_contact_kernel(packed_a,packed_b,times,
        np.array([length,width,length_b,width_b]),np.asarray(env.world_size,np.float64),
        2 if b else int(isinstance(obstacle,tuple)),circle,1.e-8))
