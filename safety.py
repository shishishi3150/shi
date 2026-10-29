"""Continuous capsule checks on the integrator's piecewise substep path.

Within a numerical substep translation/heading are interpolated. A Lipschitz
bound certifies empty intervals; uncertain intervals are bisected to 1e-8 m.
The physical integration values themselves are never modified.
"""
import math
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


def pose(points,t):
    value=t*(len(points)-1)
    i=min(int(value),len(points)-2); f=value-i
    p0,h0=points[i]; p1,h1=points[i+1]
    return (1-f)*np.asarray(p0)+f*np.asarray(p1),float(h0+f*wrap_angle(h1-h0))


def path_contact(env,old,new,other_old=None,other_new=None,obstacle="boundary"):
    a=trace(new,old)
    b=trace(other_new,other_old) if other_new is not None else None
    times=sorted(set(np.linspace(0,1,len(a)).tolist()+(np.linspace(0,1,len(b)).tolist() if b else [])))
    def dimensions(ship):
        return ((env.submarine_length,env.submarine_collision_width) if ship.unit_type=="submarine"
                else (env.surface_length,env.surface_collision_width))
    length,width=dimensions(new)
    if b: length_b,width_b=dimensions(other_new)
    for start,end in zip(times[:-1],times[1:]):
        p0,h0=pose(a,start); p1,h1=pose(a,end)
        bound=float(np.linalg.norm(p1-p0))+.5*length*abs(float(wrap_angle(h1-h0)))
        if b:
            q0,g0=pose(b,start); q1,g1=pose(b,end)
            bound+=float(np.linalg.norm(q1-q0))+.5*length_b*abs(float(wrap_angle(g1-g0)))
        def clearance(t):
            p,h=pose(a,start+t*(end-start)); cap=capsule_segment(p,h,length,width)
            if b:
                q,g=pose(b,start+t*(end-start))
                return capsule_clearance(cap,capsule_segment(q,g,length_b,width_b))
            if obstacle=="island": return capsule_circle_clearance(cap,env.island_center,env.island_radius)
            return boundary_clearance(cap,env.world_size)
        if swept_contact(clearance,bound): return True
    return False


def landing_crossed(env,old,new):
    if not env.island_active: return False
    points=trace(new,old)
    return any(point_segment_distance(env.island_center,a[0],b[0])<=env.island_radius+env.landing_buffer
               for a,b in zip(points[:-1],points[1:]))
