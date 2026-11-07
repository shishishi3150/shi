"""Compiled numerical kernel for I1; no target points or privileged observations.

No fast-math: the force model, five integration substeps, float32 position
rounding at environment boundaries and swept-capsule bounds match dynamics.py.
The independent object-model implementation is exercised by regression tests.
"""
import math
import numpy as np
from numba import njit


def model_arrays(model):
    """Read the actual model parameters once, never hard-code a second model."""
    p = model.parameters
    if 'm11' in p:
        damping = [p['d11'], p['d11_quad'], 0., p['d22'], p['d22_quad'], 0.,
                   0., 0., 0., 0., 0., 0., p['d33'], 0., p['d33_quad']]
    else:
        damping = [-p[k] for k in ('x_u', 'x_abs_u_u', 'x_uuu',
            'y_v', 'y_abs_v_v', 'y_abs_r_v', 'y_r', 'y_abs_v_r', 'y_abs_r_r',
            'n_v', 'n_abs_v_v', 'n_abs_r_v', 'n_r', 'n_abs_v_r', 'n_abs_r_r')]
    limits = np.array([model.min_speed, model.max_speed, model.max_sway_speed,
        model.max_yaw_rate, model.max_surge_force, model.max_yaw_moment])
    return np.asarray(damping), model.mass_matrix, model.inverse_mass_matrix, limits


@njit(cache=True)
def _loads(u, v, r, damping, mass):
    c13 = -(mass[1, 1]*v + mass[1, 2]*r)
    c23 = mass[0, 0]*u
    d11 = damping[0] + damping[1]*abs(u) + damping[2]*u*u
    d22 = damping[3] + damping[4]*abs(v) + damping[5]*abs(r)
    d23 = damping[6] + damping[7]*abs(v) + damping[8]*abs(r)
    d32 = damping[9] + damping[10]*abs(v) + damping[11]*abs(r)
    d33 = damping[12] + damping[13]*abs(v) + damping[14]*abs(r)
    return (c13*r + d11*u, c23*r + (d22*v + d23*r),
            (-c13*u - c23*v) + (d32*v + d33*r))


@njit(cache=True)
def brake_action(q, damping, mass, limits, dt):
    """Bounded surge/yaw damping, not a position or heading reference."""
    load0, _, load2 = _loads(q[2], q[3], q[4], damping, mass)
    force = load0 - mass[0, 0]*q[2]/dt
    moment = load2 - mass[2, 2]*q[4]/dt
    return np.array([min(limits[4], max(-limits[4], force)),
                     min(limits[5], max(-limits[5], moment))])


@njit(cache=True)
def step_state(q, action, damping, mass, inverse, limits, dt, substeps):
    """Return boundary-rounded state and the unrounded integration trace."""
    following = np.empty(6, np.float64)
    trace = np.empty((substeps+1, 3))
    _step_state_into(q, action, damping, mass, inverse, limits, dt, substeps, following, trace)
    return following, trace


@njit(cache=True)
def _step_state_into(q, action, damping, mass, inverse, limits, dt, substeps, following, trace):
    """The same arithmetic into caller-owned state/trace buffers.

    Read every starting scalar before writing, so q and following may alias.
    No integration, rounding, safety or braking step is removed.
    """
    x, y, u, v, r, heading = q
    trace[0, 0], trace[0, 1], trace[0, 2] = x, y, heading
    force = min(limits[4], max(-limits[4], action[0]))
    moment = min(limits[5], max(-limits[5], action[1]))
    h = dt/substeps
    for k in range(substeps):
        l0, l1, l2 = _loads(u, v, r, damping, mass)
        f0, f1, f2 = force-l0, -l1, moment-l2
        du = inverse[0, 0]*f0 + inverse[0, 1]*f1 + inverse[0, 2]*f2
        dv = inverse[1, 0]*f0 + inverse[1, 1]*f1 + inverse[1, 2]*f2
        dr = inverse[2, 0]*f0 + inverse[2, 1]*f1 + inverse[2, 2]*f2
        u = min(limits[1], max(limits[0], u+du*h))
        v = min(limits[2], max(-limits[2], v+dv*h))
        r = min(limits[3], max(-limits[3], r+dr*h))
        heading = (heading+r*h+math.pi) % (2.*math.pi)-math.pi
        cosine, sine = math.cos(heading), math.sin(heading)
        x += (cosine*u-sine*v)*h
        y += (sine*u+cosine*v)*h
        trace[k+1, 0], trace[k+1, 1], trace[k+1, 2] = x, y, heading
    following[0], following[1] = float(np.float32(x)), float(np.float32(y))
    following[2], following[3], following[4], following[5] = u, v, r, heading


@njit(cache=True)
def _point_distance(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    denom = dx*dx+dy*dy
    alpha = min(1., max(0., ((px-ax)*dx+(py-ay)*dy)/denom)) if denom > 1.e-16 else 0.
    return math.hypot(px-ax-alpha*dx, py-ay-alpha*dy)


@njit(cache=True)
def _segment_distance(ax, ay, bx, by, cx, cy, dx, dy):
    rx, ry, sx, sy = bx-ax, by-ay, dx-cx, dy-cy
    denominator = rx*sy-ry*sx
    if abs(denominator) > 1.e-12:
        qx, qy = cx-ax, cy-ay
        t, u = (qx*sy-qy*sx)/denominator, (qx*ry-qy*rx)/denominator
        if 0. <= t <= 1. and 0. <= u <= 1.:
            return 0.
    return min(_point_distance(ax, ay, cx, cy, dx, dy),
        _point_distance(bx, by, cx, cy, dx, dy),
        _point_distance(cx, cy, ax, ay, bx, by),
        _point_distance(dx, dy, ax, ay, bx, by))


@njit(cache=True)
def _trace_pose(points, t):
    value = t*(len(points)-1)
    index = min(int(value), len(points)-2)
    fraction = value-index
    first, following = points[index], points[index+1]
    angle = (following[2]-first[2]+math.pi) % (2.*math.pi)-math.pi
    return ((1.-fraction)*first[0]+fraction*following[0],
            (1.-fraction)*first[1]+fraction*following[1], first[2]+fraction*angle)


@njit(cache=True)
def _path_clearance(a, b, t, dimensions, world, kind, circle):
    x, y, heading = _trace_pose(a, t)
    length, width, length_b, width_b = dimensions
    radius = .5*width
    half = max(.5*length-radius, 0.)
    dx, dy = half*math.cos(heading), half*math.sin(heading)
    ax, ay, bx, by = x-dx, y-dy, x+dx, y+dy
    if kind == 0:
        return min(ax-radius, ay-radius, bx-radius, by-radius,
            world[0]-ax-radius, world[1]-ay-radius,
            world[0]-bx-radius, world[1]-by-radius)
    if kind == 1:
        return _point_distance(circle[0], circle[1], ax, ay, bx, by)-radius-circle[2]
    x, y, heading = _trace_pose(b, t)
    radius_b = .5*width_b
    half_b = max(.5*length_b-radius_b, 0.)
    dx, dy = half_b*math.cos(heading), half_b*math.sin(heading)
    return _segment_distance(ax, ay, bx, by, x-dx, y-dy, x+dx, y+dy)-radius-radius_b


@njit(cache=True)
def _swept_interval(a, b, dimensions, world, kind, circle,
                    start, span, motion_bound, left, right, fa, fb, tolerance):
    bound = motion_bound*(right-left)
    if (fa+fb-bound)/2. > tolerance:
        return False
    middle = (left+right)/2.
    fm = _path_clearance(a, b, start+middle*span, dimensions, world, kind, circle)
    if fm <= tolerance:
        return True
    if bound <= tolerance:
        return False
    return (_swept_interval(a, b, dimensions, world, kind, circle, start, span,
            motion_bound, left, middle, fa, fm, tolerance) or
        _swept_interval(a, b, dimensions, world, kind, circle, start, span,
            motion_bound, middle, right, fm, fb, tolerance))


@njit(cache=True)
def path_contact_kernel(a, b, times, dimensions, world, kind, circle, tolerance=1.e-8):
    """Original continuous capsule test, including adaptive interval bisection.

    All original integration segments and the 1e-8 contact tolerance remain.
    This kernel only removes Python closures and temporary geometry arrays.
    """
    for index in range(len(times)-1):
        start, end = times[index], times[index+1]
        x0, y0, h0 = _trace_pose(a, start)
        x1, y1, h1 = _trace_pose(a, end)
        angle = (h1-h0+math.pi) % (2.*math.pi)-math.pi
        bound = math.hypot(x1-x0, y1-y0)+.5*dimensions[0]*abs(angle)
        if kind == 2:
            x0, y0, h0 = _trace_pose(b, start)
            x1, y1, h1 = _trace_pose(b, end)
            angle = (h1-h0+math.pi) % (2.*math.pi)-math.pi
            bound += math.hypot(x1-x0, y1-y0)+.5*dimensions[2]*abs(angle)
        fa = _path_clearance(a, b, start, dimensions, world, kind, circle)
        fb = _path_clearance(a, b, end, dimensions, world, kind, circle)
        if min(fa, fb) <= 0.:
            return True
        if _swept_interval(a, b, dimensions, world, kind, circle, start, end-start,
                bound, 0., 1., fa, fb, tolerance):
            return True
    return False


@njit(cache=True)
def _clearance(x, y, heading, seconds, length, width, world, reefs, obstacles):
    radius = .5*width
    half = max(.5*length-radius, 0.)
    dx, dy = half*math.cos(heading), half*math.sin(heading)
    ax, ay, bx, by = x-dx, y-dy, x+dx, y+dy
    result = min(ax-radius, bx-radius, ay-radius, by-radius,
                 world[0]-ax-radius, world[0]-bx-radius,
                 world[1]-ay-radius, world[1]-by-radius)
    for j in range(len(reefs)):
        result = min(result, _point_distance(reefs[j, 0], reefs[j, 1], ax, ay, bx, by)-radius-reefs[j, 2])
    for j in range(len(obstacles)):
        vx, vy = seconds*obstacles[j, 2], seconds*obstacles[j, 3]
        # NumPy preserves float32 for array * Python scalar. Reproduce that
        # rounding for received float32 positions/velocities.
        if obstacles[j, 6]:
            vx = float(np.float32(np.float32(seconds)*np.float32(obstacles[j, 2])))
            vy = float(np.float32(np.float32(seconds)*np.float32(obstacles[j, 3])))
        px, py = obstacles[j, 0]+vx, obstacles[j, 1]+vy
        if obstacles[j, 7] and obstacles[j, 6]:
            px, py = float(np.float32(px)), float(np.float32(py))
        result = min(result, _point_distance(px, py, ax, ay, bx, by)-radius-obstacles[j, 4]-obstacles[j, 5])
    return result


@njit(cache=True)
def assess_motion(q, action, damping, mass, inverse, limits, dt, substeps,
                  length, width, world, reefs, obstacles, horizon, backup_steps,
                  stop_speed, stop_yaw_rate, margin, clearance_cutoff=-np.inf):
    """Constant candidate followed by bounded feedback braking to near rest.

    The backup is a feasibility witness under the local zero-wave/CV model,
    not a proof against unobserved disturbances or unknown vessels.
    """
    minimum = _clearance(q[0], q[1], q[5], 0., length, width, world, reefs, obstacles)
    # Running minimum cannot rise. Stop only after this trajectory is unsafe
    # AND cannot beat the incumbent fallback. The default retains full detail;
    # +inf is used only when the caller needs a boolean safety decision.
    if minimum < margin and minimum <= clearance_cutoff:
        return minimum, False, 0.
    max_peer_speed = 0.
    for j in range(len(obstacles)):
        max_peer_speed = max(max_peer_speed, math.hypot(obstacles[j, 2], obstacles[j, 3]))
    tube = (math.hypot(limits[1], limits[2])+max_peer_speed)*(horizon+backup_steps)*dt+length
    # Only use this certificate when every bounded trajectory (including the
    # backup) lies in free space. Otherwise integrate the unchanged dynamics.
    if not backup_steps and minimum-tube >= margin:
        return minimum-tube, True, 0.
    current = q.astype(np.float64)
    trace = np.empty((substeps+1, 3))
    recovered = not backup_steps
    duration = 0.
    for step in range(horizon+backup_steps):
        control = action
        if step >= horizon:
            control = brake_action(current, damping, mass, limits, dt)
        _step_state_into(current, control, damping, mass, inverse, limits, dt, substeps, current, trace)
        previous_clearance = _clearance(trace[0, 0], trace[0, 1], trace[0, 2],
            step*dt, length, width, world, reefs, obstacles)
        for k in range(substeps):
            t0 = (step+k/substeps)*dt
            t1 = (step+(k+1)/substeps)*dt
            p0, p1 = trace[k], trace[k+1]
            angle = (p1[2]-p0[2]+math.pi) % (2.*math.pi)-math.pi
            bound = math.hypot(p1[0]-p0[0], p1[1]-p0[1])+.5*length*abs(angle)+max_peer_speed*(t1-t0)
            following_clearance = _clearance(p1[0], p1[1], p1[2], t1,
                length, width, world, reefs, obstacles)
            lower = min(previous_clearance, following_clearance)-.5*bound
            minimum = min(minimum, lower)
            if minimum < margin and minimum <= clearance_cutoff:
                return minimum, False, 0.
            previous_clearance = following_clearance
        if step >= horizon:
            duration = (step-horizon+1)*dt
            if math.hypot(current[2], current[3]) <= stop_speed and abs(current[4]) <= stop_yaw_rate:
                recovered = True
                break
    return minimum, recovered, duration


@njit(cache=True)
def _rectangle_distance(px, py, low_x, low_y, high_x, high_y):
    return math.hypot(max(low_x-px, 0., px-high_x), max(low_y-py, 0., py-high_y))


@njit(cache=True)
def _full_motion_traces(q, action, damping, mass, inverse, limits, dt, substeps,
                        horizon, backup_steps, stop_speed, stop_yaw_rate):
    """Record the unchanged candidate and feedback-braking integration once."""
    traces = np.empty((horizon+backup_steps, substeps+1, 3))
    current = q.astype(np.float64)
    recovered = not backup_steps
    duration = 0.
    count = 0
    for step in range(horizon+backup_steps):
        control = action if step < horizon else brake_action(current, damping, mass, limits, dt)
        _step_state_into(current, control, damping, mass, inverse, limits, dt, substeps, current, traces[step])
        count += 1
        if step >= horizon:
            duration = (step-horizon+1)*dt
            if math.hypot(current[2], current[3]) <= stop_speed and abs(current[4]) <= stop_yaw_rate:
                recovered = True
                break
    return traces[:count], recovered, duration


@njit(cache=True)
def _trajectory_enclosure(q, traces, dt, substeps, length, width, world, reefs, obstacles, margin):
    """Lower bound over EVERY recorded substep, including moving peers."""
    radius = max(.5*length, .5*width)
    # The enclosure only grows. If its initial circle is inconclusive, skip
    # the enclosure work and use the original exact geometric assessment.
    initial = min(q[0], q[1], world[0]-q[0], world[1]-q[1])-radius
    for j in range(len(reefs)):
        initial = min(initial, math.hypot(q[0]-reefs[j, 0], q[1]-reefs[j, 1])-radius-reefs[j, 2])
    relative = np.empty((len(obstacles), 4))
    for j in range(len(obstacles)):
        # Preserve discontinuous float32 packet propagation via the full path.
        if obstacles[j, 6] or obstacles[j, 7]:
            return -np.inf
        x, y = q[0]-obstacles[j, 0], q[1]-obstacles[j, 1]
        initial = min(initial, math.hypot(x, y)-radius-obstacles[j, 4]-obstacles[j, 5])
        relative[j, 0], relative[j, 1] = x, y
        relative[j, 2], relative[j, 3] = x, y
    if initial < margin+1.e-8:
        return -np.inf
    peer_speed = 0.
    for j in range(len(obstacles)):
        peer_speed = max(peer_speed, math.hypot(obstacles[j, 2], obstacles[j, 3]))
    low_x = high_x = q[0]
    low_y = high_y = q[1]
    max_bound = 0.
    for step in range(len(traces)):
        trace = traces[step]
        for k in range(substeps+1):
            point = trace[k]
            seconds = (step+k/substeps)*dt
            low_x, low_y = min(low_x, point[0]), min(low_y, point[1])
            high_x, high_y = max(high_x, point[0]), max(high_y, point[1])
            for j in range(len(obstacles)):
                x = point[0]-(obstacles[j, 0]+seconds*obstacles[j, 2])
                y = point[1]-(obstacles[j, 1]+seconds*obstacles[j, 3])
                relative[j, 0], relative[j, 1] = min(relative[j, 0], x), min(relative[j, 1], y)
                relative[j, 2], relative[j, 3] = max(relative[j, 2], x), max(relative[j, 3], y)
            if k:
                previous = trace[k-1]
                angle = (point[2]-previous[2]+math.pi) % (2.*math.pi)-math.pi
                t0 = (step+(k-1)/substeps)*dt
                bound = math.hypot(point[0]-previous[0], point[1]-previous[1])+.5*length*abs(angle)+peer_speed*(seconds-t0)
                max_bound = max(max_bound, bound)
    lower = min(low_x, low_y, world[0]-high_x, world[1]-high_y)-radius
    for j in range(len(reefs)):
        lower = min(lower, _rectangle_distance(reefs[j, 0], reefs[j, 1],
            low_x, low_y, high_x, high_y)-radius-reefs[j, 2])
    for j in range(len(obstacles)):
        lower = min(lower, _rectangle_distance(0., 0., relative[j, 0], relative[j, 1],
            relative[j, 2], relative[j, 3])-radius-obstacles[j, 4]-obstacles[j, 5])
    # Outward numerical guard; a strict test never rounds a boundary into safety.
    lower -= .5*max_bound+1.e-8
    return lower


@njit(cache=True)
def _radial_motion_enclosure(q, traces, limits, dt, substeps,
                             length, width, world, reefs, obstacles, margin):
    """Cheap sufficient bound around the FULL integrated braking trajectory.

    One own-trajectory scan replaces a scan per moving obstacle. Peers are
    bounded by their entire CV segment, so crossings between sampled times
    cannot be missed. An inconclusive bound falls through to the tighter box
    certificate and then the original swept-capsule checks.
    """
    radius = max(.5*length, .5*width)
    x, y = q[0], q[1]
    lower = min(x, y, world[0]-x, world[1]-y)-radius
    for j in range(len(reefs)):
        lower = min(lower, math.hypot(x-reefs[j, 0], y-reefs[j, 1])-radius-reefs[j, 2])
    duration = len(traces)*dt
    peer_speed = 0.
    for j in range(len(obstacles)):
        # The reference explicitly reproduces float32 packet propagation.
        # Its discontinuities retain the existing exact fallback.
        if obstacles[j, 6] or obstacles[j, 7]:
            return -np.inf
        px, py, vx, vy = obstacles[j, :4]
        peer_speed = max(peer_speed, math.hypot(vx, vy))
        lower = min(lower, _point_distance(x, y, px, py,
            px+duration*vx, py+duration*vy)-radius-obstacles[j, 4]-obstacles[j, 5])
    if lower < margin+1.e-6:
        return -np.inf
    displacement_squared = 0.
    for step in range(len(traces)):
        for k in range(substeps+1):
            dx, dy = traces[step, k, 0]-x, traces[step, k, 1]-y
            displacement_squared = max(displacement_squared, dx*dx+dy*dy)
    # The exact assessment subtracts half the substep motion bound. Velocity
    # clamps give an upper bound on that same translation/rotation/CV term.
    h = dt/substeps
    speed = math.hypot(max(abs(limits[0]), abs(limits[1])), limits[2])
    motion = (speed+peer_speed)*h+.5*length*min(limits[3]*h, math.pi)
    return lower-math.sqrt(displacement_squared)-.5*motion-1.e-6


@njit(cache=True)
def certify_free_motion(q, action, damping, mass, inverse, limits, dt, substeps,
                        length, width, world, reefs, obstacles, horizon, backup_steps,
                        stop_speed, stop_yaw_rate, margin):
    """Sufficient free-space certificate using the full original braking tail."""
    traces, recovered, duration = _full_motion_traces(q, action, damping, mass, inverse,
        limits, dt, substeps, horizon, backup_steps, stop_speed, stop_yaw_rate)
    lower = _trajectory_enclosure(q, traces, dt, substeps, length, width, world, reefs, obstacles, margin)
    return recovered and lower >= margin, lower, duration


@njit(cache=True)
def assess_nominal_motion(q, action, damping, mass, inverse, limits, dt, substeps,
                        length, width, world, reefs, obstacles, horizon, backup_steps,
                        stop_speed, stop_yaw_rate, margin, decision_only=False):
    """Full nominal assessment with a sufficient geometry shortcut.

    Every original physical substep and the entire feedback braking tail are
    integrated once. A conservative enclosure may certify a recovered safe
    nominal. Otherwise the ORIGINAL geometric minimum is evaluated on those
    same traces. Unsafe fallbacks retain their exact ranking and slack.
    Boolean screening may stop once rejection is known; its partial clearance
    must not be exposed as a full diagnostic or used for fallback ranking.
    """
    if decision_only:
        initial = _clearance(q[0], q[1], q[5], 0., length, width, world, reefs, obstacles)
        if initial < margin:
            return initial, False, 0.
    if not backup_steps:
        return assess_motion(q, action, damping, mass, inverse, limits, dt, substeps,
            length, width, world, reefs, obstacles, horizon, backup_steps,
            stop_speed, stop_yaw_rate, margin, np.inf if decision_only else -np.inf)
    traces, recovered, duration = _full_motion_traces(q, action, damping, mass, inverse,
        limits, dt, substeps, horizon, backup_steps, stop_speed, stop_yaw_rate)
    if decision_only and not recovered:
        return -np.inf, False, duration
    if recovered:
        lower = _radial_motion_enclosure(q, traces, limits, dt, substeps,
            length, width, world, reefs, obstacles, margin)
        if lower >= margin:
            return lower, True, duration
        lower = _trajectory_enclosure(q, traces, dt, substeps, length, width, world, reefs, obstacles, margin)
        if lower >= margin:
            return lower, True, duration
    # Reuse the integrated path when the enclosure cannot prove safety.
    minimum = _clearance(q[0], q[1], q[5], 0., length, width, world, reefs, obstacles)
    max_peer_speed = 0.
    for j in range(len(obstacles)):
        max_peer_speed = max(max_peer_speed, math.hypot(obstacles[j, 2], obstacles[j, 3]))
    for step in range(len(traces)):
        trace = traces[step]
        previous_clearance = _clearance(trace[0, 0], trace[0, 1], trace[0, 2],
            step*dt, length, width, world, reefs, obstacles)
        for k in range(substeps):
            t0 = (step+k/substeps)*dt
            t1 = (step+(k+1)/substeps)*dt
            p0, p1 = trace[k], trace[k+1]
            angle = (p1[2]-p0[2]+math.pi) % (2.*math.pi)-math.pi
            bound = math.hypot(p1[0]-p0[0], p1[1]-p0[1])+.5*length*abs(angle)+max_peer_speed*(t1-t0)
            following_clearance = _clearance(p1[0], p1[1], p1[2], t1,
                length, width, world, reefs, obstacles)
            minimum = min(minimum, min(previous_clearance, following_clearance)-.5*bound)
            if decision_only and minimum < margin:
                return minimum, False, 0.
            previous_clearance = following_clearance
    return minimum, recovered, duration
