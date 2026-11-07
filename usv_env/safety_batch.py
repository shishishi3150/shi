"""Batch implementation of the receiver-local I1 filter.

Independent trajectories run on CPU workers. Candidate order, force dynamics,
integration substeps, recovery tests and local information match the scalar
projector. The scalar path remains useful for small batches and as an oracle.
The compiled motion kernel reuses caller-owned integration buffers. Keep this
caller source in sync with safety_kernel changes to invalidate its Numba cache.
The nominal path first tries a radial bound over the complete braking trace.
"""
import math
from collections import OrderedDict
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads

from .history import schema_for_counts
from .safety_kernel import assess_motion, assess_nominal_motion, brake_action


@njit(cache=True)
def _project_motion(q, nominal, actuator_limits, damping, mass, inverse, limits,
                    dt, substeps, length, width, world, reefs, obstacles,
                    horizon, backup_steps, stop_speed, stop_yaw, margin,
                    levels, reject_only, decision_only=False):
    nominal = np.minimum(np.maximum(nominal, -actuator_limits), actuator_limits)
    if not reject_only:
        initial, recovered, seconds = assess_nominal_motion(q, nominal, damping, mass, inverse,
            limits, dt, substeps, length, width, world, reefs, obstacles,
            horizon, backup_steps, stop_speed, stop_yaw, margin)
    elif decision_only:
        initial, recovered, seconds = assess_nominal_motion(q, nominal, damping, mass, inverse,
            limits, dt, substeps, length, width, world, reefs, obstacles,
            horizon, backup_steps, stop_speed, stop_yaw, margin, True)
    else:
        initial, recovered, seconds = assess_motion(q, nominal, damping, mass, inverse,
            limits, dt, substeps, length, width, world, reefs, obstacles,
            horizon, backup_steps, stop_speed, stop_yaw, margin,
            -np.inf)
    safe = initial >= margin and recovered
    # Columns match the scalar project_one diagnostics (except wall time).
    if safe or reject_only:
        return nominal, np.array([1.*safe, 0., 0., 1.*(not safe),
            max(0., margin-initial), initial, 1., 1.*recovered, seconds, 0.])
    size = levels*levels+2+int(backup_steps > 0)
    candidates = np.empty((size, 2))
    force = np.linspace(-actuator_limits[0], actuator_limits[0], levels)
    yaw = np.linspace(-actuator_limits[1], actuator_limits[1], levels)
    cursor = 0
    for f in force:
        for y in yaw:
            candidates[cursor, 0], candidates[cursor, 1] = f, y
            cursor += 1
    candidates[cursor, 0], candidates[cursor, 1] = -actuator_limits[0], nominal[1]
    candidates[cursor+1, 0], candidates[cursor+1, 1] = 0., nominal[1]
    if backup_steps:
        candidates[-1] = brake_action(q, damping, mass, limits, dt)
    distances = np.empty(size)
    for j in range(size):
        df = (candidates[j, 0]-nominal[0])/actuator_limits[0]
        dy = (candidates[j, 1]-nominal[1])/actuator_limits[1]
        distances[j] = df*df+dy*dy
    # Stable insertion sort retains the scalar filter's tie-breaking order.
    for j in range(1, size):
        candidate, distance = candidates[j].copy(), distances[j]
        k = j
        while k > 0 and distances[k-1] > distance:
            distances[k], candidates[k] = distances[k-1], candidates[k-1]
            k -= 1
        distances[k], candidates[k] = distance, candidate
    best, clearance = nominal.copy(), initial
    best_recovered, best_seconds = recovered, seconds
    evaluated = 1
    for candidate in candidates:
        value, recovered, seconds = assess_motion(q, candidate, damping, mass,
            inverse, limits, dt, substeps, length, width, world, reefs, obstacles,
            horizon, backup_steps, stop_speed, stop_yaw, margin, clearance)
        evaluated += 1
        if value > clearance:
            best, clearance = candidate.copy(), value
            best_recovered, best_seconds = recovered, seconds
        if value >= margin and recovered:
            df, dy = (candidate-nominal)/actuator_limits
            return candidate.copy(), np.array([1., 1., math.sqrt(df*df+dy*dy),
                0., 0., value, float(evaluated), 1.*recovered, seconds, 0.])
    df, dy = (best-nominal)/actuator_limits
    changed = best[0] != nominal[0] or best[1] != nominal[1]
    return best, np.array([0., 1.*changed, math.sqrt(df*df+dy*dy), 1.,
        max(0., margin-clearance), clearance, float(evaluated),
        1.*best_recovered, best_seconds, 1.])


@njit(cache=True, parallel=True)
def _project_many(own, obstacles, present, active, selected, roles, actions,
                  actuator_limits, damping, mass, inverse, limits, dt, substeps,
                  dimensions, world, reefs, horizon, backup_steps, stop_speed,
                  stop_yaw, margin, levels, reject_only, decision_only):
    count = len(own)
    result = actions.copy()
    diagnostics = np.zeros((count, 10))
    for row in prange(count):
        if not selected[row] or not active[row]:
            diagnostics[row, 0] = 1.
            diagnostics[row, 5] = max(world[0], world[1])
            diagnostics[row, 7] = 1.
            if selected[row] and not active[row]:
                result[row] = 0.
            continue
        role = roles[row]
        packed = np.empty((int(present[row].sum()), 8))
        cursor = 0
        for j in range(obstacles.shape[1]):
            if present[row, j]:
                packed[cursor] = obstacles[row, j]
                cursor += 1
        result[row], diagnostics[row] = _project_motion(own[row], actions[row],
            actuator_limits[role], damping[role], mass[role], inverse[role],
            limits[role], dt, substeps[role], dimensions[role, 0],
            dimensions[role, 1], world, reefs, packed, horizon, backup_steps,
            stop_speed, stop_yaw, margin, levels, reject_only, decision_only)
    return result, diagnostics


@njit(cache=True)
def _receiver_rotations(rows):
    """Same double-precision scalar trigonometry without Python row loops."""
    heading = np.empty(len(rows), np.float64)
    cosine = np.empty(len(rows), np.float64)
    sine = np.empty(len(rows), np.float64)
    for row in range(len(rows)):
        angle = math.atan2(np.float64(rows[row, 6]), np.float64(rows[row, 7]))
        heading[row], cosine[row], sine[row] = angle, math.cos(angle), math.sin(angle)
    return heading, cosine, sine


def _inputs(cfg, state, frames, selected):
    """Decode every obstacle from that receiver's observation only."""
    batch, count = frames.shape[:2]
    surface_count = cfg['env']['num_surface_blue']
    schema = schema_for_counts(count)
    f = frames.reshape(-1, frames.shape[-1]).astype(np.float64)
    rows = state[:, :9*count].reshape(-1, 9)
    scale = max(cfg['env']['world_size'])
    roles = np.tile(np.arange(count) >= surface_count, batch).astype(np.int64)
    active = rows[:, 0] != 0
    selection = np.tile(np.array([i in selected for i in range(count)]), batch)
    # Same float32 position boundary and float64 local reconstruction as scalar.
    pos = (rows[:, 2:4]*scale).astype(np.float32)
    heading, cosine, sine = _receiver_rotations(rows)
    rotation = np.empty((len(f), 2, 2))
    rotation[:, 0, 0] = rotation[:, 1, 1] = cosine
    rotation[:, 0, 1], rotation[:, 1, 0] = -sine, sine
    vmax = np.array([cfg['ship']['max_speed_blue'], cfg['ship']['max_speed_submarine']])[roles]
    surge = f[:, schema.index('surge')]*vmax
    sway = f[:, schema.index('sway')]*cfg['ship']['max_sway_speed']
    velocity = np.stack((np.cos(heading)*surge-np.sin(heading)*sway,
                         np.sin(heading)*surge+np.cos(heading)*sway), 1).astype(np.float32)
    own = np.column_stack((pos, surge, sway,
        f[:, schema.index('yaw_rate')]*cfg['ship']['max_yaw_rate'], heading))
    obstacles = np.zeros((len(f), count, 8))
    present = np.zeros((len(f), count), bool)
    d = cfg['depth']
    separated = d['auv_depth_m']-d['surface_keel_depth_m'] > d['auv_vertical_radius_m']+d['vertical_safety_margin_m']
    for j in range(count):
        if j < count-1:
            prefix = f'peer_{j}.'
            present[:, j] = f[:, schema.index(prefix+'active')] != 0
            surface = f[:, schema.index(prefix+'surface')] != 0
        else:
            prefix = 'estimated_target_' if 'estimated_target_x' in schema.fields else 'red_'
            present[:, j] = f[:, schema.index('target_valid')] != 0
            surface = np.ones(len(f), bool)
        if separated:
            present[:, j] &= surface == (roles == 0)
        local_position = f[:, [schema.index(prefix+'x'), schema.index(prefix+'y')]]
        local_velocity = f[:, [schema.index(prefix+'vx'), schema.index(prefix+'vy')]]
        obstacles[:, j, :2] = pos+(rotation@local_position[:, :, None])[:, :, 0]*scale
        obstacles[:, j, 2:4] = velocity+(rotation@local_velocity[:, :, None])[:, :, 0]*1.5
        obstacles[:, j, 4] = .5*np.where(surface,
            cfg['ship']['surface_length_m'], cfg['ship']['submarine_collision_length_m'])
        obstacles[:, j, 5] = cfg['i1']['uncertainty_margin_m']
        # Reconstructed positions and velocities above are float64 in both paths.
    return own, obstacles, present, active, selection, roles


def project_batch(predictor, cfg, state, frames, actions, selected, reject_only=False,
                  decision_only=False):
    if decision_only and not reject_only:
        raise ValueError('decision_only requires reject_only')
    inputs = _inputs(cfg, state, frames, selected)
    ship, settings = cfg['ship'], cfg['i1']
    data = predictor.model_data
    actuator_limits = np.array([[ship['max_surge_force'], ship['max_yaw_moment']],
        [ship['submarine_dynamics']['max_surge_force'], ship['submarine_dynamics']['max_yaw_moment']]])
    dimensions = np.array([[ship['surface_length_m'], ship['surface_collision_width_m']],
        [ship['submarine_collision_length_m'], ship['submarine_collision_width_m']]])
    own, obstacles, present, active, selection, roles = inputs
    flat_actions = actions.reshape(-1, 2).astype(np.float64)
    result = flat_actions.copy()
    detail = np.zeros((len(own), 10))
    detail[:, 0] = detail[:, 7] = 1.
    detail[:, 5] = max(cfg['env']['world_size'])
    result[selection & ~active] = 0.
    jobs = np.flatnonzero(selection & active)
    cache = getattr(predictor, '_batch_projection_cache', None)
    if cache is None:
        cache = predictor._batch_projection_cache = OrderedDict()
        predictor._batch_cache_hits = predictor._batch_cache_misses = 0
    # A projection depends only on this receiver, its received obstacles and
    # this action. Exact byte keys retain the scalar filter's float semantics.
    records = np.ascontiguousarray(np.column_stack((own[jobs],
        obstacles[jobs].reshape(len(jobs), obstacles.shape[1]*8), present[jobs], roles[jobs],
        np.clip(flat_actions[jobs], -actuator_limits[roles[jobs]], actuator_limits[roles[jobs]]))))
    missing, keys, aliases, pending = [], [], [], {}
    for row, record in zip(jobs, records):
        key = (reject_only, decision_only, record.tobytes())
        if key in cache:
            result[row], detail[row] = cache[key]
            cache.move_to_end(key)
            predictor._batch_cache_hits += 1
        elif key in pending:
            aliases.append((row, pending[key]))
            predictor._batch_cache_hits += 1
        else:
            pending[key] = row
            missing.append(row); keys.append(key)
            predictor._batch_cache_misses += 1
    missing = np.asarray(missing, np.int64)
    previous_threads = get_num_threads()
    # A small bounded pool avoids oversubscribing CPU/GPU work on laptops.
    set_num_threads(min(8, previous_threads))
    try:
        if len(missing):
            computed, diagnostics = _project_many(*(value[missing] for value in inputs),
                flat_actions[missing], actuator_limits,
                *(np.stack([role[i] for role in data]) for i in range(4)),
                cfg['env']['dt'], np.array([m.internal_substeps for m in predictor.models]),
                dimensions, predictor.world, predictor.reefs, int(settings['horizon_steps']),
                int(settings.get('backup_steps', 0)), settings.get('stop_speed_mps', .02),
                settings.get('stop_yaw_rate_rps', .02), settings['margin_m'],
                int(settings.get('candidate_levels', 3)), reject_only, decision_only)
            result[missing], detail[missing] = computed, diagnostics
            for key, output, diagnostic in zip(keys, computed, diagnostics):
                cache[key] = (output.copy(), diagnostic.copy())
                if len(cache) > 32768:
                    cache.popitem(last=False)
        for row, source in aliases:
            result[row], detail[row] = result[source], detail[source]
    finally:
        set_num_threads(previous_threads)
    detail = detail.reshape(*actions.shape[:2], -1)
    diagnostics = dict(safe_mask=detail[:, :, 0].astype(bool).all(1),
        intervention=detail[:, :, 1].astype(bool).any(1),
        projection_norm=detail[:, :, 2].sum(1),
        infeasible=detail[:, :, 3].astype(bool).any(1), slack=detail[:, :, 4].max(1))
    if decision_only:
        # Pruned failures do not have a complete minimum/slack. Do not expose
        # them as full diagnostics or reuse them through the full-detail memo.
        diagnostics = {'safe_mask': diagnostics['safe_mask']}
    return result.reshape(actions.shape).astype(np.float32), diagnostics
