"""Positive task earnings with explicit historical cost/PBRS compatibility.

Truth is used only here for training rewards. Task gates and Actor inputs are
unchanged. Accidents earn zero and cannot escape a negative living cost.
Historical schemes retain their original equations and recorded settings.
"""
import math
import numpy as np
from .geometry import point_segment_distance, segment_distance
from .task import encirclement_geometry

PBRS_COMPONENTS = ('auv_potential_reward', 'auv_individual_progress',
    'usv_potential_reward', 'usv_shapley_credit', 'discovery',
    'safety_penalty', 'action_penalty', 'yaw_rate')
COST_COMPONENTS = ('auv_distance_cost', 'auv_individual_cost', 'auv_gap_cost',
    'auv_hull_cost', 'auv_tracking_cost', 'auv_hold_cost', 'usv_approach_cost',
    'usv_blocking_cost', 'usv_support_cost', 'usv_shapley_cost', 'time_cost')
POSITIVE_AUV_COMPONENTS = ('auv_individual_reward', 'auv_distance_reward',
    'auv_gap_reward', 'auv_hull_reward', 'auv_tracking_reward', 'auv_hold_reward')
POSITIVE_USV_COMPONENTS = ('usv_approach_reward', 'usv_blocking_reward',
    'usv_support_reward', 'usv_contribution_reward')
POSITIVE_V2_AUV_COMPONENTS = POSITIVE_AUV_COMPONENTS + ('auv_velocity_reward', 'auv_heading_reward')
POSITIVE_V2_USV_COMPONENTS = POSITIVE_USV_COMPONENTS + ('usv_interception_position_reward',)
POSITIVE_COMPONENTS = POSITIVE_AUV_COMPONENTS + POSITIVE_USV_COMPONENTS + (
    'usv_withdrawal_reward', 'capture_bonus', 'auv_velocity_reward', 'auv_heading_reward',
    'usv_interception_position_reward')
REWARD_COMPONENTS = PBRS_COMPONENTS + COST_COMPONENTS + POSITIVE_COMPONENTS
FAILURE_REASONS = ('breakthrough', 'collision', 'reef_collision', 'out_of_bounds',
                   'timeout', 'scenario_anomaly', 'numerical_error')
REWARD_DIAGNOSTICS = ('auv_distance_quality', 'auv_gap_quality', 'auv_hull_quality',
    'auv_tracking_quality', 'auv_hold_quality', 'auv_potential',
    'usv_counterfactual_gap', 'usv_readiness', 'usv_potential')
# Zero columns for old plots, NOT paid aliases that would double-count sums.
LEGACY_ZERO_COMPONENTS = ('blocking', 'usv_contribution', 'search', 'tracking',
    'observation_geometry', 'encirclement', 'information', 'time', 'safety', 'action')


def capsule_clearance(left, right):
    return segment_distance(left[0], left[1], right[0], right[1])-left[2]-right[2]


def boundary_clearance(capsule, world_size):
    points = np.stack(capsule[:2]); radius = capsule[2]
    return float(min(np.min(points-radius), np.min(np.asarray(world_size)-points-radius)))


def validate_settings(cfg, gamma=None, max_steps=None):
    if cfg.get('scheme') not in ('cooperative_pbrs_v1', 'cooperative_cost_v2',
                               'cooperative_positive_v1', 'cooperative_positive_v2'):
        raise ValueError('reward scheme unsupported; historical reward requires its original source')
    positive = {'potential': ('distance_scale_m', 'hull_guidance_m', 'hull_power',
        'tracking_error_scale_m', 'tracking_variance_scale_m2', 'tracking_fresh_seconds'),
        'usv': ('gap_scale_m', 'shapley_scale_m')}
    for group, keys in positive.items():
        for key in keys:
            if cfg[group][key] <= 0: raise ValueError(f'reward.{group}.{key} must be positive')
    for group in ('potential', 'usv', 'auv'):
        for key, value in cfg[group].items():
            if value < 0: raise ValueError(f'reward.{group}.{key} must be nonnegative')
    if not 0 <= cfg['usv']['readiness_floor'] <= 1:
        raise ValueError('reward readiness_floor must be in [0,1]')
    for key in ('distance_mean_fraction', 'tracking_mean_fraction'):
        if not 0 <= cfg['potential'][key] <= 1:
            raise ValueError(f'reward.{key} must be in [0,1]')
    if not np.isclose(sum(cfg['usv'][k] for k in ('readiness_distance_weight',
            'readiness_gap_weight', 'readiness_hull_weight')), 1.):
        raise ValueError('reward readiness weights must sum to one')
    if cfg['scheme'] in ('cooperative_positive_v1', 'cooperative_positive_v2'):
        positive = cfg['positive']
        for key in ('step_budget', 'episode_budget', 'distance_scale_m',
                    'usv_distance_scale_m', 'full_clearance_m'):
            if not np.isfinite(positive[key]) or positive[key] <= 0:
                raise ValueError(f'reward.positive.{key} must be finite and positive')
        for key in ('capture_bonus', 'usv_minimum_distance_m'):
            if not np.isfinite(positive[key]) or positive[key] < 0:
                raise ValueError(f'reward.positive.{key} must be finite and nonnegative')
        for key in ('action_moderation_fraction', 'yaw_moderation_fraction'):
            if not np.isfinite(positive[key]) or not 0 <= positive[key] < 1:
                raise ValueError(f'reward.positive.{key} must be in [0,1)')
        roles = (('auv', POSITIVE_V2_AUV_COMPONENTS), ('usv', POSITIVE_V2_USV_COMPONENTS)) if cfg['scheme'].endswith('_v2') else (
            ('auv', POSITIVE_AUV_COMPONENTS), ('usv', POSITIVE_USV_COMPONENTS))
        if cfg['scheme'].endswith('_v2'):
            for key in ('closing_speed_mps', 'closing_scale_m', 'velocity_error_scale_mps',
                        'heading_speed_floor_mps', 'motion_clearance_m', 'interception_lateral_scale_m'):
                value = positive[key]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'reward.positive.{key} must be finite and positive')
        for role, names in roles:
            weights = positive[role+'_weights']
            if (set(weights) != set(names) or not all(np.isfinite(v) and v >= 0 for v in weights.values())
                    or not np.isclose(sum(weights.values()), 1., rtol=0., atol=1e-12)):
                raise ValueError(f'reward.positive.{role}_weights must be nonnegative and sum to one')
        terminal = cfg['terminal']
        if (set(terminal) != {'success', *FAILURE_REASONS}
                or not np.isfinite(terminal['success']) or terminal['success'] <= 0
                or any(terminal[k] != 0. for k in FAILURE_REASONS)):
            raise ValueError('reward positive scheme requires success>0 and all failure rewards=0')
        if max_steps is not None:
            bound = max_steps*positive['step_budget']+positive['capture_bonus']+terminal['success']
            if bound > positive['episode_budget']+1e-10:
                raise ValueError('reward positive process and milestone budgets exceed episode_budget')
    if cfg['scheme'] == 'cooperative_cost_v2':
        cost = cfg['cost']
        for key in ('step_budget', 'episode_budget', 'distance_scale_m', 'usv_distance_scale_m'):
            if not np.isfinite(cost[key]) or cost[key] <= 0:
                raise ValueError(f'reward.cost.{key} must be finite and positive')
        common = {'time_cost', 'safety_penalty', 'action_penalty', 'yaw_rate'}
        roles = {'auv': set(COST_COMPONENTS[:6]) | common,
                 'usv': set(COST_COMPONENTS[6:10]) | common}
        for role, names in roles.items():
            weights = cost[role+'_weights']
            if (set(weights) != names or not all(np.isfinite(v) and v >= 0 for v in weights.values())
                    or not np.isclose(sum(weights.values()), 1., rtol=0., atol=1e-12)):
                raise ValueError(f'reward.cost.{role}_weights must be nonnegative and sum to one')
        terminal = cfg['terminal']
        if set(terminal) != {'success', *FAILURE_REASONS} or terminal['success'] != 0:
            raise ValueError('reward cost scheme requires success=0 and all failure reasons')
        if any(not np.isfinite(terminal[k]) or terminal[k] >= 0 for k in FAILURE_REASONS):
            raise ValueError('reward cost scheme requires negative failure penalties')
        if gamma is not None:
            future_bound = cost['step_budget'] / (1.-gamma)
            if min(-terminal[k] for k in FAILURE_REASONS) + 1e-10 < future_bound:
                raise ValueError('reward failure penalty must cover the discounted future cost; early failure could escape costs')
        if max_steps is not None:
            bound = max_steps*cost['step_budget'] + max(-terminal[k] for k in FAILURE_REASONS)
            if bound > cost['episode_budget'] + 1e-10:
                raise ValueError('reward process and failure budgets exceed episode_budget')


def distance_quality(radii, minimum, maximum, scale):
    violation = np.maximum(minimum-np.asarray(radii), 0.) + np.maximum(np.asarray(radii)-maximum, 0.)
    return np.exp(-violation/scale)


def smooth_distance_quality(radii, minimum, maximum, scale):
    """Non-saturating far-field guidance; the real capture band is unchanged."""
    violation = np.maximum(minimum-np.asarray(radii), 0.) + np.maximum(np.asarray(radii)-maximum, 0.)
    return 1. / (1. + violation/scale)


def gap_quality(gap, maximum):
    return float(np.clip((2*math.pi-gap)/(2*math.pi-maximum), 0., 1.))


def hull_quality(margin, minimum, guidance, power):
    return float(np.clip((margin+guidance)/(guidance+minimum), 0., 1.)**power)


def potentials(env, geometry=None):
    """Pure current-state evaluation, shared by both transition endpoints.

    Incomplete AUV teams have zero team potential/readiness. Remaining AUVs
    retain only their bounded individual distance potential. No two-boat hull
    is substituted for the original three-boat task.
    """
    cfg = env.config['reward']; task = env.config['task']; p = cfg['potential']
    active = env.agent_active_mask(); auvs = active[env.num_surface_blue:]
    positions = np.asarray([s.pos for s in env.submarine_states], dtype=float)
    radii = np.linalg.norm(positions-env.red_state.pos, axis=1)
    if cfg['scheme'] in ('cooperative_cost_v2', 'cooperative_positive_v1', 'cooperative_positive_v2'):
        settings = cfg['cost'] if cfg['scheme'] == 'cooperative_cost_v2' else cfg['positive']
        individual = smooth_distance_quality(radii, task['capture_min_radius_m'],
            task['capture_max_radius_m'], settings['distance_scale_m']) * auvs
    else:
        individual = distance_quality(radii, task['capture_min_radius_m'],
            task['capture_max_radius_m'], p['distance_scale_m']) * auvs
    complete = bool(np.all(auvs) and len(auvs) == 3)
    result = {key: 0. for key in REWARD_DIAGNOSTICS}
    result['auv_individual_quality'] = individual
    result['auv_local_tracking_quality'] = np.zeros(len(auvs))
    if complete:
        if geometry is None:
            geometry = encirclement_geometry(env.red_state.pos, positions)
        mix = p['distance_mean_fraction']
        d = mix*float(individual.mean()) + (1.-mix)*float(individual.min())
        g = gap_quality(geometry['maximum_angle_gap_rad'], math.radians(task['maximum_observer_angle_gap_degrees']))
        h = hull_quality(geometry['hull_margin_m'], task['capture_min_hull_margin_m'],
            p['hull_guidance_m'], p['hull_power'])
        now = env.step_count*env.dt; qualities = []
        for i in range(env.num_surface_blue, env.num_friendly):
            estimate = env.estimators[i].predict(now)
            q = 0.
            if estimate.valid and 0 <= now-estimate.t_obs <= p['tracking_fresh_seconds']:
                error = float(np.linalg.norm(estimate.position-env.red_state.pos))
                variance = max(0., float(np.trace(estimate.covariance[:2, :2])))
                q = math.exp(-error/p['tracking_error_scale_m'])*float(np.clip(estimate.confidence, 0., 1.))
                q /= 1.+variance/p['tracking_variance_scale_m2']
            qualities.append(q)
        mix = p['tracking_mean_fraction']
        q = mix*float(np.mean(qualities)) + (1.-mix)*float(np.min(qualities))
        hold = float(np.clip(env.task_state.hold_steps/env.required_hold_steps, 0., 1.))
        result.update(auv_distance_quality=d, auv_gap_quality=g, auv_hull_quality=h,
            auv_tracking_quality=q, auv_hold_quality=hold,
            auv_local_tracking_quality=np.asarray(qualities),
            auv_potential=p['distance_weight']*d+p['gap_weight']*g+p['hull_weight']*h
                +p['tracking_weight']*q+p['hold_weight']*hold)
        u = cfg['usv']
        result['usv_readiness'] = float(np.clip(u['readiness_distance_weight']*d
            +u['readiness_gap_weight']*g+u['readiness_hull_weight']*h, 0., 1.))
    result['usv_counterfactual_gap'] = float(env.counterfactual_gap)
    if active[:env.num_surface_blue].any():
        u = cfg['usv']; readiness = result['usv_readiness']
        result['usv_potential'] = math.tanh(env.counterfactual_gap/u['gap_scale_m'])*(u['readiness_floor']
            +(1.-u['readiness_floor'])*readiness)
    return result


def snapshot(env):
    result = potentials(env)
    result.update(team_first_detection_step=env.team_first_detection_step,
                  step=env.step_count, active=env.agent_active_mask())
    return result


def evaluate(previous, env, terminal_reason='', *, geometry=None):
    if env.config['reward']['scheme'] in ('cooperative_positive_v1', 'cooperative_positive_v2'):
        return _evaluate_positive(env, terminal_reason, geometry=geometry)
    if env.config['reward']['scheme'] == 'cooperative_cost_v2':
        return _evaluate_cost(env, terminal_reason, geometry=geometry)
    n = env.num_friendly; cfg = env.config['reward']; dt = env.dt
    gamma = float(env.config['training']['gamma'])
    active = env.agent_active_mask(); nu = int(active[:env.num_surface_blue].sum())
    na = int(active[env.num_surface_blue:].sum()); current = potentials(env, geometry)
    following_a = 0. if terminal_reason else current['auv_potential']
    following_u = 0. if terminal_reason else current['usv_potential']
    following_b = np.zeros(env.num_submarines) if terminal_reason else current['auv_individual_quality']
    parts = {name: np.zeros(n, np.float64) for name in REWARD_COMPONENTS}
    audit = env.counterfactual_audit
    for i, ship in enumerate(env.friendly_states()):
        if not active[i]: continue
        role = 'usv' if i < env.num_surface_blue else 'auv'; weights = cfg[role]
        parts['safety_penalty'][i] = -weights['safety']*dt*env._safety_cost(i)
        parts['action_penalty'][i] = -weights['action']*dt*float(np.mean(
            (env.last_executed_actions[i]/env.agent_action_limits[i])**2))
        parts['yaw_rate'][i] = -weights['yaw_rate']*dt*max(0., abs(ship.yaw_rate)
            -weights['yaw_soft_threshold_rps'])/env.max_yaw_rate
        if role == 'usv':
            parts['usv_potential_reward'][i] = weights['potential_weight']/max(nu, 1)*(
                gamma*following_u-previous['usv_potential'])
            if env.counterfactual_audit_new and audit.get('valid', False):
                phi = audit['shapley_effect_m'][i]
                readiness = weights['readiness_floor']+(1.-weights['readiness_floor'])*current['usv_readiness']
                parts['usv_shapley_credit'][i] = weights['shapley_weight']*float(
                    np.clip(phi/weights['shapley_scale_m'], -1., 1.))*readiness
        else:
            j = i-env.num_surface_blue
            parts['auv_potential_reward'][i] = weights['potential_weight']*(gamma*following_a-previous['auv_potential'])
            parts['auv_individual_progress'][i] = weights['individual_weight']*(gamma*following_b[j]
                -previous['auv_individual_quality'][j])
            if previous['team_first_detection_step'] is None and env.team_first_detection_step is not None:
                parts['discovery'][i] = weights['discovery']/max(na, 1)
    dense = sum(parts.values())
    terminal = float(cfg['terminal'].get(terminal_reason, 0.))
    terminal_values = terminal*active
    rewards = (dense+terminal_values).astype(np.float32)
    terms = {name: value.tolist() for name, value in parts.items()}
    terms.update({name: [0.]*n for name in LEGACY_ZERO_COMPONENTS})
    for name in REWARD_DIAGNOSTICS:
        role_mask = active.copy()
        if name.startswith('auv_'): role_mask[:env.num_surface_blue] = 0
        else: role_mask[env.num_surface_blue:] = 0
        terms[name] = (current[name]*role_mask).tolist()
    terms.update(terminal_reward=terminal_values.tolist(), reward_gamma=gamma,
        next_auv_potential=following_a, next_usv_potential=following_u,
        next_auv_individual_quality=following_b.tolist(),
        team_terminal_reward=float(terminal_values.mean()), terminal_reward_per_active_agent=terminal,
        canonical_team_dense_reward=float(dense.mean()),
        reward_component_means={name: float(np.mean(value)) for name,value in parts.items()},
        usv_contributions={name: values[:env.num_surface_blue].tolist() for name,values in parts.items()},
        auv_contributions={name: values[env.num_surface_blue:].tolist() for name,values in parts.items()})
    return rewards, terms


def pursuit_motion_quality(env, ship):
    """AUV velocity/heading credit for a free capture annulus, never a slot.

    Outside the distance band, add bounded radial closing to target velocity;
    inside, match target velocity to keep the enclosure moving. A static-map
    gate suppresses direct pursuit credit through a reef, leaving detours to
    the distance/enclosure terms. No waypoint, controller action or time-based
    curriculum is introduced, and no progress history can be farmed in loops.
    """
    p = env.config['reward']['positive']; task = env.config['task']
    delta = np.asarray(env.red_state.pos, float)-np.asarray(ship.pos, float)
    radius = float(np.linalg.norm(delta))
    direction = delta/max(radius, 1e-12)
    violation = radius-float(np.clip(radius, task['capture_min_radius_m'], task['capture_max_radius_m']))
    reference = np.asarray(env.red_state.velocity, float)+p['closing_speed_mps']*math.tanh(
        violation/p['closing_scale_m'])*direction
    speed = float(np.linalg.norm(reference)); limit = env.config['ship']['max_speed_submarine']
    reference *= min(1., limit/max(speed, 1e-12)); speed = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(np.asarray(ship.velocity, float)-reference))
    velocity = 1./(1.+(error/p['velocity_error_scale_mps'])**2)
    forward = np.array([math.cos(ship.heading), math.sin(ship.heading)])
    heading = .5*(1.+float(forward@reference)/speed) if speed > 1e-12 else 0.
    heading *= min(1., speed/p['heading_speed_floor_mps'])
    endpoint = np.asarray(ship.pos, float)+violation*direction
    hull_radius = .5*env.config['ship']['submarine_collision_length_m']
    shallow = env.config['depth']['reef_shallow_buffer_m']
    clearance = min((point_segment_distance(np.asarray(r['center']), ship.pos, endpoint)
        -r['radius']-shallow-hull_radius for r in env.reefs), default=float('inf'))
    path_gate = float(np.clip(clearance/p['motion_clearance_m'], 0., 1.))
    return dict(velocity=path_gate*float(np.clip(velocity, 0., 1.)),
                heading=path_gate*float(np.clip(heading, 0., 1.)),
                path_gate=path_gate, reference_velocity=reference.tolist())


def interception_position_quality(env, ship):
    """Goal-side geometry is guidance, not a claim of causal blocking."""
    goal = np.asarray(env.protected_center, float)-np.asarray(env.red_state.pos, float)
    magnitude = float(np.linalg.norm(goal))
    if magnitude < 1e-12 or env.capture_success: return 0.
    axis = goal/magnitude
    offset = np.asarray(ship.pos, float)-np.asarray(env.red_state.pos, float)
    distance = float(np.linalg.norm(offset))
    if distance < 1e-12: return 0.
    front = float(np.clip((offset@axis)/distance, 0., 1.))
    lateral = float(np.linalg.norm(offset-(offset@axis)*axis))
    scale = env.config['reward']['positive']['interception_lateral_scale_m']
    return front/(1.+(lateral/scale)**2)


def _evaluate_positive(env, terminal_reason, *, geometry=None):
    """Earn bounded task quality; safety and motion quality only gate earnings.

    There is no time tax, method-specific credit or reward for mere survival.
    A failed transition earns exactly zero, including a simultaneous capture.
    After a shared prefix, stopping cannot improve return by avoiding costs.
    The completion bonus is configurable; it does not guarantee that success
    dominates every possible raw or discounted failed-episode return.
    """
    cfg = env.config['reward']; p = cfg['positive']; n = env.num_friendly
    motion_transfer = cfg['scheme'] == 'cooperative_positive_v2'
    if terminal_reason and terminal_reason not in cfg['terminal']:
        raise ValueError(f'unknown reward terminal reason: {terminal_reason}')
    active = env.agent_active_mask(); current = potentials(env, geometry)
    parts = {name: np.zeros(n, np.float64) for name in REWARD_COMPONENTS}
    gates = np.zeros(n, np.float64)
    failed = terminal_reason in FAILURE_REASONS
    audit = env.counterfactual_audit
    evidence = not failed and env.counterfactual_evidence_valid()
    readiness = cfg['usv']['readiness_floor']+(1.-cfg['usv']['readiness_floor'])*current['usv_readiness']
    for i, ship in enumerate(env.friendly_states()):
        if not active[i] or failed: continue
        role = 'usv' if i < env.num_surface_blue else 'auv'
        clearance = min(env._clearances(i))
        safety = float(np.clip(clearance/p['full_clearance_m'], 0., 1.))**2
        energy = float(np.clip(np.mean((env.last_executed_actions[i]/env.agent_action_limits[i])**2), 0., 1.))
        threshold = cfg[role]['yaw_soft_threshold_rps']
        yaw = float(np.clip((abs(ship.yaw_rate)-threshold)/max(env.max_yaw_rate-threshold, 1e-12), 0., 1.))
        gates[i] = safety*(1.-p['action_moderation_fraction']*energy)*(1.-p['yaw_moderation_fraction']*yaw)
        if role == 'auv':
            j = i-env.num_surface_blue
            distance = current['auv_distance_quality']
            values = dict(auv_individual_reward=current['auv_individual_quality'][j],
                auv_distance_reward=distance,
                auv_gap_reward=distance*current['auv_gap_quality'],
                auv_hull_reward=distance*current['auv_hull_quality'],
                auv_tracking_reward=current['auv_individual_quality'][j]*current['auv_local_tracking_quality'][j],
                auv_hold_reward=current['auv_hold_quality'])
            if motion_transfer:
                motion = pursuit_motion_quality(env, ship)
                distance_gate = current['auv_individual_quality'][j]
                values.update(auv_velocity_reward=distance_gate*motion['velocity'],
                              auv_heading_reward=distance_gate*motion['heading'])
        else:
            distance = float(np.linalg.norm(ship.pos-env.red_state.pos))
            approach = smooth_distance_quality(distance, p['usv_minimum_distance_m'],
                env.config['task']['minimum_usv_engagement_m'], p['usv_distance_scale_m'])
            blocking = contribution = 0.
            if evidence:
                blocking = max(0., math.tanh(audit['total_effect_m']/cfg['usv']['shapley_scale_m']))*readiness
                contribution = float(np.clip(audit.get('shapley_effect_m', [0.]*env.num_surface_blue)[i]
                    /cfg['usv']['shapley_scale_m'], 0., 1.))*readiness
            values = dict(usv_approach_reward=approach, usv_blocking_reward=blocking,
                usv_support_reward=approach*current['usv_readiness'], usv_contribution_reward=contribution)
            if motion_transfer:
                values['usv_interception_position_reward'] = approach*interception_position_quality(env, ship)
            if env.capture_success:
                # Withdrawal has no intrusion counterfactual: earn the retired
                # weights through ongoing close pursuit and AUV support.
                weight = p['usv_weights']['usv_blocking_reward']+p['usv_weights']['usv_contribution_reward']
                if motion_transfer: weight += p['usv_weights']['usv_interception_position_reward']
                parts['usv_withdrawal_reward'][i] = p['step_budget']*gates[i]*weight*approach*current['usv_readiness']
        for name, weight in p[role+'_weights'].items():
            parts[name][i] = p['step_budget']*gates[i]*weight*float(np.clip(values[name], 0., 1.))
    if not failed and env.capture_success and env.first_capture_step == env.step_count:
        parts['capture_bonus'] = p['capture_bonus']*active
    dense = sum(parts.values())
    terminal = float(cfg['terminal'].get(terminal_reason, 0.))
    terminal_values = terminal*active
    rewards = (dense+terminal_values).astype(np.float32)
    terms = {name: value.tolist() for name, value in parts.items()}
    terms.update({name: [0.]*n for name in LEGACY_ZERO_COMPONENTS})
    for name in REWARD_DIAGNOSTICS:
        mask = active.copy()
        if name.startswith('auv_'): mask[:env.num_surface_blue] = 0
        else: mask[env.num_surface_blue:] = 0
        terms[name] = (current[name]*mask).tolist()
    terms.update(terminal_reward=terminal_values.tolist(), reward_gamma=float(env.config['training']['gamma']),
        reward_scheme=cfg['scheme'], reward_safety_motion_gate=gates.tolist(),
        team_terminal_reward=float(terminal_values.mean()), terminal_reward_per_active_agent=terminal,
        canonical_team_dense_reward=float(dense.mean()),
        reward_component_means={name: float(value.mean()) for name, value in parts.items()},
        usv_contributions={name: value[:env.num_surface_blue].tolist() for name,value in parts.items()},
        auv_contributions={name: value[env.num_surface_blue:].tolist() for name,value in parts.items()})
    return rewards, terms


def _evaluate_cost(env, terminal_reason, *, geometry=None):
    """Every paid component is nonpositive and the process sum is in [-a,0].

    Success zeros only this step, never refunds earlier costs. All failures
    keep their real terminal state and pay a bounded penalty. With b>=a/(1-gamma),
    immediate failure cannot beat any feasible finite successful continuation
    from the same state merely by avoiding future process costs.
    """
    cfg = env.config['reward']; cost = cfg['cost']; n = env.num_friendly
    if terminal_reason and terminal_reason not in cfg['terminal']:
        raise ValueError(f'unknown reward terminal reason: {terminal_reason}')
    current = potentials(env, geometry); active = env.agent_active_mask()
    parts = {name: np.zeros(n, np.float64) for name in REWARD_COMPONENTS}
    audit = env.counterfactual_audit
    for i, ship in enumerate(env.friendly_states()):
        if not active[i] or terminal_reason == 'success': continue
        role = 'usv' if i < env.num_surface_blue else 'auv'
        values = dict(time_cost=1., safety_penalty=np.clip(env._safety_cost(i)/2., 0., 1.),
            action_penalty=np.clip(np.mean((env.last_executed_actions[i]/env.agent_action_limits[i])**2), 0., 1.),
            yaw_rate=np.clip((abs(ship.yaw_rate)-cfg[role]['yaw_soft_threshold_rps']) /
                max(env.max_yaw_rate-cfg[role]['yaw_soft_threshold_rps'], 1e-12), 0., 1.))
        if role == 'auv':
            j = i-env.num_surface_blue
            values.update(auv_distance_cost=1.-current['auv_distance_quality'],
                auv_individual_cost=1.-current['auv_individual_quality'][j],
                auv_gap_cost=1.-current['auv_gap_quality'],
                auv_hull_cost=1.-current['auv_hull_quality'],
                auv_tracking_cost=1.-current['auv_tracking_quality'],
                auv_hold_cost=1.-current['auv_hold_quality'])
        else:
            distance = np.linalg.norm(ship.pos-env.red_state.pos)
            approach = smooth_distance_quality(distance, 0.,
                env.config['task']['minimum_usv_engagement_m'], cost['usv_distance_scale_m'])
            # A zero contribution has neutral quality. Only a new, valid audit
            # can reduce/increase this bounded cost; no positive payout is made.
            shapley = .5
            if env.counterfactual_audit_new and audit.get('valid', False):
                shapley += .5*float(np.clip(audit['shapley_effect_m'][i]/cfg['usv']['shapley_scale_m'], -1., 1.))
            values.update(usv_approach_cost=1.-approach,
                usv_blocking_cost=1.-max(0., math.tanh(env.counterfactual_gap/cfg['usv']['gap_scale_m'])),
                usv_support_cost=1.-current['usv_readiness'], usv_shapley_cost=1.-shapley)
            if getattr(env, 'capture_success', False):
                # Intrusion-only counterfactual evidence is retired at capture.
                # Its zero gap/missing audit must not become an unavoidable
                # withdrawal penalty. Keep the other actionable costs and their
                # original weights; this is no bonus, reweighting or refund.
                values.update(usv_blocking_cost=0., usv_shapley_cost=0.)
        for name, weight in cost[role+'_weights'].items():
            parts[name][i] = -cost['step_budget']*weight*float(np.clip(values[name], 0., 1.))
    dense = sum(parts.values())
    terminal = float(cfg['terminal'].get(terminal_reason, 0.))
    terminal_values = terminal*active
    rewards = (dense+terminal_values).astype(np.float32)
    terms = {name: value.tolist() for name, value in parts.items()}
    terms.update({name: [0.]*n for name in LEGACY_ZERO_COMPONENTS})
    for name in REWARD_DIAGNOSTICS:
        mask = active.copy()
        if name.startswith('auv_'): mask[:env.num_surface_blue] = 0
        else: mask[env.num_surface_blue:] = 0
        terms[name] = (current[name]*mask).tolist()
    terms.update(terminal_reward=terminal_values.tolist(), reward_gamma=float(env.config['training']['gamma']),
        reward_scheme=cfg['scheme'], team_terminal_reward=float(terminal_values.mean()),
        terminal_reward_per_active_agent=terminal, canonical_team_dense_reward=float(dense.mean()),
        reward_component_means={name: float(value.mean()) for name, value in parts.items()},
        usv_contributions={name: value[:env.num_surface_blue].tolist() for name, value in parts.items()},
        auv_contributions={name: value[env.num_surface_blue:].tolist() for name, value in parts.items()})
    return rewards, terms
