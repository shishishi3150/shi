from __future__ import annotations

from collections import deque
from functools import lru_cache
import heapq
import math
from typing import Any, Iterable

import numpy as np

from .geometry import wrap_angle
from .types import ShipState


def _reef_parts(reef: Any) -> tuple[np.ndarray, float]:
    if isinstance(reef, dict):
        center, radius = reef["center"], reef["radius"]
    else:
        center, radius = reef
    point = np.asarray(center, dtype=np.float64)
    value = float(radius)
    if point.shape != (2,) or not np.isfinite(point).all() or not math.isfinite(value) or value <= 0:
        raise ValueError("each reef must have a finite center and positive radius")
    return point, value


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    magnitude = float(np.linalg.norm(value))
    if magnitude <= 1e-12:
        return np.asarray([1., 0.])
    return value / magnitude


def _segment_clear(a: np.ndarray, b: np.ndarray, circles: list[tuple[np.ndarray, float]]) -> bool:
    delta = b - a
    denom = float(delta @ delta)
    for center, radius in circles:
        t = 0.0 if denom <= 1e-15 else float(np.clip(((center - a) @ delta) / denom, 0.0, 1.0))
        if float(np.linalg.norm(a + t * delta - center)) <= radius + 1e-6:
            return False
    return True


@lru_cache(maxsize=64)
def _static_visibility_graph(circle_key: tuple, world_key: tuple,
                             margin: float, side_bias: int) -> tuple[tuple, tuple]:
    """Cache the fixed-reef part; online replanning adds only start/goal edges."""
    circles = [(np.asarray(center, dtype=np.float64), radius) for center, radius in circle_key]
    world_size = np.asarray(world_key, dtype=np.float64)
    nodes: list[np.ndarray] = []
    angles = np.arange(16, dtype=np.float64) * (2 * math.pi / 16) + (0.08 * side_bias)
    for center, radius in circles:
        for angle in angles:
            point = center + (radius + margin) * np.asarray([math.cos(angle), math.sin(angle)])
            if np.all(point >= margin) and np.all(point <= world_size - margin):
                nodes.append(point)
    graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _segment_clear(nodes[i], nodes[j], circles):
                cost = float(np.linalg.norm(nodes[i] - nodes[j]))
                graph[i].append((j, cost))
                graph[j].append((i, cost))
    return (tuple(tuple(p) for p in nodes), tuple(tuple(e) for e in graph))


def _route(start: np.ndarray, goal: np.ndarray, circles: list[tuple[np.ndarray, float]],
           world_size: np.ndarray, margin: float, side_bias: int) -> list[np.ndarray]:
    """Find a short path through deterministic points around inflated reefs."""
    if not circles or _segment_clear(start, goal, circles):
        return [goal.copy()]
    circle_key = tuple((tuple(center), float(radius)) for center, radius in circles)
    fixed_nodes, fixed_edges = _static_visibility_graph(circle_key, tuple(world_size), float(margin), side_bias)
    nodes = [start.copy(), goal.copy()] + [np.asarray(point, dtype=np.float64) for point in fixed_nodes]
    graph: list[list[tuple[int, float]]] = [[], []]
    graph.extend([[(j + 2, cost) for j, cost in edges] for edges in fixed_edges])
    for i in (0, 1):
        for j in range(2, len(nodes)):
            if _segment_clear(nodes[i], nodes[j], circles):
                cost = float(np.linalg.norm(nodes[i] - nodes[j]))
                graph[i].append((j, cost))
                graph[j].append((i, cost))
    axis = goal - start
    for i, edges in enumerate(graph):
        for k, (j, cost) in enumerate(edges):
            relative = .5 * (nodes[i] + nodes[j]) - start
            cross = float(axis[0] * relative[1] - axis[1] * relative[0])
            preferred_side = -1. if side_bias > 0 else 1.
            side_cost = .02 if cross * preferred_side < 0. else 0.
            edges[k] = (j, cost + 1e-5 * (i + j) + side_cost)
    distance = [math.inf] * len(nodes)
    previous = [-1] * len(nodes)
    distance[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        value, node = heapq.heappop(queue)
        if value != distance[node]:
            continue
        if node == 1:
            break
        for other, cost in graph[node]:
            candidate = value + cost
            if candidate < distance[other]:
                distance[other], previous[other] = candidate, node
                heapq.heappush(queue, (candidate, other))
    if previous[1] < 0:
        return [goal.copy()]
    path, node = [], 1
    while node != 0:
        path.append(nodes[node].copy())
        node = previous[node]
    path.reverse()
    return path


class RedNavigator:
    """APF navigation with an optional, side-latched USV bypass response.

    Historical configs without bypass_gain retain pure APF. Withdrawal is an
    explicit mission-abort signal, never inferred from a timer or AUV truth.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(clearance_m=1.25, attraction_gain=1.0,
            reef_avoidance_range_m=6.0, reef_avoidance_gain=2.0, reef_field_x_scale=1.0,
            surface_avoidance_range_m=5.0, surface_avoidance_gain=1.5,
            boundary_avoidance_range_m=2.5, boundary_avoidance_gain=1.0,
            repulsion_limit=4.0, field_speed_gain=1.5, heading_gain=1.8, yaw_rate_factor=0.25,
            target_speed=0.4, accel_gain=2.0, stagnation_window_steps=40,
            stagnation_progress_m=0.25, bypass_gain=0.0,
            withdrawal_yaw_rate_factor=0.4)
        if config:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f'unknown red APF settings: {sorted(unknown)}')
            self.config.update(config)
        for key, value in self.config.items():
            if not math.isfinite(float(value)) or float(value) < 0.:
                raise ValueError(f'invalid red APF setting: {key}')
        for key in ('reef_avoidance_range_m', 'surface_avoidance_range_m',
                    'boundary_avoidance_range_m', 'repulsion_limit'):
            if self.config[key] <= 0.:
                raise ValueError(f'red APF {key} must be positive')
        if type(self.config['stagnation_window_steps']) is not int or self.config['stagnation_window_steps'] < 1:
            raise ValueError('stagnation_window_steps must be a positive integer')
        if self.config['reef_field_x_scale'] < 1.:
            raise ValueError('reef_field_x_scale must be at least one')
        if not 0. < self.config['withdrawal_yaw_rate_factor'] <= 1.:
            raise ValueError('withdrawal yaw factor must be in (0, 1]')
        self.reset()

    def reset(self) -> None:
        self._progress = deque(maxlen=self.config['stagnation_window_steps'] + 1)
        self.recovery_count = 0
        self.recovery_active = False
        self.mode = 'ADVANCE'
        self.bypass_side = 0
        self.withdrawal_goal = None
        self.withdrawal_reason = ''

    def request_withdrawal(self, goal, *, reason):
        """Latch a verified mission-abort decision; do not oscillate back in."""
        point = np.asarray(goal, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all() or not reason:
            raise ValueError('withdrawal needs a finite goal and an explicit reason')
        if self.withdrawal_goal is not None:
            return False
        self.withdrawal_goal = point.copy()
        self.withdrawal_reason = str(reason)
        self.mode = 'WITHDRAW'
        self.bypass_side = 0
        self._progress.clear()
        return True

    def _repulsion(self, gap, influence, gain):
        if gap >= influence:
            return 0.
        # Negative gradient of k*d0/3 * (1-d/d0)^3 inside the influence
        # range. Unlike inverse-distance repulsion, this has useful strength
        # before a finite-turn-radius vessel is already close to an obstacle.
        distance = max(float(gap), 0.)
        return min(float(self.config['repulsion_limit']),
                   gain * (1. - distance / influence) ** 2)

    def potential_forces(self, red, goal, reefs, surface_states, world_size, ship_cfg):
        goal = np.asarray(goal, dtype=np.float64)
        world = np.asarray(world_size, dtype=np.float64)
        position = np.asarray(red.pos, dtype=np.float64)
        if (goal.shape != (2,) or world.shape != (2,) or position.shape != (2,)
                or not np.isfinite(np.r_[goal, world, position]).all() or np.any(world <= 0.)):
            raise ValueError('goal, position and world_size must be finite two-vectors; world must be positive')
        delta = goal - position
        forces = dict(goal=self.config['attraction_gain'] * delta / max(1., float(np.linalg.norm(delta))),
                      reef=np.zeros(2), usv=np.zeros(2), boundary=np.zeros(2))
        hull_radius = .5 * float(ship_cfg.get('surface_length_m', 1.255))
        reef_metric = np.asarray([self.config['reef_field_x_scale'] ** -2, 1.])
        for center, radius in (_reef_parts(item) for item in reefs):
            away = position - center
            distance = float(np.linalg.norm(away))
            gap = distance - radius - hull_radius - self.config['clearance_m']
            strength = self._repulsion(gap, self.config['reef_avoidance_range_m'], self.config['reef_avoidance_gain'])
            forces['reef'] += strength * away / max(distance, 1e-8)
            if self.config['reef_field_x_scale'] == 1.:
                continue
            # rho defines elliptical equipotentials, not a changed reef hull.
            # Keep the actual negative gradient, including the metric factor:
            # -grad U = strength(rho) * M*(position-center)/rho.
            # This bends the approach earlier without intermediate targets.
            gradient = reef_metric * away
            distance = math.sqrt(float(away @ gradient))
            gap = distance - radius - hull_radius - self.config['clearance_m']
            strength = self._repulsion(gap, self.config['reef_avoidance_range_m'], self.config['reef_avoidance_gain'])
            forces['reef'] += strength * gradient / max(distance, 1e-8)
        for state in surface_states:
            away = position - np.asarray(state.pos, dtype=np.float64)
            distance = float(np.linalg.norm(away))
            influence = max(self.config['surface_avoidance_range_m'] - 2. * hull_radius, .05)
            strength = self._repulsion(distance - 2. * hull_radius, influence, self.config['surface_avoidance_gain'])
            forces['usv'] += strength * away / max(distance, 1e-8)
        for gap, direction in ((position[0] - hull_radius, [1., 0.]),
                               (world[0] - position[0] - hull_radius, [-1., 0.]),
                               (position[1] - hull_radius, [0., 1.]),
                               (world[1] - position[1] - hull_radius, [0., -1.])):
            strength = self._repulsion(float(gap), self.config['boundary_avoidance_range_m'], self.config['boundary_avoidance_gain'])
            forces['boundary'] += strength * np.asarray(direction)
        return forces

    def _bypass_vector(self, red, goal, reefs, surfaces, world_size, ship_cfg):
        if self.withdrawal_goal is not None or self.config['bypass_gain'] == 0.:
            return np.zeros(2)
        axis = _unit(np.asarray(goal) - red.pos)
        radius = self.config['surface_avoidance_range_m']
        nearby = [s for s in surfaces if np.linalg.norm(s.pos - red.pos) < radius + 1.]
        if not nearby:
            self.bypass_side = 0
        ahead = [s for s in nearby if float((s.pos - red.pos) @ axis) > 0.
                 and np.linalg.norm(s.pos - red.pos) < radius]
        if not ahead:
            self.mode = 'ADVANCE'
            return np.zeros(2)
        nearest = min(ahead, key=lambda s: float(np.linalg.norm(s.pos - red.pos)))
        away = _unit(red.pos - nearest.pos)
        perpendicular = np.array([-away[1], away[0]])
        hull_radius = .5 * float(ship_cfg.get('surface_length_m', 1.255))
        if not self.bypass_side:
            # Pick a side once using forward progress and public-map clearance.
            # This local tangent is a control vector, not a commanded waypoint.
            scores = {}
            for side in (-1, 1):
                direction = side * perpendicular
                probe = red.pos + 3. * direction
                clearance = min(float(np.min(probe - hull_radius)),
                                float(np.min(np.asarray(world_size) - probe - hull_radius)))
                for center, reef_radius in (_reef_parts(r) for r in reefs):
                    clearance = min(clearance, float(np.linalg.norm(probe - center))
                                    - reef_radius - hull_radius - self.config['clearance_m'])
                scores[side] = float(direction @ axis) + .08 * float(np.clip(clearance, -10., 10.))
            self.bypass_side = max(scores, key=scores.get)
        distance = float(np.linalg.norm(red.pos - nearest.pos))
        influence = max(radius - 2. * hull_radius, .05)
        strength = self._repulsion(distance - 2. * hull_radius, influence,
                                   self.config['surface_avoidance_gain'])
        self.mode = 'BYPASS'
        return self.config['bypass_gain'] * strength * self.bypass_side * perpendicular

    def reference(self, red, goal, reefs, surface_states, world_size, ship_cfg):
        goal = self.withdrawal_goal if self.withdrawal_goal is not None else goal
        reefs, surface_states = list(reefs), list(surface_states)
        forces = self.potential_forces(red, goal, reefs, surface_states, world_size, ship_cfg)
        vector = sum(forces.values(), np.zeros(2))
        vector += self._bypass_vector(red, goal, reefs, surface_states, world_size, ship_cfg)
        distance = float(np.linalg.norm(np.asarray(goal) - red.pos))
        self._progress.append((np.asarray(red.pos, dtype=np.float64).copy(), distance))
        if len(self._progress) == self._progress.maxlen:
            progress = self._progress[0][1] - self._progress[-1][1]
            moved = float(np.linalg.norm(self._progress[-1][0] - self._progress[0][0]))
            threshold = self.config['stagnation_progress_m']
            self.recovery_active = progress < threshold and moved < threshold
            if self.recovery_active:
                self.recovery_count += 1
                self._progress.clear()
        magnitude = float(np.linalg.norm(vector))
        heading = math.atan2(float(vector[1]), float(vector[0])) if magnitude > 1e-8 else red.heading
        error = float(wrap_angle(heading - red.heading))
        yaw_factor = (self.config['withdrawal_yaw_rate_factor'] if self.withdrawal_goal is not None
                      else self.config['yaw_rate_factor'])
        yaw_limit = float(ship_cfg['max_yaw_rate']) * yaw_factor
        yaw_rate = float(np.clip(error * self.config['heading_gain'], -yaw_limit, yaw_limit))
        speed = min(self.config['target_speed'], float(ship_cfg['max_speed_red']))
        speed *= min(magnitude * self.config['field_speed_gain'], 1.) * max(0., math.cos(error))
        limit = float(ship_cfg['dynamics']['reference_max_acceleration'])
        acceleration = float(np.clip((speed - red.speed) * self.config['accel_gain'], -limit, limit))
        return np.asarray([yaw_rate, acceleration], dtype=np.float32)


def _estimate_fields(estimate: Any) -> tuple[bool, np.ndarray, np.ndarray, float]:
    if estimate is None:
        return False, np.zeros(2), np.zeros(2), math.inf
    if isinstance(estimate, dict):
        valid = estimate.get("valid", False)
        position = estimate.get("position", estimate.get("position_estimate", (0., 0.)))
        velocity = estimate.get("velocity", estimate.get("velocity_estimate", (0., 0.)))
        covariance = estimate.get("covariance")
    else:
        valid = getattr(estimate, "valid", False)
        position = getattr(estimate, "position", getattr(estimate, "position_estimate", (0., 0.)))
        velocity = getattr(estimate, "velocity", getattr(estimate, "velocity_estimate", (0., 0.)))
        covariance = getattr(estimate, "covariance", None)
    p, v, cov = np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    if p.shape != (2,) or v.shape != (2,) or not np.isfinite(p).all() or not np.isfinite(v).all():
        return False, np.zeros(2), np.zeros(2), math.inf
    uncertainty = (float(np.trace(cov[:2, :2]))
                   if cov.shape in ((2, 2), (4, 4)) and np.isfinite(cov).all() else math.inf)
    return bool(valid), p, v, uncertainty


def _safe_route_target(position: np.ndarray, target: np.ndarray, reefs: list[tuple[np.ndarray, float]],
                       world_size: np.ndarray, hull_radius: float, clearance: float) -> np.ndarray:
    circles = [(center, radius + hull_radius + clearance) for center, radius in reefs]
    for center, radius in circles:
        delta = np.asarray(position, dtype=np.float64) - center
        if np.linalg.norm(delta) < radius:
            # A finite-turn-radius vessel can enter the warning band while
            # rounding a waypoint. Steer outward before planning onward.
            return center + _unit(delta) * (radius + .2)
    point = np.clip(np.asarray(target, dtype=np.float64), hull_radius + clearance,
                    np.asarray(world_size, dtype=np.float64) - hull_radius - clearance)
    for _ in range(len(circles) + 1):
        changed = False
        for center, radius in circles:
            delta = point - center
            distance = float(np.linalg.norm(delta))
            if distance < radius:
                if distance <= 1e-9:
                    delta = np.asarray([1., 0.])
                    distance = 1.0
                point = center + delta / distance * (radius + 1e-3)
                point = np.clip(point, hull_radius + clearance,
                                np.asarray(world_size, dtype=np.float64) - hull_radius - clearance)
                changed = True
        if not changed:
            break
    if any(float(np.linalg.norm(point - center)) <= radius for center, radius in circles):
        candidates = []
        for center, radius in circles:
            for angle in np.arange(32) * (2 * math.pi / 32):
                candidate = np.clip(center + (radius + 0.05) * np.asarray([math.cos(angle), math.sin(angle)]),
                                    hull_radius + clearance,
                                    np.asarray(world_size, dtype=np.float64) - hull_radius - clearance)
                if all(float(np.linalg.norm(candidate - other)) > other_radius
                       for other, other_radius in circles):
                    candidates.append(candidate)
        if not candidates:
            return point.astype(np.float32)
        point = min(candidates, key=lambda item: float(np.linalg.norm(item - target)))
    path = _route(np.asarray(position, dtype=np.float64), point, circles,
                  np.asarray(world_size, dtype=np.float64), hull_radius + clearance, 1)
    return path[0]


def _reference_action(state: ShipState, target: np.ndarray, target_speed: float,
                      dynamics: Any, *, heading_gain: float = 1.8,
                      accel_gain: float = 1.5,
                      yaw_acceleration_limit: float | None = None) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(state.pos, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance <= 0.05:
        target_speed = 0.0
        desired_heading = state.heading
    else:
        desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    error = float(wrap_angle(desired_heading - state.heading))
    yaw_ref = float(np.clip(heading_gain * error, -dynamics.max_yaw_rate, dynamics.max_yaw_rate))
    if yaw_acceleration_limit is not None:
        yaw_ref = float(np.clip(yaw_ref, -.2, .2))
        increment = yaw_acceleration_limit * dynamics.dt
        yaw_ref = float(state.yaw_rate + np.clip(yaw_ref - state.yaw_rate, -increment, increment))
    speed_ref = float(np.clip(target_speed, 0., dynamics.max_speed))
    speed_ref *= float(np.clip(math.cos(error), 0., 1.))
    if yaw_acceleration_limit is not None:
        # Brake before a large course change. Moving through a half-turn
        # pushes a flank screen several metres off station.
        speed_ref *= float(np.clip(math.cos(2. * error), 0., 1.)) if abs(error) < math.pi / 2 else 0.
    speed_ref = min(speed_ref, .6 * distance)
    acceleration_ref = float(np.clip(accel_gain * (speed_ref - state.speed),
                                     -dynamics.reference_max_acceleration,
                                     dynamics.reference_max_acceleration))
    return dynamics.reference_action(state, desired_yaw_rate=yaw_ref,
                                     desired_surge_acceleration=acceleration_ref)


class RuleController:
    """A physically actuated rule baseline using only supplied target estimates.

    Two USVs press the same flank of the estimated protection approach, with
    a lateral screen and a forward cover that can both affect the target.
    AUVs search fixed sectors until an estimate is valid, then maintain distinct
    observation bearings around the estimated (never true) target state.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            # A lateral screen and a forward cover, both inside Red's APF
            # influence. A distant second slot never helped the first vessel.
            "usv_block_distance_m": 0.8,
            "usv_side_offset_m": -3.4,
            "usv_stagger_m": 2.3,
            "usv_cover_inset_m": 2.5,
            "usv_lookahead_s": 1.0,
            "auv_observation_radius_m": 4.0,
            # Moving slots can change the desired turn near a reef. Reserve
            # braking distance (v^2/(2a)) plus the surface half-length.
            "path_clearance_m": 1.8,
            "target_speed_usv_mps": 0.38,
            "target_speed_auv_mps": 0.45,
            "teammate_avoidance_range_m": 3.0,
            "teammate_stop_distance_m": 1.8,
            "surface_target_standoff_m": 2.8,
            "uncertain_covariance_trace_m2": 16.0,
            "search_stage_seconds": 35.0,
            "search_routes": [[[16., 29.], [24., 30.], [34., 33.]],
                              [[16., 24.], [25., 24.], [34., 22.]],
                              [[23., 27.], [30., 23.], [38., 25.]]],
        }
        if config:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f"unknown rule controller settings: {sorted(unknown)}")
            self.config.update(config)
        routes = np.asarray(self.config["search_routes"], dtype=np.float64)
        if routes.ndim != 3 or routes.shape[0] != 3 or routes.shape[2] != 2 or not np.isfinite(routes).all():
            raise ValueError("search_routes must be finite three-vessel waypoint routes")
        self.search_routes = routes
        self.reset()

    def reset(self) -> None:
        self._path_cache: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}
        self._velocity_memory: dict[int, tuple[float, np.ndarray]] = {}

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.actions(*args, **kwargs)

    def actions(self, surface_states: list[ShipState], submarine_states: list[ShipState],
                target_estimate: Any, protected_center: np.ndarray,
                reefs: Iterable[Any], world_size: np.ndarray,
                surface_dynamics: Any, submarine_dynamics: Any,
                time_seconds: float, *, target_estimates: Iterable[Any] | None = None,
                active_mask: Iterable[float] | None = None) -> np.ndarray:
        if len(surface_states) != 2 or len(submarine_states) != 3:
            raise ValueError("Rule controller requires exactly 2 USVs and 3 AUVs")
        surface_count=len(surface_states); agent_count=surface_count+len(submarine_states)
        active=np.ones(agent_count,bool) if active_mask is None else np.asarray(active_mask,bool)
        if active.shape!=(agent_count,): raise ValueError('invalid rule-controller active mask')
        protected = np.asarray(protected_center, dtype=np.float64)
        world = np.asarray(world_size, dtype=np.float64)
        if protected.shape != (2,) or world.shape != (2,) or not np.isfinite(protected).all() or not np.isfinite(world).all():
            raise ValueError("protected_center and world_size must be finite two-vectors")
        if not math.isfinite(float(time_seconds)) or time_seconds < 0:
            raise ValueError("time_seconds must be finite and nonnegative")
        reef_list = [_reef_parts(item) for item in reefs]
        if target_estimates is not None:
            estimates = list(target_estimates)
        elif isinstance(target_estimate, (list, tuple)):
            estimates = list(target_estimate)
        else:
            estimates = [target_estimate] * agent_count
        if len(estimates) != agent_count:
            raise ValueError("target_estimates must contain one received estimate per friendly vessel")
        actions: list[np.ndarray] = []
        # These private rule-controller navigation references are diagnostic
        # heuristics. They never enter the environment, learned input or reward.
        for index, state in enumerate(surface_states):
            if not active[index]: actions.append(np.zeros(2,np.float32)); continue
            valid, estimate_pos, estimate_vel, uncertainty = _estimate_fields(estimates[index])
            if valid:
                previous = self._velocity_memory.get(index)
                if previous is not None:
                    elapsed = max(float(time_seconds) - previous[0], surface_dynamics.dt)
                    alpha = 1. - math.exp(-elapsed / 2.)
                    estimate_vel = previous[1] + alpha * (estimate_vel - previous[1])
                velocity_norm = float(np.linalg.norm(estimate_vel))
                if velocity_norm > surface_dynamics.max_speed:
                    estimate_vel = estimate_vel * (surface_dynamics.max_speed / velocity_norm)
                self._velocity_memory[index] = (float(time_seconds), estimate_vel.copy())
                predicted = estimate_pos + estimate_vel * float(self.config["usv_lookahead_s"])
                forward = protected - predicted
                norm = float(np.linalg.norm(forward))
                forward = forward / norm if norm > 1e-8 else np.asarray([1., 0.])
                left = np.asarray([-forward[1], forward[0]])
                lead = float(self.config["usv_block_distance_m"]) + index * float(self.config["usv_stagger_m"])
                flank = math.copysign(1., float(self.config['usv_side_offset_m']))
                side = float(self.config['usv_side_offset_m']) - index * flank * self.config['usv_cover_inset_m']
                raw_target = predicted + lead * forward + side * left
            else:
                # Same southern approach while awaiting a legal received track.
                raw_target = world * np.asarray([0.57, 0.40])
                raw_target[0] += index * max(3., float(self.config["usv_stagger_m"]))
            raw_target = np.clip(raw_target, 1.0, world - 1.0)
            screen = raw_target.copy()
            if reef_list:
                # Use a public-map screen only while the target is unknown.
                # A received track must trigger the same-flank response even
                # before Red reaches the old deployment longitude.
                front, radius = max(reef_list, key=lambda item: item[0][0])
                anchor = front + np.asarray([-4.3 + 3.5 * index, -radius - 4.2 + 1.4 * index])
                if not valid:
                    raw_target = np.clip(anchor, 1., world - 1.)
                    screen = None  # Stationary deployment: no motion feedforward.
            # Uncertain estimates call for a wider, more conservative outer screen.
            speed = float(self.config["target_speed_usv_mps"])
            if uncertainty > float(self.config["uncertain_covariance_trace_m2"]):
                speed *= 0.8
            target = raw_target
            peers = [s for i, s in enumerate(surface_states) if i != index and active[i]]
            # Resolve the final local request once. Re-routing a first waypoint
            # after every avoidance adjustment used to reverse turns near C.
            hull_radius = 0.5 * float(getattr(surface_dynamics, "hull_length", 1.255))
            target = self._waypoint(index, state.pos, target, reef_list, world,
                                    hull_radius)
            actions.append(self._surface_action(state, target, screen, speed,
                surface_dynamics, reef_list, hull_radius,
                estimate_pos if valid else None, estimate_vel if valid else None, peers,
                _unit(protected - estimate_pos) if valid else np.asarray([1., 0.])))

        for index, state in enumerate(submarine_states):
            if not active[surface_count+index]: actions.append(np.zeros(2,np.float32)); continue
            valid, estimate_pos, estimate_vel, uncertainty = _estimate_fields(estimates[surface_count + index])
            if valid:
                # The public protection approach is a stable path direction.
                # Instantaneous CV velocity can rotate sharply under noisy
                # measurements while Red maneuvers around a screen.
                axis = _unit(protected - estimate_pos)
                # Three fixed view directions follow the estimated route heading;
                # there is no prescribed orbit rate around the target.
                theta = math.atan2(float(axis[1]), float(axis[0])) + (2 * math.pi / 3, -2 * math.pi / 3, 0.)[index]
                center = estimate_pos
                radius = float(self.config["auv_observation_radius_m"])
                speed = float(self.config["target_speed_auv_mps"])
            else:
                route_index = min(int(float(time_seconds) / float(self.config["search_stage_seconds"])),
                                  self.search_routes.shape[1] - 1)
                center = self.search_routes[index, route_index]
                theta, radius = 0., 0.
                speed = float(self.config["target_speed_auv_mps"])
            raw_target = np.clip(center + radius * np.asarray([math.cos(theta), math.sin(theta)]),
                                 1.0, world - 1.0)
            target = self._waypoint(surface_count + index, state.pos, raw_target, reef_list, world,
                                    0.5 * float(getattr(submarine_dynamics, "hull_width", 0.31)))
            target, speed = self._avoid_teammates(state, target, speed,
                                                 [s for i, s in enumerate(submarine_states) if i != index and active[surface_count+i]])
            target = _safe_route_target(state.pos, target, reef_list, world,
                                        .155, float(self.config["path_clearance_m"]))
            actions.append(_reference_action(state, target, speed, submarine_dynamics))
        result = np.asarray(actions, dtype=np.float32)
        if result.shape != (agent_count, 2) or not np.isfinite(result).all():
            raise FloatingPointError("rule controller produced invalid five-agent actions")
        return result

    def _surface_action(self, state, target, screen, speed, dynamics, reefs,
                        hull_radius, estimate_pos, estimate_vel, peers, forward):
        """Approach, match the estimated motion and brake before close contact.

        This belongs only to the rule baseline. It does not change the learned
        observations, reward, dynamics or the I1 filter.
        """
        delta = np.asarray(target) - state.pos
        distance = float(np.linalg.norm(delta))
        velocity = speed * _unit(delta) * min(distance / 2., 1.)
        if (screen is not None and estimate_vel is not None and distance < 6.
                and np.linalg.norm(np.asarray(target) - screen) < .8):
            # Position-only pursuit overshoots a moving screen, then commands
            # a 180-degree return. Match its motion before reaching the slot.
            velocity = np.asarray(estimate_vel) + .18 * delta
        if estimate_pos is not None:
            offset = state.pos - estimate_pos
            separation = float(np.linalg.norm(offset))
            away = offset / max(separation, 1e-8)
            braking_time = max(0., float(state.speed)) / dynamics.reference_max_acceleration + 2.
            minimum_rate = -max(separation - self.config['surface_target_standoff_m'], 0.) / braking_time
            radial_rate = float((velocity - estimate_vel) @ away)
            if radial_rate < minimum_rate:
                velocity += (minimum_rate - radial_rate) * away
        for peer in peers:
            offset = state.pos - peer.pos
            separation = float(np.linalg.norm(offset))
            if separation >= 6.:
                continue
            away = offset / max(separation, 1e-8)
            braking_time = (max(0., float(state.speed)) + max(0., float(peer.speed))) / dynamics.reference_max_acceleration + 2.
            minimum_rate = -max(separation - self.config['teammate_stop_distance_m'], 0.) / braking_time
            radial_rate = float((velocity - peer.velocity) @ away)
            if radial_rate < minimum_rate:
                velocity += (minimum_rate - radial_rate) * away
        speed = min(speed, float(np.linalg.norm(velocity)))
        # The path is collision-free geometrically; a moving hull still needs
        # space to brake while its heading catches up with a route change.
        motion = np.asarray(state.velocity)
        ray = _unit(motion) if np.linalg.norm(motion) > .03 else np.asarray([math.cos(state.heading), math.sin(state.heading)])
        for center, radius in reefs:
            to_reef = center - state.pos
            along = float(to_reef @ ray)
            cross_squared = max(float(to_reef @ to_reef) - along ** 2, 0.)
            warning_radius = radius + hull_radius + .7
            if along > 0. and cross_squared < warning_radius ** 2:
                free_path = max(along - math.sqrt(warning_radius ** 2 - cross_squared), 0.)
                speed = min(speed, math.sqrt(2. * dynamics.reference_max_acceleration * max(free_path - .3, 0.)))
        if distance < 2. and (screen is None or np.linalg.norm(target - screen) < .8):
            # Keep a useful blocking heading while matching a slow target.
            # Forward-only station keeping: brake rather than request a
            # backward position correction.
            body_axis = np.asarray([math.cos(state.heading), math.sin(state.heading)])
            # Either bow direction can hold a flank. Preserve the nearer one
            # rather than forcing a vessel that arrived facing west through
            # a half-turn while Red passes it.
            station_axis = forward if float(body_axis @ forward) >= 0. else -forward
            lateral = np.asarray([-station_axis[1], station_axis[0]])
            course_error = float(velocity @ lateral)
            longitudinal = float(velocity @ station_axis)
            heading = math.atan2(float(station_axis[1]), float(station_axis[0]))
            correction = math.atan2(course_error, .25 + abs(longitudinal))
            heading += float(np.clip(correction if longitudinal >= 0. else -correction, -.6, .6))
            error = float(wrap_angle(heading - state.heading))
            yaw_ref = float(np.clip(.8 * error, -.2, .2))
            yaw_ref = float(state.yaw_rate + np.clip(yaw_ref - state.yaw_rate,
                -.08 * dynamics.dt, .08 * dynamics.dt))
            surge = float(np.clip(velocity @ body_axis, dynamics.min_speed, speed)) if abs(error) < math.pi / 3 else 0.
            acceleration = float(np.clip(1.5 * (surge - state.speed),
                -dynamics.reference_max_acceleration, dynamics.reference_max_acceleration))
            action = dynamics.reference_action(state, desired_yaw_rate=yaw_ref,
                desired_surge_acceleration=acceleration)
            return self._surface_braking_action(state, action, dynamics,
                estimate_pos, estimate_vel, peers)
        emergency = False
        if estimate_pos is not None:
            offset = state.pos - estimate_pos
            relative = motion - estimate_vel
            closest_time = float(np.clip(-float(offset @ relative) / max(float(relative @ relative), 1e-8),
                                        0., 8.))
            emergency = (float(offset @ relative) < 0.
                and np.linalg.norm(offset + closest_time * relative) < self.config['surface_target_standoff_m'])
            if emergency:
                speed = 0.
                velocity = _unit(offset) + _unit(velocity)
        if np.linalg.norm(velocity) > .03:
            target = state.pos + 4. * _unit(velocity)
        else:
            target = state.pos  # Hold heading at rest instead of chasing noise.
        # I1 holds each physical command during prediction. Saturating moment
        # on a small heading error predicts an unnecessary full turn and makes
        # the filter reject steering repeatedly. Bound rule steering changes.
        action = _reference_action(state, target, speed, dynamics,
                                   heading_gain=.8, yaw_acceleration_limit=.08)
        return self._surface_braking_action(state, action, dynamics,
            estimate_pos, estimate_vel, peers)

    def _surface_braking_action(self, state, action, dynamics,
                                estimate_pos, estimate_vel, peers):
        """Recheck contact closing rates after mapping velocity to body surge.

        A safe planar reference need not remain safe when an underactuated
        hull projects it onto its current heading. Include the measured sway
        and reserve stopping time in the final physical surge command.
        """
        axis = np.asarray([math.cos(state.heading), math.sin(state.heading)])
        sway = state.sway_speed * np.asarray([-axis[1], axis[0]])
        lower, upper = dynamics.min_speed, dynamics.max_speed
        contacts = [(p.pos, p.velocity, self.config['teammate_stop_distance_m']) for p in peers]
        if estimate_pos is not None:
            contacts.append((estimate_pos, estimate_vel, self.config['surface_target_standoff_m']))
        for position, velocity, standoff in contacts:
            offset = state.pos - position
            separation = float(np.linalg.norm(offset))
            if separation >= 6. or separation < 1e-8:
                continue
            away = offset / separation
            stopping_time = (abs(state.speed) + np.linalg.norm(velocity)) / dynamics.reference_max_acceleration + 2.
            minimum_rate = -max(separation - standoff, 0.) / stopping_time
            coefficient = float(axis @ away)
            required = minimum_rate - float((sway - velocity) @ away)
            if coefficient > .05:
                lower = max(lower, required / coefficient)
            elif coefficient < -.05:
                upper = min(upper, required / coefficient)
        acceleration = float(dynamics.body_acceleration(state.body_velocity, action)[0])
        requested = state.speed + acceleration / 1.5
        surge = float(np.clip(requested, lower, upper)) if lower <= upper else 0.
        acceleration = float(np.clip(1.5 * (surge - state.speed),
            -dynamics.reference_max_acceleration, dynamics.reference_max_acceleration))
        # Surge is the independent first row of the existing mass matrix.
        load = (dynamics.coriolis_matrix(state.body_velocity) @ state.body_velocity
                + dynamics.damping_matrix(state.body_velocity) @ state.body_velocity)
        result = np.asarray(action, dtype=np.float32).copy()
        result[0] = np.clip(dynamics.mass_matrix[0, 0] * acceleration + load[0],
                            -dynamics.max_surge_force, dynamics.max_surge_force)
        return result

    def _waypoint(self, index: int, position: np.ndarray, target: np.ndarray,
                  reefs: list[tuple[np.ndarray, float]], world: np.ndarray,
                  hull_radius: float) -> np.ndarray:
        clearance = float(self.config["path_clearance_m"])
        entry = self._path_cache.get(index)
        circles = [(c, r + hull_radius + clearance) for c, r in reefs]
        if (entry is None or np.linalg.norm(entry[0] - target) > .75
                or not entry[1] or not _segment_clear(np.asarray(position), entry[1][0], circles)):
            first = _safe_route_target(position, target, reefs, world, hull_radius, clearance)
            # `_safe_route_target` also repairs an infeasible endpoint. Keep the
            # first waypoint only and recompute after it is reached.
            entry = (np.asarray(target).copy(), [first])
            self._path_cache[index] = entry
        if np.linalg.norm(entry[1][0] - position) < .6:
            first = _safe_route_target(position, target, reefs, world, hull_radius, clearance)
            self._path_cache[index] = (np.asarray(target).copy(), [first])
            return first
        return entry[1][0]

    def _avoid_teammates(self, own: ShipState, target: np.ndarray, speed: float,
                         peers: list[ShipState]) -> tuple[np.ndarray, float]:
        """Local same-depth avoidance and braking; no target truth is consumed."""
        direction = _unit(np.asarray(target) - own.pos)
        correction = np.zeros(2, dtype=np.float64)
        trigger = float(self.config["teammate_avoidance_range_m"])
        stop = float(self.config["teammate_stop_distance_m"])
        for other in peers:
            delta = np.asarray(own.pos, dtype=np.float64) - np.asarray(other.pos, dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            if distance >= trigger:
                continue
            away = delta / max(distance, 1e-8)
            correction += 2.0 * max(0., trigger - distance) / max(trigger - stop, .1) * away
            if distance <= stop or float(direction @ (-away)) > .25:
                speed *= float(np.clip((distance - stop) / max(trigger - stop, .1), 0., 1.))
        if np.linalg.norm(correction) > 0:
            target = np.asarray(own.pos) + 3.0 * _unit(direction + correction)
        return np.asarray(target), speed
