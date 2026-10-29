from __future__ import annotations


import math


from dataclasses import dataclass, replace


from types import MappingProxyType


from typing import Any, Mapping


import numpy as np


UNIFIED_REWARD_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "position": 0.6,
        "progress": 1.0,
        "heading": 0.5,
        "speed": 0.5,
        "alignment": 0.0,
        "cooperation": 0.6,
        "spacing": 1.0,
        "yaw_rate": 1.0,
        "time": 1.0,
        "safety": 1.0,
    }
)


@dataclass(frozen=True)
class RewardSnapshot:
    """Only physical state consumed by the shared reward formula.

    ``phase``, ``task_name`` and the compatibility diagnostics below are
    emitted for existing analysis tools but never branch or alter the reward.
    """

    phase: str
    task_name: str
    dt: float
    is_first_step: bool
    surface_active: bool
    submarine_active: bool

    surface_positions: np.ndarray
    surface_velocities: np.ndarray
    surface_headings: np.ndarray
    surface_yaw_rates: np.ndarray
    pressure_zone_positions: np.ndarray
    pressure_zone_velocities: np.ndarray
    b1_apf_projection: float
    b2_apf_projection: float

    submarine_positions: np.ndarray
    submarine_velocities: np.ndarray
    submarine_headings: np.ndarray
    submarine_yaw_rates: np.ndarray
    slot_positions: np.ndarray
    slot_velocities: np.ndarray
    red_pos: np.ndarray

    action_limit: np.ndarray
    max_surface_speed: float
    max_submarine_speed: float
    max_yaw_rate: float
    max_red_speed: float = 1.0
    surface_influence_range: float = 1.0
    counterfactual_reference: float = 1.0
    clearance_reference: float = 1.0

    # Compatibility diagnostics. The world/island fields are also consumed by
    # continuous safety shaping; the remaining fields never branch the reward.
    stable_seconds: float = 0.0
    stabilization_seconds: float = 0.0
    formation_radius: float = 4.5
    initial_radius: float = 4.5
    compact_radius: float = 3.5
    compaction_started: bool = False
    compaction_complete: bool = False
    green_seconds: float = 0.0
    green_hold_seconds: float = 0.0
    b1_history: float = 0.0
    b2_history: float = 0.0
    combined_history: float = 0.0
    max_favorable_lateral_deviation: float = 0.0
    signed_lateral_deviation: float = 0.0
    combined_apf_projection: float = 0.0
    pressure_errors: np.ndarray | None = None
    submarine_surfaced: np.ndarray | None = None
    red_velocity: np.ndarray | None = None
    max_angle_gap: float = 0.0
    triangle_margin: float = 0.0
    world_size: np.ndarray | None = None
    island_center: np.ndarray | None = None
    island_radius: float = 0.0
    landing_buffer: float = 0.0
    island_active: bool = True
    stage_hold_seconds: float = 0.0
    stage_hold_target_seconds: float = 0.0
    formation_hold_target_seconds: float = 0.0
    surface_b1_latched: bool = False
    surface_b2_latched: bool = False
    surface_causal_latched: bool = False
    surface_complete_latched: bool = False
    surface_effective_dwell_seconds: tuple[float, float] = (0.0, 0.0)
    surface_effective_pressure_integrals: tuple[float, float] = (0.0, 0.0)
    counterfactual_cross_track_gain: float = 0.0
    clearance_gain_vs_shadow: float = 0.0
    surface_gate_progress: float = 0.0
    surface_pressure_b1_valid: bool = False
    surface_pressure_b2_valid: bool = False
    surface_pressure_b1_quality: float = 0.0
    surface_pressure_b2_quality: float = 0.0
    surface_pressure_joint_quality: float | None = None
    surface_pressure_b1_threshold_margin: float = 0.0
    surface_pressure_b2_threshold_margin: float = 0.0
    surface_b1_hold_progress: float = 0.0
    surface_b2_hold_progress: float = 0.0
    surface_causal_progress: float = 0.0
    surface_relay_dwell_seconds: float = 0.0
    surface_relay_pressure_integral: float = 0.0
    surface_spin_cost: float = 0.0
    triangle_geometry_quality: float | None = None
    triangle_containment_quality: float | None = None
    submarine_hold_progress: float | None = None
    reference_radius: float | None = None
    submarine_kinematic_inefficiency_cost: float = 0.0
    max_steps: int = 1
    compaction_enabled: bool = False
    surface_zone_qualities: tuple[float, float] = (0.0, 0.0)
    surface_motion_qualities: tuple[float, float] = (0.0, 0.0)
    surface_speeds: tuple[float, float] = (0.0, 0.0)
    recent_counterfactual_mean: float = 0.0
    recent_counterfactual_max: float = 0.0
    counterfactual_nondecreasing_ratio: float = 0.0
    predicted_island_clearance: float = 0.0
    shadow_predicted_island_clearance: float = 0.0
    recent_actual_clearance_min: float = 0.0
    surface_island_avoidance_latched: bool = False
    surface_final_hold_progress: float = 0.0
    formation45_hold_progress: float = 0.0
    final_hold_progress: float = 0.0
    slot_good_error: float = 1.0
    velocity_good_error: float = 0.75
    gap_good_degrees: float = 168.0
    surface_final_core_current: bool = False
    surface_persistence_core_current: bool = False
    submarine_tracking_core_current: bool = False
    submarine_shrink_core_current: bool = False
    submarine_final_core_current: bool = False
    friendly_active_mask: np.ndarray | None = None
    v43_quality: dict | None = None


@dataclass(frozen=True)
class RewardEvaluation:
    rewards: np.ndarray
    terms: dict[str, Any]


def _validate_reward_config(reward_config):
    from .config import finite_tree
    if reward_config.get("formulation") != "usv_role_tracking_cost_v1":
        raise ValueError("unsupported reward formulation; use the archived project for legacy algorithms")
    finite_tree(dict(reward_config))
    for name in ("tracking_length_m", "tracking_cost_per_second", "risk_cost_per_second"):
        value = reward_config["surface"].get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    tracking_weights = reward_config["surface"].get("tracking_weights", {})
    for name in ("b1_progress_scale", "b2_progress_scale"):
        progress_scale = reward_config["surface"].get(name)
        if isinstance(progress_scale, bool) or not isinstance(progress_scale, (int, float)) or not math.isfinite(progress_scale) or progress_scale < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (set(tracking_weights) != {"position", "heading", "velocity"}
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
               for v in tracking_weights.values())
        or not math.isclose(sum(tracking_weights.values()), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError("tracking_weights must contain positive position/heading/velocity weights summing to one")
    weights = dict(reward_config["weights"])
    if weights != dict(UNIFIED_REWARD_WEIGHTS):
        raise ValueError("v4.3 freezes original AUV relative weights")
    terminal = dict(reward_config["terminal"])
    if set(terminal) != {"success", "landing", "collision", "out_of_bounds", "timeout"}:
        raise ValueError("invalid terminal schema")
    if terminal["success"] < 0 or any(v >= 0 for k, v in terminal.items() if k != "success"):
        raise ValueError("invalid terminal signs")
    return weights, dict(reward_config["normalization"]), terminal


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _clip_vector_norms(vectors: np.ndarray, maximum: float) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    scales = np.minimum(1.0, float(maximum) / np.maximum(norms, 1.0e-12))
    return values * scales


def _angular_formation_quality(positions: np.ndarray, center: np.ndarray) -> float:
    angles = np.sort(np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0]))
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
    ideal = 2.0 * np.pi / len(positions)
    return float(max(0.0, 1.0 - np.std(gaps - ideal) / max(ideal / 2.0, 1.0e-8)))


def _spacing_penalty(
    positions: np.ndarray,
    *,
    safe: float,
    scale: float,
    max_penalty: float,
) -> np.ndarray:
    result = np.zeros(len(positions), dtype=np.float64)
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            value = -min((max(0.0, safe - distance) / scale) ** 2, max_penalty)
            result[left] += value
            result[right] += value
    peer_count = max(len(positions) - 1, 1)
    return result / peer_count


def _safety_penalty(
    positions: np.ndarray,
    *,
    red_pos: np.ndarray,
    world_size: np.ndarray | None,
    island_center: np.ndarray | None,
    island_radius: float,
    landing_buffer: float,
    island_active: bool,
    normalization: Mapping[str, float],
) -> np.ndarray:
    """Return a bounded warning penalty before a hard safety terminal."""

    risk = np.zeros(len(positions), dtype=np.float64)
    if world_size is not None:
        boundary_clearance = np.min(
            np.stack(
                (
                    positions[:, 0],
                    world_size[0] - positions[:, 0],
                    positions[:, 1],
                    world_size[1] - positions[:, 1],
                ),
                axis=1,
            ),
            axis=1,
        )
        boundary_risk = np.clip(
            (
                normalization["safety_boundary_margin_m"]
                - boundary_clearance
            )
            / normalization["safety_boundary_margin_m"],
            0.0,
            1.0,
        )
        risk = np.maximum(risk, boundary_risk)

    if island_active and island_center is not None and island_radius > 0.0:
        island_clearance = (
            np.linalg.norm(positions - island_center[None, :], axis=1)
            - island_radius
            - landing_buffer
        )
        island_risk = np.clip(
            (
                normalization["safety_island_margin_m"]
                - island_clearance
            )
            / normalization["safety_island_margin_m"],
            0.0,
            1.0,
        )
        risk = np.maximum(risk, island_risk)

    red_clearance = np.linalg.norm(positions - red_pos[None, :], axis=1)
    red_risk = np.clip(
        (
            normalization["safety_red_margin_m"]
            - red_clearance
        )
        / normalization["safety_red_margin_m"],
        0.0,
        1.0,
    )
    risk = np.maximum(risk, red_risk)

    return -normalization["safety_penalty_scale"] * risk


def _dense_components(
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
    current_velocities: np.ndarray,
    current_headings: np.ndarray,
    current_yaw_rates: np.ndarray,
    previous_targets: np.ndarray,
    current_targets: np.ndarray,
    current_target_velocities: np.ndarray,
    cooperation_quality: np.ndarray,
    *,
    red_pos: np.ndarray,
    world_size: np.ndarray | None,
    island_center: np.ndarray | None,
    island_radius: float,
    landing_buffer: float,
    island_active: bool,
    max_speed: float,
    max_yaw_rate: float,
    dt: float,
    first_step: bool,
    normalization: Mapping[str, float],
) -> dict[str, np.ndarray]:
    target_vectors = current_targets - current_positions
    distances = np.linalg.norm(target_vectors, axis=1)
    previous_distances = np.linalg.norm(previous_targets - previous_positions, axis=1)
    target_speeds = np.linalg.norm(current_target_velocities, axis=1)

    position = -np.divide(
        distances,
        distances + normalization["position_softening_m"],
    )
    progress = (
        np.zeros_like(distances)
        if first_step
        else np.clip(
            (previous_distances - distances) / (max_speed * dt),
            -1.0,
            1.0,
        )
    )
    # Slot-error progress is meaningful for both static and moving targets:
    # approaching is positive shaping and retreating is negative shaping.
    desired_headings = np.arctan2(target_vectors[:, 1], target_vectors[:, 0])
    # The historical heading distance is 3 m = 1.5 * the 2 m position scale.
    # Use the configured spatial scale for geometrically smaller scenes too.
    heading_relevance = np.clip(
        distances / (1.5 * normalization["position_softening_m"]),
        0.0,
        1.0,
    )
    heading = (
        -np.abs(_wrap_angles(current_headings - desired_headings))
        / np.pi
        * heading_relevance
    )

    correction_velocity = _clip_vector_norms(
        normalization["k_pos"] * target_vectors,
        normalization["correction_speed_ratio"] * max_speed,
    )
    desired_velocities = _clip_vector_norms(
        current_target_velocities + correction_velocity,
        max_speed,
    )
    speeds = np.linalg.norm(current_velocities, axis=1)
    # Under 3-DOF dynamics the world velocity is not generally parallel to the
    # bow.  Use the measured velocity vector so the existing tracking term sees
    # sway instead of silently reconstructing the old kinematic velocity.
    agent_velocities = current_velocities
    cosine = np.cos(current_headings)
    sine = np.sin(current_headings)
    surge_speeds = cosine * current_velocities[:, 0] + sine * current_velocities[:, 1]
    sway_speeds = -sine * current_velocities[:, 0] + cosine * current_velocities[:, 1]
    slip_angles = np.arctan2(sway_speeds, surge_speeds)
    desired_speeds = np.linalg.norm(desired_velocities, axis=1)
    velocity_errors = np.linalg.norm(
        agent_velocities - desired_velocities,
        axis=1,
    )
    speed = (
        -normalization["velocity_penalty_scale"]
        * np.clip(velocity_errors / max_speed, 0.0, 1.0)
    )
    alignment = np.zeros_like(speed)
    cooperation = -(
        1.0 - np.clip(cooperation_quality, 0.0, 1.0)
    ) * normalization["cooperation_penalty_scale"]
    spacing = _spacing_penalty(
        current_positions,
        safe=normalization["spacing_safe_m"],
        scale=normalization["spacing_scale_m"],
        max_penalty=normalization["spacing_max_penalty"],
    )
    yaw_rate = (
        -np.abs(current_yaw_rates)
        / max_yaw_rate
        * normalization["yaw_rate_penalty_scale"]
    )
    time = np.full(
        len(current_positions),
        -normalization["time_penalty_per_second"],
        dtype=np.float64,
    )
    safety = _safety_penalty(
        current_positions,
        red_pos=red_pos,
        world_size=world_size,
        island_center=island_center,
        island_radius=island_radius,
        landing_buffer=landing_buffer,
        island_active=island_active,
        normalization=normalization,
    )
    return {
        "position": position,
        "progress": progress,
        "heading": heading,
        "speed": speed,
        "alignment": alignment,
        "cooperation": cooperation,
        "spacing": spacing,
        "yaw_rate": yaw_rate,
        "time": time,
        "safety": safety,
        "distance": distances,
        "target_speed": target_speeds,
        "desired_speed": desired_speeds,
        "agent_speed": speeds,
        "surge_speed": surge_speeds,
        "sway_speed": sway_speeds,
        "slip_angle_degrees": np.degrees(slip_angles),
        "velocity_error": velocity_errors,
        "velocity_reward": speed,
    }


def _weighted_dense(
    components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    dt: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    contributions: dict[str, np.ndarray] = {}
    for name in UNIFIED_REWARD_WEIGHTS:
        values = np.asarray(components[name], dtype=np.float64)
        contributions[name] = weights[name] * values * dt
    total = np.sum(np.stack(list(contributions.values()), axis=0), axis=0)
    return total, contributions


from dataclasses import replace


import numpy as np


from .geometry import capsule_segment, capsule_circle_clearance, segment_distance


def clearance_quality(m, s):
    if not np.isfinite(m) or not np.isfinite(s) or s <= 0: raise ValueError("invalid clearance")
    return .5 / (1 - m / s) if m < 0 else .5 + .5 * min(m / s, 1.)


def boundary_clearance(capsule, world):
    a, b, r = capsule
    pts = np.stack((a, b))
    return float(min(pts[:, 0].min(), world[0] - pts[:, 0].max(),
                     pts[:, 1].min(), world[1] - pts[:, 1].max()) - r)


def capsule_clearance(left, right):
    return float(segment_distance(left[0], left[1], right[0], right[1]) - left[2] - right[2])


def unit(v, fallback):
    v = np.asarray(v, dtype=float)
    if not np.isfinite(v).all(): raise ValueError("nonfinite direction")
    length = np.linalg.norm(v)
    return v / length if length > 1e-10 else np.asarray(fallback, dtype=float)


def surface_reference(env, role):
    from .task_geometry import surface_role_reference
    ship, red, cfg = env.blue_states[role], env.red_state, env.reward_cfg["surface"]
    keys = ("inner", "outer", "half_angle_degrees", "k_pos", "max_speed",
            "motion_softening", "heading_speed_scale", "heading_weight", "velocity_weight")
    return surface_role_reference(ship.pos, ship.heading, ship.velocity, red.pos, red.velocity,
        env.island_center if env.island_active else None, env.natural_route_tangent,
        float(env.pressure_cfg["desired_lateral_sign"]), role, **{k: cfg[k] for k in keys})


def qualities(env):
    cfg = env.reward_cfg["surface"]
    red = env.red_state
    references = [surface_reference(env, i) for i in range(2)]
    v, horizon = red.velocity.astype(float), cfg["prediction_seconds"]
    vhat = unit(v, (math.cos(red.heading), math.sin(red.heading)))
    tau_distance = max(0., float((env.island_center-red.pos) @ vhat))
    island = (clearance_quality(float(np.linalg.norm(red.pos + tau_distance*vhat - env.island_center))
                 - env.island_radius - env.landing_buffer, cfg["safe_margin"]) if env.island_active else 1.)
    cap = env._capsule(red)
    translated = (cap[0] + horizon*v, cap[1] + horizon*v, cap[2])
    boundary = clearance_quality(min(boundary_clearance(cap, env.world_size),
                                     boundary_clearance(translated, env.world_size)), cfg["safe_margin"])
    minimum_clearances = []
    for i, ship in enumerate(env.blue_states):
        own = env._capsule(ship)
        margins = [boundary_clearance(own, env.world_size), capsule_clearance(own, cap)]
        if env.island_active: margins.append(capsule_circle_clearance(own, env.island_center, env.island_radius))
        for j, other in enumerate(env.friendly_states()):
            if j != i and env.agent_active_mask()[j] and env._friendly_pair_can_collide(ship, other):
                margins.append(capsule_clearance(own, env._capsule(other)))
        minimum_clearances.append(float(min(margins)))
    minimum_clearances = np.asarray(minimum_clearances)
    x = np.clip(minimum_clearances / cfg["clearance_scale_m"], 0., 1.)
    q_clear = x*x*(3.-2.*x)
    # One worst-object capsule risk; q_clear and the old potentials are diagnostics.
    risk = np.clip((cfg["warning"]-minimum_clearances)/cfg["warning"], 0., 1.)
    xi = float(np.clip((env.initial_radius-env.formation_radius)/(env.initial_radius-env.compact_radius), 0, 1))
    return {"safe": island*boundary, "island": island, "boundary": boundary,
            "error": np.array([r["error"] for r in references]),
            "heading_error": np.array([r["heading_error"] for r in references]),
            "velocity_error": np.array([r["velocity_error"] for r in references]),
            "motion_error": np.array([r["motion_error"] for r in references]),
            "E": np.array([r["E"] for r in references]),
            "risk": risk, "m_clear": minimum_clearances, "q_clear": q_clear, "xi": xi}


def original_auv(previous, current, cfg):
    angular = _angular_formation_quality(current.submarine_positions, current.red_pos)
    containment = float(np.clip((float(current.triangle_margin) + .5) / 1., 0., 1.))
    components = _dense_components(previous.submarine_positions, current.submarine_positions,
        current.submarine_velocities, current.submarine_headings, current.submarine_yaw_rates,
        previous.slot_positions, current.slot_positions, current.slot_velocities,
        np.full(3, .7 * angular + .3 * containment), red_pos=current.red_pos,
        world_size=current.world_size, island_center=current.island_center, island_radius=current.island_radius,
        landing_buffer=current.landing_buffer, island_active=current.island_active,
        max_speed=current.max_submarine_speed, max_yaw_rate=current.max_yaw_rate,
        dt=current.dt, first_step=current.is_first_step, normalization=cfg["normalization"])
    return _weighted_dense(components, cfg["weights"], current.dt)


def evaluate(previous, current, cfg, reason=""):
    ca = cfg["c_A"]
    if isinstance(ca, bool) or not isinstance(ca, (int, float)) or not np.isfinite(ca) or ca <= 0:
        raise ValueError("unresolved c_A in reward")
    prev, nxt, s = previous.v43_quality, current.v43_quality, cfg["surface"]
    auv, raw_parts = original_auv(previous, current, cfg)
    # Current-state execution cost, zero at exact tracking. Heading and actual
    # ground-velocity mismatch remain visible even far from or inside the region.
    # The existing reference helper supplies bounded heading/velocity errors;
    # its diagnostic E/motion gate does not enter this learning objective.
    w = s["tracking_weights"]
    tracking_error = (w["position"] * -np.expm1(-nxt["error"] / s["tracking_length_m"])
                      + w["heading"] * nxt["heading_error"]
                      + w["velocity"] * nxt["velocity_error"])
    previous_tracking_error = (w["position"] * -np.expm1(-prev["error"] / s["tracking_length_m"])
                               + w["heading"] * prev["heading_error"]
                               + w["velocity"] * prev["velocity_error"])
    # Bounded potential-difference shaping: tracking remains the main term,
    # using the same tracking objective (not a separate claim of blockade effect).
    # Ordinary differences are bounded but are not policy-invariant PBRS.
    progress_delta = np.clip(previous_tracking_error - tracking_error, -1.0, 1.0)
    b1_progress = float(s.get("b1_progress_scale", 0.0)) * progress_delta[0]
    b2_progress = float(s.get("b2_progress_scale", 0.0)) * progress_delta[1]
    progress = np.asarray([b1_progress, b2_progress], dtype=np.float64)
    uparts = {"tracking": -current.dt * s["tracking_cost_per_second"] * tracking_error,
              "b1_progress": np.asarray([b1_progress, 0.0], dtype=np.float64),
              "b2_progress": np.asarray([0.0, b2_progress], dtype=np.float64),
              "safety": -current.dt * s["risk_cost_per_second"] * nxt["risk"]}
    aparts = {k: ca * v for k,v in raw_parts.items()}
    process = np.r_[sum(uparts.values()), sum(aparts.values())]
    mask = np.asarray(current.friendly_active_mask, dtype=float)
    terminal = cfg["terminal"][reason] if reason else 0.
    rewards = ((process + terminal) * mask).astype(np.float32)
    absolute = sum(float(np.abs(v * mask[:2]).sum()) for v in uparts.values())
    absolute += sum(float(np.abs(v * mask[2:]).sum()) for v in aparts.values())
    terms = {"reward_formulation": cfg["formulation"], "terminated": bool(reason), "truncated": False,
        "timeout": reason == "timeout", "terminal_reason": reason, "bootstrap_mask": float(not reason),
        "surface_dense_reward": float(np.mean(process[:2]*mask[:2])),
        "surface_b1_progress_reward": float(b1_progress * mask[0]),
        "surface_b2_progress_reward": float(b2_progress * mask[1]),
        "submarine_dense_reward": float(np.mean(process[2:]*mask[2:])),
        "canonical_team_dense_reward": float(np.mean(process*mask)),
        "team_terminal_reward": float(terminal * mask.mean()), "process_abs_sum": absolute / 5,
        "canonical_team_reward": float(np.mean(rewards)), "shrink_progress": nxt["xi"],
        "raw_auv_contributions": raw_parts, "auv_contributions": aparts, "usv_contributions": uparts}
    for role, parts in (("surface", uparts), ("submarine", aparts)):
        terms.update({f"{role}_contribution_{k}": float(v.mean()) for k,v in parts.items()})
    if not np.isfinite(rewards).all(): raise FloatingPointError("nonfinite reward")
    return RewardEvaluation(rewards, terms)

