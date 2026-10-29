from __future__ import annotations

import math

import numpy as np

from .geometry import norm, to_local, wrap_angle
from .types import ShipState


from .history import SENSOR_DIM
ACTOR_OBSERVATION_DIM = SENSOR_DIM
ACTOR_TASK_CONTEXT_DIM = 15
CRITIC_STATE_DIM = 96
CRITIC_TASK_CONTEXT_DIM = 38

# Fixed actor-observation layout. Peer blocks are emitted in friendly-agent
# index order with the observing agent omitted. Thus every surface observer
# sees its three submarine peers in blocks 2-4, while every submarine observer
# sees B1/B2 in blocks 1-2.
ACTOR_PEER_BLOCK_START = 22
ACTOR_PEER_BLOCK_DIM = 9
SURFACE_NEW_SUBMARINE_PEER_INPUTS = tuple(range(31, 58))
SUBMARINE_NEW_SURFACE_PEER_INPUTS = tuple(range(22, 40))
ACTOR_COMPACTION_ENABLED_INPUT = 68


def build_actor_observation(
    friendly_index: int,
    friendly_states: list[ShipState],
    red_state: ShipState,
    world_size: np.ndarray,
    island_center: np.ndarray,
    island_radius: float,
    max_speed_by_type: dict[str, float],
    max_speed_red: float,
    max_sway_speed: float,
    max_yaw_rate: float,
    task_context: np.ndarray,
    active_mask: np.ndarray | None = None,
    peer_states: list[ShipState] | None = None,
    communication_features: dict[int, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Build the fixed 73-field decentralized actor observation.

    ``task_context`` contains the assigned target, guided desired velocity,
    assignment identity, previous action, and six compact task-progress fields.
    Physical quantities seen by the actor are expressed in its body frame.
    """

    if len(friendly_states) != 5:
        raise ValueError("actor observation requires exactly five friendly agents")
    context = np.asarray(task_context, dtype=np.float32).reshape(-1)
    if context.shape != (ACTOR_TASK_CONTEXT_DIM,):
        raise ValueError(
            f"actor task_context must have shape {(ACTOR_TASK_CONTEXT_DIM,)}"
        )
    mask = _activity_mask(active_mask, len(friendly_states))
    own = friendly_states[friendly_index]
    perceived = friendly_states if peer_states is None else peer_states
    own_active = bool(mask[friendly_index])
    scale = max(float(np.max(world_size)), 1.0e-6)
    own_speed_scale = max(float(max_speed_by_type[own.unit_type]), 1.0e-6)
    sway_scale = max(float(max_sway_speed), 1.0e-6)
    own_ground_speed_scale = float(np.hypot(own_speed_scale, sway_scale))
    red_ground_speed_scale = float(
        np.hypot(max(float(max_speed_red), 1.0e-6), sway_scale)
    )
    red_relative_speed_scale = red_ground_speed_scale + own_ground_speed_scale
    yaw_scale = max(float(max_yaw_rate), 1.0e-6)

    values: list[float] = [
        float(own_active),
        float(own.unit_type == "surface"),
        float(own.unit_type == "submarine"),
        float(own.speed / own_speed_scale) if own_active else 0.0,
        float(own.sway_speed / sway_scale) if own_active else 0.0,
        float(own.yaw_rate / yaw_scale) if own_active else 0.0,
        math.sin(own.heading) if own_active else 0.0,
        math.cos(own.heading) if own_active else 0.0,
    ]

    target_pos = context[0:2]
    desired_velocity = context[2:4]
    target_relative = target_pos - own.pos
    target_local = to_local(target_relative, own.heading)
    desired_local = to_local(desired_velocity, own.heading)
    values.extend(
        [
            float(target_local[0] / scale) if own_active else 0.0,
            float(target_local[1] / scale) if own_active else 0.0,
            float(norm(target_relative) / scale) if own_active else 0.0,
            float(desired_local[0] / own_speed_scale) if own_active else 0.0,
            float(desired_local[1] / own_speed_scale) if own_active else 0.0,
            *context[4:7].tolist(),
        ]
    )

    relative_red = red_state.pos - own.pos
    relative_red_local = to_local(relative_red, own.heading)
    red_velocity_local = to_local(red_state.velocity - own.velocity, own.heading)
    red_heading_delta = float(wrap_angle(red_state.heading - own.heading))
    values.extend(
        [
            float(relative_red_local[0] / scale) if own_active else 0.0,
            float(relative_red_local[1] / scale) if own_active else 0.0,
            float(red_velocity_local[0] / red_relative_speed_scale) if own_active else 0.0,
            float(red_velocity_local[1] / red_relative_speed_scale) if own_active else 0.0,
            math.sin(red_heading_delta) if own_active else 0.0,
            math.cos(red_heading_delta) if own_active else 0.0,
        ]
    )

    for other_index, other in enumerate(perceived):
        if other_index == friendly_index:
            continue
        other_active = bool(mask[other_index])
        if not own_active or not other_active:
            values.extend([0.0] * 9)
            continue
        relative_local = to_local(other.pos - own.pos, own.heading)
        relative_velocity_local = to_local(
            other.velocity - own.velocity, own.heading
        )
        other_ground_speed_scale = float(
            np.hypot(
                max(float(max_speed_by_type[other.unit_type]), 1.0e-6),
                sway_scale,
            )
        )
        relative_speed_scale = own_ground_speed_scale + other_ground_speed_scale
        values.extend(
            [
                1.0,
                float(other.unit_type == "surface"),
                float(other.surfaced),
                float(relative_local[0] / scale),
                float(relative_local[1] / scale),
                float(relative_velocity_local[0] / relative_speed_scale),
                float(relative_velocity_local[1] / relative_speed_scale),
            ]
        )
        fresh, age = (1.0, 0.0) if communication_features is None else communication_features[other_index]
        values.extend([float(fresh), float(age)])

    boundary = np.asarray(
        (
            own.pos[0],
            world_size[0] - own.pos[0],
            own.pos[1],
            world_size[1] - own.pos[1],
        ),
        dtype=np.float32,
    )
    island_active = float(island_radius) > 0.0
    island_local = (
        to_local(island_center - own.pos, own.heading)
        if island_active
        else np.zeros(2, dtype=np.float32)
    )
    island_clearance = (
        norm(island_center - own.pos) - float(island_radius)
        if island_active
        else 0.0
    )
    values.extend(
        (
            (boundary / scale).tolist()
            + [
                float(island_local[0] / scale),
                float(island_local[1] / scale),
                float(np.clip(island_clearance / scale, -1.0, 1.0)),
            ]
            if own_active
            else [0.0] * 7
        )
    )
    values.extend(context[7:9].tolist())
    values.extend(context[9:15].tolist())

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (ACTOR_OBSERVATION_DIM,):
        raise AssertionError(
            f"actor observation has shape {result.shape}, expected {(ACTOR_OBSERVATION_DIM,)}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("actor observation must be finite")
    return result


def build_critic_state(
    friendly_states: list[ShipState],
    red_state: ShipState,
    world_size: np.ndarray,
    island_center: np.ndarray,
    island_radius: float,
    landing_point: np.ndarray,
    max_speed_by_type: dict[str, float],
    max_speed_red: float,
    max_sway_speed: float,
    max_yaw_rate: float,
    task_context: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the fixed 96-field centralized critic state."""

    if len(friendly_states) != 5:
        raise ValueError("critic state requires exactly five friendly agents")
    context = np.asarray(task_context, dtype=np.float32).reshape(-1)
    if context.shape != (CRITIC_TASK_CONTEXT_DIM,):
        raise ValueError(
            f"critic task_context must have shape {(CRITIC_TASK_CONTEXT_DIM,)}"
        )
    mask = _activity_mask(active_mask, len(friendly_states))
    yaw_scale = max(float(max_yaw_rate), 1.0e-6)
    sway_scale = max(float(max_sway_speed), 1.0e-6)
    red_speed_scale = float(
        np.hypot(max(float(max_speed_red), 1.0e-6), sway_scale)
    )
    scale = max(float(np.max(world_size)), 1.0e-6)
    values: list[float] = []

    for state, active in zip(friendly_states, mask):
        if not active:
            values.extend([0.0] * 9)
            continue
        surge_scale = max(float(max_speed_by_type[state.unit_type]), 1.0e-6)
        speed_scale = float(np.hypot(surge_scale, sway_scale))
        values.extend(
            [
                1.0,
                float(state.unit_type == "surface"),
                float(state.pos[0] / world_size[0]),
                float(state.pos[1] / world_size[1]),
                float(state.velocity[0] / speed_scale),
                float(state.velocity[1] / speed_scale),
                math.sin(state.heading),
                math.cos(state.heading),
                float(state.yaw_rate / yaw_scale),
            ]
        )

    target_positions = context[0:10].reshape(5, 2)
    desired_velocities = context[10:20].reshape(5, 2)
    for index, state in enumerate(friendly_states):
        if not mask[index]:
            values.extend([0.0] * 4)
            continue
        surge_scale = max(float(max_speed_by_type[state.unit_type]), 1.0e-6)
        speed_scale = float(np.hypot(surge_scale, sway_scale))
        relative = target_positions[index] - state.pos
        values.extend(
            [
                float(relative[0] / scale),
                float(relative[1] / scale),
                float(desired_velocities[index, 0] / surge_scale),
                float(desired_velocities[index, 1] / surge_scale),
            ]
        )

    values.extend(
        [
            float(red_state.pos[0] / world_size[0]),
            float(red_state.pos[1] / world_size[1]),
            float(red_state.velocity[0] / red_speed_scale),
            float(red_state.velocity[1] / red_speed_scale),
            math.sin(red_state.heading),
            math.cos(red_state.heading),
            float(red_state.yaw_rate / yaw_scale),
            float(island_center[0] / world_size[0]),
            float(island_center[1] / world_size[1]),
            float(island_radius / scale),
            float(landing_point[0] / world_size[0]),
            float(landing_point[1] / world_size[1]),
            float(island_radius > 0.0),
        ]
    )
    values.extend(context[20:38].tolist())

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (CRITIC_STATE_DIM,):
        raise AssertionError(
            f"critic state has shape {result.shape}, expected {(CRITIC_STATE_DIM,)}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("critic state must be finite")
    return result


def _activity_mask(active_mask: np.ndarray | None, count: int) -> np.ndarray:
    if active_mask is None:
        return np.ones(count, dtype=bool)
    result = np.asarray(active_mask, dtype=np.float32).reshape(-1) > 0.5
    if result.shape != (count,):
        raise ValueError("active_mask must match friendly_states")
    return result
