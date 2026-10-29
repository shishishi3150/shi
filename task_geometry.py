from __future__ import annotations

import math

import numpy as np

from .geometry import wrap_angle

_EPSILON = 1e-8
_SLOT_COUNT = 3


def surface_role_reference(position, heading, velocity, red_position, red_velocity,
                           island_center, natural_tangent, desired_lateral_sign, role,
                           *, inner=4., outer=10., half_angle_degrees=20.,
                           k_pos=.08, max_speed=.4, motion_softening=2.,
                           heading_speed_scale=.05, heading_weight=1., velocity_weight=.5):
    """Annular sector midpoint and world ground-speed reference.

    This pure geometry helper supplies reward, observation and display; it never
    chooses an actuator command or consumes another friendly vessel's state.
    """
    position, red_position = _point(position, "position"), _point(red_position, "red_position")
    velocity, red_velocity = _point(velocity, "velocity"), _point(red_velocity, "red_velocity")
    tangent = _point(natural_tangent, "natural_tangent")
    tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 1e-10 else np.array([1., 0.])
    toward = tangent if island_center is None else _point(island_center, "island_center") - red_position
    g = toward / np.linalg.norm(toward) if np.linalg.norm(toward) > 1e-10 else tangent
    if role not in (0, 1) or desired_lateral_sign not in (-1, 1):
        raise ValueError("invalid fixed surface role or lateral sign")
    n = desired_lateral_sign * np.array([-g[1], g[0]])
    # B1 remains lateral pressure; B2 is a forward blocker on the
    # red-to-island escape line, so the roles are geometrically distinct.
    axis = -n if role == 0 else g
    # The angular bisector at the radial midpoint is independent of own position.
    # It follows Red and the role axis without sliding along the sector boundary.
    radius = .5 * (inner + outer)
    target = red_position + radius*axis
    error = float(np.linalg.norm(position-target))
    requested = red_velocity + k_pos*(target-position)
    speed = float(np.linalg.norm(requested))
    requested *= min(1., max_speed/max(speed, 1e-12))
    speed = float(np.linalg.norm(requested))
    a = (min(speed/heading_speed_scale, 1.) *
         (1.-float(np.clip(np.array([math.cos(heading), math.sin(heading)]) @ (requested/speed), -1., 1.)))/2.
         if speed > 1e-10 else 0.)
    b = min(float(np.sum((velocity-requested)**2))/(2*max_speed)**2, 1.)
    gate = error/(error+motion_softening)
    motion = gate*(heading_weight*a + velocity_weight*b)
    return dict(axis=axis, target=target, velocity=requested, error=error,
                heading_error=a, velocity_error=b, motion_error=motion, E=error+motion)


def _point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite")
    return point


def _alpha(value: float) -> float:
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def constant_velocity_prediction(
    red_pos: np.ndarray,
    red_velocity: np.ndarray,
    tau: float,
    drive_effect: np.ndarray | None = None,
    drive_gain: float = 0.0,
) -> np.ndarray:
    """Return the short-horizon red position prediction.

    ``drive_effect`` contributes only its direction. ``drive_gain`` therefore has
    velocity units and can be set to zero for a pure constant-velocity model.
    """

    position = _point(red_pos, "red_pos")
    velocity = _point(red_velocity, "red_velocity")
    horizon = float(tau)
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError("tau must be finite and non-negative")
    gain = float(drive_gain)
    if not math.isfinite(gain):
        raise ValueError("drive_gain must be finite")

    drive_direction = np.zeros(2, dtype=np.float64)
    if drive_effect is not None:
        effect = _point(drive_effect, "drive_effect")
        magnitude = float(np.linalg.norm(effect))
        if magnitude > _EPSILON:
            drive_direction = effect / magnitude

    predicted_velocity = velocity + gain * drive_direction
    return np.asarray(position + horizon * predicted_velocity, dtype=np.float32)


def low_pass_point(previous: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    previous_point = _point(previous, "previous")
    target_point = _point(target, "target")
    blend = _alpha(alpha)
    return np.asarray((1.0 - blend) * previous_point + blend * target_point, dtype=np.float32)


def smooth_heading(previous: float, target: float, alpha: float) -> float:
    previous_heading = float(previous)
    target_heading = float(target)
    if not math.isfinite(previous_heading) or not math.isfinite(target_heading):
        raise ValueError("headings must be finite")
    delta = float(wrap_angle(target_heading - previous_heading))
    return float(wrap_angle(previous_heading + _alpha(alpha) * delta))


def triangular_slots(center: np.ndarray, heading: float, radius: float) -> np.ndarray:
    slot_center = _point(center, "center")
    reference_heading = float(heading)
    reference_radius = float(radius)
    if not math.isfinite(reference_heading):
        raise ValueError("heading must be finite")
    if not math.isfinite(reference_radius) or reference_radius < 0.0:
        raise ValueError("radius must be finite and non-negative")

    angles = reference_heading + np.arange(_SLOT_COUNT, dtype=np.float64) * (
        2.0 * math.pi / _SLOT_COUNT
    )
    offsets = reference_radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return np.asarray(slot_center + offsets, dtype=np.float32)


def limit_reference(
    previous: np.ndarray,
    target: np.ndarray,
    max_speed: float,
    dt: float,
) -> np.ndarray:
    """Rate-limit one reference point or an array of reference points."""

    previous_points = np.asarray(previous, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if previous_points.shape != target_points.shape or previous_points.shape[-1:] != (2,):
        raise ValueError("previous and target must have matching (..., 2) shapes")
    if not np.all(np.isfinite(previous_points)) or not np.all(np.isfinite(target_points)):
        raise ValueError("reference points must be finite")
    speed_limit = float(max_speed)
    time_step = float(dt)
    if not math.isfinite(speed_limit) or speed_limit < 0.0:
        raise ValueError("max_speed must be finite and non-negative")
    if not math.isfinite(time_step) or time_step < 0.0:
        raise ValueError("dt must be finite and non-negative")

    delta = target_points - previous_points
    distances = np.linalg.norm(delta, axis=-1, keepdims=True)
    maximum_step = speed_limit * time_step
    scale = np.ones_like(distances)
    np.divide(maximum_step, distances, out=scale, where=distances > _EPSILON)
    scale = np.minimum(scale, 1.0)
    scale = np.where(distances > _EPSILON, scale, 0.0)
    return np.asarray(previous_points + delta * scale, dtype=np.float32)


def fixed_slot_ids(count: int = _SLOT_COUNT) -> np.ndarray:
    slot_count = int(count)
    if slot_count != _SLOT_COUNT:
        raise ValueError("dynamic predictive encirclement requires exactly three fixed slots")
    return np.arange(_SLOT_COUNT, dtype=np.int64)


def fixed_slot_reference(slots: np.ndarray, slot_id: int) -> np.ndarray:
    references = np.asarray(slots, dtype=np.float32)
    if references.shape != (_SLOT_COUNT, 2):
        raise ValueError("slots must have shape (3, 2)")
    identity = int(slot_id)
    if identity < 0 or identity >= _SLOT_COUNT:
        raise IndexError(f"slot_id must be in [0, {_SLOT_COUNT - 1}]")
    return references[identity].copy()


def triangle_margin(point: np.ndarray, triangle: np.ndarray) -> float:
    """Return signed minimum edge distance; non-negative means inside.

    A collinear or collapsed physical formation has no interior, so it is
    represented as a finite, non-positive margin instead of raising.  Invalid
    shapes and non-finite inputs remain programmer errors.
    """

    query = _point(point, "point")
    vertices = np.asarray(triangle, dtype=np.float64)
    if vertices.shape != (_SLOT_COUNT, 2) or not np.all(np.isfinite(vertices)):
        raise ValueError("triangle must be a finite array with shape (3, 2)")

    twice_area = _cross_2d(vertices[1] - vertices[0], vertices[2] - vertices[0])
    if abs(twice_area) <= _EPSILON:
        segment_distances: list[float] = []
        for index in range(_SLOT_COUNT):
            start = vertices[index]
            edge = vertices[(index + 1) % _SLOT_COUNT] - start
            squared_length = float(np.dot(edge, edge))
            if squared_length <= _EPSILON * _EPSILON:
                continue
            projection = float(
                np.clip(
                    np.dot(query - start, edge) / squared_length,
                    0.0,
                    1.0,
                )
            )
            closest = start + projection * edge
            segment_distances.append(float(np.linalg.norm(query - closest)))
        if segment_distances:
            return -float(min(segment_distances))
        return -float(np.linalg.norm(query - vertices[0]))
    orientation = 1.0 if twice_area > 0.0 else -1.0

    signed_distances: list[float] = []
    for index in range(_SLOT_COUNT):
        start = vertices[index]
        edge = vertices[(index + 1) % _SLOT_COUNT] - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= _EPSILON:
            raise ValueError("triangle edges must have positive length")
        signed_distance = orientation * _cross_2d(edge, query - start) / edge_length
        signed_distances.append(signed_distance)
    return float(min(signed_distances))


def max_angular_gap(positions: np.ndarray, center: np.ndarray) -> float:
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("positions must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    origin = _point(center, "center")
    relative = points - origin
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * math.pi)
    angles.sort()
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * math.pi)))
    return float(np.max(gaps))


def surface_repulsion_force(
    red_pos: np.ndarray,
    surface_pos: np.ndarray,
    repulsion_range: float,
    repulsion_gain: float,
) -> np.ndarray:
    """Compute one surface vessel's APF repulsion on the red vessel."""

    red_position = _point(red_pos, "red_pos")
    surface_position = _point(surface_pos, "surface_pos")
    interaction_range = float(repulsion_range)
    gain = float(repulsion_gain)
    if not math.isfinite(interaction_range) or interaction_range <= 0.0:
        raise ValueError("repulsion_range must be finite and positive")
    if not math.isfinite(gain):
        raise ValueError("repulsion_gain must be finite")

    offset = red_position - surface_position
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-6 or distance >= interaction_range:
        return np.zeros(2, dtype=np.float32)
    strength = ((interaction_range - distance) / interaction_range) ** 2
    return np.asarray(gain * strength * offset / distance, dtype=np.float32)


def combined_surface_repulsion_force(
    red_pos: np.ndarray,
    surface_positions: np.ndarray,
    repulsion_range: float,
    repulsion_gain: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.asarray(surface_positions, dtype=np.float64)
    if positions.size == 0:
        return np.zeros(2, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.all(np.isfinite(positions)):
        raise ValueError("surface_positions must be a finite array with shape (N, 2)")

    if active_mask is None:
        mask = np.ones(len(positions), dtype=np.float64)
    else:
        mask = np.asarray(active_mask, dtype=np.float64)
        if mask.shape != (len(positions),) or not np.all(np.isfinite(mask)):
            raise ValueError("active_mask must have shape (N,) and be finite")

    force = np.zeros(2, dtype=np.float64)
    for surface_position, active in zip(positions, mask):
        force += float(active) * surface_repulsion_force(
            red_pos,
            surface_position,
            repulsion_range,
            repulsion_gain,
        )
    return np.asarray(force, dtype=np.float32)


def lateral_projection(force: np.ndarray, direction: np.ndarray) -> float:
    force_vector = _point(force, "force")
    lateral_direction = _point(direction, "direction")
    direction_norm = float(np.linalg.norm(lateral_direction))
    if direction_norm <= _EPSILON:
        raise ValueError("direction must be non-zero")
    return float(np.dot(force_vector, lateral_direction / direction_norm))


def direction_error(force: np.ndarray, desired_direction: np.ndarray) -> float:
    force_vector = _point(force, "force")
    desired = _point(desired_direction, "desired_direction")
    desired_norm = float(np.linalg.norm(desired))
    if desired_norm <= _EPSILON:
        raise ValueError("desired_direction must be non-zero")
    force_norm = float(np.linalg.norm(force_vector))
    if force_norm <= _EPSILON:
        return math.pi
    cosine = float(np.clip(np.dot(force_vector / force_norm, desired / desired_norm), -1.0, 1.0))
    return float(math.acos(cosine))


def b2_marginal_correction(
    b1_force: np.ndarray,
    b2_force: np.ndarray,
    desired_direction: np.ndarray,
) -> float:
    """Positive values mean B2 improves B1's force direction."""

    first_force = _point(b1_force, "b1_force")
    second_force = _point(b2_force, "b2_force")
    before = direction_error(first_force, desired_direction)
    after = direction_error(first_force + second_force, desired_direction)
    return float(before - after)
