from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + math.pi) % TWO_PI - math.pi


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float32)))


def angle_of(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float32)
    return float(math.atan2(float(value[1]), float(value[0])))


def to_local(vector: np.ndarray, heading: float) -> np.ndarray:
    c = math.cos(-heading)
    s = math.sin(-heading)
    x, y = np.asarray(vector, dtype=np.float32)
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def rotate(vector: np.ndarray, heading: float) -> np.ndarray:
    c = math.cos(heading)
    s = math.sin(heading)
    x, y = np.asarray(vector, dtype=np.float32)
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def capsule_segment(position: np.ndarray, heading: float, length: float, width: float) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(position, dtype=np.float64)
    radius = 0.5 * float(width)
    half_line = max(0.5 * float(length) - radius, 0.0)
    direction = np.asarray((math.cos(heading), math.sin(heading)), dtype=np.float64)
    return center - half_line * direction, center + half_line * direction, radius


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start
    denom = float(delta @ delta)
    if denom <= 1.0e-16:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(((point - start) @ delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - start - t * delta))


def segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    def cross(left: np.ndarray, right: np.ndarray) -> float:
        return float(left[0] * right[1] - left[1] * right[0])
    p, q = np.asarray(a0, dtype=np.float64), np.asarray(b0, dtype=np.float64)
    r = np.asarray(a1, dtype=np.float64) - p
    s = np.asarray(b1, dtype=np.float64) - q
    denominator = cross(r, s)
    if abs(denominator) > 1.0e-12:
        t, u = cross(q - p, s) / denominator, cross(q - p, r) / denominator
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(point_segment_distance(p, q, q + s), point_segment_distance(p + r, q, q + s),
               point_segment_distance(q, p, p + r), point_segment_distance(q + s, p, p + r))


def capsules_overlap(left: tuple[np.ndarray, np.ndarray, float], right: tuple[np.ndarray, np.ndarray, float]) -> bool:
    return segment_distance(left[0], left[1], right[0], right[1]) <= left[2] + right[2]


def capsule_circle_clearance(capsule: tuple[np.ndarray, np.ndarray, float], center: np.ndarray, radius: float) -> float:
    return point_segment_distance(center, capsule[0], capsule[1]) - capsule[2] - float(radius)


def capsule_inside_rectangle(capsule: tuple[np.ndarray, np.ndarray, float], world_size: np.ndarray) -> bool:
    points = np.stack((capsule[0], capsule[1]))
    radius = capsule[2]
    limit = np.asarray(world_size, dtype=np.float64)
    return bool(np.all(points >= radius) and np.all(points <= limit - radius))
