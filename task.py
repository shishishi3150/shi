from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskEnvironment:
    red_speed: float
    red_speed_range: tuple[float, float]
    submarine_dynamics: bool
    static_slots: bool
    submarine_start_interpolation: float
    compaction_enabled: bool
    position_jitter: float
    heading_jitter_degrees: float


@dataclass(frozen=True)
class TaskSuccess:
    max_slot_error: float
    mean_slot_error: float
    mean_velocity_error: float
    max_angle_gap_degrees: float
    hold_seconds: float
    require_surface_intervention: bool
    require_compaction: bool
    require_triangle: bool


@dataclass(frozen=True)
class TaskTiming:
    final_hold_steps: int
    formation45_hold_steps: int
    minimum_shrink_steps: int


@dataclass(frozen=True)
class FinalTask:
    active_roles: tuple[str, ...]
    max_steps: int
    environment: TaskEnvironment
    success: TaskSuccess
    timing: TaskTiming


def resolve_final_task(config: Mapping[str, Any]) -> FinalTask:
    raw = config["final_task"]
    environment = raw["environment"]
    success = raw["success"]
    timing = raw["timing"]
    roles = tuple(str(value) for value in raw["active_roles"])
    if roles != ("surface", "submarine"):
        raise ValueError("final_task.active_roles must be [surface, submarine]")
    max_steps = int(raw["max_steps"])
    if max_steps <= 0:
        raise ValueError("final_task.max_steps must be positive")
    result = FinalTask(
        active_roles=roles,
        max_steps=max_steps,
        environment=TaskEnvironment(
            red_speed=float(environment["red_speed"]),
            red_speed_range=(float(environment["red_speed"]), float(environment["red_speed"])),
            submarine_dynamics=bool(environment["submarine_dynamics"]),
            static_slots=bool(environment["static_slots"]),
            submarine_start_interpolation=float(environment["submarine_start_interpolation"]),
            compaction_enabled=bool(environment["compaction_enabled"]),
            position_jitter=float(environment["position_jitter"]),
            heading_jitter_degrees=float(environment["heading_jitter_degrees"]),
        ),
        success=TaskSuccess(
            max_slot_error=float(success["max_slot_error"]),
            mean_slot_error=float(success["mean_slot_error"]),
            mean_velocity_error=float(success["mean_velocity_error"]),
            max_angle_gap_degrees=float(success["max_angle_gap_degrees"]),
            hold_seconds=float(success["hold_seconds"]),
            require_surface_intervention=bool(success["require_surface_intervention"]),
            require_compaction=bool(success["require_compaction"]),
            require_triangle=bool(success["require_triangle"]),
        ),
        timing=TaskTiming(
            final_hold_steps=int(timing["final_hold_steps"]),
            formation45_hold_steps=int(timing["formation45_hold_steps"]),
            minimum_shrink_steps=int(timing["minimum_shrink_steps"]),
        ),
    )
    if result.environment.red_speed < 0.0:
        raise ValueError("final_task.environment.red_speed must be non-negative")
    if min(result.timing.final_hold_steps, result.timing.formation45_hold_steps,
           result.timing.minimum_shrink_steps) < 0:
        raise ValueError("final_task timing values must be non-negative")
    return result
