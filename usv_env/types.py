from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ShipState:
    pos: np.ndarray
    speed: float
    heading: float
    yaw_rate: float
    unit_type: str
    surfaced: bool = True
    sway_speed: float = 0.0

    @property
    def body_velocity(self) -> np.ndarray:
        """Return the 3-DOF body velocity ``[surge, sway, yaw_rate]``."""

        return np.asarray(
            [self.speed, self.sway_speed, self.yaw_rate], dtype=np.float64
        )

    @property
    def ground_speed(self) -> float:
        """Return the translational speed magnitude ``sqrt(u^2 + v^2)``."""

        return float(np.hypot(self.speed, self.sway_speed))

    @property
    def velocity(self) -> np.ndarray:
        cosine = float(np.cos(self.heading))
        sine = float(np.sin(self.heading))
        return np.asarray(
            [
                cosine * float(self.speed) - sine * float(self.sway_speed),
                sine * float(self.speed) + cosine * float(self.sway_speed),
            ],
            dtype=np.float32,
        )


@dataclass
class EnvInfo:
    step: int
    success: bool
    hold_steps: int
    collision: bool
    boundary_violation: bool
    landing_success: bool
    surface_intervention_success: bool = False
    cooperative_task_success: bool = False
    task_name: str = "reef_interception"
    stage_hold_seconds: float = 0.0
    stage_success_gates: dict[str, bool] = field(default_factory=dict)
    unmet_success_gates: tuple[str, ...] = ()
    support_gates: dict[str, bool] = field(default_factory=dict)
    capture_geometry: dict[str, Any] = field(default_factory=dict)
    phase: str = "SEARCH"
    terminated: bool = False
    truncated: bool = False
    timeout: bool = False
    terminal_reason: str = ""
    collision_pair: str = ""
    reward_terms: dict[str, Any] = field(default_factory=dict)
    safety_diagnostics: dict[str, Any] = field(default_factory=dict)
    counterfactual_audit: dict[str, Any] = field(default_factory=dict)
    red_incident: dict[str, Any] = field(default_factory=dict)
    capture_success: bool = False
    capture_just_confirmed: bool = False
    first_capture_step: int | None = None
    first_capture_time_s: float | None = None
    post_capture_steps: int = 0
    withdrawal_cleared_entry: bool = False
    terminal_observation: np.ndarray | None = None
    terminal_state: np.ndarray | None = None
