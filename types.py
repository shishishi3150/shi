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


@dataclass(frozen=True)
class ObservationSchema:
    version: str
    actor_dim: int
    critic_dim: int
    actor_context_fields: tuple[str, ...]
    critic_context_fields: tuple[str, ...]


@dataclass
class EnvInfo:
    step: int
    success: bool
    hold_steps: int
    collision: bool
    boundary_violation: bool
    landing_success: bool
    surface_intervention_success: bool = False
    physical_capture_success: bool = False
    cooperative_task_success: bool = False
    relaxed_cooperative_success: bool = False
    strict_success: bool = False
    stage_success_relaxed: bool = False
    physical_capture: bool = False
    task_name: str = "final_task"
    stage_hold_seconds: float = 0.0
    stage_success_gates: dict[str, bool] = field(default_factory=dict)
    unmet_success_gates: tuple[str, ...] = ()
    physical_hold_credit: float = 0.0
    phase: str = "CHASE_AND_PRESSURE"
    terminated: bool = False
    truncated: bool = False
    timeout: bool = False
    terminal_reason: str = ""
    wall_contact: bool = False
    wall_contact_agents: tuple[str, ...] = ()
    collision_pair: str = ""
    collision_step: int | None = None
    collision_stage: str = ""
    first_physical_gate_step: int | None = None
    first_physical_capture_step: int | None = None
    first_cooperative_success_step: int | None = None
    first_hold_step: int | None = None
    first_encircle_step: int | None = None
    stable_steps: int = 0
    green_hold_steps: int = 0
    reward_terms: dict[str, Any] = field(default_factory=dict)
    premature_success_attempt: bool = False
    surface_b1_latched: bool = False
    surface_b2_latched: bool = False
    surface_causal_latched: bool = False
    surface_island_avoidance_latched: bool = False
    surface_complete_latched: bool = False
    red_actual_cross_track: float = 0.0
    red_shadow_cross_track: float = 0.0
    counterfactual_cross_track_gain: float = 0.0
    maximum_counterfactual_cross_track_gain: float = 0.0
    predicted_island_clearance: float = 0.0
    shadow_predicted_island_clearance: float = 0.0
    clearance_gain_vs_shadow: float = 0.0
    surface_gate_progress: float = 0.0
    b2_marginal_correction: float = 0.0
    surface_pressure_threshold_margins: tuple[float, float] = (0.0, 0.0)
    surface_net_displacements: tuple[float, float] = (0.0, 0.0)
    surface_path_lengths: tuple[float, float] = (0.0, 0.0)
    surface_path_efficiencies: tuple[float, float] = (0.0, 0.0)
    surface_useful_velocity_ratios: tuple[float, float] = (0.0, 0.0)
    surface_corridor_dwell_seconds: tuple[float, float] = (0.0, 0.0)
    surface_active_pressure_integrals: tuple[float, float] = (0.0, 0.0)
    surface_yaw_saturation_fractions: tuple[float, float] = (0.0, 0.0)
    surface_mean_abs_yaw_rates: tuple[float, float] = (0.0, 0.0)
    surface_mean_normalized_action_changes: tuple[float, float] = (0.0, 0.0)
    surface_mean_speeds: tuple[float, float] = (0.0, 0.0)
    counterfactual_gain_at_b1_latch: float = 0.0
    b2_incremental_counterfactual_gain: float = 0.0
    submarine_net_displacements: tuple[float, float, float] = (0.0, 0.0, 0.0)
    formation45_hold_steps: int = 0
    shrink_steps: int = 0
    final_hold_steps: int = 0
    reference_radius: float = 4.5
    surface_b1_latch_step: int | None = None
    surface_b2_latch_step: int | None = None
    surface_causal_latch_step: int | None = None
    surface_island_latch_step: int | None = None
    surface_complete_step: int | None = None
    formation45_hold_start_step: int | None = None
    formation45_hold_complete_step: int | None = None
    shrink_started_step: int | None = None
    radius35_step: int | None = None
    final_hold_start_step: int | None = None

    @property
    def landing_failure(self) -> bool:
        return self.landing_success
