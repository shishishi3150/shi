from __future__ import annotations

import math
from collections import deque
from itertools import permutations
from typing import Any

import numpy as np

from .config import TASK_CONTRACT
from .task import FinalTask, resolve_final_task
from .dynamics import AUVDynamics3DOF, USVDynamics
from .geometry import (angle_of, norm, unit, wrap_angle, capsule_segment,
                       capsules_overlap, capsule_circle_clearance, capsule_inside_rectangle)
from .communication import CommunicationNetwork
from .waves import RegularLinearWave
from .observation import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    build_actor_observation,
    build_critic_state,
)
from .red_policy import red_policy_reference
from .reward import RewardSnapshot, qualities, evaluate, surface_reference
from .config import validate
from dataclasses import replace
from .seeding import derive_episode_seeds
from .task_geometry import (
    b2_marginal_correction,
    constant_velocity_prediction,
    fixed_slot_ids,
    max_angular_gap,
    surface_repulsion_force,
    triangle_margin,
    triangular_slots,
)
from .types import EnvInfo, ObservationSchema, ShipState


PHASES = ("CHASE_AND_PRESSURE", "HOLD", "ENCIRCLE", "SUCCESS")
RED_BOUNDARY_CLEARANCE_MIN = 4.0


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or array[0] > array[1]:
        raise ValueError(f"{name} must be a finite [low, high] pair")
    return float(array[0]), float(array[1])


class PursuitEvasionEnv:
    """Single dynamic predictive encirclement environment.

    Dynamics, red APF, collision, boundary, and island landing remain physical.
    Only the task geometry, state machine, observations, rewards, and success
    contract are defined here.
    """


    def _initialize_physics(self, config: dict[str, Any], seed: int | None = None) -> None:
        self.config = config
        self.geometry_scale = float(config.get("scene_geometry", {}).get("scale", 1.0))
        self.task_cfg = dict(config["task"])
        if str(self.task_cfg["contract"]) != TASK_CONTRACT:
            raise ValueError(f"task.contract must be {TASK_CONTRACT}")
        self.encirclement_cfg = dict(self.task_cfg["encirclement"])
        self.pressure_cfg = dict(self.task_cfg["surface_pressure"])
        self.intervention_cfg = dict(self.pressure_cfg["intervention_success"])
        self.tracking_success_cfg = dict(self.pressure_cfg["tracking_success"])
        self.physical_cfg = dict(self.task_cfg["physical_capture"])
        self.reward_cfg = dict(config["reward"])
        self.scenario_cfg = dict(config["scenario"])
        self.spawn_cfg = dict(self.scenario_cfg["task_spawn"])
        configured_seed = int(self.scenario_cfg.get("seed", 0))
        self.seed = configured_seed if seed is None else int(seed)
        initial_seeds = derive_episode_seeds(self.seed)
        self.master_seed = initial_seeds.master_seed
        self._master_sequence_rng = np.random.default_rng(self.master_seed)
        self.scenario_seed = initial_seeds.scenario_seed
        self.rng = np.random.default_rng(self.scenario_seed)

        env_cfg = config["env"]
        ship_cfg = config["ship"]
        island_cfg = config["island"]
        self.world_size = np.asarray(env_cfg["world_size"], dtype=np.float32)
        self.dt = float(env_cfg["dt"])
        self.base_max_steps = int(env_cfg["max_steps"])
        self.max_steps = self.base_max_steps
        self.num_surface_blue = int(env_cfg["num_surface_blue"])
        self.num_submarines = int(env_cfg["num_submarines"])
        if self.num_surface_blue != 2 or self.num_submarines != 3:
            raise ValueError("dynamic predictive encirclement requires exactly 2 surface boats and 3 submarines")
        self.num_friendly = self.num_surface_blue + self.num_submarines
        self.surface_slice = slice(0, self.num_surface_blue)
        self.submarine_slice = slice(self.num_surface_blue, self.num_friendly)
        self.slot_ids = fixed_slot_ids(self.num_submarines)
        self.submarine_slot_assignment = self.slot_ids.copy()
        self.semi_major = float(ship_cfg["semi_major"])
        self.semi_minor = float(ship_cfg["semi_minor"])
        self.surface_length = float(ship_cfg.get("surface_length_m", 1.255))
        self.surface_collision_width = float(ship_cfg.get("surface_collision_width_m", 0.29))
        self.submarine_length = float(ship_cfg.get("submarine_collision_length_m", 0.60))
        self.submarine_collision_width = float(ship_cfg.get("submarine_collision_width_m", 0.31))
        self.wave_cfg = dict(config.get("wave", {}))
        self.communication_cfg = dict(config.get("communication", {}))
        self.wave_seed = initial_seeds.wave_seed
        self.communication_seed = initial_seeds.communication_seed
        # Optional experimental channel used by the slot-observation probe.
        # It is disabled by default so existing training/evaluation contracts
        # remain unchanged until a new model is trained for this observation.
        self.slot_observation_enabled = False
        self.slot_observation_loss = 0.0
        self.slot_observation_delay_steps = (0, 0)
        self.slot_observation_seed_offset = 7919
        self._slot_observation_rng = np.random.default_rng(self.communication_seed + self.slot_observation_seed_offset)
        self._slot_observation_cache = None
        self._slot_observation_velocity_cache = None
        self._slot_observation_last_step = None
        self._slot_observation_pending = None
        self.max_speed_by_type = {
            "surface": float(ship_cfg["max_speed_blue"]),
            "submarine": float(ship_cfg["max_speed_submarine"]),
        }
        self.max_speed_red = float(ship_cfg["max_speed_red"])
        self.max_yaw_rate = float(ship_cfg["max_yaw_rate"])
        self.max_surge_force = float(ship_cfg["max_surge_force"])
        self.max_yaw_moment = float(ship_cfg["max_yaw_moment"])
        self.max_sway_speed = float(ship_cfg["max_sway_speed"])
        self.action_dim = 2
        self.surface_action_limit = np.asarray(
            [self.max_surge_force, self.max_yaw_moment], dtype=np.float32
        )
        # Keep ``action_limit`` as the public surface-vessel compatibility
        # alias.  Heterogeneous consumers must use ``agent_action_limits``.
        self.action_limit = self.surface_action_limit.copy()
        submarine_dynamics_cfg = dict(ship_cfg["submarine_dynamics"])
        self.submarine_max_surge_force = float(
            submarine_dynamics_cfg["max_surge_force"]
        )
        self.submarine_max_yaw_moment = float(
            submarine_dynamics_cfg["max_yaw_moment"]
        )
        self.submarine_action_limit = np.asarray(
            [self.submarine_max_surge_force, self.submarine_max_yaw_moment],
            dtype=np.float32,
        )
        self.agent_action_limits = np.concatenate(
            (
                np.repeat(
                    self.surface_action_limit[None, :],
                    self.num_surface_blue,
                    axis=0,
                ),
                np.repeat(
                    self.submarine_action_limit[None, :],
                    self.num_submarines,
                    axis=0,
                ),
            ),
            axis=0,
        ).astype(np.float32)
        dynamics_cfg = dict(ship_cfg["dynamics"])
        dynamics_kwargs = {
            "dt": self.dt,
            "allow_reverse": bool(ship_cfg["allow_reverse"]),
            "max_yaw_rate": self.max_yaw_rate,
            "max_surge_force": self.max_surge_force,
            "max_yaw_moment": self.max_yaw_moment,
            "max_sway_speed": self.max_sway_speed,
            "internal_substeps": int(dynamics_cfg["internal_substeps"]),
            "reference_max_acceleration": float(
                dynamics_cfg["reference_max_acceleration"]
            ),
            "reference_max_yaw_acceleration": float(
                dynamics_cfg["reference_max_yaw_acceleration"]
            ),
            "reference_sway_damping": float(
                dynamics_cfg["reference_sway_damping"]
            ),
            "parameters": dict(dynamics_cfg["parameters"]),
        }
        self.blue_dynamics = USVDynamics(
            max_speed=self.max_speed_by_type["surface"],
            reverse_speed_ratio=float(ship_cfg["blue_reverse_speed_ratio"]),
            **dynamics_kwargs,
        )
        submarine_dynamics_kwargs = {
            "dt": self.dt,
            "allow_reverse": bool(ship_cfg["allow_reverse"]),
            "max_yaw_rate": self.max_yaw_rate,
            "max_surge_force": self.submarine_max_surge_force,
            "max_yaw_moment": self.submarine_max_yaw_moment,
            "max_sway_speed": self.max_sway_speed,
            "internal_substeps": int(submarine_dynamics_cfg["internal_substeps"]),
            "reference_max_acceleration": float(
                submarine_dynamics_cfg["reference_max_acceleration"]
            ),
            "reference_max_yaw_acceleration": float(
                submarine_dynamics_cfg["reference_max_yaw_acceleration"]
            ),
            "reference_sway_damping": float(
                submarine_dynamics_cfg["reference_sway_damping"]
            ),
            "parameters": dict(submarine_dynamics_cfg["parameters"]),
        }
        self.submarine_dynamics = AUVDynamics3DOF(
            max_speed=self.max_speed_by_type["submarine"],
            reverse_speed_ratio=float(ship_cfg["submarine_reverse_speed_ratio"]),
            **submarine_dynamics_kwargs,
        )
        self.red_dynamics = USVDynamics(
            max_speed=self.max_speed_red,
            reverse_speed_ratio=1.0,
            **dynamics_kwargs,
        )

        self.island_center = (
            self.world_size * np.asarray(island_cfg["center_ratio"], dtype=np.float32)
        ).astype(np.float32)
        self.island_radius = float(min(self.world_size)) * float(
            island_cfg["radius_short_side_factor"]
        )
        self.landing_buffer = float(island_cfg["landing_buffer"])
        self.route_endpoint = self.island_center.copy()
        self.nominal_red_position = np.asarray(
            self.scenario_cfg["red_start"], dtype=np.float32
        )
        nominal_route = self.route_endpoint - self.nominal_red_position
        if norm(nominal_route) <= 1.0e-6:
            raise ValueError("scenario.red_start and the island must differ")
        self.nominal_route_tangent = unit(angle_of(nominal_route))
        nominal_left_normal = np.asarray(
            [-self.nominal_route_tangent[1], self.nominal_route_tangent[0]],
            dtype=np.float32,
        )
        desired_sign = float(self.pressure_cfg["desired_lateral_sign"])
        if desired_sign not in (-1.0, 1.0):
            raise ValueError("desired_lateral_sign must be -1 or +1")
        self.nominal_desired_lateral_direction = (
            nominal_left_normal * desired_sign
        ).astype(np.float32)
        self.surface_spawn_lateral_direction = (
            -self.nominal_desired_lateral_direction
        ).astype(np.float32)

        self.initial_radius = float(self.encirclement_cfg["initial_radius"])
        self.compact_radius = float(self.encirclement_cfg["compact_radius"])
        self.prediction_horizon_seconds = float(
            self.encirclement_cfg["prediction_horizon_seconds"]
        )
        self.slot_arrival_tolerance = float(self.encirclement_cfg["slot_arrival_tolerance"])
        self.stabilization_seconds = float(self.encirclement_cfg["stabilization_seconds"])
        self.shrink_rate_mps = float(self.encirclement_cfg.get("shrink_rate_mps", 0.05))
        if not math.isclose(self.shrink_rate_mps, 0.05, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("final task requires shrink_rate_mps=0.05")
        self.minimum_shrink_steps = int(
            self.encirclement_cfg.get("minimum_shrink_steps", 134)
        )
        required_shrink_steps = math.ceil(
            (self.initial_radius - self.compact_radius) / (self.shrink_rate_mps * self.dt) - 1e-10
        )
        if self.minimum_shrink_steps < required_shrink_steps:
            raise ValueError("minimum shrink steps cannot be shorter than radius travel time")
        self.green_hold_seconds = float(self.encirclement_cfg["green_hold_seconds"])
        self.radius_ref = self.initial_radius
        self.base_active_surface_mask = np.asarray(
            self.task_cfg["active_surface_mask"], dtype=np.float32
        )
        if self.base_active_surface_mask.shape != (2,) or np.any(
            (self.base_active_surface_mask != 0.0)
            & (self.base_active_surface_mask != 1.0)
        ):
            raise ValueError("task.active_surface_mask must contain exactly two binary entries")
        self.active_surface_mask = self.base_active_surface_mask.copy()
        self._agent_active_mask = np.ones(self.num_friendly, dtype=np.float32)

        self.history_profile_name = str(self.pressure_cfg["active_history_profile"])
        profiles = self.pressure_cfg["history_profiles"]
        if self.history_profile_name not in profiles:
            raise ValueError(f"unknown surface pressure history profile: {self.history_profile_name}")
        self.history_profile = dict(profiles[self.history_profile_name])

        self.evaluation_variant = "fixed"
        self.task_config: FinalTask = resolve_final_task(config)
        self._stage_position_jitter = 0.0
        self._stage_heading_jitter_degrees = 0.0
        self._sampled_submarine_headings = np.zeros(self.num_submarines, dtype=np.float32)
        self.set_evaluation_variant("fixed")

        self.blue_states: list[ShipState] = []
        self.submarine_states: list[ShipState] = []
        self.red_state = ShipState(np.zeros(2, dtype=np.float32), 0.0, 0.0, 0.0, "red")
        self.observation_schema: ObservationSchema
        self.observation_dim = 0
        self.critic_state_dim = 0
        obs, state = self.reset(seed=self.seed)
        self.observation_dim = int(obs.shape[1])
        self.critic_state_dim = int(state.shape[0])
        actor_context, actor_names = self._actor_task_context(0)
        critic_context, critic_names = self._critic_task_context()
        del actor_context, critic_context
        self.observation_schema = ObservationSchema(
            version="cooperative_blockade_reference_free_surface_v2_wave_comm_capsule_auv06",
            actor_dim=self.observation_dim,
            critic_dim=self.critic_state_dim,
            actor_context_fields=tuple(actor_names),
            critic_context_fields=tuple(critic_names),
        )


    @property
    def obs_dim(self) -> int:
        return self.observation_dim


    @property
    def state_dim(self) -> int:
        return self.critic_state_dim


    @property
    def active_roles(self) -> tuple[str, ...]:
        return self.task_config.active_roles


    def scripted_reference_action(
        self,
        state: ShipState,
        *,
        desired_yaw_rate: float,
        desired_surge_acceleration: float,
    ) -> np.ndarray:
        """Map a deterministic controller reference to force and moment."""

        if state.unit_type == "surface":
            dynamics = self.blue_dynamics
        elif state.unit_type == "submarine":
            dynamics = self.submarine_dynamics
        elif state.unit_type == "red":
            dynamics = self.red_dynamics
        else:
            raise ValueError(f"unknown ship unit_type: {state.unit_type}")
        return dynamics.reference_action(
            state,
            desired_yaw_rate=desired_yaw_rate,
            desired_surge_acceleration=desired_surge_acceleration,
        )


    def set_evaluation_variant(self, evaluation_variant: str = "fixed") -> FinalTask:
        """Select fixed, randomized, or training deployment for the final task."""
        normalized_variant = str(evaluation_variant).lower()
        if normalized_variant in {"jittered", "perturbed"}:
            normalized_variant = "randomized"
        if normalized_variant not in {"fixed", "randomized", "training"}:
            raise ValueError(
                "evaluation_variant must be 'fixed', 'randomized', or 'training'"
            )
        self.evaluation_variant = normalized_variant
        self.island_active = True
        self.surface_apf_enabled = True
        self.surface_dynamics_enabled = True
        self.max_steps = self.task_config.max_steps
        self.final_hold_target_steps = int(self.task_config.timing.final_hold_steps)
        self.surface_final_hold_target_steps = int(
            self.tracking_success_cfg["window_steps"]
        )
        self.formation45_hold_target_steps = int(
            self.task_config.timing.formation45_hold_steps
        )
        self.minimum_shrink_steps = max(
            self.minimum_shrink_steps,
            int(self.task_config.timing.minimum_shrink_steps),
        )
        jitter_enabled = normalized_variant in {"randomized", "training"}
        self._stage_position_jitter = (
            self.task_config.environment.position_jitter if jitter_enabled else 0.0
        )
        self._stage_heading_jitter_degrees = (
            self.task_config.environment.heading_jitter_degrees if jitter_enabled else 0.0
        )
        self.red_policy_cfg = dict(self.config["red_policy"])
        self.red_policy_cfg["target_speed"] = self.task_config.environment.red_speed
        surface_active = 1.0
        submarine_active = 1.0
        self.active_surface_mask = (
            self.base_active_surface_mask * surface_active
        ).astype(np.float32)
        self._agent_active_mask = np.concatenate(
            (
                self.active_surface_mask,
                np.full(self.num_submarines, submarine_active, dtype=np.float32),
            )
        ).astype(np.float32)
        return self.task_config


    def agent_active_mask(self, count: int | None = None) -> np.ndarray:
        if count is not None and int(count) != self.num_friendly:
            raise ValueError(
                f"agent_active_mask count must be {self.num_friendly}, got {count}"
            )
        return self._agent_active_mask.copy()


    def reset(
        self,
        seed: int | None = None,
        *,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        options = dict(options or {})
        if options:
            requested_variant = str(
                options.get("evaluation_variant", self.evaluation_variant)
            )
            self.set_evaluation_variant(requested_variant)
        if seed is None:
            master_seed = int(
                self._master_sequence_rng.integers(
                    0, np.iinfo(np.uint32).max, dtype=np.uint32
                )
            )
        else:
            master_seed = int(seed)
            self._master_sequence_rng = np.random.default_rng(master_seed)
        derived = derive_episode_seeds(master_seed)
        self.master_seed = derived.master_seed
        self.seed = derived.master_seed
        self.scenario_seed = int(options.get("scenario_seed", derived.scenario_seed))
        self.wave_seed = int(options.get("wave_seed", derived.wave_seed))
        self.communication_seed = int(
            options.get("communication_seed", derived.communication_seed)
        )
        self.rng = np.random.default_rng(self.scenario_seed)
        self.step_count = 0
        self._episode_done = False
        self.terminal_reason = ""
        self.wall_contact = False
        self.wall_contact_agents = ()
        self.collision_pair = ""
        self.collision_step: int | None = None
        self.collision_stage = ""
        self.final_gate_evaluation_steps = 0
        self.final_gate_failure_counts = {
            "distance_band_failed": 0,
            "core_band_failed": 0,
            "triangle_failed": 0,
            "angular_gap_failed": 0,
            "slot_error_failed": 0,
            "relative_speed_failed": 0,
            "collision_failed": 0,
        }
        self.phase = PHASES[0]
        self.physical_capture_success = False
        self.cooperative_task_success = False
        self.relaxed_cooperative_success = False
        self.strict_success = False
        self.stage_success = False
        self.stage_hold_seconds = 0.0
        self.stage_hold_step_count = 0
        self.surface_final_hold_steps = 0
        self.surface_pressure_valid_steps = 0
        self.surface_pressure_total_steps = 0
        self.formation45_hold_steps = 0
        self.shrink_steps = 0
        self.final_geometry_hold_steps = 0
        self.premature_success_attempt = False
        self.surface_b1_latched = False
        self.surface_b2_latched = False
        self.surface_causal_latched = False
        self.surface_island_avoidance_latched = False
        self.surface_complete_latched = False
        self.surface_b1_latch_step: int | None = None
        self.surface_b2_latch_step: int | None = None
        self.surface_causal_latch_step: int | None = None
        self.surface_island_latch_step: int | None = None
        self.surface_complete_step: int | None = None
        self.formation45_hold_start_step: int | None = None
        self.formation45_hold_complete_step: int | None = None
        self.shrink_started_step: int | None = None
        self.radius35_step: int | None = None
        self.final_hold_start_step: int | None = None
        self.first_stage_success_step: int | None = None
        self.first_relaxed_success_step: int | None = None
        self.physical_hold_credit = 0.0
        self.first_physical_gate_step: int | None = None
        self.first_physical_capture_step: int | None = None
        self.first_cooperative_success_step: int | None = None
        self.first_hold_step: int | None = None
        self.first_encircle_step: int | None = None
        self.physical_core_gate = False
        self.previous_actions = np.zeros((self.num_friendly, self.action_dim), dtype=np.float32)
        self.last_action_delta = np.zeros_like(self.previous_actions)
        self.last_executed_actions = np.zeros_like(self.previous_actions)
        self.episode_action_history: list[np.ndarray] = []
        self._clamp_start = {
            "surface": dict(self.blue_dynamics.clamp_counts),
            "submarine": dict(self.submarine_dynamics.clamp_counts),
            "red": dict(self.red_dynamics.clamp_counts),
        }
        self.surface_apf_forces = np.zeros((2, 2), dtype=np.float32)
        self.current_b1_projection = 0.0
        self.current_b2_projection = 0.0
        self.current_combined_projection = 0.0
        self.current_b2_marginal = 0.0
        self.b1_history_contribution = 0.0
        self.b2_history_contribution = 0.0
        self.combined_pressure_history = 0.0
        self.b1_apf_projection_integral = 0.0
        self.b2_apf_projection_integral = 0.0
        self.combined_apf_projection_integral = 0.0
        self.max_favorable_lateral_deviation = 0.0
        self.current_lateral_deviation = 0.0
        self.current_lateral_velocity = 0.0
        self.red_path_length = 0.0
        self.red_heading_change = 0.0
        self.red_actual_cross_track = 0.0
        self.red_shadow_cross_track = 0.0
        self.counterfactual_cross_track_gain = 0.0
        self.maximum_counterfactual_cross_track_gain = 0.0
        self.predicted_island_clearance = 0.0
        self.shadow_predicted_island_clearance = 0.0
        self.clearance_gain_vs_shadow = 0.0
        self.surface_gate_progress = 0.0
        self.surface_causal_hold_steps = 0
        self.surface_island_hold_steps = 0

        randomized_deployment = self.evaluation_variant in {"randomized", "training"}
        self.submarine_slot_assignment = fixed_slot_ids(self.num_submarines)
        deployment_attempts = (
            min(100, int(self.spawn_cfg.get("max_sampling_attempts", 1000)))
            if randomized_deployment and "submarine" in self.task_config.active_roles
            else 1
        )
        deployment_error: RuntimeError | None = None
        for _ in range(deployment_attempts):
            red_position, red_heading, red_speed = self._sample_red_reset()
            self.route_origin = red_position.copy()
            natural_route = self.route_endpoint - red_position
            if norm(natural_route) <= 1.0e-6:
                raise RuntimeError(f"seed={self.seed}: red spawn coincides with island reference")
            self.natural_route_tangent = unit(angle_of(natural_route))
            natural_left_normal = np.asarray(
                [-self.natural_route_tangent[1], self.natural_route_tangent[0]],
                dtype=np.float32,
            )
            self.desired_lateral_direction = (
                natural_left_normal * float(self.pressure_cfg["desired_lateral_sign"])
            ).astype(np.float32)
            self.n_drive = self.desired_lateral_direction.copy()
            self.surface_spawn_lateral_direction = (-self.n_drive).astype(np.float32)
            self.red_state = ShipState(
                red_position.astype(np.float32),
                red_speed,
                float(wrap_angle(red_heading)),
                0.0,
                "red",
                True,
            )
            self.shadow_red_state = self._copy_ship_state(self.red_state)
            self.initial_red_heading = self.red_state.heading
            self.pressure_zones = self._build_pressure_zones()
            self.blue_states = self._spawn_surface_states()

            self.prediction_center = self.red_state.pos.copy()
            self.prediction_heading = float(self.red_state.heading)
            self._encircle_slot_heading = None
            self.encircle_beta = 0.0
            self.slot_center = self.prediction_center.copy()
            self.radius_ref = self.initial_radius
            initial_center = constant_velocity_prediction(
                self.red_state.pos,
                self.red_state.velocity,
                self.prediction_horizon_seconds,
            )
            raw_slots = triangular_slots(
                initial_center,
                self.red_state.heading,
                self.radius_ref,
            ).astype(np.float32)
            initial_slots = raw_slots[[2, 0, 1]]
            try:
                submarine_states = self._spawn_submarine_states(initial_slots)
            except RuntimeError as error:
                deployment_error = error
                continue
            self.submarine_states = submarine_states
            break
        else:
            raise RuntimeError(
                f"seed={self.seed}: unable to sample a jointly reachable final-task deployment"
            ) from deployment_error

        self.slot_ids = self.submarine_slot_assignment.copy()
        assigned_initial_slots = initial_slots[self.submarine_slot_assignment]
        self.submarine_target_slots = assigned_initial_slots.copy()
        self._static_submarine_target_slots = assigned_initial_slots.copy()
        self._static_slot_center = initial_center.copy()
        self._static_slot_heading = float(self.red_state.heading)
        self.formation_radius = self.initial_radius
        self.stable_seconds = 0.0
        self.green_seconds = 0.0
        self.compaction_started = False
        self.compaction_complete = False
        self.submarine_initial_positions = np.asarray(
            [state.pos for state in self.submarine_states], dtype=np.float32
        )
        self.submarine_path_lengths = np.zeros(self.num_submarines, dtype=np.float64)
        self.submarine_net_displacements = np.zeros(self.num_submarines, dtype=np.float64)
        self.surface_initial_positions = np.asarray(
            [state.pos for state in self.blue_states], dtype=np.float32
        )
        self.surface_previous_positions = self.surface_initial_positions.copy()
        self.submarine_previous_positions = self.submarine_initial_positions.copy()
        self.submarine_target_velocities = np.zeros((3, 2), dtype=np.float32)
        self.pressure_zone_velocities = np.zeros((2, 2), dtype=np.float32)
        self._reset_intervention_windows()
        self._update_dynamic_references(initialize=True)
        self._reset_slot_observation_channel()
        self._refresh_geometry_metrics()
        self._update_counterfactual_metrics(initialize=True)
        self.wave = RegularLinearWave.sample(self.wave_cfg, self.wave_seed)
        self.wave_velocity_samples = {"surface": [], "red": []}
        self.communication = CommunicationNetwork(
            self.num_friendly, self.communication_cfg, self.communication_seed
        )
        self.communication.reset(self.friendly_states(), self.step_count)
        observations = self._observations()
        state = self._critic_state()
        if self.stage_success_predicate():
            raise AssertionError("reset must not satisfy a final-task success gate")
        return observations, state

    def configure_slot_observation(
        self,
        *,
        loss_rate: float,
        delay_steps: tuple[int, int] = (0, 0),
        seed_offset: int = 7919,
    ) -> None:
        """Enable a probe channel that makes each submarine's own slot stale/drop-prone.

        The environment truth remains untouched for rewards and gates. Only the
        actor task context uses the cached slot message. This is intentionally
        an opt-in probe; trained checkpoints still expect the original direct
        slot observation unless they are retrained with this channel enabled.
        """
        loss = float(loss_rate)
        delay = tuple(int(x) for x in delay_steps)
        if not np.isfinite(loss) or not 0.0 <= loss < 1.0:
            raise ValueError("slot observation loss_rate must be in [0, 1)")
        if len(delay) != 2 or delay[0] < 0 or delay[0] > delay[1]:
            raise ValueError("slot observation delay_steps must be ordered nonnegative integers")
        self.slot_observation_enabled = True
        self.slot_observation_loss = loss
        self.slot_observation_delay_steps = delay
        self.slot_observation_seed_offset = int(seed_offset)
        self._slot_observation_rng = np.random.default_rng(
            int(self.communication_seed) + self.slot_observation_seed_offset
        )
        if hasattr(self, "submarine_target_slots"):
            self._reset_slot_observation_channel()

    def _reset_slot_observation_channel(self) -> None:
        if not self.slot_observation_enabled or not hasattr(self, "submarine_target_slots"):
            self._slot_observation_cache = None
            self._slot_observation_velocity_cache = None
            self._slot_observation_last_step = None
            self._slot_observation_pending = None
            return
        self._slot_observation_cache = self.submarine_target_slots.copy()
        self._slot_observation_velocity_cache = self.submarine_target_velocities.copy()
        self._slot_observation_last_step = np.zeros(self.num_submarines, dtype=np.int64)
        self._slot_observation_pending = [[] for _ in range(self.num_submarines)]
        self._slot_observation_rng = np.random.default_rng(
            int(self.communication_seed) + self.slot_observation_seed_offset
        )

    def _advance_slot_observation_channel(self) -> None:
        if not self.slot_observation_enabled:
            return
        if self._slot_observation_cache is None:
            self._reset_slot_observation_channel()
        for local_index in range(self.num_submarines):
            if self._slot_observation_rng.random() >= self.slot_observation_loss:
                delay = int(
                    self._slot_observation_rng.integers(
                        self.slot_observation_delay_steps[0],
                        self.slot_observation_delay_steps[1] + 1,
                    )
                )
                self._slot_observation_pending[local_index].append(
                    (
                        int(self.step_count + delay),
                        self.submarine_target_slots[local_index].copy(),
                        self.submarine_target_velocities[local_index].copy(),
                    )
                )
            due = [
                item
                for item in self._slot_observation_pending[local_index]
                if item[0] <= self.step_count
            ]
            self._slot_observation_pending[local_index] = [
                item
                for item in self._slot_observation_pending[local_index]
                if item[0] > self.step_count
            ]
            if due:
                delivery_step, position, velocity = max(due, key=lambda item: item[0])
                self._slot_observation_cache[local_index] = position
                self._slot_observation_velocity_cache[local_index] = velocity
                self._slot_observation_last_step[local_index] = int(delivery_step)

    def _observed_submarine_target(self, local_index: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.slot_observation_enabled or self._slot_observation_cache is None:
            return (
                self.submarine_target_slots[local_index],
                self.submarine_target_velocities[local_index],
            )
        return (
            self._slot_observation_cache[local_index],
            self._slot_observation_velocity_cache[local_index],
        )


    @staticmethod
    def _copy_ship_state(state: ShipState) -> ShipState:
        return ShipState(
            pos=state.pos.copy(),
            speed=float(state.speed),
            heading=float(state.heading),
            yaw_rate=float(state.yaw_rate),
            unit_type=str(state.unit_type),
            surfaced=bool(state.surfaced),
            sway_speed=float(state.sway_speed),
        )


    def _sample_red_reset(self) -> tuple[np.ndarray, float, float]:
        """Sample the final task's red state."""

        randomize = self.evaluation_variant in {"randomized", "training"}
        spawn_submarine = self.spawn_cfg["submarine"]
        profile = self._submarine_spawn_profile()
        attempts = int(self.spawn_cfg.get("max_sampling_attempts", 1000))

        for _ in range(attempts if randomize else 1):
            s_low, s_high = _range_pair(profile["route_progress"], "final profile route_progress")
            l_low, l_high = _range_pair(profile["lateral_perturbation"], "final profile lateral_perturbation")
            s_value = float(self.rng.uniform(s_low, s_high)) if randomize else 0.0
            l_value = float(self.rng.uniform(l_low, l_high)) if randomize else 0.0
            red_position = (
                self.nominal_red_position
                + self.nominal_route_tangent * s_value
                + self.nominal_desired_lateral_direction * l_value
            ).astype(np.float32)
            heading_jitter = float(profile["red_heading_perturbation_degrees"]) if randomize else 0.0
            red_heading = float(self.scenario_cfg["red_heading"]) + float(
                self.rng.uniform(-math.radians(heading_jitter), math.radians(heading_jitter))
            )
            speed_low, speed_high = _range_pair(profile["red_speed_range"], "final profile red_speed_range")
            red_speed = float(self.rng.uniform(speed_low, speed_high)) if randomize else float(self.task_config.environment.red_speed)

            clearance = float(spawn_submarine.get("boundary_clearance", 2.0))
            island_buffer = float(spawn_submarine.get("island_safety_buffer", 2.0))
            if not self._spawn_position_safe(red_position, clearance, island_buffer):
                continue
            self.red_policy_cfg["target_speed"] = red_speed
            return (
                red_position.astype(np.float32),
                float(wrap_angle(red_heading)),
                float(red_speed),
            )

        raise RuntimeError(
            f"seed={self.seed}: unable to sample a safe red state for "
            "for the final task"
        )


    def _submarine_spawn_profile(self) -> dict[str, Any]:
        profiles = self.spawn_cfg["submarine"]["profiles"]
        return dict(profiles["final"])


    def _reset_intervention_windows(self) -> None:
        projection_steps = max(
            1,
            int(math.ceil(
                float(self.intervention_cfg["recent_window_seconds"]) / self.dt
            )),
        )
        trend_steps = max(
            1,
            int(math.ceil(
                float(self.intervention_cfg["trend_window_seconds"]) / self.dt
            )),
        )
        self._surface_projection_window: deque[tuple[float, float, float]] = deque(
            maxlen=projection_steps
        )
        self._surface_trend_window: deque[bool] = deque(maxlen=trend_steps)
        self._counterfactual_trend_window: deque[bool] = deque(maxlen=trend_steps)
        self._counterfactual_gain_window: deque[float] = deque(maxlen=trend_steps)
        clearance_steps = max(
            1,
            int(
                math.ceil(
                    float(
                        self.intervention_cfg["recent_clearance_window_seconds"]
                    )
                    / self.dt
                )
            ),
        )
        self._actual_clearance_window: deque[float] = deque(
            maxlen=clearance_steps
        )
        self._surface_success_window: deque[bool] = deque(
            maxlen=int(self.tracking_success_cfg["window_steps"])
        )
        self._surface_motion_windows: list[deque[tuple[float, float, float, float]]] = [
            deque(maxlen=projection_steps) for _ in range(self.num_surface_blue)
        ]
        self._surface_action_change_windows: list[deque[float]] = [
            deque(maxlen=projection_steps) for _ in range(self.num_surface_blue)
        ]
        self.surface_path_lengths = np.zeros(self.num_surface_blue, dtype=np.float64)
        self.surface_net_displacements = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_path_efficiencies = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_useful_velocity_ratios = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_yaw_saturation_fractions = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_mean_abs_yaw_rates = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_mean_speeds = np.zeros(self.num_surface_blue, dtype=np.float64)
        self.surface_mean_normalized_action_changes = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_corridor_current = np.zeros(self.num_surface_blue, dtype=bool)
        self.surface_pressure_valid = np.zeros(self.num_surface_blue, dtype=bool)
        self.surface_corridor_dwell_seconds = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_active_pressure_integrals = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_effective_dwell_seconds = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_effective_pressure_integrals = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_relay_dwell_seconds = 0.0
        self.surface_relay_pressure_integral = 0.0
        self.counterfactual_gain_at_b1_latch = 0.0
        self.b2_incremental_counterfactual_gain = 0.0
        self.surface_pressure_quality = np.zeros(self.num_surface_blue, dtype=np.float64)
        self.surface_pressure_threshold_margins = np.zeros(
            self.num_surface_blue, dtype=np.float64
        )
        self.surface_pressure_joint_quality = 0.0
        self.surface_spin_cost = 0.0
        self.surface_b1_recent = 0.0
        self.surface_b2_recent = 0.0
        self.surface_combined_recent = 0.0
        self.surface_b1_direction_efficiency = 0.0
        self.surface_b2_direction_efficiency = 0.0
        self.surface_combined_direction_efficiency = 0.0
        self.surface_trend_fraction = 0.0
        self.surface_projected_landing_clearance = float("-inf")
        self.surface_projected_route_clear = False


    def _build_pressure_zones(self) -> np.ndarray:
        zones = list(self.pressure_cfg["zones"])
        if len(zones) != 2:
            raise ValueError("surface_pressure.zones must define B1 and B2")
        points = []
        for index, zone in enumerate(zones):
            along = float(zone["along_distance"])
            lateral = float(zone["lateral_offset"])
            if along <= 0.0 or lateral < 0.0 or (index == 0 and lateral <= 0.0):
                raise ValueError("pressure-zone along must be positive and lateral offsets nonnegative (B1 positive)")
            along_axis, lateral_axis = self._surface_zone_basis(index)
            point = (
                self.red_state.pos
                + along_axis * along
                - lateral_axis * lateral
            )
            clearance = float(self.scenario_cfg["boundary_clearance"])
            points.append(
                np.clip(point, clearance, self.world_size - clearance).astype(np.float32)
            )
        points_array = np.asarray(points, dtype=np.float32)
        return points_array

    def _surface_zone_basis(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return along/lateral axes for the role-specific surface zone."""
        if int(index) == 0:
            return self.natural_route_tangent, self.desired_lateral_direction
        direction = np.asarray(self.island_center - self.red_state.pos, dtype=np.float64)
        norm_direction = float(np.linalg.norm(direction))
        along = (direction / norm_direction if norm_direction > 1.0e-8
                 else np.asarray(self.natural_route_tangent, dtype=np.float64))
        lateral = np.asarray([-along[1], along[0]], dtype=np.float64)
        return along.astype(np.float32), lateral.astype(np.float32)


    def _spawn_surface_states(self) -> list[ShipState]:
        if "surface" not in self.task_config.active_roles:
            positions = np.asarray(self.scenario_cfg["blue_starts"], dtype=np.float32)
            speed = 0.0
            heading = angle_of(self.natural_route_tangent)
            return [
                ShipState(
                    position.copy(),
                    speed,
                    heading,
                    0.0,
                    "surface",
                    True,
                )
                for position in positions
            ]

        use_nominal = self.evaluation_variant == "fixed" or (
            self.evaluation_variant == "training"
            and float(self.rng.random())
            < float(self.spawn_cfg["nominal_probability"])
        )
        if use_nominal:
            positions = np.asarray(self.scenario_cfg["blue_starts"], dtype=np.float32)
        else:
            positions = self._sample_surface_positions()
        surface_spawn = self.spawn_cfg["surface"]
        if self.evaluation_variant == "fixed":
            heading_jitter = 0.0
        else:
            heading_jitter = math.radians(
                float(surface_spawn.get("heading_jitter_degrees", 0.0))
            )
        speed = float(self.scenario_cfg["initial_surface_speed"])
        states: list[ShipState] = []
        natural_heading = angle_of(self.natural_route_tangent)
        for position in positions:
            sampled = position.copy()
            sampled = self._clip_inside_world(sampled)
            heading = natural_heading + float(
                self.rng.uniform(-heading_jitter, heading_jitter)
            )
            states.append(
                ShipState(sampled, speed, float(wrap_angle(heading)), 0.0, "surface", True)
            )
        return states


    def _spawn_submarine_states(self, reference_slots: np.ndarray) -> list[ShipState]:
        if "submarine" not in self.task_config.active_roles:
            positions = self._nominal_submarine_positions()
            speed = 0.0
            return [
                ShipState(
                    position.copy(),
                    speed,
                    0.0,
                    0.0,
                    "submarine",
                    False,
                )
                for position in positions
            ]

        if self.evaluation_variant == "fixed":
            positions = self._nominal_submarine_positions()
        else:
            positions = self._sample_submarine_positions(reference_slots)
        assigned_slots = np.asarray(reference_slots, dtype=np.float32)[
            self.submarine_slot_assignment
        ]
        headings = self._sample_submarine_headings(positions, assigned_slots)
        speed = (
            float(self.scenario_cfg["initial_submarine_speed"])
            if self.task_config.environment.submarine_dynamics
            else 0.0
        )
        states: list[ShipState] = []
        for position, heading in zip(positions, headings):
            sampled = position.copy()
            sampled = self._clip_inside_world(sampled)
            states.append(
                ShipState(sampled, speed, float(wrap_angle(heading)), 0.0, "submarine", False)
            )
        return states


    def _nominal_submarine_positions(self) -> np.ndarray:
        configured = np.asarray(self.scenario_cfg["submarine_starts"], dtype=np.float32)
        return configured.copy()


    def _sample_surface_positions(self) -> np.ndarray:
        spawn = self.spawn_cfg["surface"]
        attempts = int(self.spawn_cfg.get("max_sampling_attempts", 1000))
        translation = np.asarray(spawn.get("random_translation", [0.0, 0.0]), dtype=np.float32)
        for _ in range(attempts):
            values = (
                (
                    float(self.rng.uniform(*_range_pair(spawn["b1_s"], "surface b1 s"))),
                    float(self.rng.uniform(*_range_pair(spawn["b1_l"], "surface b1 l"))),
                ),
                (
                    float(self.rng.uniform(*_range_pair(spawn["b2_s"], "surface b2 s"))),
                    float(self.rng.uniform(*_range_pair(spawn["b2_l"], "surface b2 l"))),
                ),
            )
            positions = np.asarray(
                [
                    self.red_state.pos
                    + self.natural_route_tangent * along
                    + self.surface_spawn_lateral_direction * lateral
                    + translation
                    for along, lateral in values
                ],
                dtype=np.float32,
            )
            if any(
                not self._inside_spawn_bounds(position, float(spawn["boundary_clearance"]))
                for position in positions
            ):
                continue
            if norm(positions[0] - positions[1]) < float(spawn["minimum_pair_distance"]):
                continue
            if any(
                norm(position - self.red_state.pos) < float(spawn["minimum_red_distance"])
                for position in positions
            ):
                continue
            if any(
                float((position - self.red_state.pos) @ self.natural_route_tangent) <= 0.0
                for position in positions
            ):
                continue
            landing_distance = max(
                0.0,
                norm(self.red_state.pos - self.island_center)
                - self.island_radius
                - self.landing_buffer,
            )
            available_seconds = landing_distance / max(self.red_state.speed, 1.0e-6)
            travel_seconds = [
                norm(position - target) / self.max_speed_by_type["surface"]
                for position, target in zip(positions, self.pressure_zones)
            ]
            if any(seconds >= available_seconds for seconds in travel_seconds):
                continue
            return positions
        raise RuntimeError("unable to sample a valid route-relative surface deployment")


    def _sample_submarine_positions(self, reference_slots: np.ndarray) -> np.ndarray:
        spawn = self.spawn_cfg["submarine"]
        profile = self._submarine_spawn_profile()
        radius_low, radius_high = _range_pair(
            profile["spawn_radius"], "submarine spawn radius"
        )
        attempts = int(self.spawn_cfg.get("max_sampling_attempts", 1000))
        boundary_clearance = max(2.0 * self.geometry_scale, float(spawn.get("boundary_clearance", 2.0 * self.geometry_scale)))
        island_buffer = float(spawn.get("island_safety_buffer", 2.0))
        minimum_red_distance = float(spawn.get("minimum_red_distance", 90.0))
        minimum_slot_distance = float(spawn.get("minimum_any_slot_error", 3.5))
        minimum_intercept = float(spawn.get("minimum_intercept_seconds", 10.0))
        maximum_intercept = (
            float(spawn.get("maximum_reachability_fraction", 0.60))
            * self.max_steps
            * self.dt
        )
        sector_width = math.radians(
            float(spawn.get("lower_half_sector_degrees", 160.0))
        )
        if sector_width < math.radians(90.0) or sector_width > math.pi:
            raise ValueError("lower-half submarine spawn sector must span 90 to 180 degrees")
        render_down_angle = angle_of(self._visual_lower_direction())
        sector_start = render_down_angle - 0.5 * sector_width
        subsector_width = sector_width / self.num_submarines
        translation = np.asarray(spawn.get("random_translation", [0.0, 0.0]), dtype=np.float32)
        for _ in range(attempts):
            positions: list[np.ndarray] = []
            for slot_index in range(self.num_submarines):
                radius = float(self.rng.uniform(radius_low, radius_high))
                subsector_start = sector_start + slot_index * subsector_width
                angle = float(
                    self.rng.uniform(
                        subsector_start,
                        subsector_start + subsector_width,
                    )
                )
                positions.append(
                    self.red_state.pos
                    + radius * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)
                    + translation
                )
            array = np.asarray(positions, dtype=np.float32)
            if any(
                not self._spawn_position_safe(
                    position, boundary_clearance, island_buffer
                )
                for position in array
            ):
                continue
            pair_distances = [
                norm(array[left] - array[right])
                for left in range(self.num_submarines)
                for right in range(left + 1, self.num_submarines)
            ]
            if min(pair_distances, default=float("inf")) < float(spawn["minimum_pair_distance"]):
                continue
            if np.any(
                np.linalg.norm(array - self.red_state.pos[None, :], axis=1)
                < minimum_red_distance
            ):
                continue
            all_slot_distances = np.linalg.norm(
                array[:, None, :] - np.asarray(reference_slots)[None, :, :], axis=2
            )
            if np.any(all_slot_distances < minimum_slot_distance):
                continue
            candidate_margin = triangle_margin(self.red_state.pos, array)
            candidate_gap = max_angular_gap(array, self.red_state.pos)
            gap_limit = self.task_config.success.max_angle_gap_degrees
            if candidate_margin >= 0.0:
                continue
            if gap_limit is not None and candidate_gap <= math.radians(gap_limit):
                continue
            assignment = self._select_submarine_slot_assignment(
                array,
                np.asarray(reference_slots, dtype=np.float32),
            )
            assigned_slots = np.asarray(reference_slots, dtype=np.float32)[assignment]
            own_errors = np.linalg.norm(array - assigned_slots, axis=1)
            max_limit = self.task_config.success.max_slot_error
            mean_limit = self.task_config.success.mean_slot_error
            if (
                max_limit is not None
                and mean_limit is not None
                and float(np.max(own_errors)) <= max_limit
                and float(np.mean(own_errors)) <= mean_limit
            ):
                continue
            intercept_times = np.asarray(
                [
                    self._constant_velocity_intercept_time(
                        position,
                        slot,
                        self.red_state.velocity,
                        self.max_speed_by_type["submarine"],
                    )
                    for position, slot in zip(array, assigned_slots)
                ],
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(intercept_times))
                or np.any(intercept_times < minimum_intercept)
                or np.any(intercept_times > maximum_intercept)
            ):
                continue
            if any(
                not self._has_safe_maneuver_direction(
                    position,
                    array,
                    boundary_clearance,
                    island_buffer,
                    float(spawn.get("safe_maneuver_probe_distance", 2.0)),
                )
                for position in array
            ):
                continue
            headings = self._candidate_submarine_headings(array, assigned_slots)
            turn_times = np.asarray(
                [
                    abs(float(wrap_angle(angle_of(slot - position) - heading)))
                    / max(self.max_yaw_rate, 1.0e-8)
                    for position, slot, heading in zip(array, assigned_slots, headings)
                ],
                dtype=np.float64,
            )
            if np.any(intercept_times + turn_times > maximum_intercept):
                continue
            self.submarine_slot_assignment = assignment
            self._sampled_submarine_headings = headings.astype(np.float32)
            return array
        raise RuntimeError(
            f"seed={self.seed}: unable to sample constrained independent submarine "
            "deployment for the final task"
        )


    @staticmethod
    def _visual_lower_direction() -> np.ndarray:
        """World-space direction rendered below an object.

        ``Renderer.world_to_screen`` maps world y to ``world_height - y``;
        therefore increasing screen y (visually lower) is decreasing world y.
        Keeping that convention behind a named helper avoids embedding an
        unexplained coordinate-sign assumption in deployment sampling.
        """

        return np.asarray([0.0, -1.0], dtype=np.float32)


    def _candidate_submarine_headings(
        self,
        positions: np.ndarray,
        reference_slots: np.ndarray | None = None,
        *,
        apply_jitter: bool = True,
    ) -> np.ndarray:
        profile = self._submarine_spawn_profile()
        jitter = math.radians(
            float(profile.get("submarine_heading_jitter_degrees", 45.0))
            if apply_jitter
            else 0.0
        )
        target = constant_velocity_prediction(
            self.red_state.pos, self.red_state.velocity, self.prediction_horizon_seconds
        )
        targets = np.tile(target, (len(positions), 1))
        return np.asarray(
            [
                wrap_angle(
                    angle_of(target - position)
                    + float(self.rng.uniform(-jitter, jitter))
                )
                for position, target in zip(
                    np.asarray(positions, dtype=np.float32), targets
                )
            ],
            dtype=np.float32,
        )


    def _sample_submarine_headings(
        self, positions: np.ndarray, reference_slots: np.ndarray
    ) -> np.ndarray:
        if self.evaluation_variant == "fixed":
            return self._candidate_submarine_headings(
                positions, reference_slots, apply_jitter=False
            )
        if self.evaluation_variant != "fixed" and self._sampled_submarine_headings.shape == (
            self.num_submarines,
        ):
            return self._sampled_submarine_headings.copy()
        headings = self._candidate_submarine_headings(positions, reference_slots)
        self._sampled_submarine_headings = headings.copy()
        return headings


    def _spawn_position_safe(
        self, position: np.ndarray, boundary_clearance: float, island_buffer: float
    ) -> bool:
        if not self._inside_spawn_bounds(position, boundary_clearance):
            return False
        island_clearance = norm(
            np.asarray(position, dtype=np.float32) - self.island_center
        ) - self.island_radius - self.semi_major
        return bool(island_clearance >= island_buffer)


    def _has_safe_maneuver_direction(
        self,
        position: np.ndarray,
        peers: np.ndarray,
        boundary_clearance: float,
        island_buffer: float,
        probe_distance: float,
    ) -> bool:
        for angle in np.linspace(-math.pi, math.pi, 16, endpoint=False):
            candidate = np.asarray(position, dtype=np.float32) + float(
                probe_distance
            ) * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)
            if not self._spawn_position_safe(
                candidate, boundary_clearance, island_buffer
            ):
                continue
            peer_distances = np.linalg.norm(
                np.asarray(peers, dtype=np.float32) - candidate[None, :], axis=1
            )
            non_self = peer_distances > 1.0e-5
            if np.any(peer_distances[non_self] < self.collision_radius + 0.2):
                continue
            return True
        return False


    @staticmethod
    def _constant_velocity_intercept_time(
        pursuer_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        pursuer_speed: float,
    ) -> float:
        delta = np.asarray(target_position, dtype=np.float64) - np.asarray(
            pursuer_position, dtype=np.float64
        )
        velocity = np.asarray(target_velocity, dtype=np.float64)
        speed = float(pursuer_speed)
        a = float(velocity @ velocity - speed * speed)
        b = float(2.0 * delta @ velocity)
        c = float(delta @ delta)
        if c <= 1.0e-12:
            return 0.0
        if abs(a) <= 1.0e-12:
            return float("inf") if b >= 0.0 else max(0.0, -c / b)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return float("inf")
        root = math.sqrt(discriminant)
        roots = [
            value
            for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
            if value >= 0.0
        ]
        return min(roots, default=float("inf"))


    def _slot_assignment_metrics(
        self,
        positions: np.ndarray,
        slots: np.ndarray,
        assignment: tuple[int, ...],
    ) -> tuple[float, float, float]:
        """Predict synchronous clearance and arrival times for one slot mapping."""

        starts = np.asarray(positions, dtype=np.float64)
        references = np.asarray(slots, dtype=np.float64)[
            np.asarray(assignment, dtype=np.int64)
        ]
        target_velocity = np.asarray(self.red_state.velocity, dtype=np.float64)
        pursuer_speed = self.max_speed_by_type["submarine"]
        arrival_times = np.asarray(
            [
                self._constant_velocity_intercept_time(
                    position,
                    slot,
                    target_velocity,
                    pursuer_speed,
                )
                for position, slot in zip(starts, references)
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(arrival_times)):
            arrival_times = np.linalg.norm(references - starts, axis=1) / max(
                pursuer_speed, 1.0e-8
            )

        intercepts = references + arrival_times[:, None] * target_velocity[None, :]
        maximum_time = float(np.max(arrival_times))
        sample_count = max(1, int(math.ceil(maximum_time / self.dt)))
        minimum_clearance = float("inf")
        for time_value in np.linspace(
            0.0,
            maximum_time,
            sample_count + 1,
            dtype=np.float64,
        ):
            predicted: list[np.ndarray] = []
            for start, intercept, arrival_time in zip(
                starts,
                intercepts,
                arrival_times,
            ):
                if arrival_time <= 1.0e-12 or time_value >= arrival_time:
                    position = intercept + target_velocity * max(
                        0.0, time_value - arrival_time
                    )
                else:
                    position = start + (intercept - start) * (
                        time_value / arrival_time
                    )
                predicted.append(position)
            predicted_array = np.asarray(predicted, dtype=np.float64)
            minimum_clearance = min(
                minimum_clearance,
                *(
                    float(np.linalg.norm(predicted_array[left] - predicted_array[right]))
                    for left in range(self.num_submarines)
                    for right in range(left + 1, self.num_submarines)
                ),
            )
        return (
            minimum_clearance,
            maximum_time,
            float(np.sum(arrival_times)),
        )


    def _select_submarine_slot_assignment(
        self,
        positions: np.ndarray,
        slots: np.ndarray,
    ) -> np.ndarray:
        """Choose one collision-aware, minimum-completion-time slot permutation."""

        starts = np.asarray(positions, dtype=np.float32)
        references = np.asarray(slots, dtype=np.float32)
        expected_shape = (self.num_submarines, 2)
        if starts.shape != expected_shape or references.shape != expected_shape:
            raise ValueError(
                f"submarine positions and slots must both have shape {expected_shape}"
            )

        candidates: list[tuple[tuple[int, ...], float, float, float]] = []
        for assignment in permutations(range(self.num_submarines)):
            clearance, maximum_time, total_time = self._slot_assignment_metrics(
                starts,
                references,
                assignment,
            )
            candidates.append(
                (assignment, clearance, maximum_time, total_time)
            )

        safe_distance = float(
            self.reward_cfg["normalization"]["spacing_safe_m"]
        )
        safe_candidates = [
            candidate for candidate in candidates if candidate[1] >= safe_distance
        ]
        if safe_candidates:
            selected = min(
                safe_candidates,
                key=lambda candidate: (
                    candidate[2],
                    candidate[3],
                    -candidate[1],
                    candidate[0],
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda candidate: (
                    -candidate[1],
                    candidate[2],
                    candidate[3],
                    candidate[0],
                ),
            )
        return np.asarray(selected[0], dtype=np.int64)


    def _inside_spawn_bounds(self, position: np.ndarray, clearance: float) -> bool:
        hull_margin = 0.5 * max(self.surface_length, self.submarine_length)
        clearance = float(clearance) + hull_margin
        return bool(
            clearance <= float(position[0]) <= float(self.world_size[0] - clearance)
            and clearance <= float(position[1]) <= float(self.world_size[1] - clearance)
        )


    def _step_physics(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, EnvInfo]:
        if self._episode_done:
            raise RuntimeError("episode has terminated; call reset() before step()")
        actions_array = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_friendly, self.action_dim)
        if actions_array.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}, got {actions_array.shape}")
        if not np.all(np.isfinite(actions_array)):
            raise ValueError("actions must contain only finite values")
        executed = np.clip(
            actions_array, -self.agent_action_limits, self.agent_action_limits
        )
        executed *= self._agent_active_mask[:, None]
        executed = executed.astype(np.float32)
        reward_previous = self._reward_snapshot()
        previous_actions = self.previous_actions.copy()
        self.last_action_delta = executed - previous_actions
        self.previous_actions = executed.copy()
        self.last_executed_actions = executed.copy()
        self.episode_action_history.append(executed.copy())
        previous_red_position = self.red_state.pos.copy()
        previous_surface_positions = np.asarray(
            [state.pos for state in self.blue_states], dtype=np.float32
        )
        previous_submarine_positions = np.asarray(
            [state.pos for state in self.submarine_states], dtype=np.float32
        )
        previous_slots = self.submarine_target_slots.copy()
        previous_pressure_zones = self.pressure_zones.copy()

        self.step_count += 1
        if (
            "surface" in self.task_config.active_roles
            and self.surface_dynamics_enabled
        ):
            self.blue_states = [
                self.blue_dynamics.step(
                    state,
                    action,
                    self.wave.horizontal_velocity(state.pos, self.step_count * self.dt),
                )
                for state, action in zip(self.blue_states, executed[self.surface_slice])
            ]
            self.wave_velocity_samples["surface"].extend(
                float(norm(self.wave.horizontal_velocity(state.pos, self.step_count * self.dt)))
                for state in self.blue_states
            )
        if self.task_config.environment.submarine_dynamics:
            self.submarine_states = [
                self.submarine_dynamics.step(state, action)
                for state, action in zip(
                    self.submarine_states, executed[self.submarine_slice]
                )
            ]
        wall_contact_agents = []
        for index, state in enumerate(self.blue_states):
            if self._confine_state_to_world(state):
                wall_contact_agents.append("B1" if index == 0 else "B2")
        for index, state in enumerate(self.submarine_states):
            if self._confine_state_to_world(state):
                wall_contact_agents.append(f"AUV{index + 1}")
        current_surface_positions = np.asarray(
            [state.pos for state in self.blue_states], dtype=np.float32
        )
        current_submarine_positions = np.asarray(
            [state.pos for state in self.submarine_states], dtype=np.float32
        )
        self.surface_path_lengths += np.linalg.norm(
            current_surface_positions - previous_surface_positions, axis=1
        )
        self.surface_net_displacements = np.linalg.norm(
            current_surface_positions - self.surface_initial_positions, axis=1
        )
        self.submarine_path_lengths += np.linalg.norm(
            current_submarine_positions - previous_submarine_positions, axis=1
        )
        self.submarine_net_displacements = np.linalg.norm(
            current_submarine_positions - self.submarine_initial_positions, axis=1
        )
        self.surface_apf_forces = self._current_surface_apf_forces()
        active_surfaces = [
            state
            for index, state in enumerate(self.blue_states)
            if self.surface_apf_enabled and self.active_surface_mask[index] > 0.5
        ]
        red_reference = red_policy_reference(
            self.red_state, self._landing_point(), active_surfaces,
            self.red_policy_cfg, self.config["ship"],
        )
        shadow_red_reference = red_policy_reference(
            self.shadow_red_state, self._landing_point(), [],
            self.red_policy_cfg, self.config["ship"],
        )
        red_action = self.red_dynamics.reference_action(
            self.red_state,
            desired_yaw_rate=float(red_reference[0]),
            desired_surge_acceleration=float(red_reference[1]),
        )
        shadow_red_action = self.red_dynamics.reference_action(
            self.shadow_red_state,
            desired_yaw_rate=float(shadow_red_reference[0]),
            desired_surge_acceleration=float(shadow_red_reference[1]),
        )
        red_wave = self.wave.horizontal_velocity(
            self.red_state.pos, self.step_count * self.dt
        )
        self.wave_velocity_samples["red"].append(float(norm(red_wave)))
        self.red_state = self.red_dynamics.step(self.red_state, red_action, red_wave)
        if self._confine_state_to_world(self.red_state):
            wall_contact_agents.append("Red")
        self.shadow_red_state = self.red_dynamics.step(
            self.shadow_red_state,
            shadow_red_action,
            self.wave.horizontal_velocity(
                self.shadow_red_state.pos, self.step_count * self.dt
            ),
        )
        self._confine_state_to_world(self.shadow_red_state)
        self.wall_contact_agents = tuple(wall_contact_agents)
        self.wall_contact = bool(wall_contact_agents)
        self.red_path_length += norm(self.red_state.pos - previous_red_position)
        self.red_heading_change = abs(
            float(wrap_angle(self.red_state.heading - self.initial_red_heading))
        )
        self._update_counterfactual_metrics(initialize=False)
        # Pressure corridors and dynamic slots belong to the new red state.
        # Refresh them before measuring this transition's approach/pressure
        # quality so the gates do not lag their moving references by one step.
        self._update_dynamic_references(initialize=False)
        self._update_pressure_history()
        self._update_surface_maneuver_metrics()
        self._update_surface_event_latches()
        self._refresh_geometry_metrics()
        self.communication.step(self.friendly_states(), self.step_count)

        collision = self._collision()
        boundary = self._boundary_violation()
        landing = self._red_landed()
        radius_changed = (False if getattr(self, "v43", False) and (collision or boundary or landing)
                          else self._advance_formation())
        if radius_changed:
            self._update_dynamic_references(initialize=False)
            self._refresh_geometry_metrics()
        self.submarine_target_velocities = (
            (self.submarine_target_slots - previous_slots) / self.dt
        ).astype(np.float32)
        self.pressure_zone_velocities = (
            (self.pressure_zones - previous_pressure_zones) / self.dt
        ).astype(np.float32)
        self._refresh_geometry_metrics()
        self._record_final_gate_failures(collision)
        core_gate = self._physical_core_gate(collision, boundary, landing)
        self.physical_core_gate = bool(core_gate)
        self._advance_green_hold(core_gate)
        self._update_stage_success_windows(collision, boundary, landing)
        self.physical_hold_credit = float(self.green_seconds)
        if core_gate and self.first_physical_gate_step is None:
            self.first_physical_gate_step = self.step_count
        self.physical_capture_success = bool(self.compaction_complete and core_gate)
        if self.physical_capture_success and self.first_physical_capture_step is None:
            self.first_physical_capture_step = self.step_count
        self.strict_success = self.strict_success_predicate()
        self.cooperative_task_success = self.strict_success
        if self.cooperative_task_success and self.first_cooperative_success_step is None:
            self.first_cooperative_success_step = self.step_count
        stage_core_gate = self._current_stage_core_gate(collision, boundary, landing)
        self._advance_stage_hold(stage_core_gate)
        gates_met = self.stage_success_predicate(
            collision=collision,
            boundary=boundary,
            landing=landing,
        )
        horizon_reached = self.step_count >= self.max_steps
        self.stage_success = gates_met
        self.relaxed_cooperative_success = bool(
            True
            and self.stage_success
        )
        if self.stage_success and self.first_stage_success_step is None:
            self.first_stage_success_step = self.step_count
        if (
            self.relaxed_cooperative_success
            and self.first_relaxed_success_step is None
        ):
            self.first_relaxed_success_step = self.step_count
        task_success = self._canonical_task_success()
        terminal_reason = ""
        terminated = False
        truncated = False
        if landing:
            terminal_reason, terminated = "landing", True
        elif collision:
            terminal_reason, terminated = "collision", True
        elif task_success:
            terminal_reason, terminated = "success", True
            self.phase = "SUCCESS"
        elif horizon_reached:
            terminal_reason = "timeout"
            if getattr(self, "v43", False):
                terminated = True
            else:
                truncated = True
        self.terminal_reason = terminal_reason

        rewards, reward_terms = self._rewards(
            previous=reward_previous,
            previous_actions=previous_actions,
            current_actions=executed,
            terminal_reason=terminal_reason,
            collision=collision,
            boundary=boundary,
            landing=landing,
            core_gate=core_gate,
        )
        done = bool(terminated or truncated)
        self._episode_done = done
        info = EnvInfo(
            step=self.step_count,
            success=task_success,
            hold_steps=int(math.ceil(self.stable_seconds / max(self.dt, 1.0e-9))),
            collision=collision,
            boundary_violation=boundary,
            landing_success=landing,
            surface_intervention_success=self.surface_intervention_success,
            physical_capture_success=self.physical_capture_success,
            cooperative_task_success=self.cooperative_task_success,
            relaxed_cooperative_success=self.relaxed_cooperative_success,
            strict_success=self.strict_success,
            stage_success_relaxed=self.stage_success,
            physical_capture=self.physical_capture_success,
            task_name="final_task",
            stage_hold_seconds=self.stage_hold_seconds,
            stage_success_gates=self.stage_success_gates(
                collision=collision,
                boundary=boundary,
                landing=landing,
            ),
            unmet_success_gates=self.unmet_success_gates(
                collision=collision,
                boundary=boundary,
                landing=landing,
            ),
            physical_hold_credit=self.physical_hold_credit,
            phase=self.phase,
            terminated=terminated,
            truncated=truncated,
            timeout=terminal_reason == "timeout",
            terminal_reason=terminal_reason,
            collision_pair=self.collision_pair,
            collision_step=self.collision_step,
            collision_stage=self.collision_stage,
            wall_contact=self.wall_contact,
            wall_contact_agents=self.wall_contact_agents,
            first_physical_gate_step=self.first_physical_gate_step,
            first_physical_capture_step=self.first_physical_capture_step,
            first_cooperative_success_step=self.first_cooperative_success_step,
            first_hold_step=self.first_hold_step,
            first_encircle_step=self.first_encircle_step,
            stable_steps=int(math.ceil(self.stable_seconds / max(self.dt, 1.0e-9))),
            green_hold_steps=int(math.ceil(self.green_seconds / max(self.dt, 1.0e-9))),
            reward_terms=reward_terms,
            premature_success_attempt=self.premature_success_attempt,
            surface_b1_latched=self.surface_b1_latched,
            surface_b2_latched=self.surface_b2_latched,
            surface_causal_latched=self.surface_causal_latched,
            surface_island_avoidance_latched=self.surface_island_avoidance_latched,
            surface_complete_latched=self.surface_complete_latched,
            red_actual_cross_track=self.red_actual_cross_track,
            red_shadow_cross_track=self.red_shadow_cross_track,
            counterfactual_cross_track_gain=self.counterfactual_cross_track_gain,
            maximum_counterfactual_cross_track_gain=self.maximum_counterfactual_cross_track_gain,
            predicted_island_clearance=self.predicted_island_clearance,
            shadow_predicted_island_clearance=self.shadow_predicted_island_clearance,
            clearance_gain_vs_shadow=self.clearance_gain_vs_shadow,
            surface_gate_progress=self.surface_gate_progress,
            b2_marginal_correction=self.current_b2_marginal,
            surface_pressure_threshold_margins=tuple(
                float(value) for value in self.surface_pressure_threshold_margins
            ),
            surface_net_displacements=tuple(float(v) for v in self.surface_net_displacements),
            surface_path_lengths=tuple(float(v) for v in self.surface_path_lengths),
            surface_path_efficiencies=tuple(float(v) for v in self.surface_path_efficiencies),
            surface_useful_velocity_ratios=tuple(
                float(v) for v in self.surface_useful_velocity_ratios
            ),
            surface_corridor_dwell_seconds=tuple(
                float(v) for v in self.surface_corridor_dwell_seconds
            ),
            surface_active_pressure_integrals=tuple(
                float(v) for v in self.surface_active_pressure_integrals
            ),
            surface_yaw_saturation_fractions=tuple(
                float(v) for v in self.surface_yaw_saturation_fractions
            ),
            surface_mean_abs_yaw_rates=tuple(
                float(v) for v in self.surface_mean_abs_yaw_rates
            ),
            surface_mean_normalized_action_changes=tuple(
                float(v) for v in self.surface_mean_normalized_action_changes
            ),
            surface_mean_speeds=tuple(float(v) for v in self.surface_mean_speeds),
            counterfactual_gain_at_b1_latch=float(
                self.counterfactual_gain_at_b1_latch
            ),
            b2_incremental_counterfactual_gain=float(
                self.b2_incremental_counterfactual_gain
            ),
            submarine_net_displacements=tuple(
                float(v) for v in self.submarine_net_displacements
            ),
            formation45_hold_steps=self.formation45_hold_steps,
            shrink_steps=self.shrink_steps,
            final_hold_steps=self.stage_hold_step_count,
            reference_radius=self.radius_ref,
            surface_b1_latch_step=self.surface_b1_latch_step,
            surface_b2_latch_step=self.surface_b2_latch_step,
            surface_causal_latch_step=self.surface_causal_latch_step,
            surface_island_latch_step=self.surface_island_latch_step,
            surface_complete_step=self.surface_complete_step,
            formation45_hold_start_step=self.formation45_hold_start_step,
            formation45_hold_complete_step=self.formation45_hold_complete_step,
            shrink_started_step=self.shrink_started_step,
            radius35_step=self.radius35_step,
            final_hold_start_step=self.final_hold_start_step,
        )
        # Deliver the next slot-observation messages only after the physical
        # truth and gates have been updated for this transition.
        self._advance_slot_observation_channel()
        return self._observations(), self._critic_state(), rewards, done, info


    def _current_surface_apf_forces(self) -> np.ndarray:
        if (
            "surface" not in self.task_config.active_roles
            or not self.surface_apf_enabled
        ):
            return np.zeros((self.num_surface_blue, 2), dtype=np.float32)
        policy = self.config["red_policy"]
        values = []
        for index, state in enumerate(self.blue_states):
            if self.active_surface_mask[index] <= 0.5:
                values.append(np.zeros(2, dtype=np.float32))
            else:
                values.append(
                    surface_repulsion_force(
                        self.red_state.pos,
                        state.pos,
                        float(policy["repulsion_range"]),
                        float(policy["repulsion_gain"]),
                    )
                )
        return np.asarray(values, dtype=np.float32)


    def _update_pressure_history(self) -> None:
        b1_force, b2_force = self.surface_apf_forces
        combined = b1_force + b2_force
        self.current_b1_projection = float(b1_force @ self.desired_lateral_direction)
        self.current_b2_projection = float(b2_force @ self.desired_lateral_direction)
        self.current_combined_projection = float(
            combined @ self.desired_lateral_direction
        )
        self.current_b2_marginal = float(
            b2_marginal_correction(
                b1_force, b2_force, self.desired_lateral_direction
            )
        )
        self.b1_apf_projection_integral += self.current_b1_projection * self.dt
        self.b2_apf_projection_integral += self.current_b2_projection * self.dt
        self.combined_apf_projection_integral += (
            self.current_combined_projection * self.dt
        )
        offset = self.red_state.pos - self.route_origin
        self.current_lateral_deviation = float(
            offset @ self.desired_lateral_direction
        )
        self.current_lateral_velocity = float(
            self.red_state.velocity @ self.desired_lateral_direction
        )
        self.max_favorable_lateral_deviation = max(
            self.max_favorable_lateral_deviation,
            self.current_lateral_deviation,
        )
        if "surface" not in self.task_config.active_roles:
            return

        self._surface_projection_window.append(
            (
                self.current_b1_projection,
                self.current_b2_projection,
                self.current_combined_projection,
            )
        )
        self._surface_trend_window.append(self._current_surface_trend_is_favorable())
        reference = max(
            float(self.intervention_cfg["apf_projection_reference"]), 1.0e-6
        )
        values = np.asarray(self._surface_projection_window, dtype=np.float64)
        self.surface_b1_recent = float(np.mean(values[:, 0]) / reference)
        self.surface_b2_recent = float(np.mean(values[:, 1]) / reference)
        self.surface_combined_recent = float(np.mean(values[:, 2]) / reference)
        for index, name in enumerate(
            (
                "surface_b1_direction_efficiency",
                "surface_b2_direction_efficiency",
                "surface_combined_direction_efficiency",
            )
        ):
            projections = values[:, index] if index < 2 else values[:, 2]
            denominator = float(np.sum(np.abs(projections)))
            setattr(
                self,
                name,
                float(np.sum(np.maximum(0.0, projections)) / (denominator + 1.0e-8)),
            )
        self.surface_trend_fraction = float(
            np.mean(np.asarray(self._surface_trend_window, dtype=np.float32))
        )
        self.surface_projected_landing_clearance = self.predicted_island_clearance
        minimum_clearance = float(
            self.intervention_cfg.get("projected_actual_clearance_min", 0.0)
        )
        self.surface_projected_route_clear = bool(
            len(self._surface_projection_window)
            == self._surface_projection_window.maxlen
            and self.surface_projected_landing_clearance >= minimum_clearance
        )


    def _update_counterfactual_metrics(self, *, initialize: bool) -> None:
        previous_gain = float(self.counterfactual_cross_track_gain)
        self.red_actual_cross_track = float(
            (self.red_state.pos - self.route_origin) @ self.n_drive
        )
        self.red_shadow_cross_track = float(
            (self.shadow_red_state.pos - self.route_origin) @ self.n_drive
        )
        self.counterfactual_cross_track_gain = float(
            self.red_actual_cross_track - self.red_shadow_cross_track
        )
        self.maximum_counterfactual_cross_track_gain = max(
            self.maximum_counterfactual_cross_track_gain,
            self.counterfactual_cross_track_gain,
        )
        self.predicted_island_clearance = self._projected_landing_clearance(
            self.red_state
        )
        self.shadow_predicted_island_clearance = self._projected_landing_clearance(
            self.shadow_red_state
        )
        self.clearance_gain_vs_shadow = float(
            self.predicted_island_clearance
            - self.shadow_predicted_island_clearance
        )
        if not initialize:
            self._counterfactual_trend_window.append(
                self.counterfactual_cross_track_gain + 1.0e-6 >= previous_gain
            )
        self._counterfactual_gain_window.append(
            float(self.counterfactual_cross_track_gain)
        )
        self._actual_clearance_window.append(
            float(self.predicted_island_clearance)
        )


    def _update_surface_maneuver_metrics(self) -> None:
        if "surface" not in self.task_config.active_roles:
            return
        cfg = self.intervention_cfg
        zones = list(self.pressure_cfg["zones"])
        for index, (state, zone) in enumerate(zip(self.blue_states, zones)):
            influence_range = max(
                float(self.red_policy_cfg["repulsion_range"]), 1.0e-8
            )
            favorable_projection = max(
                float(self.surface_apf_forces[index] @ self.desired_lateral_direction),
                0.0,
            )
            effectively_engaged = bool(
                norm(state.pos - self.red_state.pos) <= influence_range
                and favorable_projection > 0.0
            )
            if effectively_engaged:
                self.surface_effective_dwell_seconds[index] += self.dt
                self.surface_effective_pressure_integrals[index] += (
                    favorable_projection * self.dt
                )
            elif not (self.surface_b1_latched if index == 0 else self.surface_b2_latched):
                self.surface_effective_dwell_seconds[index] = 0.0

            self.surface_path_efficiencies[index] = float(
                self.surface_net_displacements[index]
                / max(self.surface_path_lengths[index], 1.0e-8)
            )
            direction_to_zone = self.pressure_zones[index] - state.pos
            distance = norm(direction_to_zone)
            direction = (
                np.zeros(2, dtype=np.float32)
                if distance <= 1.0e-8
                else direction_to_zone / distance
            )
            speed = float(state.ground_speed)
            useful_velocity = max(0.0, float(state.velocity @ direction))
            window = self._surface_motion_windows[index]
            window.append(
                (float(state.yaw_rate), speed, useful_velocity, max(speed, 1.0e-8))
            )
            values = np.asarray(window, dtype=np.float64)
            self.surface_useful_velocity_ratios[index] = float(
                np.sum(values[:, 2]) / max(np.sum(values[:, 3]), 1.0e-8)
            )
            self.surface_yaw_saturation_fractions[index] = float(
                np.mean(np.abs(values[:, 0]) >= 0.95 * self.max_yaw_rate)
            )
            self.surface_mean_abs_yaw_rates[index] = float(
                np.mean(np.abs(values[:, 0]))
            )
            self.surface_mean_speeds[index] = float(np.mean(values[:, 1]))
            action_change = float(
                np.mean(
                    np.abs(
                        self.last_action_delta[index]
                        / np.maximum(self.action_limit, 1.0e-8)
                    )
                )
            )
            action_window = self._surface_action_change_windows[index]
            action_window.append(action_change)
            self.surface_mean_normalized_action_changes[index] = float(
                np.mean(action_window)
            )

            zone_error = state.pos - self.pressure_zones[index]
            along_axis, lateral_axis = self._surface_zone_basis(index)
            in_corridor = bool(
                abs(float(zone_error @ along_axis))
                <= float(zone["along_tolerance"])
                and abs(float(zone_error @ lateral_axis))
                <= float(zone["lateral_tolerance"])
            )
            projection = (
                self.current_b1_projection
                if index == 0
                else self.current_b2_projection
            )
            projection_min = float(
                cfg[
                    "b1_apf_projection_min"
                    if index == 0
                    else "b2_apf_projection_min"
                ]
            )
            valid = bool(
                in_corridor
                and projection >= projection_min
            )
            self.surface_corridor_current[index] = in_corridor
            self.surface_pressure_valid[index] = valid
            if valid:
                self.surface_corridor_dwell_seconds[index] += self.dt
                self.surface_active_pressure_integrals[index] += projection * self.dt
            elif not (self.surface_b1_latched if index == 0 else self.surface_b2_latched):
                self.surface_corridor_dwell_seconds[index] = 0.0
            along_tolerance = max(float(zone["along_tolerance"]), 1.0e-8)
            lateral_tolerance = max(float(zone["lateral_tolerance"]), 1.0e-8)
            minimum_speed = max(float(cfg["minimum_speed"]), 1.0e-8)
            minimum_path_efficiency = max(
                float(cfg["path_efficiency_min"]), 1.0e-8
            )
            minimum_useful_velocity = max(
                float(cfg["useful_velocity_ratio_min"]), 1.0e-8
            )
            maximum_yaw_fraction = max(
                float(cfg["yaw_saturation_fraction_max"]), 1.0e-8
            )
            maximum_action_change = max(
                float(cfg["mean_normalized_action_change_max"]), 1.0e-8
            )
            along_error = abs(float(zone_error @ along_axis))
            lateral_error = abs(float(zone_error @ lateral_axis))
            quality_factors = (
                np.clip(1.0 - along_error / along_tolerance, 0.0, 1.0),
                np.clip(1.0 - lateral_error / lateral_tolerance, 0.0, 1.0),
                np.clip(len(window) / max(window.maxlen or 1, 1), 0.0, 1.0),
                np.clip(speed / minimum_speed, 0.0, 1.0),
                np.clip(
                    self.surface_path_efficiencies[index]
                    / minimum_path_efficiency,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.surface_useful_velocity_ratios[index]
                    / minimum_useful_velocity,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (2.0 * maximum_yaw_fraction
                     - self.surface_yaw_saturation_fractions[index])
                    / maximum_yaw_fraction,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (
                        2.0 * maximum_action_change
                        - self.surface_mean_normalized_action_changes[index]
                    )
                    / maximum_action_change,
                    0.0,
                    1.0,
                ),
                np.clip(projection / max(projection_min, 1.0e-8), 0.0, 1.0),
            )
            self.surface_pressure_quality[index] = float(min(quality_factors))
            normalized_margins = (
                1.0 - along_error / along_tolerance,
                1.0 - lateral_error / lateral_tolerance,
                len(window) / max(window.maxlen or 1, 1) - 1.0,
                speed / minimum_speed - 1.0,
                self.surface_path_efficiencies[index]
                / minimum_path_efficiency
                - 1.0,
                self.surface_useful_velocity_ratios[index]
                / minimum_useful_velocity
                - 1.0,
                (maximum_yaw_fraction
                 - self.surface_yaw_saturation_fractions[index])
                / maximum_yaw_fraction,
                (
                    maximum_action_change
                    - self.surface_mean_normalized_action_changes[index]
                )
                / maximum_action_change,
                projection / max(projection_min, 1.0e-8) - 1.0,
            )
            self.surface_pressure_threshold_margins[index] = float(
                np.clip(min(normalized_margins), -1.0, 1.0)
            )

        if self.surface_b1_latched and self.surface_pressure_valid[1]:
            self.surface_relay_dwell_seconds += self.dt
            self.surface_relay_pressure_integral += (
                max(0.0, self.current_b2_projection) * self.dt
            )
        elif not self.surface_b2_latched:
            self.surface_relay_dwell_seconds = 0.0
            self.surface_relay_pressure_integral = 0.0

        self.b1_history_contribution = float(
            self.surface_active_pressure_integrals[0]
        )
        self.b2_history_contribution = float(
            self.surface_active_pressure_integrals[1]
        )
        if bool(np.all(self.surface_pressure_valid)):
            self.combined_pressure_history += max(
                0.0, self.current_combined_projection
            ) * self.dt
        self.surface_pressure_joint_quality = float(
            min(self.surface_pressure_quality)
        )
        low_efficiency = np.clip(1.0 - self.surface_path_efficiencies, 0.0, 1.0)
        low_useful = np.clip(1.0 - self.surface_useful_velocity_ratios, 0.0, 1.0)
        yaw_fraction = np.clip(self.surface_yaw_saturation_fractions, 0.0, 1.0)
        self.surface_spin_cost = float(
            np.clip(np.mean((low_efficiency + low_useful + yaw_fraction) / 3.0), 0.0, 1.0)
        )


    def _surface_motion_gate(self) -> bool:
        cfg = self.intervention_cfg
        return bool(
            all(
                len(window) == window.maxlen
                for window in self._surface_motion_windows
            )
            and all(
                len(window) == window.maxlen
                for window in self._surface_action_change_windows
            )
            and np.all(
                self.surface_net_displacements
                >= float(cfg["net_displacement_min"])
            )
            and np.all(
                self.surface_path_efficiencies
                >= float(cfg["path_efficiency_min"])
            )
            and np.all(
                self.surface_useful_velocity_ratios
                >= float(cfg["useful_velocity_ratio_min"])
            )
            and np.all(
                self.surface_yaw_saturation_fractions
                <= float(cfg["yaw_saturation_fraction_max"])
            )
            and np.all(
                self.surface_mean_abs_yaw_rates
                <= float(cfg["mean_abs_yaw_rate_max"])
            )
            and np.all(
                self.surface_mean_normalized_action_changes
                <= float(cfg["mean_normalized_action_change_max"])
            )
            and np.all(
                self.surface_mean_speeds >= float(cfg["mean_speed_min"])
            )
        )


    @property
    def recent_counterfactual_mean(self) -> float:
        return (
            float(np.mean(self._counterfactual_gain_window))
            if self._counterfactual_gain_window
            else 0.0
        )


    @property
    def recent_counterfactual_max(self) -> float:
        return (
            float(np.max(self._counterfactual_gain_window))
            if self._counterfactual_gain_window
            else 0.0
        )


    @property
    def counterfactual_nondecreasing_ratio(self) -> float:
        return (
            float(np.mean(self._counterfactual_trend_window))
            if self._counterfactual_trend_window
            else 0.0
        )


    @property
    def recent_actual_clearance_min(self) -> float:
        return (
            float(np.min(self._actual_clearance_window))
            if self._actual_clearance_window
            else float(self.predicted_island_clearance)
        )


    def _surface_reward_qualities(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        zones = list(self.pressure_cfg["zones"])
        zone_quality: list[float] = []
        motion_quality: list[float] = []
        cfg = self.intervention_cfg
        for index, (state, zone) in enumerate(zip(self.blue_states, zones)):
            error = state.pos - self.pressure_zones[index]
            along_axis, lateral_axis = self._surface_zone_basis(index)
            along = float(error @ along_axis) / max(
                float(zone["along_tolerance"]), 1.0e-8
            )
            lateral = float(error @ lateral_axis) / max(
                float(zone["lateral_tolerance"]), 1.0e-8
            )
            rho = math.sqrt(along * along + lateral * lateral)
            zone_quality.append(float(np.clip(1.0 - rho / 2.0, 0.0, 1.0)))
            speed_quality = float(
                np.clip(
                    float(state.ground_speed) / max(float(cfg["minimum_speed"]), 1.0e-8),
                    0.0,
                    1.0,
                )
            )
            useful_quality = float(
                np.clip(
                    self.surface_useful_velocity_ratios[index]
                    / max(float(cfg["useful_velocity_ratio_min"]), 1.0e-8),
                    0.0,
                    1.0,
                )
            )
            efficiency_quality = float(
                np.clip(
                    self.surface_path_efficiencies[index]
                    / max(float(cfg["path_efficiency_min"]), 1.0e-8),
                    0.0,
                    1.0,
                )
            )
            motion_quality.append(
                0.40 * speed_quality
                + 0.40 * useful_quality
                + 0.20 * efficiency_quality
            )
        return (
            (float(zone_quality[0]), float(zone_quality[1])),
            (float(motion_quality[0]), float(motion_quality[1])),
        )


    def _surface_final_core_current(self) -> bool:
        return self._surface_effect_core_current()


    def _surface_persistence_core_current(self) -> bool:
        return bool(
            self.surface_complete_latched
            and self._surface_effect_core_current()
        )


    def _surface_effect_core_current(self) -> bool:
        """Effect-based Surface completion; pressure-zone geometry is diagnostic only."""

        cfg = self.intervention_cfg
        trend_ready = (
            len(self._counterfactual_trend_window)
            == self._counterfactual_trend_window.maxlen
        )
        return bool(
            self.surface_b1_latched
            and self.surface_b2_latched
            and self.counterfactual_cross_track_gain
            >= float(cfg["current_counterfactual_cross_track_min"])
            and self.maximum_counterfactual_cross_track_gain
            >= float(cfg["maximum_counterfactual_cross_track_min"])
            and self.clearance_gain_vs_shadow >= float(cfg["clearance_gain_min"])
            and self.predicted_island_clearance
            >= float(self.tracking_success_cfg["predicted_clearance_min"])
            and trend_ready
            and self.counterfactual_nondecreasing_ratio
            >= float(cfg["trend_majority_fraction"])
        )


    def _surface_pressure_core_current(
        self,
        *,
        require_deflection: bool = True,
        require_corridor: bool = True,
    ) -> bool:
        cfg = self.tracking_success_cfg
        return bool(
            (not require_corridor or np.all(self.surface_corridor_current))
            and self.current_b1_projection
            >= float(self.intervention_cfg["b1_apf_projection_min"])
            and self.current_b2_projection
            >= float(self.intervention_cfg["b2_apf_projection_min"])
            and self.predicted_island_clearance
            >= float(cfg["predicted_clearance_min"])
            and (
                not require_deflection
                or self.current_lateral_velocity
                >= float(cfg["lateral_velocity_min"])
            )
        )


    def _surface_role_pressure_current(self, role_id: int) -> bool:
        threshold_key = "b1_apf_projection_min" if role_id == 0 else "b2_apf_projection_min"
        projection = (
            self.current_b1_projection if role_id == 0 else self.current_b2_projection
        )
        return bool(
            self.surface_corridor_current[role_id]
            and projection >= float(self.intervention_cfg[threshold_key])
        )


    def _update_surface_event_latches(self) -> None:
        if "surface" not in self.task_config.active_roles:
            self.surface_gate_progress = 0.0
            return
        dwell_target = float(self.intervention_cfg["corridor_dwell_seconds"])
        integral_target = float(self.intervention_cfg["active_pressure_integral_min"])
        if (
            not self.surface_b1_latched
            and self.surface_effective_dwell_seconds[0] >= dwell_target
            and self.surface_effective_pressure_integrals[0] >= integral_target
        ):
            self.surface_b1_latched = True
            self.surface_b1_latch_step = self.step_count
            self.counterfactual_gain_at_b1_latch = float(
                self.counterfactual_cross_track_gain
            )
        if (
            not self.surface_b2_latched
            and self.surface_effective_dwell_seconds[1] >= dwell_target
            and self.surface_effective_pressure_integrals[1] >= integral_target
        ):
            self.surface_b2_latched = True
            self.surface_b2_latch_step = self.step_count
            self.b2_incremental_counterfactual_gain = float(
                self.counterfactual_cross_track_gain
                - self.counterfactual_gain_at_b1_latch
            )
        safe_direction = bool(
            self.predicted_island_clearance
            >= float(self.tracking_success_cfg["predicted_clearance_min"])
            and self.current_lateral_velocity
            >= float(self.tracking_success_cfg["lateral_velocity_min"])
        )
        if safe_direction and not self.surface_causal_latched:
            self.surface_causal_latched = True
            self.surface_causal_latch_step = self.step_count
        if safe_direction and not self.surface_island_avoidance_latched:
            self.surface_island_avoidance_latched = True
            self.surface_island_latch_step = self.step_count

        core = self._surface_effect_core_current()
        self._surface_success_window.append(core)
        valid_count = int(sum(self._surface_success_window))
        self.surface_final_hold_steps = valid_count
        if core and self.final_hold_start_step is None:
            self.final_hold_start_step = self.step_count
        self.surface_pressure_total_steps += 1
        if core:
            self.surface_pressure_valid_steps += 1
        window_ready = len(self._surface_success_window) == self._surface_success_window.maxlen
        valid_fraction = valid_count / max(len(self._surface_success_window), 1)
        stable_success = bool(
            core
            and window_ready
            and valid_fraction
            >= float(self.tracking_success_cfg["minimum_valid_fraction"])
        )
        if not self.surface_complete_latched and stable_success:
            self.surface_complete_latched = True
            self.surface_complete_step = self.step_count
        self.surface_gate_progress = float(
            np.clip(
                0.5 * np.mean((self.surface_effective_dwell_seconds >= dwell_target).astype(np.float32))
                + 0.5 * valid_fraction,
                0.0,
                1.0,
            )
        )


    def _current_surface_trend_is_favorable(self) -> bool:
        heading_vector = np.asarray(
            [math.cos(self.red_state.heading), math.sin(self.red_state.heading)],
            dtype=np.float32,
        )
        along = float(heading_vector @ self.natural_route_tangent)
        lateral = float(heading_vector @ self.desired_lateral_direction)
        heading_angle = math.atan2(lateral, max(along, 1.0e-8))
        return bool(
            self.current_lateral_velocity
            >= float(self.intervention_cfg["lateral_velocity_min"])
            or heading_angle >= math.radians(
                float(self.intervention_cfg["favorable_heading_degrees_min"])
            )
        )


    def _projected_landing_clearance(self, state: ShipState) -> float:
        velocity = state.velocity.astype(np.float64)
        speed_squared = float(velocity @ velocity)
        if speed_squared <= 1.0e-10:
            velocity = np.asarray(
                [math.cos(state.heading), math.sin(state.heading)],
                dtype=np.float64,
            )
            speed_squared = 1.0
        relative = self.island_center.astype(np.float64) - state.pos.astype(np.float64)
        projection_time = max(0.0, float(relative @ velocity) / speed_squared)
        closest = state.pos.astype(np.float64) + projection_time * velocity
        center_distance = float(np.linalg.norm(closest - self.island_center))
        return center_distance - self.island_radius - self.landing_buffer


    def _update_physical_references(self, *, initialize: bool) -> None:
        del initialize
        self.pressure_zones = self._build_pressure_zones()
        if "submarine" not in self.task_config.active_roles:
            self.prediction_center = self.red_state.pos.copy()
            self.prediction_heading = float(self.red_state.heading)
            self.slot_center = self.red_state.pos.copy()
            self.encircle_beta = 0.0
            self.radius_ref = self.initial_radius
            self.submarine_target_slots = self._static_submarine_target_slots.copy()
            return
        if self.task_config.environment.static_slots:
            self.prediction_center = self._static_slot_center.copy()
            self.prediction_heading = self._static_slot_heading
            self.slot_center = self._static_slot_center.copy()
            self.encircle_beta = 0.0
            self.radius_ref = self.initial_radius
            self.submarine_target_slots = self._static_submarine_target_slots.copy()
            return
        self.prediction_center = constant_velocity_prediction(
            self.red_state.pos,
            self.red_state.velocity,
            self.prediction_horizon_seconds,
        )
        self.prediction_heading = float(self.red_state.heading)
        self.slot_center = self.prediction_center.copy()
        self.encircle_beta = 1.0
        self.radius_ref = self.formation_radius
        raw_slots = triangular_slots(
            self.slot_center,
            self.prediction_heading,
            self.formation_radius,
        ).astype(np.float32)
        canonical_slots = raw_slots[[2, 0, 1]].astype(np.float32)
        self.submarine_target_slots = canonical_slots[
            self.submarine_slot_assignment
        ].astype(np.float32)


    def _refresh_geometry_metrics(self) -> None:
        positions = np.asarray([state.pos for state in self.submarine_states], dtype=np.float32)
        velocities = np.asarray(
            [state.velocity for state in self.submarine_states], dtype=np.float32
        )
        if positions.shape != (3, 2):
            raise ValueError("exactly three submarine positions are required")
        self.red_submarine_distances = np.linalg.norm(
            positions - self.red_state.pos[None, :], axis=1
        ).astype(np.float32)
        self.slot_errors = np.linalg.norm(
            positions - self.submarine_target_slots, axis=1
        ).astype(np.float32)
        self.mean_slot_error = float(np.mean(self.slot_errors))
        self.max_slot_error = float(np.max(self.slot_errors))
        self.max_angular_gap = float(max_angular_gap(positions, self.red_state.pos))
        self.triangle_margin = float(triangle_margin(self.red_state.pos, positions))
        relative_speeds = np.linalg.norm(
            velocities - self.red_state.velocity[None, :], axis=1
        )
        self.mean_relative_speed = float(np.mean(relative_speeds))
        slot_velocity_errors = np.linalg.norm(
            velocities - self.submarine_target_velocities, axis=1
        )
        self.slot_velocity_errors = slot_velocity_errors.astype(np.float32)
        self.mean_slot_velocity_error = float(np.mean(slot_velocity_errors))
        angles = np.mod(
            np.arctan2(
                positions[:, 1] - self.red_state.pos[1],
                positions[:, 0] - self.red_state.pos[0],
            ),
            2.0 * np.pi,
        )
        ordered = np.sort(angles)
        gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
        self.angular_distribution_error = float(
            np.mean(np.abs(gaps - 2.0 * np.pi / 3.0))
        )
        self.mean_radius_error = float(
            np.mean(np.abs(self.red_submarine_distances - self.radius_ref))
        )


    def _final_surface_success(self) -> bool:
        return bool(
            True
            and self.surface_intervention_success
        )


    def _training_capture_core_gate(
        self, collision: bool, boundary: bool, landing: bool
    ) -> bool:
        all_low, all_high = _range_pair(
            self.physical_cfg["submarine_distance_range"],
            "physical_capture.submarine_distance_range",
        )
        core_low, core_high = _range_pair(
            self.physical_cfg["core_distance_range"],
            "physical_capture.core_distance_range",
        )
        in_core = int(
            np.count_nonzero(
                (self.red_submarine_distances >= core_low)
                & (self.red_submarine_distances <= core_high)
            )
        )
        return bool(
            self.formation_radius <= self.compact_radius + 1.0e-6
            and np.all(self.red_submarine_distances >= all_low)
            and np.all(self.red_submarine_distances <= all_high)
            and in_core
            >= int(self.physical_cfg["training_minimum_core_submarines"])
            and self.mean_slot_error
            <= float(self.physical_cfg["training_mean_slot_error_max"])
            and self.max_slot_error
            <= float(self.physical_cfg["training_max_slot_error_max"])
            and self.max_angular_gap
            <= math.radians(float(self.physical_cfg["max_angular_gap_degrees"]))
            and self.triangle_contains_red
            and self.mean_relative_speed
            <= float(self.physical_cfg["training_mean_relative_speed_max"])
            and not collision
            and not boundary
            and not landing
        )


    def _final_gate_components(self, collision: bool) -> dict[str, bool]:
        core_low, core_high = _range_pair(
            self.physical_cfg["core_distance_range"],
            "physical_capture.core_distance_range",
        )
        all_in_core = bool(
            np.all(self.red_submarine_distances >= core_low)
            and np.all(self.red_submarine_distances <= core_high)
        )
        shrink_cfg = self.encirclement_cfg["shrink_tracking"]
        return {
            "distance_band": all_in_core,
            "core_band": all_in_core,
            "triangle": bool(self.triangle_contains_red),
            "angular_gap": bool(
                self.max_angular_gap
                <= math.radians(float(shrink_cfg["max_angle_gap_degrees"]))
            ),
            "slot_error": bool(
                self.mean_slot_error <= float(shrink_cfg["mean_slot_error_max"])
                and self.max_slot_error <= float(shrink_cfg["max_slot_error_max"])
            ),
            # Relative speed is intentionally not part of the shared shrink/final gate.
            "relative_speed": True,
            "collision": not bool(collision),
        }


    def _red_boundary_clearance(self) -> float:
        x, y = (float(value) for value in self.red_state.pos)
        width, height = (float(value) for value in self.world_size)
        return min(x, width - x, y, height - y)


    def _update_stage_success_windows(
        self, collision: bool, boundary: bool, landing: bool
    ) -> None:
        del collision, boundary, landing


    @property
    def b1_history_ready(self) -> bool:
        if not bool(self.history_profile["require_history"]):
            return False
        return self.b1_history_contribution >= float(
            self.history_profile["b1_min"]
        )


    @property
    def b2_history_ready(self) -> bool:
        if not bool(self.history_profile["require_history"]):
            return False
        return self.b2_history_contribution >= float(
            self.history_profile["b2_min"]
        )


    @property
    def combined_pressure_ready(self) -> bool:
        if not bool(self.history_profile["require_history"]):
            return False
        return self.combined_pressure_history >= float(
            self.history_profile["combined_min"]
        )


    @property
    def lateral_deviation_ready(self) -> bool:
        if not bool(self.history_profile["require_history"]):
            return False
        return self.max_favorable_lateral_deviation >= float(
            self.history_profile["lateral_deviation_min"]
        )


    @property
    def surface_intervention_success(self) -> bool:
        return bool(
            "surface" in self.task_config.active_roles
            and self.surface_complete_latched
        )


    @property
    def triangle_contains_red(self) -> bool:
        positions = np.asarray(
            [state.pos for state in self.submarine_states], dtype=np.float64
        )
        left, middle, right = positions
        twice_area = abs(
            (middle[0] - left[0]) * (right[1] - left[1])
            - (middle[1] - left[1]) * (right[0] - left[0])
        )
        return bool(twice_area > 1.0e-6 and self.triangle_margin >= 0.0)


    def unmet_success_gates(
        self,
        *,
        collision: bool = False,
        boundary: bool = False,
        landing: bool = False,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for name, passed in self.stage_success_gates(
                collision=collision,
                boundary=boundary,
                landing=landing,
            ).items()
            if not passed
        )


    def _current_stage_core_gate(
        self, collision: bool, boundary: bool, landing: bool
    ) -> bool:
        gates = self.stage_success_gates(
            collision=collision,
            boundary=boundary,
            landing=landing,
        )
        excluded = {"final_hold", "surface_final_hold"}
        return all(passed for name, passed in gates.items() if name not in excluded)


    def _advance_stage_hold(self, core_gate: bool) -> None:
        del core_gate
        self.stage_hold_step_count = 0
        self.final_geometry_hold_steps = 0
        self.stage_hold_seconds = 0.0


    def stage_success_predicate(
        self,
        *,
        collision: bool = False,
        boundary: bool = False,
        landing: bool = False,
    ) -> bool:
        return all(
            self.stage_success_gates(
                collision=collision,
                boundary=boundary,
                landing=landing,
            ).values()
        )


    def relaxed_success_predicate(
        self,
        *,
        collision: bool = False,
        boundary: bool = False,
        landing: bool = False,
    ) -> bool:
        return bool(
            True
            and self.stage_success_predicate(
                collision=collision,
                boundary=boundary,
                landing=landing,
            )
        )


    def _physical_reward_snapshot(self) -> RewardSnapshot:
        surface_positions = np.asarray(
            [state.pos for state in self.blue_states], dtype=np.float32
        )
        submarine_positions = np.asarray(
            [state.pos for state in self.submarine_states], dtype=np.float32
        )
        submarine_velocities = np.asarray(
            [state.velocity for state in self.submarine_states], dtype=np.float32
        )
        pressure_errors = np.linalg.norm(
            surface_positions - self.pressure_zones, axis=1
        ).astype(np.float32)
        distribution_quality = float(
            np.clip(
                1.0 - self.angular_distribution_error / (2.0 * math.pi / 3.0),
                0.0,
                1.0,
            )
        )
        # Max-gap quality is scored independently by the reward evaluator.  Keep
        # this field limited to the complementary angular-distribution signal so
        # the same geometric defect is not counted twice.
        triangle_geometry_quality = distribution_quality
        triangle_margin_scale = max(
            float(
                self.reward_cfg.get("normalization", {}).get(
                    "triangle_margin_scale_m", 0.5 * self.geometry_scale
                )
            ),
            1.0e-8,
        )
        triangle_containment_quality = float(
            np.clip(
                max(0.0, self.triangle_margin) / triangle_margin_scale,
                0.0,
                1.0,
            )
        )
        if self.task_config.environment.compaction_enabled:
            # Treat the pre-shrink 4.5 m hold and the final compact hold as one
            # monotonic requirement.  Switching denominators at compaction start
            # would otherwise manufacture a one-step negative progress spike.
            completed_hold_steps = min(
                self.formation45_hold_steps, self.formation45_hold_target_steps
            ) + min(self.stage_hold_step_count, self.final_hold_target_steps)
            required_hold_steps = (
                self.formation45_hold_target_steps + self.final_hold_target_steps
            )
            submarine_hold_progress = completed_hold_steps / max(
                required_hold_steps, 1
            )
        else:
            submarine_hold_progress = self.stage_hold_step_count / max(
                self.final_hold_target_steps, 1
            )
        submarine_efficiency = np.divide(
            self.submarine_net_displacements,
            np.maximum(self.submarine_path_lengths, 1.0e-8),
        )
        submarine_kinematic_inefficiency = float(
            np.clip(np.mean(1.0 - np.clip(submarine_efficiency, 0.0, 1.0)), 0.0, 1.0)
        )
        b1_hold_progress, b2_hold_progress, causal_progress = (
            self._surface_latch_progresses()
        )
        surface_zone_qualities, surface_motion_qualities = (
            self._surface_reward_qualities()
        )
        success = self.task_config.success
        return RewardSnapshot(
            phase=self.phase,
            dt=self.dt,
            stable_seconds=self.stable_seconds,
            stabilization_seconds=self.stabilization_seconds,
            formation_radius=self.formation_radius,
            initial_radius=self.initial_radius,
            compact_radius=self.compact_radius,
            compaction_started=self.compaction_started,
            compaction_complete=self.compaction_complete,
            green_seconds=self.green_seconds,
            green_hold_seconds=self.green_hold_seconds,
            b1_history=self.b1_history_contribution,
            b2_history=self.b2_history_contribution,
            combined_history=self.combined_pressure_history,
            max_favorable_lateral_deviation=self.max_favorable_lateral_deviation,
            signed_lateral_deviation=self.current_lateral_deviation,
            b1_apf_projection=self.current_b1_projection,
            b2_apf_projection=self.current_b2_projection,
            combined_apf_projection=self.current_combined_projection,
            pressure_errors=pressure_errors,
            surface_positions=surface_positions,
            surface_velocities=tuple(
                tuple(float(value) for value in state.velocity)
                for state in self.blue_states
            ),
            surface_headings=tuple(float(state.heading) for state in self.blue_states),
            surface_yaw_rates=tuple(float(state.yaw_rate) for state in self.blue_states),
            pressure_zone_positions=tuple(
                tuple(float(value) for value in point) for point in self.pressure_zones
            ),
            pressure_zone_velocities=tuple(
                tuple(float(value) for value in velocity)
                for velocity in self.pressure_zone_velocities
            ),
            submarine_positions=submarine_positions,
            submarine_velocities=submarine_velocities,
            submarine_headings=tuple(
                float(state.heading) for state in self.submarine_states
            ),
            submarine_yaw_rates=tuple(
                float(state.yaw_rate) for state in self.submarine_states
            ),
            submarine_surfaced=np.asarray(
                [state.surfaced for state in self.submarine_states], dtype=bool
            ),
            friendly_active_mask=self._agent_active_mask.copy(),
            slot_positions=self.submarine_target_slots.copy(),
            slot_velocities=self.submarine_target_velocities.copy(),
            red_pos=self.red_state.pos.copy(),
            red_velocity=self.red_state.velocity.copy(),
            max_angle_gap=self.max_angular_gap,
            triangle_margin=self.triangle_margin,
            action_limit=self.agent_action_limits.copy(),
            max_surface_speed=float(self.max_speed_by_type["surface"]),
            max_submarine_speed=float(self.max_speed_by_type["submarine"]),
            max_yaw_rate=float(self.max_yaw_rate),
            max_red_speed=float(self.max_speed_red),
            surface_influence_range=float(self.red_policy_cfg["repulsion_range"]),
            counterfactual_reference=float(
                self.intervention_cfg["current_counterfactual_cross_track_min"]
            ),
            clearance_reference=float(self.intervention_cfg["clearance_gain_min"]),
            # Boundary contact is handled by the hard-wall response and is
            # intentionally neither terminal nor reward-shaped.
            world_size=None,
            island_center=self.island_center.copy(),
            island_radius=self.island_radius,
            landing_buffer=self.landing_buffer,
            island_active=bool(self.island_active),
            stage_hold_seconds=self.stage_hold_seconds,
            stage_hold_target_seconds=self.final_hold_target_steps * self.dt,
            formation_hold_target_seconds=(
                self.formation45_hold_target_steps * self.dt
                if self.task_config.environment.compaction_enabled
                else 0.0
            ),
            is_first_step=self.step_count <= 1,
            surface_active="surface" in self.task_config.active_roles,
            submarine_active="submarine" in self.task_config.active_roles,
            surface_b1_latched=bool(self.surface_b1_latched),
            surface_b2_latched=bool(self.surface_b2_latched),
            surface_causal_latched=bool(self.surface_causal_latched),
            surface_complete_latched=bool(self.surface_complete_latched),
            surface_effective_dwell_seconds=tuple(
                float(value) for value in self.surface_effective_dwell_seconds
            ),
            surface_effective_pressure_integrals=tuple(
                float(value) for value in self.surface_effective_pressure_integrals
            ),
            counterfactual_cross_track_gain=self.counterfactual_cross_track_gain,
            clearance_gain_vs_shadow=self.clearance_gain_vs_shadow,
            surface_gate_progress=float(np.clip(self.surface_gate_progress, 0.0, 1.0)),
            surface_pressure_b1_valid=bool(self.surface_pressure_valid[0]),
            surface_pressure_b2_valid=bool(self.surface_pressure_valid[1]),
            surface_pressure_b1_quality=float(
                np.clip(self.surface_pressure_quality[0], 0.0, 1.0)
            ),
            surface_pressure_b2_quality=float(
                np.clip(self.surface_pressure_quality[1], 0.0, 1.0)
            ),
            surface_pressure_joint_quality=float(
                np.clip(self.surface_pressure_joint_quality, 0.0, 1.0)
            ),
            surface_pressure_b1_threshold_margin=float(
                np.clip(self.surface_pressure_threshold_margins[0], -1.0, 1.0)
            ),
            surface_pressure_b2_threshold_margin=float(
                np.clip(self.surface_pressure_threshold_margins[1], -1.0, 1.0)
            ),
            surface_b1_hold_progress=b1_hold_progress,
            surface_b2_hold_progress=b2_hold_progress,
            surface_causal_progress=causal_progress,
            surface_relay_dwell_seconds=float(self.surface_relay_dwell_seconds),
            surface_relay_pressure_integral=float(
                self.surface_relay_pressure_integral
            ),
            surface_spin_cost=float(np.clip(self.surface_spin_cost, 0.0, 1.0)),
            triangle_geometry_quality=float(
                np.clip(triangle_geometry_quality, 0.0, 1.0)
            ),
            triangle_containment_quality=triangle_containment_quality,
            submarine_hold_progress=float(
                np.clip(submarine_hold_progress, 0.0, 1.0)
            ),
            reference_radius=float(self.radius_ref),
            submarine_kinematic_inefficiency_cost=submarine_kinematic_inefficiency,
            task_name="final_task",
            max_steps=self.max_steps,
            compaction_enabled=bool(
                self.task_config.environment.compaction_enabled
            ),
            surface_zone_qualities=surface_zone_qualities,
            surface_motion_qualities=surface_motion_qualities,
            surface_speeds=(
                float(self.blue_states[0].ground_speed),
                float(self.blue_states[1].ground_speed),
            ),
            recent_counterfactual_mean=self.recent_counterfactual_mean,
            recent_counterfactual_max=self.recent_counterfactual_max,
            counterfactual_nondecreasing_ratio=(
                self.counterfactual_nondecreasing_ratio
            ),
            predicted_island_clearance=float(self.predicted_island_clearance),
            shadow_predicted_island_clearance=float(
                self.shadow_predicted_island_clearance
            ),
            recent_actual_clearance_min=self.recent_actual_clearance_min,
            surface_island_avoidance_latched=bool(
                self.surface_island_avoidance_latched
            ),
            surface_final_hold_progress=float(
                np.clip(
                    self.surface_final_hold_steps
                    / max(self.surface_final_hold_target_steps, 1),
                    0.0,
                    1.0,
                )
            ),
            formation45_hold_progress=float(
                np.clip(
                    self.formation45_hold_steps
                    / max(self.formation45_hold_target_steps, 1),
                    0.0,
                    1.0,
                )
            ),
            final_hold_progress=float(
                np.clip(
                    self.stage_hold_step_count
                    / max(self.final_hold_target_steps, 1),
                    0.0,
                    1.0,
                )
            ),
            slot_good_error=float(
                1.0 if success.max_slot_error is None else success.max_slot_error
            ),
            velocity_good_error=float(
                0.75
                if success.mean_velocity_error is None
                else success.mean_velocity_error
            ),
            gap_good_degrees=float(
                168.0
                if success.max_angle_gap_degrees is None
                else success.max_angle_gap_degrees
            ),
            surface_final_core_current=self._surface_final_core_current(),
            surface_persistence_core_current=(
                self._surface_persistence_core_current()
            ),
            submarine_tracking_core_current=self._submarine_pre_shrink_gate(),
            submarine_shrink_core_current=(
                self._submarine_shrink_tracking_gate()
                if self.task_config.environment.compaction_enabled
                else False
            ),
            submarine_final_core_current=self._stage_final_capture_core_gate(
                False, False, False
            ),
        )


    def _history_scale(self, name: str) -> float:
        normalization = dict(self.pressure_cfg["history_normalization"])
        return max(float(normalization[name]), 1e-6)


    def _surface_latch_progresses(self) -> tuple[float, float, float]:
        cfg = self.intervention_cfg
        dwell_target = max(float(cfg["corridor_dwell_seconds"]), 1.0e-8)
        pressure_target = max(float(cfg["active_pressure_integral_min"]), 1.0e-8)
        b1 = 1.0 if self.surface_b1_latched else float(
            np.clip(
                min(
                    self.surface_effective_dwell_seconds[0] / dwell_target,
                    self.surface_effective_pressure_integrals[0] / pressure_target,
                ),
                0.0,
                1.0,
            )
        )
        b2 = 1.0 if self.surface_b2_latched else float(
            np.clip(
                min(
                    self.surface_effective_dwell_seconds[1] / dwell_target,
                    self.surface_effective_pressure_integrals[1] / pressure_target,
                ),
                0.0,
                1.0,
            )
        )
        trend_fraction = (
            float(np.mean(self._counterfactual_trend_window))
            if self._counterfactual_trend_window
            else 0.0
        )
        trend_ready = len(self._counterfactual_trend_window) / max(
            self._counterfactual_trend_window.maxlen or 1, 1
        )
        trend_target = max(float(cfg["trend_majority_fraction"]), 1.0e-8)
        causal = 1.0 if self.surface_causal_latched else float(
            np.clip(
                min(
                    b2,
                    self.counterfactual_cross_track_gain
                    / max(float(cfg["current_counterfactual_cross_track_min"]), 1.0e-8),
                    self.maximum_counterfactual_cross_track_gain
                    / max(float(cfg["maximum_counterfactual_cross_track_min"]), 1.0e-8),
                    self.red_actual_cross_track
                    / max(float(cfg["route_corridor_half_width"]), 1.0e-8),
                    trend_ready,
                    trend_fraction / trend_target,
                ),
                0.0,
                1.0,
            )
        )
        return b1, b2, causal


    def _common_task_context(self) -> tuple[np.ndarray, list[str]]:
        values: list[float] = []
        names: list[str] = []

        def append(name: str, value: float) -> None:
            names.append(name)
            values.append(float(value))

        def progress(current_steps: int | float, target_steps: int | float) -> float:
            current = float(current_steps)
            target = float(target_steps)
            if not math.isfinite(current) or not math.isfinite(target):
                return 0.0
            if target <= 0.0:
                return 0.0
            return float(np.clip(current / target, 0.0, 1.0))

        def clip01(value: float) -> float:
            numeric = float(value)
            if not math.isfinite(numeric):
                return 0.0
            return float(np.clip(numeric, 0.0, 1.0))

        for phase in PHASES:
            append(f"phase_{phase}", float(self.phase == phase))
        append("final_task", 1.0)
        append(
            "stage_compaction_enabled",
            float(self.task_config.environment.compaction_enabled),
        )
        append(
            "stage_submarine_dynamics_enabled",
            float(self.task_config.environment.submarine_dynamics),
        )
        append(
            "stage_static_slots",
            float(self.task_config.environment.static_slots),
        )
        # Markov task-state block. Its order is part of the observation schema.
        append("surface_b1_latched", float(self.surface_b1_latched))
        append("surface_b2_latched", float(self.surface_b2_latched))
        append("surface_causal_latched", float(self.surface_causal_latched))
        append(
            "surface_island_avoidance_latched",
            float(self.surface_island_avoidance_latched),
        )
        append("surface_complete_latched", float(self.surface_complete_latched))
        append(
            "surface_gate_progress",
            float(np.clip(self.surface_gate_progress, 0.0, 1.0))
            if math.isfinite(float(self.surface_gate_progress))
            else 0.0,
        )
        append(
            "stage_hold_progress",
            progress(self.stage_hold_step_count, self.final_hold_target_steps),
        )
        append(
            "surface_final_hold_progress",
            progress(
                self.surface_final_hold_steps,
                self.surface_final_hold_target_steps,
            ),
        )
        append(
            "formation45_hold_progress",
            progress(
                self.formation45_hold_steps,
                self.formation45_hold_target_steps,
            ),
        )
        append(
            "shrink_progress",
            progress(self.shrink_steps, self.minimum_shrink_steps),
        )
        append("compaction_started", float(self.compaction_started))
        append("compaction_complete", float(self.compaction_complete))
        intervention_cfg = self.intervention_cfg
        append(
            "counterfactual_gain_progress",
            clip01(
                self.counterfactual_cross_track_gain
                / max(float(intervention_cfg["current_counterfactual_cross_track_min"]), 1.0e-8)
            ),
        )
        append(
            "counterfactual_max_gain_progress",
            clip01(
                self.maximum_counterfactual_cross_track_gain
                / max(float(intervention_cfg["maximum_counterfactual_cross_track_min"]), 1.0e-8)
            ),
        )
        append(
            "clearance_gain_progress",
            clip01(
                self.clearance_gain_vs_shadow
                / max(float(intervention_cfg["clearance_gain_min"]), 1.0e-8)
            ),
        )
        shadow_clearance = float(self.shadow_predicted_island_clearance)
        append(
            "shadow_clearance_progress",
            0.0
            if not math.isfinite(shadow_clearance)
            else clip01(
                1.0
                - max(shadow_clearance, 0.0)
                / max(float(intervention_cfg["clearance_gain_min"]), 1.0e-8)
            ),
        )
        dwell_target = float(intervention_cfg["corridor_dwell_seconds"])
        pressure_target = float(intervention_cfg["active_pressure_integral_min"])
        append(
            "b1_corridor_dwell_progress",
            progress(self.surface_corridor_dwell_seconds[0], dwell_target),
        )
        append(
            "b2_relay_dwell_progress",
            progress(self.surface_relay_dwell_seconds, dwell_target),
        )
        append(
            "b1_active_pressure_integral_progress",
            progress(self.surface_active_pressure_integrals[0], pressure_target),
        )
        append(
            "b2_relay_pressure_integral_progress",
            progress(self.surface_relay_pressure_integral, pressure_target),
        )
        append(
            "b1_net_displacement_progress",
            clip01(self.surface_net_displacements[0] / (3.5 * self.geometry_scale)),
        )
        append(
            "b2_net_displacement_progress",
            clip01(self.surface_net_displacements[1] / (3.5 * self.geometry_scale)),
        )
        append("b1_path_efficiency", clip01(self.surface_path_efficiencies[0]))
        append("b2_path_efficiency", clip01(self.surface_path_efficiencies[1]))
        append(
            "b1_useful_velocity_ratio",
            clip01(self.surface_useful_velocity_ratios[0]),
        )
        append(
            "b2_useful_velocity_ratio",
            clip01(self.surface_useful_velocity_ratios[1]),
        )
        append(
            "b1_yaw_saturation_fraction",
            clip01(self.surface_yaw_saturation_fractions[0]),
        )
        append(
            "b2_yaw_saturation_fraction",
            clip01(self.surface_yaw_saturation_fractions[1]),
        )
        yaw_target = float(intervention_cfg["mean_abs_yaw_rate_max"])
        append(
            "b1_mean_abs_yaw_rate_fraction",
            clip01(self.surface_mean_abs_yaw_rates[0] / max(yaw_target, 1.0e-8)),
        )
        append(
            "b2_mean_abs_yaw_rate_fraction",
            clip01(self.surface_mean_abs_yaw_rates[1] / max(yaw_target, 1.0e-8)),
        )
        speed_target = float(intervention_cfg["mean_speed_min"])
        append(
            "b1_mean_speed_progress",
            clip01(self.surface_mean_speeds[0] / max(speed_target, 1.0e-8)),
        )
        append(
            "b2_mean_speed_progress",
            clip01(self.surface_mean_speeds[1] / max(speed_target, 1.0e-8)),
        )
        append(
            "surface_pressure_b1_quality",
            clip01(self.surface_pressure_quality[0]),
        )
        append(
            "surface_pressure_b2_quality",
            clip01(self.surface_pressure_quality[1]),
        )
        append(
            "surface_pressure_joint_quality",
            clip01(self.surface_pressure_joint_quality),
        )
        append(
            "surface_pressure_b1_threshold_margin",
            float(np.clip(self.surface_pressure_threshold_margins[0], -1.0, 1.0)),
        )
        append(
            "surface_pressure_b2_threshold_margin",
            float(np.clip(self.surface_pressure_threshold_margins[1], -1.0, 1.0)),
        )
        b1_hold_progress, b2_hold_progress, causal_progress = (
            self._surface_latch_progresses()
        )
        append("surface_b1_hold_progress", b1_hold_progress)
        append("surface_b2_hold_progress", b2_hold_progress)
        append("surface_causal_progress", causal_progress)
        append(
            "counterfactual_trend_fraction",
            float(np.mean(self._counterfactual_trend_window))
            if self._counterfactual_trend_window
            else 0.0,
        )
        scale = max(float(np.max(self.world_size)), 1.0e-6)
        shadow_relative_position = self.shadow_red_state.pos - self.red_state.pos
        shadow_relative_velocity = (
            self.shadow_red_state.velocity - self.red_state.velocity
        )
        shadow_heading_delta = float(
            wrap_angle(self.shadow_red_state.heading - self.red_state.heading)
        )
        append("shadow_relative_position_x", shadow_relative_position[0] / scale)
        append("shadow_relative_position_y", shadow_relative_position[1] / scale)
        append(
            "shadow_relative_velocity_x",
            shadow_relative_velocity[0] / max(self.max_speed_red, 1.0e-6),
        )
        append(
            "shadow_relative_velocity_y",
            shadow_relative_velocity[1] / max(self.max_speed_red, 1.0e-6),
        )
        append("shadow_relative_heading_sin", math.sin(shadow_heading_delta))
        append("shadow_relative_heading_cos", math.cos(shadow_heading_delta))
        append(
            "shadow_clearance_gain",
            0.0
            if not math.isfinite(float(self.clearance_gain_vs_shadow))
            else float(np.clip(self.clearance_gain_vs_shadow / scale, -1.0, 1.0)),
        )
        submarine_path_scale = max(
            self.max_speed_by_type["submarine"] * self.max_steps * self.dt,
            1.0e-6,
        )
        for submarine_id in range(self.num_submarines):
            net_displacement = float(self.submarine_net_displacements[submarine_id])
            path_length = float(self.submarine_path_lengths[submarine_id])
            append(
                f"submarine_{submarine_id}_net_displacement",
                clip01(net_displacement / scale),
            )
            append(
                f"submarine_{submarine_id}_path_length",
                clip01(path_length / submarine_path_scale),
            )
            append(
                f"submarine_{submarine_id}_path_efficiency",
                clip01(net_displacement / max(path_length, 1.0e-8)),
            )
        append("red_pos_x", self.red_state.pos[0] / self.world_size[0])
        append("red_pos_y", self.red_state.pos[1] / self.world_size[1])
        append("red_velocity_x", self.red_state.velocity[0] / max(self.max_speed_red, 1e-6))
        append("red_velocity_y", self.red_state.velocity[1] / max(self.max_speed_red, 1e-6))
        append("red_heading_sin", math.sin(self.red_state.heading))
        append("red_heading_cos", math.cos(self.red_state.heading))
        append("route_tangent_x", self.natural_route_tangent[0])
        append("route_tangent_y", self.natural_route_tangent[1])
        append("desired_lateral_x", self.desired_lateral_direction[0])
        append("desired_lateral_y", self.desired_lateral_direction[1])
        append("prediction_center_x", self.prediction_center[0] / self.world_size[0])
        append("prediction_center_y", self.prediction_center[1] / self.world_size[1])
        append("prediction_heading_sin", math.sin(self.prediction_heading))
        append("prediction_heading_cos", math.cos(self.prediction_heading))
        for slot_id, slot in enumerate(self.submarine_target_slots):
            append(f"slot_{slot_id}_x", slot[0] / self.world_size[0])
            append(f"slot_{slot_id}_y", slot[1] / self.world_size[1])
        append("encircle_beta", self.encircle_beta)
        append("radius_ref", self.radius_ref / max(self.world_size))
        append(
            "commanded_encirclement_radius",
            self.formation_radius / max(self.world_size),
        )
        append("stable_progress", self.stable_seconds / max(self.stabilization_seconds, 1.0e-6))
        append("stable_steps_normalized", self.stable_seconds / max(self.stabilization_seconds, 1.0e-6))
        append("episode_progress", self.step_count / max(self.max_steps, 1))
        append("episode_remaining", max(self.max_steps - self.step_count, 0) / max(self.max_steps, 1))
        append(
            "red_landing_clearance",
            (norm(self.red_state.pos - self.island_center) - self.island_radius - self.landing_buffer)
            / max(float(np.max(self.world_size)), 1.0e-6),
        )
        append("physical_core_gate", float(self.physical_core_gate))
        append(
            "b1_history",
            min(self.b1_history_contribution / self._history_scale("b1_contribution"), 2.0),
        )
        append(
            "b2_history",
            min(self.b2_history_contribution / self._history_scale("b2_contribution"), 2.0),
        )
        append(
            "combined_pressure_history",
            min(self.combined_pressure_history / self._history_scale("combined_pressure"), 2.0),
        )
        append(
            "max_lateral_deviation",
            self.max_favorable_lateral_deviation / max(self.world_size),
        )
        append(
            "current_lateral_deviation",
            self.current_lateral_deviation / max(self.world_size),
        )
        append(
            "current_lateral_velocity",
            self.current_lateral_velocity / max(self.max_speed_red, 1e-6),
        )
        append("surface_intervention_success", float(self.surface_intervention_success))
        append("max_angular_gap", self.max_angular_gap / (2.0 * np.pi))
        append("triangle_margin", self.triangle_margin / max(float(np.max(self.world_size)), 1e-6))
        append(
            "physical_hold_credit",
            min(self.physical_hold_credit / max(self.green_hold_seconds, 1.0e-6), 2.0),
        )
        append("surface_active_b1", self.active_surface_mask[0])
        append("surface_active_b2", self.active_surface_mask[1])
        append("submarine_active_y1", self._agent_active_mask[2])
        append("submarine_active_y2", self._agent_active_mask[3])
        append("submarine_active_y3", self._agent_active_mask[4])
        append("surface_b1_recent", min(max(self.surface_b1_recent, -2.0), 2.0))
        append("surface_b2_recent", min(max(self.surface_b2_recent, -2.0), 2.0))
        append(
            "surface_combined_recent",
            min(max(self.surface_combined_recent, -2.0), 2.0),
        )
        append("surface_trend_fraction", self.surface_trend_fraction)
        append(
            "surface_projected_landing_clearance",
            0.0
            if not np.isfinite(self.surface_projected_landing_clearance)
            else self.surface_projected_landing_clearance / max(float(np.max(self.world_size)), 1.0e-6),
        )
        for role_id, zone in enumerate(self.pressure_zones):
            append(f"pressure_zone_{role_id}_x", zone[0] / self.world_size[0])
            append(f"pressure_zone_{role_id}_y", zone[1] / self.world_size[1])
        result = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise AssertionError("common task context must be finite")
        return result, names


    def _physical_actor_task_context(
        self, friendly_index: int
    ) -> tuple[np.ndarray, list[str]]:
        state = self.friendly_states()[friendly_index]
        target_positions, target_velocities, desired_velocities = (
            self._assigned_guidance_targets()
        )
        if friendly_index >= self.num_surface_blue and self.slot_observation_enabled:
            observed_position, observed_target_velocity = self._observed_submarine_target(
                friendly_index - self.num_surface_blue
            )
            target_positions[friendly_index] = observed_position
            # Recompute the same bounded guidance velocity using the observed
            # slot. This preserves the clean-channel behavior exactly while
            # making stale slot messages affect the actor's target and desired
            # velocity together.
            state_speed = max(float(self.max_speed_by_type[state.unit_type]), 1.0e-6)
            k_pos = float(self.reward_cfg["normalization"]["k_pos"])
            correction_ratio = float(self.reward_cfg["normalization"]["correction_speed_ratio"])
            correction = k_pos * (observed_position - state.pos)
            correction_norm = norm(correction)
            correction_limit = correction_ratio * state_speed
            if correction_norm > correction_limit:
                correction *= correction_limit / max(correction_norm, 1.0e-8)
            desired = observed_target_velocity + correction
            desired_norm = norm(desired)
            if desired_norm > state_speed:
                desired *= state_speed / max(desired_norm, 1.0e-8)
            desired_velocities[friendly_index] = desired
        identity = np.zeros(3, dtype=np.float32)
        role_id = (
            friendly_index
            if friendly_index < self.num_surface_blue
            else friendly_index - self.num_surface_blue
        )
        identity[role_id] = 1.0
        scale = max(float(np.max(self.world_size)), 1.0e-6)
        previous = (
            self.previous_actions[friendly_index]
            / self.agent_action_limits[friendly_index]
        )
        own_target_error = norm(target_positions[friendly_index] - state.pos)
        local_hold_progress = float(np.clip(1.0 - own_target_error / max(self.slot_arrival_tolerance, 1.0e-6), 0.0, 1.0))
        local_gate_progress = local_hold_progress
        if friendly_index < self.num_surface_blue:
            q_intervention, engagement = self._surface_intervention_qualities()
            local_hold_progress = q_intervention
            local_gate_progress = float(engagement[friendly_index])
        values = [
            *target_positions[friendly_index].tolist(),
            *desired_velocities[friendly_index].tolist(),
            *identity.tolist(),
            *previous.tolist(),
            float(self.task_config.environment.static_slots),
            float(self.task_config.environment.compaction_enabled),
            float(self.formation_radius / scale),
            local_hold_progress,
            local_gate_progress,
            float(max(self.max_steps - self.step_count, 0) / max(self.max_steps, 1)),
        ]
        names = [
            "assigned_target_x",
            "assigned_target_y",
            "assigned_desired_velocity_x",
            "assigned_desired_velocity_y",
            "assignment_id_0",
            "assignment_id_1",
            "assignment_id_2",
            "previous_surge_force_action",
            "previous_yaw_moment_action",
            "static_target",
            "compaction_enabled",
            "target_radius",
            "local_target_hold_estimate",
            "local_surface_gate_estimate",
            "episode_remaining",
        ]
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (15,) or not np.all(np.isfinite(result)):
            raise AssertionError("compact actor task context must be a finite 15-vector")
        return result, names


    def _critic_task_context(self) -> tuple[np.ndarray, list[str]]:
        target_positions, _, desired_velocities = self._assigned_guidance_targets()
        scale = max(float(np.max(self.world_size)), 1.0e-6)
        intervention = self.intervention_cfg

        def ratio(value: float, target: float) -> float:
            return float(np.clip(value / max(target, 1.0e-8), -1.0, 1.0))

        course_values = [
            float(self.task_config.environment.static_slots),
            float(self.task_config.environment.compaction_enabled),
            float(self.task_config.environment.submarine_dynamics),
            float(self.formation_radius / scale),
            float(np.clip(self.stage_hold_step_count / max(self.final_hold_target_steps, 1), 0.0, 1.0)),
            float(np.clip(self.formation45_hold_steps / max(self.formation45_hold_target_steps, 1), 0.0, 1.0)),
            float(np.clip(self.shrink_steps / max(self.minimum_shrink_steps, 1), 0.0, 1.0)),
            float(np.clip(self.surface_gate_progress, 0.0, 1.0)),
            float(self.surface_b1_latched),
            float(self.surface_b2_latched),
            float(self.surface_causal_latched),
            float(self.surface_island_avoidance_latched),
            float(self.surface_complete_latched),
            ratio(self.counterfactual_cross_track_gain, float(intervention["current_counterfactual_cross_track_min"])),
            ratio(self.clearance_gain_vs_shadow, float(intervention["clearance_gain_min"])),
            float(np.clip(self.triangle_margin / scale, -1.0, 1.0)),
            float(self.physical_core_gate),
            float(max(self.max_steps - self.step_count, 0) / max(self.max_steps, 1)),
        ]
        names = [
            *[
                f"agent_{agent_id}_target_{axis}"
                for agent_id in range(self.num_friendly)
                for axis in ("x", "y")
            ],
            *[
                f"agent_{agent_id}_desired_velocity_{axis}"
                for agent_id in range(self.num_friendly)
                for axis in ("x", "y")
            ],
            "static_target",
            "compaction_enabled",
            "submarine_dynamics_enabled",
            "target_radius",
            "stage_hold_progress",
            "formation45_hold_progress",
            "shrink_progress",
            "surface_gate_progress",
            "surface_b1_latched",
            "surface_b2_latched",
            "surface_causal_latched",
            "surface_island_avoidance_latched",
            "surface_complete_latched",
            "counterfactual_gain_progress",
            "clearance_gain_progress",
            "triangle_margin",
            "physical_core_gate",
            "episode_remaining",
        ]
        result = np.asarray(
            [
                *target_positions.reshape(-1).tolist(),
                *desired_velocities.reshape(-1).tolist(),
                *course_values,
            ],
            dtype=np.float32,
        )
        if result.shape != (38,) or not np.all(np.isfinite(result)):
            raise AssertionError("compact critic task context must be a finite 38-vector")
        return result, names


    def _assigned_guidance_targets(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_positions = np.concatenate(
            (self.pressure_zones, self.submarine_target_slots), axis=0
        ).astype(np.float32)
        target_velocities = np.concatenate(
            (self.pressure_zone_velocities, self.submarine_target_velocities), axis=0
        ).astype(np.float32)
        states = self.friendly_states()
        desired_velocities = np.zeros((self.num_friendly, 2), dtype=np.float32)
        k_pos = float(self.reward_cfg["normalization"]["k_pos"])
        correction_ratio = float(
            self.reward_cfg["normalization"]["correction_speed_ratio"]
        )
        for index, state in enumerate(states):
            if index < self.num_surface_blue:
                reference = surface_reference(self, index)
                target_positions[index] = reference["target"]
                target_velocities[index] = reference["velocity"]
                desired_velocities[index] = reference["velocity"]
                continue
            max_speed = max(float(self.max_speed_by_type[state.unit_type]), 1.0e-6)
            correction = k_pos * (target_positions[index] - state.pos)
            correction_norm = norm(correction)
            correction_limit = correction_ratio * max_speed
            if correction_norm > correction_limit:
                correction *= correction_limit / max(correction_norm, 1.0e-8)
            desired = target_velocities[index] + correction
            desired_norm = norm(desired)
            if desired_norm > max_speed:
                desired *= max_speed / max(desired_norm, 1.0e-8)
            desired_velocities[index] = desired
        return target_positions, target_velocities, desired_velocities


    def _observations(self) -> np.ndarray:
        states = self.friendly_states()
        observations = []
        expected_names: list[str] | None = None
        for index in range(self.num_friendly):
            context, names = self._actor_task_context(index)
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise AssertionError("actor task context schema changed by role")
            perceived = list(states)
            comm_features: dict[int, tuple[float, float]] = {}
            for peer_index in range(self.num_friendly):
                if peer_index == index:
                    continue
                perceived[peer_index] = self.communication.peer_state(
                    index, peer_index, states[peer_index]
                )
                comm_features[peer_index] = self.communication.link_features(
                    index, peer_index, self.step_count
                )
            observations.append(
                build_actor_observation(
                    friendly_index=index,
                    friendly_states=states,
                    red_state=self.red_state,
                    world_size=self.world_size,
                    island_center=(
                        self.island_center
                        if self.island_active
                        else np.zeros(2, dtype=np.float32)
                    ),
                    island_radius=self.island_radius if self.island_active else 0.0,
                    max_speed_by_type=self.max_speed_by_type,
                    max_speed_red=self.max_speed_red,
                    max_sway_speed=self.max_sway_speed,
                    max_yaw_rate=self.max_yaw_rate,
                    task_context=context,
                    active_mask=self._agent_active_mask,
                    peer_states=perceived,
                    communication_features=comm_features,
                )
            )
        result = np.stack(observations).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise AssertionError("actor observations must be finite")
        if result.shape != (self.num_friendly, ACTOR_OBSERVATION_DIM):
            raise AssertionError("actor observation schema dimension changed")
        return result


    def _critic_state(self) -> np.ndarray:
        context, _ = self._critic_task_context()
        result = build_critic_state(
            friendly_states=self.friendly_states(),
            red_state=self.red_state,
            world_size=self.world_size,
            island_center=(
                self.island_center
                if self.island_active
                else np.zeros(2, dtype=np.float32)
            ),
            island_radius=self.island_radius if self.island_active else 0.0,
            landing_point=(
                self._landing_point()
                if self.island_active
                else np.zeros(2, dtype=np.float32)
            ),
            max_speed_by_type=self.max_speed_by_type,
            max_speed_red=self.max_speed_red,
            max_sway_speed=self.max_sway_speed,
            max_yaw_rate=self.max_yaw_rate,
            task_context=context,
            active_mask=self._agent_active_mask,
        )
        if result.shape != (CRITIC_STATE_DIM,):
            raise AssertionError("critic state schema dimension changed")
        return result.astype(np.float32)


    def friendly_states(self) -> list[ShipState]:
        return [*self.blue_states, *self.submarine_states]


    def disturbance_statistics(self) -> dict[str, float]:
        wave = self.wave.episode
        values: dict[str, float] = {
            "master_seed": float(self.master_seed),
            "scenario_seed": float(self.scenario_seed),
            "wave_seed": float(self.wave_seed),
            "communication_seed": float(self.communication_seed),
            "wave_height": wave.height_m,
            "wavelength": wave.wavelength_m,
            "wave_direction_rad": wave.direction_rad,
            "wave_phase_seed": float(wave.seed),
            "submarine_direct_wave_coupling": float(self.wave_cfg.get("submarine_direct_wave_coupling", 0.0)),
        }
        for role in ("surface", "red"):
            samples = self.wave_velocity_samples[role]
            values[f"{role}_wave_water_velocity_rms"] = float(np.sqrt(np.mean(np.square(samples)))) if samples else 0.0
            values[f"{role}_wave_water_velocity_max"] = float(max(samples, default=0.0))
        values.update(self.communication.statistics())
        for role, dynamics in (("surface", self.blue_dynamics), ("submarine", self.submarine_dynamics), ("red", self.red_dynamics)):
            for axis in ("surge", "sway", "yaw"):
                values[f"{role}_{axis}_velocity_clamp_count"] = float(
                    dynamics.clamp_counts[axis] - self._clamp_start[role][axis]
                )
        return values


    @property
    def collision_radius(self) -> float:
        return 0.5 * self.surface_collision_width


    def _capsule(self, state: ShipState) -> tuple[np.ndarray, np.ndarray, float]:
        if state.unit_type == "submarine":
            return capsule_segment(state.pos, state.heading, self.submarine_length, self.submarine_collision_width)
        return capsule_segment(state.pos, state.heading, self.surface_length, self.surface_collision_width)


    @staticmethod
    def _friendly_pair_can_collide(left: ShipState, right: ShipState) -> bool:
        pair_types = {left.unit_type, right.unit_type}
        if pair_types == {"surface", "submarine"}:
            return bool(left.surfaced and right.surfaced)
        return True


    def _collision_stage_label(self) -> str:
        if bool(self.compaction_complete):
            return "post_shrink"
        if bool(self.compaction_started):
            return "shrinking"
        return "pre_shrink"


    def _record_collision(self, pair: str) -> None:
        if self.collision_step is not None:
            return
        self.collision_pair = str(pair)
        self.collision_step = int(self.step_count)
        self.collision_stage = self._collision_stage_label()


    @staticmethod
    def _friendly_label(index: int) -> str:
        return ("B1", "B2", "Y1", "Y2", "Y3")[int(index)]


    def _endpoint_collision(self) -> bool:
        states = [
            (index, state)
            for index, (state, active) in enumerate(
                zip(self.friendly_states(), self._agent_active_mask)
            )
            if active > 0.5
        ]
        for offset, (left_index, left) in enumerate(states):
            if self.island_active and capsule_circle_clearance(
                self._capsule(left), self.island_center, self.island_radius
            ) <= 0.0:
                self._record_collision("island")
                return True
            for right_index, right in states[offset + 1 :]:
                if self._friendly_pair_can_collide(left, right) and capsules_overlap(
                    self._capsule(left), self._capsule(right)
                ):
                    self._record_collision(
                        f"{self._friendly_label(left_index)}-"
                        f"{self._friendly_label(right_index)}"
                    )
                    return True
            if (
                (left.unit_type == "surface" or left.surfaced)
                and capsules_overlap(self._capsule(left), self._capsule(self.red_state))
            ):
                self._record_collision(
                    "submarine-red"
                    if left.unit_type == "submarine"
                    else "surface-red"
                )
                return True
        return False


    def _endpoint_boundary_violation(self) -> bool:
        return bool(
            any(
                not capsule_inside_rectangle(self._capsule(state), self.world_size)
                for state, active in zip(self.friendly_states(), self._agent_active_mask)
                if active > 0.5
            )
            or not capsule_inside_rectangle(self._capsule(self.red_state), self.world_size)
        )


    def _inside_bounds(self, position: np.ndarray) -> bool:
        return self._inside_rect(position)


    def _inside_rect(self, position: np.ndarray) -> bool:
        return bool(
            0.0 <= position[0] <= self.world_size[0]
            and 0.0 <= position[1] <= self.world_size[1]
        )


    def _inside_island(self, position: np.ndarray) -> bool:
        return bool(
            self.island_radius > 0.0
            and norm(np.asarray(position, dtype=np.float32) - self.island_center)
            <= self.island_radius + 0.5 * self.surface_length
        )


    def _endpoint_red_landed(self) -> bool:
        return bool(
            self.island_active
            and
            norm(self.red_state.pos - self.island_center)
            <= self.island_radius + self.landing_buffer
        )


    def _landing_point(self) -> np.ndarray:
        return self.route_endpoint.copy()


    def _clip_inside_world(self, position: np.ndarray) -> np.ndarray:
        margin = self.semi_major + self.geometry_scale
        clipped = np.clip(
            np.asarray(position, dtype=np.float32),
            np.asarray([margin, margin], dtype=np.float32),
            self.world_size - margin,
        )
        if self._inside_island(clipped):
            away = clipped - self.island_center
            if norm(away) <= 1e-6:
                away = -self.natural_route_tangent
            clipped = (
                self.island_center
                + unit(angle_of(away))
                * (self.island_radius + self.semi_major + 2.0 * self.geometry_scale)
            )
        return np.asarray(clipped, dtype=np.float32)

    def _confine_state_to_world(self, state: ShipState) -> bool:
        """Apply a frictionless hard-wall response to one capsule.

        The capsule is projected back into the world rectangle and only the
        outward component of its world velocity is removed.  The contact is
        reported separately; it is deliberately not a terminal event and does
        not create a boundary reward penalty.
        """
        if state.unit_type == "submarine":
            length, width = self.submarine_length, self.submarine_collision_width
        else:
            length, width = self.surface_length, self.surface_collision_width
        start, end, radius = capsule_segment(state.pos, state.heading, length, width)
        points = np.stack((start, end), axis=0)
        shift = np.zeros(2, dtype=np.float64)
        lower = np.asarray([radius, radius], dtype=np.float64) - points.min(axis=0)
        upper = points.max(axis=0) - (self.world_size.astype(np.float64) - radius)
        shift += np.maximum(lower, 0.0)
        shift -= np.maximum(upper, 0.0)
        contacted = bool(np.any(np.abs(shift) > 1.0e-10))
        if contacted:
            state.pos = (np.asarray(state.pos, dtype=np.float64) + shift).astype(np.float32)
            velocity = np.asarray(state.velocity, dtype=np.float64)
            for axis in range(2):
                if shift[axis] > 0.0 and velocity[axis] < 0.0:
                    velocity[axis] = 0.0
                elif shift[axis] < 0.0 and velocity[axis] > 0.0:
                    velocity[axis] = 0.0
            c, s = math.cos(state.heading), math.sin(state.heading)
            state.speed = float(c * velocity[0] + s * velocity[1])
            state.sway_speed = float(-s * velocity[0] + c * velocity[1])
        return contacted


    def render_frame(self, *, show_shadow_red: bool | None = None):
        from .rendering import RenderFrame

        resolved_show_shadow = bool(
            getattr(self, "show_shadow_red", False)
            if show_shadow_red is None
            else show_shadow_red
        )

        event_markers = {
            "B1 intervention latched": self.surface_b1_latched,
            "B2 relay latched": self.surface_b2_latched,
            "causal deviation latched": self.surface_causal_latched,
            "island avoidance latched": self.surface_island_avoidance_latched,
            "surface complete": self.surface_complete_latched,
            "initial-radius hold started": self.formation45_hold_steps > 0,
            "initial-radius hold completed": bool(
                self.formation45_hold_target_steps > 0
                and self.formation45_hold_steps
                >= self.formation45_hold_target_steps
            ),
            "shrink started": self.compaction_started,
            "compact radius reached": bool(
                self.formation_radius <= self.compact_radius + 1.0e-6
            ),
            "final hold started": self.stage_hold_step_count > 0,
            "success": self.stage_success,
            "failure": bool(
                getattr(self, "terminal_reason", "")
                in {"landing", "collision", "out_of_bounds"}
            ),
        }
        diagnostics = {
            "e_actual": self.red_actual_cross_track,
            "e_shadow": self.red_shadow_cross_track,
            "e_cf": self.counterfactual_cross_track_gain,
            "actual_clearance": self.predicted_island_clearance,
            "shadow_clearance": self.shadow_predicted_island_clearance,
            "clearance_gain": self.clearance_gain_vs_shadow,
            "b1_apf_projection": self.current_b1_projection,
            "b2_apf_projection": self.current_b2_projection,
            "b1_active_pressure_integral": float(
                self.surface_active_pressure_integrals[0]
            ),
            "b2_active_pressure_integral": float(
                self.surface_active_pressure_integrals[1]
            ),
            "b1_net_displacement": float(self.surface_net_displacements[0]),
            "b2_net_displacement": float(self.surface_net_displacements[1]),
            "b1_path_efficiency": float(self.surface_path_efficiencies[0]),
            "b2_path_efficiency": float(self.surface_path_efficiencies[1]),
            "mean_slot_error": self.mean_slot_error,
            "max_slot_error": self.max_slot_error,
            "max_angular_gap_degrees": math.degrees(self.max_angular_gap),
            "triangle_margin": self.triangle_margin,
            "radius_ref": self.radius_ref,
            "hold_progress": self.stage_hold_step_count
            / max(self.final_hold_target_steps, 1),
            "formation45_hold_steps": float(self.formation45_hold_steps),
            "shrink_steps": float(self.shrink_steps),
        }

        return RenderFrame(
            world_size=self.world_size.copy(),
            island_center=self.island_center.copy(),
            island_radius=self.island_radius,
            red_state=self.red_state,
            blue_states=list(self.blue_states),
            submarine_states=list(self.submarine_states),
            submarine_target_slots=self.submarine_target_slots.copy(),
            prediction_center=self.prediction_center.copy(),
            slot_center=self.slot_center.copy(),
            pressure_zones=self.pressure_zones.copy(),
            route_start=self.route_origin.copy(),
            route_end=self.route_endpoint.copy(),
            phase=self.phase,
            encircle_beta=self.encircle_beta,
            physical_capture_success=self.physical_capture_success,
            cooperative_task_success=self.cooperative_task_success,
            semi_major=self.semi_major,
            semi_minor=self.semi_minor,
            radius_ref=self.radius_ref,
            surface_active_mask=self._agent_active_mask[self.surface_slice].copy(),
            submarine_active_mask=self._agent_active_mask[self.submarine_slice].copy(),
            island_active=self.island_active,
            show_pressure_zones=False,
            surface_role_regions=[dict(surface_reference(self, i),
                inner=self.reward_cfg["surface"]["inner"], outer=self.reward_cfg["surface"]["outer"],
                half_angle=math.radians(self.reward_cfg["surface"]["half_angle_degrees"]))
                for i in range(2) if self.active_surface_mask[i]],
            show_slots="submarine" in self.task_config.active_roles,
            task_name="final_task",
            difficulty_band=None,
            shadow_red_state=self._copy_ship_state(self.shadow_red_state),
            show_shadow_red=resolved_show_shadow,
            event_markers=event_markers,
            visualization_metadata={**(getattr(self, "visualization_metadata", {}) or {}),
                "experiment": self.config["revision"]["experiment"], "simulation_step": self.step_count},
            diagnostics=diagnostics,
            simulation_time=self.step_count * self.dt,
            termination_reason=str(getattr(self, "terminal_reason", "")),
        )


    v43 = True


    def __init__(self, config, seed=None):
        validate(config)
        self._initialize_physics(config, seed)
        for dynamics in (self.blue_dynamics,self.submarine_dynamics,self.red_dynamics):
            dynamics.record_trajectory = True
        self._step_start_states = None


    def step(self,actions):
        from copy import deepcopy
        self._step_start_states = deepcopy(self.friendly_states()+[self.red_state])
        try:
            return self._step_physics(actions)
        finally:
            self._step_start_states = None


    def _collision(self):
        if self._endpoint_collision(): return True
        old=getattr(self,"_step_start_states",None)
        if old is None: return False
        from .safety import path_contact
        states=self.friendly_states()
        for i,left in enumerate(states):
            if not self._agent_active_mask[i]: continue
            if self.island_active and path_contact(self,old[i],left,obstacle="island"):
                self._record_collision("island"); return True
            if (left.unit_type=="surface" or left.surfaced) and path_contact(self,old[i],left,old[5],self.red_state):
                self._record_collision("surface-red" if i<2 else "submarine-red"); return True
            for j in range(i+1,5):
                if self._agent_active_mask[j] and self._friendly_pair_can_collide(left,states[j]) and path_contact(self,old[i],left,old[j],states[j]):
                    self._record_collision(f"{self._friendly_label(i)}-{self._friendly_label(j)}"); return True
        return False


    def _boundary_violation(self):
        if self._endpoint_boundary_violation(): return True
        old=getattr(self,"_step_start_states",None)
        if old is None: return False
        from .safety import path_contact
        return any(path_contact(self,a,b) for a,b,active in zip(old,self.friendly_states()+[self.red_state],list(self._agent_active_mask)+[1]) if active)


    def _red_landed(self):
        if self._endpoint_red_landed(): return True
        old=getattr(self,"_step_start_states",None)
        if old is None: return False
        from .safety import landing_crossed
        return landing_crossed(self,old[5],self.red_state)


    def _update_dynamic_references(self, *, initialize):
        self._update_physical_references(initialize=initialize)
        if not self.task_config.environment.static_slots:
            self.prediction_heading = self.initial_red_heading
            self.submarine_target_slots = self._slots_for(self.formation_radius)


    def _slots_for(self, radius):
        return triangular_slots(self.slot_center, self.initial_red_heading, radius)[[2, 0, 1]][self.submarine_slot_assignment]


    def _loose_gate(self, slots):
        positions = np.asarray([s.pos for s in self.submarine_states])
        errors = np.linalg.norm(positions - slots, axis=1)
        gate = self.encirclement_cfg["shrink_tracking"]
        return bool(errors.mean() <= gate["mean_slot_error_max"] and errors.max() <= gate["max_slot_error_max"]
                    and self.max_angular_gap <= math.radians(gate["max_angle_gap_degrees"])
                    and self.triangle_contains_red)


    def _advance_formation(self):
        if not self.task_config.environment.compaction_enabled: return False
        if not self.compaction_started:
            if self._loose_gate(self.submarine_target_slots):
                if self.formation45_hold_steps == 0: self.formation45_hold_start_step = self.step_count
                self.formation45_hold_steps += 1
                self.phase = "HOLD"
                if self.first_hold_step is None: self.first_hold_step = self.step_count
            else:
                self.formation45_hold_steps = 0
                self.formation45_hold_start_step = None
                self.phase = "CHASE_AND_PRESSURE"
            self.stable_seconds = self.formation45_hold_steps * self.dt
            if self.formation45_hold_steps >= self.formation45_hold_target_steps:
                self.compaction_started = True
                self.formation45_hold_complete_step = self.step_count
            return False  # 14th step only enters compacting.
        if self.compaction_complete: return False
        self.phase = "ENCIRCLE"
        if self.first_encircle_step is None: self.first_encircle_step = self.step_count
        candidate = max(self.compact_radius, self.formation_radius - self.shrink_rate_mps * self.dt)
        if candidate >= self.formation_radius-1e-12: return False
        if not self._loose_gate(self._slots_for(candidate)): return False
        self.formation_radius = candidate
        self.shrink_steps += 1
        if self.shrink_started_step is None: self.shrink_started_step = self.step_count
        if candidate <= self.compact_radius + 1e-9 and self.shrink_steps >= self.minimum_shrink_steps:
            self.compaction_complete = True
            self.radius35_step = self.step_count
        return True


    def _submarine_pre_shrink_gate(self): return self._loose_gate(self.submarine_target_slots)


    def _submarine_tracking_gate(self): return self._loose_gate(self.submarine_target_slots)


    def _submarine_shrink_tracking_gate(self): return self._loose_gate(self.submarine_target_slots)


    def _physical_core_gate(self, collision, boundary, landing):
        return bool(self.compaction_complete and not (collision or boundary or landing))


    def _stage_final_capture_core_gate(self, collision, boundary, landing):
        return self._physical_core_gate(collision, boundary, landing)


    def _advance_green_hold(self, core_gate): self.green_seconds = 0.


    def _record_final_gate_failures(self, collision): pass


    def strict_success_predicate(self): return bool(self.physical_core_gate)


    def _canonical_task_success(self): return bool(self.physical_core_gate)


    def stage_success_gates(self, *, collision=False, boundary=False, landing=False):
        return {"compaction_complete": bool(self.compaction_complete), "safety": not (collision or boundary or landing)}


    def _reward_snapshot(self):
        return replace(self._physical_reward_snapshot(), v43_quality=qualities(self))


    def _rewards(self, *, previous, previous_actions, current_actions, terminal_reason, **kwargs):
        result = evaluate(previous, self._reward_snapshot(), self.reward_cfg, terminal_reason)
        return result.rewards, result.terms


    def _surface_intervention_qualities(self):
        # Channel 70 remains reserved; engagement is a local distance observation.
        dist = np.asarray([np.linalg.norm(s.pos - self.red_state.pos) for s in self.blue_states])
        s = self.reward_cfg["surface"]
        radial = np.where(dist < s["inner"], dist / s["inner"],
                          1 / (1 + (np.maximum(dist-s["outer"], 0) / s["tail"])**2))
        return 0., radial.astype(np.float32)


    def _actor_task_context(self,friendly_index):
        values,names=self._physical_actor_task_context(friendly_index)
        if friendly_index>=2:
            values[4:7]=0.
            values[4+int(self.submarine_slot_assignment[friendly_index-2])]=1.
        return values,names

