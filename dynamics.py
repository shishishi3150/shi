from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .geometry import wrap_angle
from .types import ShipState


class USVDynamics:
    """CyberShip II 3-DOF underactuated maneuvering model.

    The state is ``eta=[x, y, psi]`` and ``nu=[u, v, r]``.  The learned
    action is the physically meaningful generalized input
    ``[surge_force, yaw_moment]``; sway force is identically zero.
    """

    def __init__(
        self,
        dt: float,
        max_speed: float,
        allow_reverse: bool,
        max_yaw_rate: float,
        max_surge_force: float,
        max_yaw_moment: float,
        parameters: Mapping[str, float],
        *,
        reverse_speed_ratio: float = 1.0,
        internal_substeps: int = 5,
        max_sway_speed: float | None = None,
        reference_max_acceleration: float = 0.08,
        reference_max_yaw_acceleration: float = 0.50,
        reference_sway_damping: float = 1.0,
    ) -> None:
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.allow_reverse = bool(allow_reverse)
        self.reverse_speed_ratio = float(reverse_speed_ratio)
        self.min_speed = (
            -self.max_speed * self.reverse_speed_ratio if self.allow_reverse else 0.0
        )
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_surge_force = float(max_surge_force)
        self.max_yaw_moment = float(max_yaw_moment)
        self.internal_substeps = int(internal_substeps)
        self.max_sway_speed = float(
            self.max_speed if max_sway_speed is None else max_sway_speed
        )
        self.reference_max_acceleration = float(reference_max_acceleration)
        self.reference_max_yaw_acceleration = float(
            reference_max_yaw_acceleration
        )
        self.reference_sway_damping = float(reference_sway_damping)

        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(self.max_speed) or self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive and finite")
        if not math.isfinite(self.reverse_speed_ratio) or not (
            0.0 < self.reverse_speed_ratio <= 1.0
        ):
            raise ValueError("reverse_speed_ratio must be finite and in (0, 1]")
        if min(
            self.max_yaw_rate,
            self.max_surge_force,
            self.max_yaw_moment,
            self.max_sway_speed,
            self.reference_max_acceleration,
            self.reference_max_yaw_acceleration,
            self.reference_sway_damping,
        ) <= 0.0:
            raise ValueError("dynamics limits and reference gains must be positive")
        if self.internal_substeps < 1:
            raise ValueError("internal_substeps must be at least one")

        required = {
            "mass",
            "yaw_inertia",
            "center_of_gravity_x",
            "x_u_dot",
            "y_v_dot",
            "y_r_dot",
            "n_v_dot",
            "n_r_dot",
            "x_u",
            "x_abs_u_u",
            "x_uuu",
            "y_v",
            "y_abs_v_v",
            "y_r",
            "y_abs_r_v",
            "y_abs_v_r",
            "y_abs_r_r",
            "n_v",
            "n_abs_v_v",
            "n_abs_r_v",
            "n_r",
            "n_abs_v_r",
            "n_abs_r_r",
        }
        if set(parameters) != required:
            raise ValueError(
                "CyberShip II parameter keys differ: "
                f"missing={sorted(required - set(parameters))}, "
                f"extra={sorted(set(parameters) - required)}"
            )
        self.parameters = {key: float(parameters[key]) for key in required}
        if not all(math.isfinite(value) for value in self.parameters.values()):
            raise ValueError("CyberShip II parameters must be finite")
        if self.parameters["mass"] <= 0.0 or self.parameters["yaw_inertia"] <= 0.0:
            raise ValueError("mass and yaw inertia must be positive")

        p = self.parameters
        mass = p["mass"]
        x_g = p["center_of_gravity_x"]
        self.mass_matrix = np.asarray(
            [
                [mass - p["x_u_dot"], 0.0, 0.0],
                [0.0, mass - p["y_v_dot"], mass * x_g - p["y_r_dot"]],
                [0.0, mass * x_g - p["n_v_dot"], p["yaw_inertia"] - p["n_r_dot"]],
            ],
            dtype=np.float64,
        )
        if not np.allclose(self.mass_matrix, self.mass_matrix.T, atol=1.0e-12):
            raise ValueError("CyberShip II mass matrix must be symmetric")
        if np.min(np.linalg.eigvalsh(self.mass_matrix)) <= 0.0:
            raise ValueError("CyberShip II mass matrix must be positive definite")
        self.inverse_mass_matrix = np.linalg.inv(self.mass_matrix)
        self.clamp_counts = {"surge": 0, "sway": 0, "yaw": 0}

    def coriolis_matrix(self, body_velocity: np.ndarray) -> np.ndarray:
        u, v, r = np.asarray(body_velocity, dtype=np.float64)
        p = self.parameters
        mass = p["mass"]
        x_g = p["center_of_gravity_x"]
        c13 = -(
            (mass - p["y_v_dot"]) * v
            + (mass * x_g - p["y_r_dot"]) * r
        )
        c23 = (mass - p["x_u_dot"]) * u
        return np.asarray(
            [[0.0, 0.0, c13], [0.0, 0.0, c23], [-c13, -c23, 0.0]],
            dtype=np.float64,
        )

    def damping_matrix(self, body_velocity: np.ndarray) -> np.ndarray:
        u, v, r = np.asarray(body_velocity, dtype=np.float64)
        p = self.parameters
        return np.asarray(
            [
                [
                    -p["x_u"] - p["x_abs_u_u"] * abs(u) - p["x_uuu"] * u * u,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    -p["y_v"]
                    - p["y_abs_v_v"] * abs(v)
                    - p["y_abs_r_v"] * abs(r),
                    -p["y_r"]
                    - p["y_abs_v_r"] * abs(v)
                    - p["y_abs_r_r"] * abs(r),
                ],
                [
                    0.0,
                    -p["n_v"]
                    - p["n_abs_v_v"] * abs(v)
                    - p["n_abs_r_v"] * abs(r),
                    -p["n_r"]
                    - p["n_abs_v_r"] * abs(v)
                    - p["n_abs_r_r"] * abs(r),
                ],
            ],
            dtype=np.float64,
        )

    def body_acceleration(
        self, body_velocity: np.ndarray, action: np.ndarray,
        hydrodynamic_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        velocity = np.asarray(body_velocity, dtype=np.float64)
        command = np.asarray(action, dtype=np.float64)
        if velocity.shape != (3,) or command.shape != (2,):
            raise ValueError("body_velocity and action must have shapes (3,) and (2,)")
        if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(command)):
            raise ValueError("dynamics inputs must be finite")
        surge_force = float(
            np.clip(command[0], -self.max_surge_force, self.max_surge_force)
        )
        yaw_moment = float(
            np.clip(command[1], -self.max_yaw_moment, self.max_yaw_moment)
        )
        generalized_force = np.asarray(
            [surge_force, 0.0, yaw_moment], dtype=np.float64
        )
        water_relative = velocity if hydrodynamic_velocity is None else np.asarray(hydrodynamic_velocity, dtype=np.float64)
        hydrodynamic_load = (
            self.coriolis_matrix(water_relative) @ water_relative
            + self.damping_matrix(water_relative) @ water_relative
        )
        return self.inverse_mass_matrix @ (generalized_force - hydrodynamic_load)

    def reference_action(
        self,
        state: ShipState,
        *,
        desired_yaw_rate: float,
        desired_surge_acceleration: float,
    ) -> np.ndarray:
        """Convert a scripted velocity reference into generalized forces.

        This adapter is only for the APF red policy and deterministic rule
        controller.  MATD3 still outputs force and moment directly.
        """

        velocity = state.body_velocity
        desired_acceleration = np.asarray(
            [
                np.clip(
                    desired_surge_acceleration,
                    -self.reference_max_acceleration,
                    self.reference_max_acceleration,
                ),
                -self.reference_sway_damping * float(state.sway_speed),
                np.clip(
                    (float(desired_yaw_rate) - float(state.yaw_rate)) / self.dt,
                    -self.reference_max_yaw_acceleration,
                    self.reference_max_yaw_acceleration,
                ),
            ],
            dtype=np.float64,
        )
        required_load = (
            self.mass_matrix @ desired_acceleration
            + self.coriolis_matrix(velocity) @ velocity
            + self.damping_matrix(velocity) @ velocity
        )
        return np.asarray(
            [
                np.clip(required_load[0], -self.max_surge_force, self.max_surge_force),
                np.clip(required_load[2], -self.max_yaw_moment, self.max_yaw_moment),
            ],
            dtype=np.float32,
        )

    def step(
        self, state: ShipState, action: np.ndarray,
        water_velocity_world: np.ndarray | None = None,
    ) -> ShipState:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError("action must be a finite [surge_force, yaw_moment] vector")

        position = np.asarray(state.pos, dtype=np.float64).copy()
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("state position must be a finite two-vector")
        heading = float(state.heading)
        velocity = state.body_velocity.copy()
        substep_dt = self.dt / float(self.internal_substeps)
        trace = [(position.copy(), heading)] if getattr(self, "record_trajectory", False) else None

        for _ in range(self.internal_substeps):
            relative = velocity
            if water_velocity_world is not None:
                wave = np.asarray(water_velocity_world, dtype=np.float64)
                c, s = math.cos(heading), math.sin(heading)
                wave_body = np.asarray((c * wave[0] + s * wave[1], -s * wave[0] + c * wave[1]), dtype=np.float64)
                relative = velocity.copy(); relative[:2] -= wave_body
            velocity += self.body_acceleration(velocity, command, relative) * substep_dt
            # Surge is symmetric when reverse is enabled.  The existing
            # role-specific forward caps remain the magnitude limits, so the
            # surface/red/submarine contracts stay at 0.4/0.4/0.5 m/s.
            if not self.min_speed <= velocity[0] <= self.max_speed: self.clamp_counts["surge"] += 1
            velocity[0] = np.clip(velocity[0], self.min_speed, self.max_speed)
            if abs(velocity[1]) > self.max_sway_speed: self.clamp_counts["sway"] += 1
            velocity[1] = np.clip(
                velocity[1], -self.max_sway_speed, self.max_sway_speed
            )
            if abs(velocity[2]) > self.max_yaw_rate: self.clamp_counts["yaw"] += 1
            velocity[2] = np.clip(
                velocity[2], -self.max_yaw_rate, self.max_yaw_rate
            )
            heading = float(wrap_angle(heading + velocity[2] * substep_dt))
            cosine = math.cos(heading)
            sine = math.sin(heading)
            position += np.asarray(
                [
                    cosine * velocity[0] - sine * velocity[1],
                    sine * velocity[0] + cosine * velocity[1],
                ],
                dtype=np.float64,
            ) * substep_dt
            if trace is not None:
                trace.append((position.copy(), heading))

        result = ShipState(
            pos=position.astype(np.float32),
            speed=float(velocity[0]),
            heading=heading,
            yaw_rate=float(velocity[2]),
            unit_type=state.unit_type,
            surfaced=state.surfaced,
            sway_speed=float(velocity[1]),
        )
        if trace is not None:
            result.integration_trace = trace
        return result


class AUVDynamics3DOF(USVDynamics):
    """Fixed horizontal-plane AUV model reduced from the selected submarine.

    ``M nu_dot + C(nu) nu + D nu = tau`` with diagonal effective inertia and
    linear plus quadratic damping.  The learned action remains ``[surge_force, yaw_moment]``;
    sway is underactuated.  This class is intentionally separate from the
    CyberShip II surface-vessel model so the two vehicle roles no longer share
    hydrodynamic parameters merely because their action schemas match.
    """

    def __init__(
        self,
        dt: float,
        max_speed: float,
        allow_reverse: bool,
        max_yaw_rate: float,
        max_surge_force: float,
        max_yaw_moment: float,
        parameters: Mapping[str, float],
        *,
        reverse_speed_ratio: float = 1.0,
        internal_substeps: int = 5,
        max_sway_speed: float | None = None,
        reference_max_acceleration: float = 0.08,
        reference_max_yaw_acceleration: float = 0.50,
        reference_sway_damping: float = 1.0,
    ) -> None:
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.allow_reverse = bool(allow_reverse)
        self.reverse_speed_ratio = float(reverse_speed_ratio)
        self.min_speed = (
            -self.max_speed * self.reverse_speed_ratio if self.allow_reverse else 0.0
        )
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_surge_force = float(max_surge_force)
        self.max_yaw_moment = float(max_yaw_moment)
        self.internal_substeps = int(internal_substeps)
        self.max_sway_speed = float(
            self.max_speed if max_sway_speed is None else max_sway_speed
        )
        self.reference_max_acceleration = float(reference_max_acceleration)
        self.reference_max_yaw_acceleration = float(
            reference_max_yaw_acceleration
        )
        self.reference_sway_damping = float(reference_sway_damping)

        positive_limits = (
            self.dt,
            self.max_speed,
            self.max_yaw_rate,
            self.max_surge_force,
            self.max_yaw_moment,
            self.max_sway_speed,
            self.reference_max_acceleration,
            self.reference_max_yaw_acceleration,
            self.reference_sway_damping,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_limits):
            raise ValueError("AUV dynamics limits and reference gains must be positive and finite")
        if not math.isfinite(self.reverse_speed_ratio) or not (
            0.0 < self.reverse_speed_ratio <= 1.0
        ):
            raise ValueError("reverse_speed_ratio must be finite and in (0, 1]")
        if self.internal_substeps < 1:
            raise ValueError("internal_substeps must be at least one")

        required = {"m11", "m22", "m33", "d11", "d22", "d33", "d11_quad", "d22_quad", "d33_quad"}
        if set(parameters) != required:
            raise ValueError(
                "AUV parameter keys differ: "
                f"missing={sorted(required - set(parameters))}, "
                f"extra={sorted(set(parameters) - required)}"
            )
        self.parameters = {key: float(parameters[key]) for key in required}
        if not all(
            math.isfinite(value) and value > 0.0
            for value in self.parameters.values()
        ):
            raise ValueError("AUV inertia and damping parameters must be positive and finite")

        p = self.parameters
        self.mass_matrix = np.diag([p["m11"], p["m22"], p["m33"]]).astype(
            np.float64
        )
        self.inverse_mass_matrix = np.diag(
            [1.0 / p["m11"], 1.0 / p["m22"], 1.0 / p["m33"]]
        ).astype(np.float64)
        self.clamp_counts = {"surge": 0, "sway": 0, "yaw": 0}

    def coriolis_matrix(self, body_velocity: np.ndarray) -> np.ndarray:
        u, v, _ = np.asarray(body_velocity, dtype=np.float64)
        m11 = self.parameters["m11"]
        m22 = self.parameters["m22"]
        # Skew symmetry makes nu.T @ C(nu) @ nu identically zero.
        return np.asarray(
            [
                [0.0, 0.0, -m22 * v],
                [0.0, 0.0, m11 * u],
                [m22 * v, -m11 * u, 0.0],
            ],
            dtype=np.float64,
        )

    def damping_matrix(self, body_velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(body_velocity, dtype=np.float64)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("body_velocity must be a finite three-vector")
        p = self.parameters
        return np.diag([
            p["d11"] + p["d11_quad"] * abs(velocity[0]),
            p["d22"] + p["d22_quad"] * abs(velocity[1]),
            p["d33"] + p["d33_quad"] * abs(velocity[2]),
        ]).astype(np.float64)
