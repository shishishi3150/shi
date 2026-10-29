from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import wrap_angle
from .task_geometry import combined_surface_repulsion_force
from .types import ShipState


def red_policy_reference(
    red: ShipState,
    landing_point: np.ndarray,
    surface_states: list[ShipState],
    policy_cfg: dict[str, Any],
    ship_cfg: dict[str, Any],
) -> np.ndarray:
    """Return APF yaw-rate and surge-acceleration references.

    Submarine state is deliberately absent because submerged pursuers are not
    observable to the red policy.  The environment converts this scripted
    reference through the same CyberShip II dynamics used by learned agents.
    """

    attraction = np.asarray(landing_point, dtype=np.float32) - red.pos
    distance = float(np.linalg.norm(attraction))
    if distance > 1e-6:
        attraction = attraction / distance
    vector = float(policy_cfg["attraction_gain"]) * attraction

    vector += combined_surface_repulsion_force(
        red_pos=red.pos,
        surface_positions=np.asarray(
            [state.pos for state in surface_states], dtype=np.float32
        ),
        repulsion_range=float(policy_cfg["repulsion_range"]),
        repulsion_gain=float(policy_cfg["repulsion_gain"]),
    )

    if float(np.linalg.norm(vector)) <= 1e-6:
        vector = attraction
    desired_heading = float(np.arctan2(vector[1], vector[0]))
    heading_error = float(wrap_angle(desired_heading - red.heading))
    max_yaw = float(ship_cfg["max_yaw_rate"]) * float(policy_cfg["yaw_rate_factor"])
    yaw_rate = float(np.clip(heading_error * float(policy_cfg["heading_gain"]), -max_yaw, max_yaw))
    target_speed = min(float(policy_cfg["target_speed"]), float(ship_cfg["max_speed_red"]))
    acceleration_limit = float(
        ship_cfg["dynamics"]["reference_max_acceleration"]
    )
    acceleration = float(
        np.clip(
            (target_speed - red.speed) * float(policy_cfg["accel_gain"]),
            -acceleration_limit,
            acceleration_limit,
        )
    )
    return np.array([yaw_rate, acceleration], dtype=np.float32)
