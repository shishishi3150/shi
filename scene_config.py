"""One geometry conversion for training, ablations and checkpoint evaluation.

The historical 55 m template is immutable. This is a geometry-only benchmark
conversion, not a hydrodynamically similar scale model of the physical vessels.
"""
from copy import deepcopy
import math
import numpy as np


def scale_scene(config, side_m=55.0):
    if "scene_geometry" in config:
        raise ValueError("scene geometry is already resolved; do not scale it twice")
    side = float(side_m)
    if not math.isfinite(side) or side <= 0:
        raise ValueError("scene side must be finite and positive")
    if list(config["env"]["world_size"]) != [55.0, 55.0]:
        raise ValueError("geometry conversion requires the original 55 x 55 m template")
    cfg = deepcopy(config)
    ratio = side / 55.0
    changes = {}

    def multiply(path, factor=ratio):
        keys = path.split(".")
        owner = cfg
        for key in keys[:-1]:
            owner = owner[int(key)] if isinstance(owner, list) else owner[key]
        key = keys[-1]
        old = deepcopy(owner[key])
        def times(value):
            return [times(x) for x in value] if isinstance(value, list) else float(value) * factor
        owner[key] = times(old)
        changes[path] = {"before": old, "after": deepcopy(owner[key])}

    paths = [
        "env.world_size", "scenario.red_start", "scenario.blue_starts",
        "scenario.submarine_starts", "scenario.position_jitter", "scenario.boundary_clearance",
        "island.landing_buffer", "red_policy.repulsion_range",
        "final_task.environment.position_jitter",
        "final_task.success.max_slot_error", "final_task.success.mean_slot_error",
    ]
    paths += ["ship." + key for key in (
        "semi_major", "semi_minor", "surface_length_m", "surface_collision_width_m",
        "submarine_collision_length_m", "submarine_collision_width_m")]
    paths += ["task.encirclement." + key for key in (
        "initial_radius", "compact_radius", "slot_arrival_tolerance",
        # Fixed speed: scale the travel distance represented by the lookahead.
        "prediction_horizon_seconds", "shrink_tracking.mean_slot_error_max",
        "shrink_tracking.max_slot_error_max")]
    paths += ["task.physical_capture." + key for key in (
        "submarine_distance_range", "core_distance_range", "triangle_margin_min",
        "max_slot_error", "training_mean_slot_error_max", "training_max_slot_error_max")]
    paths += ["task.surface_pressure.tracking_success.predicted_clearance_min"]
    paths += ["task.surface_pressure.intervention_success." + key for key in (
        "net_displacement_min", "b2_incremental_counterfactual_gain_min",
        "current_counterfactual_cross_track_min", "recent_counterfactual_mean_min",
        "maximum_counterfactual_cross_track_min", "clearance_gain_min",
        "recent_actual_clearance_min", "route_corridor_half_width",
        "projected_actual_clearance_min", "projected_shadow_clearance_max",
        "final_persistence_counterfactual_min", "final_persistence_counterfactual_mean_min",
        "final_persistence_actual_clearance_min")]
    for index in range(2):
        paths += [f"task.surface_pressure.zones.{index}." + key for key in (
            "along_distance", "lateral_offset", "along_tolerance", "lateral_tolerance")]
    for profile in cfg["task"]["surface_pressure"]["history_profiles"]:
        paths += [f"task.surface_pressure.history_profiles.{profile}.lateral_deviation_min"]
    paths += ["scenario.task_spawn.surface." + key for key in (
        "random_translation", "b1_s", "b1_l", "b2_s", "b2_l",
        "minimum_pair_distance", "minimum_red_distance", "boundary_clearance")]
    paths += ["scenario.task_spawn.submarine." + key for key in (
        "random_translation", "fixed_reference_center_s", "minimum_pair_distance",
        "minimum_red_distance", "boundary_clearance", "island_safety_buffer",
        "minimum_any_slot_error", "minimum_intercept_seconds", "safe_maneuver_probe_distance",
        "profiles.final.route_progress", "profiles.final.lateral_perturbation",
        "profiles.final.spawn_radius")]
    paths += ["reward.normalization." + key for key in (
        "position_softening_m", "spacing_safe_m", "spacing_scale_m",
        "safety_boundary_margin_m", "safety_island_margin_m", "safety_red_margin_m")]
    paths += ["reward.surface." + key for key in (
        "engagement_inner_m", "engagement_outer_m", "engagement_cutoff_m")]
    paths += ["revision.i3.margin_tolerance_m", "revision.i3.progress_tolerance_m",
              "revision.i3.acceptance_tolerance"]
    for path in paths:
        multiply(path)
    # k_pos converts displacement to velocity. Preserve the same speed request
    # for the same relative slot error, without changing any reward term weight.
    multiply("reward.normalization.k_pos", 1.0 / ratio)
    enc = cfg["task"]["encirclement"]
    steps = math.ceil((enc["initial_radius"] - enc["compact_radius"]) /
                      (enc["shrink_rate_mps"] * cfg["env"]["dt"]) - 1e-10)
    for owner, path in ((enc, "task.encirclement.minimum_shrink_steps"),
                        (cfg["final_task"]["timing"], "final_task.timing.minimum_shrink_steps")):
        changes[path] = {"before": owner["minimum_shrink_steps"], "after": steps}
        owner["minimum_shrink_steps"] = steps
    # QP operates in reference-distance units so shrinking metres does not
    # implicitly weaken its soft constraints by ratio**2.
    cfg["revision"]["i3"]["margin_unit_m"] = ratio
    cfg["scene_geometry"] = {
        "profile": "uniform_geometry_v1", "reference_world_size_m": [55.0, 55.0],
        "world_size_m": [side, side], "scale": ratio, "resolved": True,
        "dynamics_scaled": False, "collision_hulls_scaled": True,
        "changes": changes,
    }
    cfg["revision"]["scene_profile"] = "uniform_geometry_v1"
    cfg["revision"]["world_size_m"] = [side, side]
    return cfg


def validate_scene(config, expected=None):
    """Reject untagged checkpoints rather than guessing their scene geometry."""
    scene = config.get("scene_geometry", {})
    if scene.get("profile") != "uniform_geometry_v1" or scene.get("resolved") is not True:
        raise ValueError("checkpoint has no resolved scene identity; use a checkpoint with explicit scene geometry")
    world = config["env"]["world_size"]
    if (world != scene["world_size_m"] or world != config["revision"].get("world_size_m")
            or not np.allclose(np.asarray(world), 55.0 * scene["scale"], rtol=0, atol=1e-9)):
        raise ValueError("inconsistent scene geometry identity")
    if expected is not None and scene != expected.get("scene_geometry"):
        raise ValueError("checkpoint scene differs from the selected training scene; retrain")
