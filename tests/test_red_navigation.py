import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from usv_env.dynamics import AUVDynamics3DOF, USVDynamics
from usv_env.geometry import capsule_segment, capsule_circle_clearance, capsules_overlap
from usv_env.red_policy import RedNavigator, RuleController
from usv_env.types import ShipState


REEFS = [([20., 37.], 3.5), ([20., 17.], 3.5), ([32., 28.], 3.5)]
WORLD = np.asarray([55., 55.])


def dynamics_pair():
    ship = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/scene_template.yaml").read_text(encoding="utf-8"))["ship"]
    surface = USVDynamics(
        dt=.15, max_speed=.4, allow_reverse=True, max_yaw_rate=ship["max_yaw_rate"],
        max_surge_force=ship["max_surge_force"], max_yaw_moment=ship["max_yaw_moment"],
        parameters=ship["dynamics"]["parameters"], internal_substeps=5,
        max_sway_speed=ship["max_sway_speed"],
        reference_max_acceleration=ship["dynamics"]["reference_max_acceleration"],
        reference_max_yaw_acceleration=ship["dynamics"]["reference_max_yaw_acceleration"],
        reference_sway_damping=ship["dynamics"]["reference_sway_damping"],
    )
    submarine_cfg = ship["submarine_dynamics"]
    submarine = AUVDynamics3DOF(
        dt=.15, max_speed=.5, allow_reverse=True, max_yaw_rate=ship["max_yaw_rate"],
        max_surge_force=submarine_cfg["max_surge_force"], max_yaw_moment=submarine_cfg["max_yaw_moment"],
        parameters=submarine_cfg["parameters"], internal_substeps=5,
        max_sway_speed=ship["max_sway_speed"],
        reference_max_acceleration=submarine_cfg["reference_max_acceleration"],
        reference_max_yaw_acceleration=submarine_cfg["reference_max_yaw_acceleration"],
        reference_sway_damping=submarine_cfg["reference_sway_damping"],
    )
    return ship, surface, submarine


def test_red_pure_apf_never_calls_route_or_candidate_helpers(monkeypatch):
    import usv_env.red_policy as policy
    ship, _, _ = dynamics_pair()
    def forbidden(*args, **kwargs):
        raise AssertionError("Red APF must not call a route planner or direction search")
    for name in ("_route", "_safe_route_target", "_segment_clear"):
        monkeypatch.setattr(policy, name, forbidden)
    red = ShipState(np.asarray([25., 25.]), .3, 0., 0., "surface")
    blue = ShipState(np.asarray([27., 25.]), .3, math.pi, 0., "surface")
    reference = RedNavigator().reference(red, np.asarray([49., 27.]), REEFS, [blue], WORLD, ship)
    assert np.isfinite(reference).all()


def test_red_apf_is_the_sum_of_goal_reef_usv_and_boundary_forces():
    ship, _, _ = dynamics_pair()
    red = ShipState(np.asarray([5., 10.]), .2, 0., 0., "surface")
    blue = ShipState(np.asarray([6.5, 10.]), 0., 0., 0., "surface")
    nav = RedNavigator()
    forces = nav.potential_forces(red, np.asarray([49., 10.]), [([7., 12.], .5)], [blue], WORLD, ship)
    assert set(forces) == {"goal", "reef", "usv", "boundary"}
    assert forces["goal"][0] > 0. and forces["usv"][0] < 0.
    assert forces["reef"][0] < 0. and forces["reef"][1] < 0.
    near_edge = ShipState(np.asarray([.9, 10.]), .2, 0., 0., "surface")
    boundary = nav.potential_forces(near_edge, [49., 10.], [], [], WORLD, ship)["boundary"]
    assert boundary[0] > 0.


def test_red_apf_points_to_goal_outside_obstacle_influence_and_is_deterministic():
    ship, _, _ = dynamics_pair()
    red = ShipState(np.asarray([11., 27.]), .4, 0., 0., "red")
    left, right = RedNavigator(), RedNavigator()
    left_ref = left.reference(red, np.asarray([49., 27.]), REEFS, [], WORLD, ship)
    right_ref = right.reference(red, np.asarray([49., 27.]), REEFS, [], WORLD, ship)
    np.testing.assert_array_equal(left_ref, right_ref)
    assert np.isfinite(left_ref).all()
    np.testing.assert_allclose(left_ref, [0., 0.], atol=1e-6)


def test_red_stagnation_telemetry_does_not_add_escape_steering():
    ship, _, _ = dynamics_pair()
    nav = RedNavigator({"stagnation_window_steps": 4, "stagnation_progress_m": .1})
    red = ShipState(np.asarray([11., 27.]), 0., 0., 0., "red")
    initial = nav.reference(red, np.asarray([49., 27.]), REEFS, [], WORLD, ship)
    for _ in range(4):
        np.testing.assert_array_equal(nav.reference(red, np.asarray([49., 27.]), REEFS, [], WORLD, ship), initial)
    assert nav.recovery_count == 1 and nav.recovery_active
    for _ in range(5):
        nav.reference(red, np.asarray([49., 27.]), REEFS, [], WORLD, ship)
    assert nav.recovery_count == 2


@pytest.mark.parametrize("blue_pos", [[25.5, 23.], [26., 23.]])
@pytest.mark.parametrize("reef_field_x_scale", [1., 1.8])
def test_red_apf_flank_encounter_keeps_reef_clearance(blue_pos, reef_field_x_scale):
    """Keep the old crash fixtures, without requiring APF to escape a minimum."""
    ship, dynamics, _ = dynamics_pair()
    red = ShipState(np.asarray([24., 24.5]), .4, -.22, 0., "surface")
    blue = ShipState(np.asarray(blue_pos), 0., 0., 0., "surface")
    navigator = RedNavigator({"reef_field_x_scale": reef_field_x_scale})
    for _ in range(180):
        reference = navigator.reference(red, np.asarray([49., 27.]), REEFS, [blue], WORLD, ship)
        red = dynamics.step(red, dynamics.reference_action(
            red, desired_yaw_rate=float(reference[0]),
            desired_surge_acceleration=float(reference[1])))
        hull = capsule_segment(red.pos, red.heading, 1.255, .29)
        assert min(capsule_circle_clearance(hull, np.asarray(c), r + .4)
                   for c, r in REEFS) > 0.


def test_red_apf_balanced_field_stays_still_without_hidden_escape():
    ship, _, _ = dynamics_pair()
    red = ShipState(np.asarray([10., 10.]), 0., 0., 0., "surface")
    blue = ShipState(np.asarray([12., 10.]), 0., 0., 0., "surface")
    hull_length = ship["surface_length_m"]
    gap, influence = 2. - hull_length, 3. - hull_length
    gain = 1. / (1. - gap / influence) ** 2
    navigator = RedNavigator({"surface_avoidance_gain": gain,
                              "surface_avoidance_range_m": 3., "stagnation_window_steps": 4})
    for _ in range(12):
        np.testing.assert_allclose(navigator.reference(red, [49., 10.], [], [blue], WORLD, ship),
                                   [0., 0.], atol=1e-6)
    assert navigator.recovery_count > 0


def test_red_apf_repels_parallel_usv_without_a_nonclosing_exception():
    ship, _, _ = dynamics_pair()
    red = ShipState(np.asarray([10., 10.]), .4, 0., 0., "surface")
    blue = ShipState(np.asarray([11.2, 8.9]), .4, 0., 0., "surface")
    reference = RedNavigator().reference(red, np.asarray([49., 10.]), [], [blue], WORLD, ship)
    assert reference[0] > 0. and reference[1] < 0.


def test_red_apf_responds_before_close_reef_or_usv_contact():
    ship, _, _ = dynamics_pair()
    navigator = RedNavigator()
    red = ShipState(np.asarray([23.5, 27.]), .4, 0., 0., "surface")
    reference = navigator.reference(red, [49., 27.], [([32.8, 28.1], 3.5)], [], WORLD, ship)
    assert reference[0] < -.01  # Already turning, over 5 m from the reef edge.
    blue = ShipState(red.pos + np.asarray([3.8, -1.]), .3, math.pi, 0., "surface")
    field = navigator.potential_forces(red, [49., 27.], [], [blue], WORLD, ship)
    assert np.linalg.norm(field['usv']) > .1 and field['usv'][1] > 0.


def test_red_apf_curves_into_bc_channel_before_reaching_c_reef():
    """The real hull should already bend south at x=20, not wait until x=25."""
    from usv_env.config import assemble
    config = assemble(device="cpu")
    ship, dynamics, _ = dynamics_pair()
    navigator = RedNavigator(config["red_policy"])
    red = ShipState(np.asarray(config["scenario"]["red_start"]), .4, config['scenario']['red_heading'], 0., "surface")
    first_bend_x = float(red.pos[0]) + 1.
    developed_bend_x = float(red.pos[0]) + 4.
    samples = {}
    minimum_clearance = math.inf
    for _ in range(config["env"]["max_steps"]):
        reference = navigator.reference(red, config["protected_zone"]["center"],
            config["reefs"], [], WORLD, ship)
        red = dynamics.step(red, dynamics.reference_action(red,
            desired_yaw_rate=float(reference[0]),
            desired_surge_acceleration=float(reference[1])))
        hull = capsule_segment(red.pos, red.heading, 1.255, .29)
        minimum_clearance = min(minimum_clearance, *(capsule_circle_clearance(
            hull, np.asarray(reef["center"]), reef["radius"] + .4) for reef in config["reefs"]))
        for x in (first_bend_x, developed_bend_x, 20., 25.):
            if x not in samples and red.pos[0] >= x:
                samples[x] = (float(red.pos[1]), red.heading)
        if np.linalg.norm(red.pos - np.asarray(config["protected_zone"]["center"])) <= 4.:
            break
    assert samples[first_bend_x][0] < config['scenario']['red_start'][1] - .01
    assert samples[developed_bend_x][0] < config['scenario']['red_start'][1] - .3
    assert samples[20.][0] < 27.6 and samples[20.][1] < -.1
    assert 21. < samples[25.][0] < 25.
    assert minimum_clearance > 0.
    assert np.linalg.norm(red.pos - np.asarray(config["protected_zone"]["center"])) <= 4.


def test_red_anticipatory_reef_force_is_a_potential_gradient():
    """The early bend still comes from APF, without a hidden steering force."""
    ship, _, _ = dynamics_pair()
    navigator = RedNavigator({"reef_field_x_scale": 1.8})
    center, radius = np.array([32.8, 28.1]), 3.5
    position = np.array([22., 25.])
    config = navigator.config

    def potential(point):
        offset = point - center
        distances = (np.linalg.norm(offset), np.linalg.norm(offset / [1.8, 1.]))
        gaps = [distance - radius - .5 * ship["surface_length_m"] - config['clearance_m']
                for distance in distances]
        return sum(config['reef_avoidance_gain'] * config['reef_avoidance_range_m'] / 3.
            * max(1. - gap / config['reef_avoidance_range_m'], 0.) ** 3 for gap in gaps)

    epsilon = 1e-5
    numerical_force = np.array([-(potential(position + delta) - potential(position - delta)) / (2. * epsilon)
                               for delta in np.eye(2) * epsilon])
    red = ShipState(position, .4, 0., 0., "surface")
    force = navigator.potential_forces(red, [49., 27.], [(center, radius)], [], WORLD, ship)['reef']
    np.testing.assert_allclose(force, numerical_force, rtol=1e-5, atol=1e-7)


def test_rule_screen_keeps_standoff_during_an_open_water_encounter():
    _, sd, ad = dynamics_pair()
    surface = [ShipState(np.asarray(p), 0., math.pi, 0., "surface")
               for p in ([35., 21.], [39., 21.])]
    subs = [ShipState(np.asarray(p), 0., 0., 0., "submarine")
            for p in ([12., 36.], [13., 19.], [25., 30.5])]
    controller = RuleController()
    for step in range(400):
        red_pos = np.asarray([11. + .4 * .15 * step, 27.])
        estimate = dict(valid=True, position=red_pos, velocity=[.4, 0.], covariance=np.eye(2) * .1)
        action = controller.actions(surface, subs, estimate, [49., 27.], [], WORLD,
                                    sd, ad, step * .15)
        assert all(np.linalg.norm(s.pos - red_pos) >= 2. for s in surface)
        surface = [sd.step(s, a) for s, a in zip(surface, action[:2])]


def test_rule_station_brakes_with_actual_heading_and_sway_near_teammate():
    """A converging flank pair must brake before their hull envelopes meet."""
    _, dynamics, _ = dynamics_pair()
    surface = [
        ShipState(np.array([29.5975, 21.1269]), .175, -.56, -.089,
                  "surface", sway_speed=.093),
        ShipState(np.array([31.4312, 21.3438]), 0., -.957, .2,
                  "surface", sway_speed=-.037),
    ]
    targets = [np.array([30.5, 19.7]), np.array([32.5, 20.8])]
    controller = RuleController()
    minimum_distance = math.inf
    for _ in range(120):
        actions = [controller._surface_action(own, target, target, .38,
            dynamics, [], .6275, np.array([27.5, 22.6]), np.zeros(2),
            [surface[1 - index]], np.array([1., 0.]))
            for index, (own, target) in enumerate(zip(surface, targets))]
        surface = [dynamics.step(own, action) for own, action in zip(surface, actions)]
        minimum_distance = min(minimum_distance,
                               float(np.linalg.norm(surface[0].pos - surface[1].pos)))
    assert minimum_distance >= 1.6


def test_rule_station_uses_reverse_instead_of_turning_a_west_facing_hull():
    _, dynamics, _ = dynamics_pair()
    own = ShipState(np.array([30., 20.]), 0., math.pi, 0., "surface")
    target = np.array([31., 20.])
    action = RuleController()._surface_action(own, target, target, .38,
        dynamics, [], .6275, np.array([27., 23.]), np.array([.1, 0.]),
        [], np.array([1., 0.]))
    acceleration = dynamics.body_acceleration(own.body_velocity, action)
    assert abs(acceleration[2]) < .01
    assert acceleration[0] < -.01


def test_rule_controller_returns_five_physical_actions_from_estimates_only():
    ship, surface_dyn, submarine_dyn = dynamics_pair()
    surface = [
        ShipState(np.asarray([35., 21.]), .3, math.pi, 0., "surface"),
        ShipState(np.asarray([39., 21.]), .3, math.pi, 0., "surface"),
    ]
    subs = [
        ShipState(np.asarray([12., 36.]), .3, 0., 0., "submarine", surfaced=False),
        ShipState(np.asarray([13., 19.]), .3, 0., 0., "submarine", surfaced=False),
        ShipState(np.asarray([25., 30.5]), .3, 0., 0., "submarine", surfaced=False),
    ]
    controller = RuleController()
    args = (surface, subs)
    estimate = {"valid": True, "position": [25., 27.], "velocity": [.4, 0.],
                "covariance": [[.25, 0.], [0., .25]]}
    actions = controller.actions(*args, target_estimate=estimate, protected_center=[49., 27.],
                                 reefs=REEFS, world_size=WORLD, surface_dynamics=surface_dyn,
                                 submarine_dynamics=submarine_dyn, time_seconds=0.)
    moved_estimate = dict(estimate, position=[40., 12.])
    other = controller.actions(*args, target_estimate=moved_estimate, protected_center=[49., 27.],
                               reefs=REEFS, world_size=WORLD, surface_dynamics=surface_dyn,
                               submarine_dynamics=submarine_dyn, time_seconds=0.)
    assert actions.shape == (5, 2) and np.isfinite(actions).all()
    assert not np.allclose(actions, other)
    limits = np.asarray([[ship["max_surge_force"], ship["max_yaw_moment"]]] * 2 +
                        [[ship["submarine_dynamics"]["max_surge_force"],
                          ship["submarine_dynamics"]["max_yaw_moment"]]] * 3)
    assert np.all(np.abs(actions) <= limits + 1e-6)


def test_rule_controller_refuses_wrong_counts_and_bad_target():
    _, surface_dyn, submarine_dyn = dynamics_pair()
    controller = RuleController()
    state = ShipState(np.asarray([10., 10.]), 0., 0., 0., "surface")
    with pytest.raises(ValueError, match="exactly 2"):
        controller.actions([state], [], None, [49., 27.], REEFS, WORLD,
                           surface_dyn, submarine_dyn, 0.)
    surface = [state] * 2
    subs = [ShipState(np.asarray([12., 12.]), 0., 0., 0., "submarine")] * 3
    invalid = controller.actions(surface, subs, {"valid": True, "position": [np.nan, 0.],
                                                  "velocity": [0., 0.]}, [49., 27.], REEFS,
                                 WORLD, surface_dyn, submarine_dyn, 0.)
    unknown = controller.actions(surface, subs, None, [49., 27.], REEFS,
                                 WORLD, surface_dyn, submarine_dyn, 0.)
    np.testing.assert_array_equal(invalid, unknown)


@pytest.mark.parametrize("known", [False, True])
def test_rule_usvs_use_same_flank_and_staggered_references(monkeypatch, known):
    """Both slots stay south; once informed, both can influence Red."""
    _, sd, ad = dynamics_pair()
    surface = [ShipState(np.asarray(p), 0., math.pi, 0., "surface")
               for p in ([35., 21.], [39., 21.])]
    subs = [ShipState(np.asarray(p), 0., 0., 0., "submarine", surfaced=False)
            for p in ([12., 36.], [13., 19.], [25., 30.5])]
    controller = RuleController()
    references = {}

    def record(index, position, target, *args):
        references[index] = target.copy()
        return target

    monkeypatch.setattr(controller, "_waypoint", record)
    estimate = (dict(valid=True, position=[25., 27.], velocity=[.4, 0.],
                     covariance=np.eye(2)) if known else None)
    controller.actions(surface, subs, estimate, [49., 27.], [], WORLD,
                       sd, ad, 0.)
    assert references[0][1] < 27. and references[1][1] < 27.
    assert np.linalg.norm(references[0] - references[1]) >= 3. - 1e-8
    if known:
        for index in (0, 1):
            # Leave room inside the 4 m engagement gate, including the
            # one-second velocity feedforward, instead of skimming its edge.
            assert 2.8 < np.linalg.norm(references[index] - estimate['position']) < 3.8


def test_rule_usvs_respond_to_received_track_before_c_longitude(monkeypatch):
    _, sd, ad = dynamics_pair()
    surface = [ShipState(np.asarray(p), 0., math.pi, 0., "surface")
               for p in ([24., 24.], [27.5, 24.])]
    subs = [ShipState(np.asarray(p), 0., 0., 0., "submarine", surfaced=False)
            for p in ([12., 36.], [29., 13.], [40., 36.])]
    controller = RuleController()
    references = {}
    def record(index, position, target, *args):
        references[index] = target.copy()
        return target
    monkeypatch.setattr(controller, '_waypoint', record)
    estimate = dict(valid=True, position=np.array([18., 28.]), velocity=[.35, -.1],
                    covariance=np.eye(2))
    controller.actions(surface, subs, estimate, [49., 27.], REEFS, WORLD, sd, ad, 10.)
    assert all(np.linalg.norm(references[i] - estimate['position']) < 4. for i in range(2))


def test_current_scene_deploys_usvs_on_same_forward_patrol_flank():
    from usv_env.config import assemble
    from usv_env.scene_config import validate_scene
    config = assemble(device="cpu")
    starts = np.asarray(config["scenario"]["blue_starts"])
    zone = config["protected_zone"]
    center = np.asarray(zone["center"])
    front = max(config['reefs'], key=lambda reef: reef['center'][0])
    assert np.all(starts[:, 1] < front['center'][1] - front['radius'])
    assert np.all(starts[:, 0] > front['center'][0] + front['radius'])
    assert np.all(starts[:, 0] < center[0] - zone['radius'])
    for position, heading in zip(starts, config["scenario"]["blue_headings"]):
        left, right, radius = capsule_segment(position, heading,
            config["ship"]["surface_length_m"], config["ship"]["surface_collision_width_m"])
        assert min(np.linalg.norm(left - center), np.linalg.norm(right - center)) - radius > zone["radius"]
    assert np.linalg.norm(starts[0] - starts[1]) >= 3.
    assert validate_scene(config)["initial_hulls"] == 6


def test_current_reef_pairs_have_ten_percent_more_center_separation():
    from usv_env.config import assemble
    reefs = assemble(device="cpu")["reefs"]
    original = np.asarray([[20., 37.], [20., 17.], [32., 28.]])
    actual = np.asarray([reef["center"] for reef in reefs])
    for i in range(3):
        assert reefs[i]["radius"] == 3.5
        for j in range(i + 1, 3):
            np.testing.assert_allclose(np.linalg.norm(actual[i] - actual[j]),
                1.1 * np.linalg.norm(original[i] - original[j]), rtol=1e-12)


def test_rule_controller_uses_receiver_estimates_and_fixed_observation_bearings(monkeypatch):
    _, sd, ad = dynamics_pair()
    surface = [ShipState(np.asarray(p), .3, math.pi, 0., "surface")
               for p in ([35., 21.], [39., 21.])]
    subs = [ShipState(np.asarray(p), .3, 0., 0., "submarine", surfaced=False)
            for p in ([12., 36.], [13., 19.], [25., 30.5])]
    estimate = dict(valid=True, position=[25., 27.], velocity=[.4, 0.], covariance=np.eye(2))
    controller = RuleController()
    references = {}
    original_waypoint = controller._waypoint
    def record(index, position, target, *args):
        references[index] = np.asarray(target).copy()
        return original_waypoint(index, position, target, *args)
    monkeypatch.setattr(controller, '_waypoint', record)
    args = (surface, subs, None, [49., 27.], REEFS, WORLD, sd, ad)
    initial = controller.actions(*args, 0., target_estimates=[estimate] * 5)
    later = controller.actions(*args, 15., target_estimates=[estimate] * 5)
    np.testing.assert_array_equal(initial, later)
    previous_references = {i: p.copy() for i, p in references.items()}
    changed = [estimate] * 5
    changed[0] = dict(estimate, position=[40., 12.])
    local = controller.actions(*args, 15., target_estimates=changed)
    np.testing.assert_array_equal(later[1:], local[1:])
    for index in range(1, 5):
        np.testing.assert_array_equal(previous_references[index], references[index])
    # Different local requests can yield the same saturated braking command.
    assert not np.allclose(previous_references[0], references[0])


def test_red_reaches_protected_zone_using_real_underactuated_dynamics():
    ship, dynamics, _ = dynamics_pair()
    red = ShipState(np.asarray([11., 27.]), .4, 0., 0., "surface")
    navigator = RedNavigator()
    minimum_clearance = math.inf
    reached = False
    for step in range(900):
        reference = navigator.reference(red, np.asarray([49., 27.]), REEFS, [], WORLD, ship)
        action = dynamics.reference_action(red, desired_yaw_rate=float(reference[0]),
                                           desired_surge_acceleration=float(reference[1]))
        red = dynamics.step(red, action)
        capsule = capsule_segment(red.pos, red.heading, 1.255, .29)
        minimum_clearance = min(minimum_clearance, *(capsule_circle_clearance(capsule, np.asarray(c), r)
                                                    for c, r in REEFS))
        if np.linalg.norm(red.pos - np.asarray([49., 27.])) <= 4.:
            reached = True
            break
    assert reached and minimum_clearance > 0.0
    assert navigator.recovery_count == 0


def test_five_rule_vessels_execute_physical_episode():
    ship, sd, ad = dynamics_pair()
    surface = [ShipState(np.asarray(p), .3, math.pi, 0., "surface")
               for p in ([35., 21.], [39., 21.])]
    subs = [ShipState(np.asarray(p), .4, 0., 0., "submarine", surfaced=False)
            for p in ([12., 36.], [13., 19.], [25., 30.5])]
    red = ShipState(np.asarray([11., 27.]), .4, 0., 0., "surface")
    navigator, controller = RedNavigator(), RuleController()
    for step in range(900):
        # Explicit ideal-information controller diagnostic, not an actor input.
        estimate = dict(valid=True, position=red.pos, velocity=red.velocity, covariance=np.eye(2) * .1)
        action = controller.actions(surface, subs, None, [49., 27.], REEFS, WORLD, sd, ad,
                                    step * .15, target_estimates=[estimate] * 5)
        surface = [sd.step(state, command) for state, command in zip(surface, action[:2])]
        subs = [ad.step(state, command) for state, command in zip(subs, action[2:])]
        reference = navigator.reference(red, np.asarray([49., 27.]), REEFS, surface, WORLD, ship)
        red = sd.step(red, sd.reference_action(red, desired_yaw_rate=float(reference[0]),
                                              desired_surge_acceleration=float(reference[1])))
        for states, length, width in ((surface + [red], 1.255, .29), (subs, .6, .31)):
            capsules = [capsule_segment(s.pos, s.heading, length, width) for s in states]
            for i, capsule in enumerate(capsules):
                clearances = [capsule_circle_clearance(capsule, np.asarray(c), r) for c, r in REEFS]
                assert min(clearances) > 0., (step, length, i, states[i].pos, clearances)
                assert all(not capsules_overlap(capsule, other) for other in capsules[i + 1:]), (
                    step, length, i, [j for j in range(i + 1, len(capsules)) if capsules_overlap(capsule, capsules[j])],
                    [s.pos for s in states])
        assert np.isfinite(np.asarray([s.pos for s in surface + subs + [red]])).all()
        if np.linalg.norm(red.pos - np.asarray([49., 27.])) <= 4.:
            break
    assert step > 200
