"""Equal-input information ablations, peer-link controls and Actor boundaries."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from usv_env.communication import CommunicationNetwork
from usv_env.history import SCHEMA, schema_for_counts
from usv_env.observation import build_actor_observation, build_critic_state, critic_state_dim
from usv_env.target_tracking import (
    INFORMATION_MODES, ConstantVelocityTargetEstimator, RangeBearingSensor,
    TargetEstimate, TargetMessageBus, actor_target_features, information_mode,
)
from usv_env.types import ShipState


def ship(position, kind="surface"):
    return ShipState(np.asarray(position, np.float32), 0., 0., 0., kind, kind == "surface")


def track(source=2, time=0., x=10., variance=.2, confidence=.9):
    return TargetEstimate(True, np.array([x, 20., .4, 0.]), np.eye(4)*variance,
                          time, time, confidence, source, f"{source}:{time}")


def peer_config(mode="normal", extra=3):
    return dict(enabled=True, initial_sync=False, delay_steps=[0, 0],
                target_loss_range=[0., 0.], loss_model="independent_bernoulli",
                peer_mode=mode, peer_extra_delay_steps=extra)


@pytest.mark.parametrize("mode,enabled", [("hold", False), ("cv", False), ("age_uncertainty", True)])
def test_information_mode_is_explicit_and_agrees_with_mechanism(mode, enabled):
    cfg=dict(information=dict(mode=mode), mechanisms=dict(age_uncertainty_compensation=enabled))
    assert information_mode(cfg) == mode
    cfg["mechanisms"]["age_uncertainty_compensation"] = not enabled
    with pytest.raises(ValueError, match="disagree"):
        information_mode(cfg)


def test_base_hold_does_not_extrapolate_or_propagate_covariance_even_in_actor_helper():
    original=track()
    bus=TargetMessageBus(5, dict(delay_steps=3, loss_rate=0.), 0, mode="hold")
    bus.broadcast(2, original, 0.)
    bus.advance(.45)
    held=bus.latest(0, 3.)
    np.testing.assert_array_equal(held.state, original.state)
    np.testing.assert_array_equal(held.covariance, original.covariance)
    assert held.state_time == 0. and held.t_obs == 0.
    assert held.contribution_source_ids == (2,) and held.contribution_weights == (1.,)
    features=actor_target_features(held, ship([0., 0.]), 3.)
    assert features[1] == original.position[0] and features[6] == 3.
    assert not bus.latest(0, 12.01).valid  # Shared expiry still protects Base.


def test_traditional_cv_predicts_latest_source_without_joint_fusion():
    bus=TargetMessageBus(5, dict(delay_steps=0, loss_rate=0.), 0, mode="cv")
    a,b=track(2, 0., 10.),track(3, .15, 11.)
    bus.broadcast(2,a,0.); bus.broadcast(3,b,.15); bus.advance(.15)
    result=bus.latest(0, 3.)
    assert result.position[0] == pytest.approx(11.+.4*2.85)
    assert result.contribution_source_ids == (3,)
    assert result.confidence == b.confidence
    assert np.trace(result.covariance) > np.trace(b.covariance)


def test_i2_exposes_each_source_age_and_weight_while_scalar_timestamp_is_newest():
    bus=TargetMessageBus(5, dict(delay_steps=0, loss_rate=0.), 0, mode="age_uncertainty")
    bus.broadcast(2,track(2,0.),0.)
    bus.broadcast(3,track(3,1.5),1.5); bus.advance(1.5)
    result=bus.latest(0, 3.)
    assert result.processing_mode == "age_uncertainty"
    assert result.t_obs == 1.5
    assert result.contribution_source_ids == (2,3)
    assert result.contribution_observation_times == (0.,1.5)
    assert sum(result.contribution_weights) == pytest.approx(1.)
    assert result.contribution_weights[1] > result.contribution_weights[0]
    assert result.confidence < .9


@pytest.mark.parametrize("mode", ["normal", "disabled", "delayed", "shuffled"])
def test_peer_ablation_changes_only_peer_delivery_and_rejects_old_packets(mode):
    cfg=peer_config(mode)
    peers=CommunicationNetwork(5,cfg,17)
    states=[ship([float(i),5.]) for i in range(5)]
    peers.reset(states)
    target=TargetMessageBus(5,dict(cfg,loss_rate=0.),17)
    for step in range(5):
        for item in states: item.pos[0] += 1.
        peers.step(states,step)
        target.broadcast(2,track(2,step*.15),step*.15)
        target.advance(step*.15)
    assert target.sent == target.delivered == 20 and target.dropped == 0
    assert target.latest(0,.6).t_obs == .6
    if mode == "disabled":
        assert not peers.cache and peers.sent == 0
    elif mode == "delayed":
        assert peers.cache[(0,1)].send_step == 1
    elif mode == "shuffled":
        assert peers.cache[(0,1)].send_step == 3 and peers.out_of_order > 0
    else:
        assert peers.cache[(0,1)].send_step == 4


def test_removed_agent_has_no_ghost_peer_or_target_links():
    states=[ship([float(i),5.]) for i in range(5)]
    mask=np.array([1,1,0,1,1])
    peers=CommunicationNetwork(5,dict(peer_config(),initial_sync=True),17)
    peers.reset(states,active_mask=mask)
    peers.step(states,1,active_mask=mask)
    assert peers.sent == peers.delivered == 12
    assert all(2 not in link for link in peers.cache)
    target=TargetMessageBus(5,dict(delay_steps=0,loss_rate=0.),17,active_mask=mask)
    assert target.broadcast(2,track(2),0.) == 0
    assert target.broadcast(3,track(3),0.) == 3
    received=target.advance(0.)
    assert not received[2] and not target.latest(2,0.,track(2)).valid
    assert target.sent == target.delivered == 3


def test_disabling_communication_never_performs_hidden_initial_truth_sync():
    cfg=dict(peer_config(),enabled=False)
    peers=CommunicationNetwork(5,cfg,17)
    states=[ship([float(i),5.]) for i in range(5)]
    peers.reset(states)
    assert not peers.cache
    peers.step(states,0)
    assert not peers.cache and peers.sent == 0
    target=TargetMessageBus(5,dict(cfg,loss_rate=0.),17)
    assert target.broadcast(2,track(2),0.) == 0 and target.sent == 0
    unspecified=peer_config(); unspecified.pop("initial_sync")
    normal=CommunicationNetwork(5,unspecified,17); normal.reset(states)
    assert not normal.cache


def test_actor_has_no_slot_inputs_or_instantaneous_private_peer_truth():
    states=[ship([10.+i, 10.], "surface" if i<2 else "submarine") for i in range(5)]
    peers=CommunicationNetwork(5,peer_config(),0); peers.reset(states)
    env=SimpleNamespace(observation_schema=SCHEMA, num_friendly=5, num_surface_blue=2,
        friendly_states=lambda: states, agent_active_mask=lambda: np.ones(5),
        world_size=np.array([55.,55.]), max_speed_by_type=dict(surface=.4,submarine=.5),
        max_sway_speed=.4,max_yaw_rate=1.,target_estimates=[TargetEstimate.unknown() for _ in states],
        communication=peers,step_count=0,dt=.15,reefs=[dict(center=p,radius=3.5) for p in [(20,37),(20,17),(32,28)]],
        protected_center=np.array([49.,27.]),protected_radius=4.,
        last_executed_actions=np.zeros((5,2)),agent_action_limits=np.ones((5,2)),
        sensor_detected=np.zeros(3),max_steps=900,hold_steps=0,required_hold_steps=54,
        red_state=ship([11.,27.]),shadow_red_state=ship([11.,27.]),counterfactual_gap=0.,
        critic_state_dim=critic_state_dim())
    assert SCHEMA.agent_count == 5 and SCHEMA.sensor_dim == 76 and SCHEMA.frame_dim == 91
    assert not any("slot" in name or "desired" in name or name in ("target_x","target_y","target_distance")
                   for name in SCHEMA.fields)
    before=build_actor_observation(env,0)
    states[1].pos += 10.  # No message means no access to this new private state.
    env.red_state.pos += 5.
    np.testing.assert_array_equal(before,build_actor_observation(env,0))
    assert before[SCHEMA.index("target_valid")] == 0.
    assert all(before[SCHEMA.index(name)] == 0. for name in ("red_x","red_y","red_vx","red_vy"))
    centralized=build_critic_state(env)
    assert centralized.shape == (99,)
    np.testing.assert_allclose(centralized[:9],[1.,1.,10./55,10./55,0.,0.,0.,1.,0.])
    assert schema_for_counts(6).frame_dim > SCHEMA.frame_dim  # Counts remain explicit, never hard-coded.


def information_replay_audit(steps=180):
    """Prescribed synthetic sensor replay; this is not a controller/task rollout."""
    dt=.15
    sensors={i:RangeBearingSensor(seed=700+i) for i in (2,3,4)}
    estimators={i:ConstantVelocityTargetEstimator(i) for i in sensors}
    observers={i:ship(p,"submarine") for i,p in zip(sensors,[(16.,25.),(20.,30.),(24.,26.)])}
    stream=[]; truth=[]; raw=[]
    for step in range(steps):
        now=step*dt
        target=ship([15.+.32*now,27.+np.sin(.1*now)])
        truth.append(target.pos.astype(float))
        events=[]
        if not 60<=step<90:
            for source,sensor in sensors.items():
                measurement=sensor.measure(source,observers[source],target,now)
                if measurement is not None:
                    events.append((source,estimators[source].update(measurement)))
                    raw.append([step,source,measurement.source_measurement_id,measurement.range_m,measurement.bearing_rad])
        stream.append(events)
    rows=[]
    for delay in (0,3,10,20,40):
        for loss in (0.,.05,.1,.2,.3):
            buses={mode:TargetMessageBus(5,dict(delay_steps=delay,loss_rate=loss),911,mode=mode)
                   for mode in INFORMATION_MODES}
            packets={mode:[] for mode in buses}; errors={mode:[] for mode in buses}
            ages={mode:[] for mode in buses}; weighted_ages={mode:[] for mode in buses}
            traces={mode:[] for mode in buses}; recovered={mode:None for mode in buses}
            for step,events in enumerate(stream):
                now=step*dt
                for mode,bus in buses.items():
                    for source,estimate in events: bus.broadcast(source,estimate,now)
                    received=bus.advance(now)
                    for receiver,messages in received.items():
                        for p in messages:
                            packets[mode].append([receiver,p.sender_id,p.source_measurement_id,p.sequence_id,p.t_obs,
                                p.t_send,p.t_recv,p.state_time,p.position.tolist(),p.velocity.tolist(),
                                p.covariance.tolist(),p.confidence])
                    estimate=bus.latest(0,now)
                    if estimate.valid:
                        errors[mode].append(float(np.linalg.norm(estimate.position-truth[step])))
                        ages[mode].append(now-estimate.t_obs)
                        weighted_ages[mode].append(sum(w*(now-t) for w,t in zip(
                            estimate.contribution_weights,estimate.contribution_observation_times)))
                        traces[mode].append(float(np.trace(estimate.covariance[:2,:2])))
                        if step>=90 and recovered[mode] is None and max(estimate.contribution_observation_times)>=90*dt:
                            recovered[mode]=now-90*dt
            digests={mode:hashlib.sha256(json.dumps(packets[mode],sort_keys=True).encode()).hexdigest() for mode in buses}
            assert len(set(digests.values())) == 1
            assert len({(b.sent,b.dropped,b.delivered) for b in buses.values()}) == 1
            row=dict(delay_steps=delay,loss_rate=loss,identical_delivery_payloads=True,
                     delivery_sha256=digests["hold"],send_attempts=buses["hold"].sent,
                     dropped=buses["hold"].dropped,delivered=buses["hold"].delivered,modes={})
            for mode in buses:
                row["modes"][mode]=dict(valid_receiver_steps=len(errors[mode]),
                    position_rmse_m=float(np.sqrt(np.mean(np.square(errors[mode])))),
                    newest_aoi_mean_seconds=float(np.mean(ages[mode])),newest_aoi_p95_seconds=float(np.percentile(ages[mode],95)),
                    weighted_source_age_mean_seconds=float(np.mean(weighted_ages[mode])),
                    position_covariance_trace_mean_m2=float(np.mean(traces[mode])),
                    first_post_outage_observation_delay_seconds=recovered[mode])
            rows.append(row)
    return dict(kind="fixed_prescribed_synthetic_sensor_replay",steps=steps,dt_seconds=dt,
        observer_positions={str(i):s.pos.tolist() for i,s in observers.items()},
        truth_trajectory="x=15+0.32t, y=27+sin(0.1t); prescribed replay, no controller",
        sensor_seeds={str(i):700+i for i in sensors},transport_seed=911,
        sensor_outage_steps=[60,90],outage_end_exclusive=True,
        raw_measurements=len(raw),raw_measurements_sha256=hashlib.sha256(json.dumps(raw).encode()).hexdigest(),
        same_measurements_same_local_filters_same_send_opportunities=True,
        source_quality_formula="q_i=clip(source_confidence,0.001,1)*exp(-AoI_i/12s)",
        fusion_formula="C_i=P_pred_i/q_i; w_i=(1/trace(C_i))/sum_j(1/trace(C_j)); P_f^-1=sum_i(w_i C_i^-1)",
        scalar_aoi_semantics="Age of newest contributing observation; weighted source age is separately reported",
        limits="Synthetic information replay only; no policy, interception decisions, breakthrough or success rates measured.",
        conditions=rows)


def test_equal_input_stress_replay_covers_all_required_delays_and_losses():
    audit=information_replay_audit(110)
    assert len(audit["conditions"]) == 25
    assert all(row["identical_delivery_payloads"] for row in audit["conditions"])
    assert all(set(row["modes"]) == set(INFORMATION_MODES) for row in audit["conditions"])
