import numpy as np
import pytest

from usv_env.target_tracking import (
    ConstantVelocityTargetEstimator,
    LocalTargetSensor,
    RangeBearingSensor,
    TargetEstimate,
    TargetEstimator,
    TargetMessageBus,
    TargetObservationNetwork,
    TargetMeasurement,
    actor_target_features,
    fuse_estimates,
    predict_estimate,
)
from usv_env.types import ShipState


def state(pos, heading=0.0, speed=0.0, unit_type="submarine", surfaced=False):
    return ShipState(np.asarray(pos, dtype=np.float32), speed, heading, 0.0, unit_type,
                     surfaced=surfaced)


def measurement(observer_index=2, seq="2:0", t=0.0, pos=(5.0, 0.0)):
    observer = state((0.0, 0.0))
    target = state(pos, unit_type="surface", surfaced=True)
    # Use a deterministic direct measurement fixture; the sensor itself is
    # covered separately for probability, range and bearing behavior.
    return TargetMeasurement(observer_index, seq, float(np.linalg.norm(target.pos)), 0.0,
                            0.04, 0.0009, observer.pos.astype(np.float64), 0.0, t, 0.95)


def test_initial_auv_positions_are_outside_candidate_sensor_range():
    red = state((11.0, 27.0), unit_type="surface", surfaced=True)
    auvs = [state((12.0, 36.0)), state((13.0, 19.0)), state((25.0, 30.5))]
    sensor = RangeBearingSensor(seed=1)
    assert [sensor.detection_probability(np.linalg.norm(red.pos - auv.pos)) for auv in auvs] == [0.0, 0.0, 0.0]
    assert sensor.measure(0, auvs[0], red, 0.0) is None


def test_sensor_is_local_noisy_range_bearing_and_has_range_rolloff():
    sensor = RangeBearingSensor({"max_range_m": 6.0, "range_sigma_m": 0.01,
                                 "bearing_sigma_rad": 0.01}, seed=42)
    observer, target = state((0, 0), heading=0.5), state((3, 0), unit_type="surface", surfaced=True)
    samples = [sensor.measure(1, observer, target, k * 0.15) for k in range(100)]
    samples = [sample for sample in samples if sample is not None]
    assert samples and all(sample.observer_index == 1 for sample in samples)
    assert abs(samples[0].range_m - 3.0) < 0.05
    assert abs(samples[0].bearing_rad - (-0.5)) < 0.05
    assert sensor.detection_probability(5.5) < sensor.detection_probability(1.0)
    assert sensor.detection_probability(6.01) == 0.0


def test_estimator_updates_and_prediction_expand_covariance():
    estimator = ConstantVelocityTargetEstimator(2)
    first = estimator.update(measurement(t=0.0, pos=(5.0, 1.0)))
    assert first.valid and first.source_measurement_id == "2:0"
    later = estimator.predict(3.0)
    assert later.t_obs == 0.0 and later.state_time == 3.0
    assert np.trace(later.covariance) > np.trace(first.covariance)


@pytest.mark.parametrize("delay", [0, 3, 10, 20, 40])
def test_network_preserves_timestamps_and_delivery_delay(delay):
    network = TargetObservationNetwork(2, {"delay_steps": delay, "loss_rate": 0.0}, seed=7)
    estimate = ConstantVelocityTargetEstimator(0).update(measurement(t=0.0))
    assert network.send(0, estimate, step=20) == 1
    delivery_step = 20 + delay
    assert network.receive(1, delivery_step - 1 if delay else 19) == []
    messages = network.receive(1, delivery_step)
    assert len(messages) == 1
    packet = messages[0]
    assert packet.sender_id == 0 and packet.source_measurement_id == "2:0"
    assert packet.sequence_id == 0 and packet.valid
    assert packet.t_obs == 0.0
    assert packet.t_send == 3.0
    assert packet.t_recv == delivery_step * 0.15
    assert packet.t_recv - packet.t_obs > packet.t_recv - packet.t_send


def test_network_drops_expired_messages_and_does_not_clip_age_to_legacy_cap():
    estimate = ConstantVelocityTargetEstimator(0).update(measurement(t=0.0))
    network = TargetObservationNetwork(2, {"delay_steps": 40, "loss_rate": 0.0,
                                         "max_age_seconds": 20.0}, seed=1)
    network.send(0, estimate, step=0)
    delivered = network.receive(1, 40)
    assert delivered[0].t_recv - delivered[0].t_obs == pytest.approx(6.0)
    assert network.fused_estimate(1, 6.0).t_obs == 0.0

    expired = TargetObservationNetwork(2, {"delay_steps": 40, "loss_rate": 0.0,
                                          "max_age_seconds": 5.0}, seed=1)
    expired.send(0, estimate, step=0)
    assert expired.receive(1, 40) == []
    assert expired.expired == 1


def test_network_deduplicates_repeated_forward_of_same_measurement():
    estimate = ConstantVelocityTargetEstimator(0).update(measurement(t=0.0))
    network = TargetObservationNetwork(2, {"delay_steps": 0, "loss_rate": 0.0}, seed=1)
    network.send(0, estimate, step=0)
    assert len(network.receive(1, 0)) == 1
    network.send(0, estimate, step=1)
    assert network.receive(1, 1) == []
    assert network.duplicates == 1


def test_supported_loss_rates_and_invalid_rate_validation():
    for loss in TargetObservationNetwork.SUPPORTED_LOSS_RATES:
        network = TargetObservationNetwork(2, {"delay_steps": 0, "loss_rate": loss}, seed=5)
        assert network.loss_rate == loss
    with pytest.raises(ValueError, match="loss_rate"):
        TargetObservationNetwork(2, {"delay_steps": 0, "loss_rate": 0.31}, seed=5)


def test_loss_statistics_count_all_attempts_and_age_uses_measurement_time():
    bus=TargetMessageBus(2,{"delay_steps":3,"loss_rate":.3},seed=5)
    estimate=ConstantVelocityTargetEstimator(0).update(measurement(t=0.0))
    for _ in range(500): bus.broadcast(0,estimate,1.5)
    bus.advance(1.95)
    statistics=bus.statistics()
    assert bus.sent==500 and 100<bus.dropped<200
    assert statistics['realized_packet_loss']==bus.dropped/500
    assert statistics['message_age_mean_seconds']==pytest.approx(1.95)
    assert statistics['transport_delay_mean_seconds']==pytest.approx(.45)


def test_unknown_actor_observation_contains_no_target_position_or_velocity():
    observer = state((2.0, 2.0), heading=1.2)
    features = actor_target_features(TargetEstimate.unknown(), observer, 0.0)
    np.testing.assert_array_equal(features, np.zeros(8, dtype=np.float32))


def test_environment_facing_sensor_estimator_and_bus_api():
    observer = state((0.0, 0.0), unit_type="submarine")
    red = state((3.0, 0.0), unit_type="surface", surfaced=True)
    sensor = LocalTargetSensor({"minimum_detection_probability": 1.0}, seed=9, observer_index=2)
    detection = sensor.measure(observer, red, 0.0)
    assert detection is not None and detection.observer_index == 2
    estimator = TargetEstimator(2)
    estimate = estimator.update_detection(detection)
    bus = TargetMessageBus(3, {"delay_steps": 0, "loss_rate": 0.0}, seed=2)
    assert bus.broadcast(2, estimate, 0.0) == 2
    assert bus.advance(0.0)[0][0].t_recv == 0.0
    shared = bus.latest(0, 0.0)
    assert shared.valid and shared.source_agent_id == 2


def test_age_prediction_and_message_fusion_use_covariance_and_confidence():
    a = TargetEstimate(True, np.asarray([2., 1., .3, 0.]), np.eye(4) * .2,
                       0.0, 0.0, .9, 0, "0:1")
    b = TargetEstimate(True, np.asarray([2.5, 1., .3, 0.]), np.eye(4) * .5,
                       0.0, 0.0, .8, 1, "1:1")
    predicted = predict_estimate(a, 2.0)
    assert predicted.position[0] == pytest.approx(2.6)
    assert np.trace(predicted.covariance) > np.trace(a.covariance)
    network = TargetObservationNetwork(2, {"delay_steps": 0, "loss_rate": 0.0}, seed=3,
                                       mode="age_uncertainty")
    fused = network.fused_estimate(0, 1.0, local_estimate=a)
    assert fused.valid and fused.source_agent_id == 0
    # Standalone fusion is used by the same network once received peer data exists.
    network.send(1, b, 0)
    network.receive(0, 0)
    fused = network.fused_estimate(0, 1.0, local_estimate=a)
    assert fused.valid and fused.t_obs == 0.0
    assert 2.0 <= fused.position[0] <= 2.5
    assert fused.state_time == 1.0


def test_communication_module_exports_target_bus_without_changing_peer_network():
    from usv_env.communication import CommunicationNetwork, TargetMessageBus as PublicBus
    assert PublicBus is TargetMessageBus
    assert hasattr(CommunicationNetwork, "peer_state")


def test_correlated_duplicate_estimates_do_not_manufacture_information():
    original = TargetEstimate(True, np.asarray([2., 1., .3, 0.]), np.eye(4) * .2,
                              0., 0., 1., 0, "0:1")
    fused = fuse_estimates([original] * 20, 0.)
    np.testing.assert_allclose(fused.covariance, original.covariance, atol=1e-8)
    np.testing.assert_allclose(fused.state, original.state)


@pytest.mark.parametrize("mode", ["hold", "cv", "age_uncertainty"])
def test_long_repeated_forwarding_does_not_shrink_receiver_covariance(mode):
    bus = TargetMessageBus(3, {"delay_steps": 0, "loss_rate": 0.0}, seed=3, mode=mode)
    original = TargetEstimate(True, np.asarray([2., 1., .3, 0.]), np.eye(4) * .2,
                              0., 0., 1., 0, "0:1")
    bus.broadcast(0, original, 0.)
    bus.advance(0.)
    first = bus.latest(2, 0.)
    for _ in range(500):
        bus.broadcast(1, bus.latest(1, 0.), 0.)
        bus.advance(0.)
    final = bus.latest(2, 0.)
    np.testing.assert_allclose(final.covariance, first.covariance, atol=1e-8)
    np.testing.assert_allclose(final.state, first.state, atol=1e-8)
    assert bus.duplicates > 400


def test_reordered_old_message_does_not_replace_newer_source_track():
    bus = TargetMessageBus(2, {"delay_steps": 0, "loss_rate": 0.0}, seed=3)
    older = TargetEstimate(True, np.asarray([2., 1., .3, 0.]), np.eye(4),
                           0., 0., 1., 0, "0:1")
    newer = TargetEstimate(True, np.asarray([2.3, 1., .3, 0.]), np.eye(4),
                           1., 1., 1., 0, "0:2")
    bus.broadcast(0, newer, 1.05)
    bus.advance(1.05)
    bus.broadcast(0, older, 1.2)
    bus.advance(1.2)
    assert bus.latest(1, 1.2).t_obs == 1.0

