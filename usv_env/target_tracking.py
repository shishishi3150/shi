"""Local target sensing, constant-velocity estimation, and delayed sharing.

The module carries estimates derived from range/bearing measurements. It never
accepts a target truth state except at the sensor boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from .geometry import wrap_angle
from .types import ShipState


INFORMATION_MODES = ("hold", "cv", "age_uncertainty")


def information_mode(config: Mapping[str, object]) -> str:
    """Resolve the explicit receiver algorithm without changing its data access."""
    enabled = bool(config.get("mechanisms", {}).get("age_uncertainty_compensation", False))
    mode = str(config.get("information", {}).get("mode", "age_uncertainty" if enabled else "hold"))
    if mode not in INFORMATION_MODES:
        raise ValueError(f"information.mode must be one of {INFORMATION_MODES}")
    if enabled != (mode == "age_uncertainty"):
        raise ValueError("information.mode and age_uncertainty_compensation disagree")
    return mode


@dataclass(frozen=True)
class TargetMeasurement:
    observer_index: int
    source_measurement_id: str
    range_m: float
    bearing_rad: float
    range_variance: float
    bearing_variance: float
    observer_position: np.ndarray
    observer_heading: float
    t_obs: float
    confidence: float

    def cartesian(self) -> tuple[np.ndarray, np.ndarray]:
        angle = self.observer_heading + self.bearing_rad
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        tangent = np.asarray([-np.sin(angle), np.cos(angle)], dtype=np.float64)
        position = np.asarray(self.observer_position, dtype=np.float64) + self.range_m * direction
        jacobian = np.column_stack((direction, self.range_m * tangent))
        polar_cov = np.diag([self.range_variance, self.bearing_variance])
        covariance = jacobian @ polar_cov @ jacobian.T
        return position.astype(np.float64), covariance.astype(np.float64)


@dataclass(frozen=True)
class TargetEstimate:
    valid: bool
    state: np.ndarray
    covariance: np.ndarray
    t_obs: float
    state_time: float
    confidence: float
    source_agent_id: int
    source_measurement_id: str
    contribution_source_ids: tuple[int, ...] = ()
    contribution_measurement_ids: tuple[str, ...] = ()
    contribution_observation_times: tuple[float, ...] = ()
    contribution_weights: tuple[float, ...] = ()
    processing_mode: str = "local_kalman"

    @classmethod
    def unknown(cls, now: float = 0.0, source_agent_id: int = -1) -> "TargetEstimate":
        return cls(False, np.zeros(4, dtype=np.float64), np.eye(4, dtype=np.float64) * 1e6,
                   float(now), float(now), 0.0, int(source_agent_id), "")

    @property
    def position(self) -> np.ndarray:
        return self.state[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:].copy()


@dataclass(frozen=True)
class TargetMessage:
    sender_id: int
    source_agent_id: int
    source_measurement_id: str
    sequence_id: int
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    confidence: float
    t_obs: float
    t_send: float
    t_recv: float
    valid: bool
    delivery_step: int
    state_time: float | None = None

    def estimate(self) -> TargetEstimate:
        return TargetEstimate(
            self.valid,
            np.concatenate((self.position, self.velocity)).astype(np.float64),
            self.covariance.astype(np.float64), self.t_obs,
            self.t_send if self.state_time is None else self.state_time,
            self.confidence, self.source_agent_id, self.source_measurement_id,
        )


class RangeBearingSensor:
    """Configurable local surface-target detector with range-bearing noise."""

    def __init__(self, config: Mapping[str, float] | None = None, seed: int = 0) -> None:
        cfg = dict(config or {})
        self.max_range_m = float(cfg.get("max_range_m", 6.0))
        self.range_sigma_m = float(cfg.get("range_sigma_m", 0.20))
        self.bearing_sigma_rad = float(cfg.get("bearing_sigma_rad", np.deg2rad(3.0)))
        self.minimum_detection_probability = float(cfg.get("minimum_detection_probability", 0.10))
        self.rng = np.random.default_rng(int(seed))
        if (not np.isfinite(self.max_range_m) or self.max_range_m <= 0
                or not np.isfinite(self.range_sigma_m) or self.range_sigma_m <= 0
                or not np.isfinite(self.bearing_sigma_rad) or self.bearing_sigma_rad <= 0
                or not 0 <= self.minimum_detection_probability <= 1):
            raise ValueError("invalid range-bearing sensor configuration")
        self.measurement_sequence = 0
        self.detections = 0
        self.misses = 0

    def detection_probability(self, distance_m: float) -> float:
        distance = float(distance_m)
        if not np.isfinite(distance) or distance < 0 or distance > self.max_range_m:
            return 0.0
        # Smooth quadratic rolloff: near-range probability is high, and it
        # reaches the configured floor immediately inside the range boundary.
        fraction = distance / self.max_range_m
        return float(np.clip(1.0 - (1.0 - self.minimum_detection_probability) * fraction**2,
                             self.minimum_detection_probability, 1.0))

    def measure(self, observer_index: int, observer: ShipState, target: ShipState,
                t_obs: float) -> TargetMeasurement | None:
        delta = np.asarray(target.pos, dtype=np.float64) - np.asarray(observer.pos, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        probability = self.detection_probability(distance)
        if probability <= 0.0 or self.rng.random() >= probability:
            self.misses += 1
            return None
        bearing = float(wrap_angle(np.arctan2(delta[1], delta[0]) - observer.heading))
        measured_range = max(0.0, distance + float(self.rng.normal(0.0, self.range_sigma_m)))
        measured_bearing = float(wrap_angle(bearing + self.rng.normal(0.0, self.bearing_sigma_rad)))
        measurement_id = f"{int(observer_index)}:{self.measurement_sequence}"
        self.measurement_sequence += 1
        self.detections += 1
        return TargetMeasurement(
            int(observer_index), measurement_id, measured_range, measured_bearing,
            self.range_sigma_m**2, self.bearing_sigma_rad**2,
            np.asarray(observer.pos, dtype=np.float64).copy(), float(observer.heading),
            float(t_obs), probability,
        )


class ConstantVelocityTargetEstimator:
    """Linear Kalman filter over target ``[x, y, vx, vy]`` state."""

    def __init__(self, source_agent_id: int, *, acceleration_noise: float = 0.35,
                 initial_velocity_variance: float = 0.25) -> None:
        self.source_agent_id = int(source_agent_id)
        self.acceleration_noise = float(acceleration_noise)
        self.initial_velocity_variance = float(initial_velocity_variance)
        if min(self.acceleration_noise, self.initial_velocity_variance) <= 0:
            raise ValueError("estimator noise parameters must be positive")
        self.estimate = TargetEstimate.unknown(source_agent_id=self.source_agent_id)

    def reset(self) -> None:
        self.estimate = TargetEstimate.unknown(source_agent_id=self.source_agent_id)

    def predict(self, now: float) -> TargetEstimate:
        current = self.estimate
        if not current.valid:
            return TargetEstimate.unknown(now, self.source_agent_id)
        dt = float(now) - current.state_time
        if dt < -1e-9:
            raise ValueError("prediction time precedes estimator state time")
        if dt <= 0:
            return current
        f = np.asarray([[1., 0., dt, 0.], [0., 1., 0., dt],
                        [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float64)
        q = self._process_noise(dt, self.acceleration_noise)
        return replace(current, state=f @ current.state, covariance=f @ current.covariance @ f.T + q,
                       state_time=float(now))

    def update(self, measurement: TargetMeasurement) -> TargetEstimate:
        z, r = measurement.cartesian()
        current = self.estimate
        if not current.valid:
            state = np.asarray([z[0], z[1], 0., 0.], dtype=np.float64)
            covariance = np.zeros((4, 4), dtype=np.float64)
            covariance[:2, :2] = r
            covariance[2:, 2:] = np.eye(2) * self.initial_velocity_variance
            result = TargetEstimate(True, state, covariance, measurement.t_obs, measurement.t_obs,
                                    measurement.confidence, self.source_agent_id,
                                    measurement.source_measurement_id)
        else:
            predicted = self.predict(measurement.t_obs)
            h = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=np.float64)
            innovation = z - h @ predicted.state
            s = h @ predicted.covariance @ h.T + r
            gain = predicted.covariance @ h.T @ np.linalg.inv(s)
            state = predicted.state + gain @ innovation
            identity = np.eye(4, dtype=np.float64)
            covariance = (identity - gain @ h) @ predicted.covariance @ (identity - gain @ h).T + gain @ r @ gain.T
            result = TargetEstimate(True, state, covariance, measurement.t_obs, measurement.t_obs,
                                    measurement.confidence, self.source_agent_id,
                                    measurement.source_measurement_id)
        self.estimate = result
        return result

    @staticmethod
    def _process_noise(dt: float, acceleration_noise: float) -> np.ndarray:
        q = float(acceleration_noise) ** 2
        d2, d3, d4 = dt**2, dt**3, dt**4
        return q * np.asarray([[d4/4, 0, d3/2, 0], [0, d4/4, 0, d3/2],
                               [d3/2, 0, d2, 0], [0, d3/2, 0, d2]], dtype=np.float64)


class TargetObservationNetwork:
    """Directed delayed/lossy links for target estimates, with source dedupe."""

    SUPPORTED_DELAYS = (0, 3, 10, 20, 40)
    SUPPORTED_LOSS_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)

    def __init__(self, agent_count: int, config: Mapping[str, object], seed: int, dt: float = 0.15,
                 *, mode: str = "hold", information: Mapping[str, object] | None = None,
                 active_mask=None) -> None:
        self.agent_count = int(agent_count)
        mask = np.ones(self.agent_count) if active_mask is None else np.asarray(active_mask)
        if mask.shape != (self.agent_count,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("target active_mask must be a binary agent_count vector")
        self.active_mask = mask.astype(bool, copy=True)
        self.config = dict(config)
        self.mode = str(mode)
        if self.mode not in INFORMATION_MODES:
            raise ValueError(f"information mode must be one of {INFORMATION_MODES}")
        self.information = dict(information or {})
        self.age_decay_seconds = float(self.information.get("age_decay_seconds", 12.0))
        if not np.isfinite(self.age_decay_seconds) or self.age_decay_seconds <= 0:
            raise ValueError("information.age_decay_seconds must be positive")
        self.dt = float(dt)
        self.enabled = bool(self.config.get("enabled", True))
        self.delay_steps = self._resolve_delay(self.config.get("delay_steps", 0))
        self.loss_rate = float(self.config.get("loss_rate", 0.0))
        self.max_age_seconds = float(self.config.get("max_age_seconds", 12.0))
        self.process_acceleration_noise = float(self.config.get("process_acceleration_noise", 0.35))
        if self.agent_count < 2 or self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("network requires at least two agents and a positive dt")
        if not 0 <= self.loss_rate <= 0.30 or not np.isfinite(self.loss_rate):
            raise ValueError("loss_rate must be in [0, 0.30]")
        if self.max_age_seconds <= 0 or not np.isfinite(self.max_age_seconds):
            raise ValueError("max_age_seconds must be positive")
        self.rng = np.random.default_rng(int(seed))
        self.queues: dict[tuple[int, int], list[TargetMessage]] = {
            (receiver, sender): [] for receiver in range(agent_count)
            for sender in range(agent_count) if receiver != sender
        }
        self.sequences = np.zeros(agent_count, dtype=np.int64)
        self.seen: list[set[str]] = [set() for _ in range(agent_count)]
        self.latest_by_source: list[dict[int, TargetEstimate]] = [dict() for _ in range(agent_count)]
        self.sent = self.dropped = self.delivered = self.duplicates = self.expired = self.out_of_order = 0
        self.delay_samples: list[int] = []
        self.observation_age_samples: list[float] = []
        self.transport_delay_samples: list[float] = []

    @classmethod
    def _resolve_delay(cls, value: object) -> tuple[int, int]:
        if isinstance(value, (int, np.integer)):
            pair = (int(value), int(value))
        else:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("delay_steps must be a supported integer or [low, high] pair")
            pair = (int(value[0]), int(value[1]))
        if pair[0] < 0 or pair[0] > pair[1] or pair[1] not in cls.SUPPORTED_DELAYS:
            raise ValueError(f"delay_steps must use supported steps {cls.SUPPORTED_DELAYS}")
        if pair[0] not in cls.SUPPORTED_DELAYS:
            raise ValueError(f"delay_steps must use supported steps {cls.SUPPORTED_DELAYS}")
        return pair

    def reset(self, active_mask=None) -> None:
        if active_mask is not None:
            mask = np.asarray(active_mask)
            if mask.shape != (self.agent_count,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("target active_mask must be a binary agent_count vector")
            self.active_mask = mask.astype(bool, copy=True)
        for queue in self.queues.values():
            queue.clear()
        self.sequences.fill(0)
        for cache in self.latest_by_source:
            cache.clear()
        for ids in self.seen:
            ids.clear()
        self.sent = self.dropped = self.delivered = self.duplicates = self.expired = self.out_of_order = 0
        self.delay_samples.clear()
        self.observation_age_samples.clear(); self.transport_delay_samples.clear()

    def send(self, sender_id: int, estimate: TargetEstimate, step: int) -> int:
        sender = int(sender_id)
        if not 0 <= sender < self.agent_count:
            raise ValueError("sender_id outside network")
        if not estimate.valid or not self.enabled or not self.active_mask[sender]:
            return 0
        t_send = int(step) * self.dt
        # The COMMON sender transports its last local measurement-time track.
        # Receiver mode must never change payloads, send chances or RNG draws.
        if estimate.state_time > t_send + 1e-9 or estimate.t_obs > t_send + 1e-9:
            raise ValueError("cannot send target information from the future")
        forwarded = estimate
        queued = 0
        for receiver in range(self.agent_count):
            if receiver == sender or not self.active_mask[receiver]:
                continue
            self.sent += 1
            if self.rng.random() < self.loss_rate:
                self.dropped += 1
                continue
            delay = int(self.rng.integers(self.delay_steps[0], self.delay_steps[1] + 1))
            sequence = int(self.sequences[sender])
            self.sequences[sender] += 1
            message = TargetMessage(
                sender, forwarded.source_agent_id, forwarded.source_measurement_id, sequence,
                forwarded.position, forwarded.velocity, forwarded.covariance.copy(),
                forwarded.confidence, forwarded.t_obs, t_send, -1.0,
                True, int(step) + delay, forwarded.state_time,
            )
            self.queues[(receiver, sender)].append(message)
            self.delay_samples.append(delay)
            queued += 1
        return queued

    def receive(self, receiver_id: int, step: int) -> list[TargetMessage]:
        receiver = int(receiver_id)
        if not 0 <= receiver < self.agent_count:
            raise ValueError("receiver_id outside network")
        if not self.active_mask[receiver]:
            self.latest_by_source[receiver].clear()
            return []
        now = int(step) * self.dt
        ready: list[TargetMessage] = []
        for link, queue in list(self.queues.items()):
            target, _ = link
            if target != receiver:
                continue
            future = []
            for message in queue:
                if message.delivery_step <= int(step):
                    ready.append(replace(message, t_recv=message.delivery_step * self.dt))
                else:
                    future.append(message)
            self.queues[link] = future
        accepted: list[TargetMessage] = []
        for message in sorted(ready, key=lambda item: (item.t_recv, item.sequence_id)):
            age = now - message.t_obs
            if age > self.max_age_seconds:
                self.expired += 1
                continue
            key = f"{message.source_agent_id}:{message.source_measurement_id}"
            if key in self.seen[receiver]:
                self.duplicates += 1
                continue
            self.seen[receiver].add(key)
            cached = self.latest_by_source[receiver].get(message.source_agent_id)
            if cached is not None and message.t_obs < cached.t_obs:
                self.out_of_order += 1
                continue
            self.delivered += 1
            self.observation_age_samples.append(message.t_recv-message.t_obs)
            self.transport_delay_samples.append(message.t_recv-message.t_send)
            accepted.append(message)
            self.latest_by_source[receiver][message.source_agent_id] = message.estimate()
        return accepted

    def receiver_estimate(self, receiver_id: int, now: float,
                          local_estimate: TargetEstimate | None = None) -> TargetEstimate:
        """Apply hold, ordinary newest-track CV, or I2 to identical received data."""
        receiver = int(receiver_id)
        if not 0 <= receiver < self.agent_count:
            raise ValueError("receiver_id outside network")
        if not self.active_mask[receiver]:
            return replace(TargetEstimate.unknown(float(now)), processing_mode=self.mode)
        candidates = list(self.latest_by_source[receiver].values())
        if local_estimate is not None and local_estimate.valid:
            candidates = [item for item in candidates if item.source_agent_id != local_estimate.source_agent_id]
            candidates.append(local_estimate)
        candidates = [item for item in candidates if item.valid
                      and 0 <= float(now) - item.t_obs <= self.max_age_seconds]
        if not candidates:
            return replace(TargetEstimate.unknown(float(now)), processing_mode=self.mode)
        if self.mode == "age_uncertainty":
            return fuse_estimates(candidates, float(now), self.process_acceleration_noise,
                                  self.age_decay_seconds)
        # Latest OBSERVATION, not latest arrival: late older packets cannot
        # overwrite current information. Quality only resolves exact-time ties.
        newest = max(candidates, key=lambda item: (item.t_obs, item.confidence, -item.source_agent_id))
        chosen = predict_estimate(newest, float(now), self.process_acceleration_noise) if self.mode == "cv" else newest
        return replace(chosen, processing_mode=self.mode,
                       contribution_source_ids=(chosen.source_agent_id,),
                       contribution_measurement_ids=(chosen.source_measurement_id,),
                       contribution_observation_times=(chosen.t_obs,), contribution_weights=(1.,))

    def fused_estimate(self, receiver_id: int, now: float,
                       local_estimate: TargetEstimate | None = None) -> TargetEstimate:
        """Compatibility alias; fusion is active only in age_uncertainty mode."""
        return self.receiver_estimate(receiver_id, now, local_estimate)

    def statistics(self) -> dict[str, float]:
        return {"sent": float(self.sent), "dropped": float(self.dropped),
                "delivered": float(self.delivered), "duplicates": float(self.duplicates),
                "expired": float(self.expired), "out_of_order": float(self.out_of_order),
                "realized_packet_loss": self.dropped / self.sent if self.sent else 0.0,
                "delay_mean_steps": float(np.mean(self.delay_samples)) if self.delay_samples else 0.0,
                "delay_max_steps": float(max(self.delay_samples, default=0)),
                "message_age_mean_seconds": float(np.mean(self.observation_age_samples)) if self.observation_age_samples else None,
                "message_age_p95_seconds": float(np.percentile(self.observation_age_samples,95)) if self.observation_age_samples else None,
                "transport_delay_mean_seconds": float(np.mean(self.transport_delay_samples)) if self.transport_delay_samples else None}


def predict_estimate(estimate: TargetEstimate, now: float, acceleration_noise: float = 0.35) -> TargetEstimate:
    """Predict an estimate to ``now`` while preserving its original ``t_obs``."""
    if not estimate.valid:
        return TargetEstimate.unknown(now, estimate.source_agent_id)
    dt = float(now) - estimate.state_time
    if dt < -1e-9:
        raise ValueError("prediction time precedes estimate state time")
    if dt <= 0:
        return estimate
    f = np.asarray([[1., 0., dt, 0.], [0., 1., 0., dt],
                    [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float64)
    q = ConstantVelocityTargetEstimator._process_noise(dt, float(acceleration_noise))
    return replace(estimate, state=f @ estimate.state, covariance=f @ estimate.covariance @ f.T + q,
                   state_time=float(now))


def fuse_estimates(estimates: Iterable[TargetEstimate], now: float,
                   acceleration_noise: float = 0.35,
                   age_decay_seconds: float = 12.0) -> TargetEstimate:
    """Covariance-intersection fusion for unknown inter-track correlation.

    Each source is a recursive Kalman track, so received estimates cannot be
    treated as independent raw observations. Convex information weights give a
    conservative covariance even for correlated/repeatedly forwarded tracks.
    """
    unique: dict[int, TargetEstimate] = {}
    for item in estimates:
        current = unique.get(item.source_agent_id)
        if item.valid and (current is None or item.t_obs > current.t_obs):
            unique[item.source_agent_id] = item
    items = [predict_estimate(item, now, acceleration_noise) for item in unique.values()]
    if not items:
        return TargetEstimate.unknown(now)
    precision = np.zeros((4, 4), dtype=np.float64)
    weighted = np.zeros(4, dtype=np.float64)
    confidences, source_ids, measurement_ids, obs_times = [], [], [], []
    if not np.isfinite(age_decay_seconds) or age_decay_seconds <= 0:
        raise ValueError("age_decay_seconds must be positive")
    # q_i = source_quality_i exp(-AoI_i / tau). C_i = predicted_P_i / q_i.
    # w_i is normalized inverse trace(C_i); CI retains unknown correlation.
    source_quality = [float(np.clip(item.confidence, 1e-3, 1.0))
                      * np.exp(-max(0., float(now)-item.t_obs)/age_decay_seconds) for item in items]
    covariances = [item.covariance / max(quality, 1e-6) + np.eye(4)*1e-9
                   for item, quality in zip(items, source_quality)]
    if len(items) == 1:
        # One-source covariance intersection has weight 1: inv(inv(C))=C
        # and C @ inv(C) @ x=x. Keep AoI inflation and all provenance.
        item = items[0]
        return TargetEstimate(True, item.state.copy(), covariances[0], item.t_obs,
            float(now), float(source_quality[0]), item.source_agent_id,
            item.source_measurement_id, (item.source_agent_id,),
            (item.source_measurement_id,), (item.t_obs,), (1.,), 'age_uncertainty')
    quality = np.asarray([1. / max(np.trace(cov), 1e-9) for cov in covariances])
    weights = quality / quality.sum()
    for item, covariance, weight, confidence in zip(items, covariances, weights, source_quality):
        information = np.linalg.inv(covariance)
        precision += weight * information
        weighted += weight * information @ item.state
        confidences.append(confidence)
        source_ids.append(item.source_agent_id)
        measurement_ids.append(item.source_measurement_id)
        obs_times.append(item.t_obs)
    covariance = np.linalg.inv(precision)
    state = covariance @ weighted
    # Scalar AoI describes the newest contribution, never the whole fused set.
    # Every contributing time and weight remains available for audit/logging.
    newest = int(np.argmax(obs_times))
    return TargetEstimate(True, state, covariance, max(obs_times), float(now),
                          float(np.average(confidences, weights=weights)),
                          source_ids[newest], measurement_ids[newest],
                          tuple(source_ids), tuple(measurement_ids), tuple(obs_times),
                          tuple(float(weight) for weight in weights), "age_uncertainty")


def actor_target_features(estimate: TargetEstimate, observer: ShipState, now: float) -> np.ndarray:
    """Return mask + body-frame estimate fields; unknown targets are all zero."""
    if not estimate.valid:
        return np.zeros(8, dtype=np.float32)
    # Receiver processing already selected hold/CV/I2. Never silently upgrade Base.
    predicted = estimate
    c, s = np.cos(-observer.heading), np.sin(-observer.heading)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    position = rotation @ (predicted.position - np.asarray(observer.pos, dtype=np.float64))
    velocity = rotation @ (predicted.velocity - observer.velocity)
    age = max(0.0, float(now) - estimate.t_obs)
    return np.asarray([1.0, position[0], position[1], velocity[0], velocity[1],
                       np.trace(predicted.covariance[:2, :2]), age,
                       float(np.clip(estimate.confidence, 0.0, 1.0))], dtype=np.float32)


@dataclass
class Detection:
    """Compatibility name for a local range-bearing target measurement."""

    measurement: TargetMeasurement

    @property
    def observer_index(self) -> int:
        return self.measurement.observer_index

    @property
    def source_measurement_id(self) -> str:
        return self.measurement.source_measurement_id

    @property
    def t_obs(self) -> float:
        return self.measurement.t_obs


class LocalTargetSensor(RangeBearingSensor):
    """Environment-facing sensor with one observer-bound ``measure`` call."""

    def __init__(self, config: Mapping[str, float] | None = None, seed: int = 0,
                 observer_index: int = 0) -> None:
        super().__init__(config, seed)
        self.observer_index = int(observer_index)

    def measure(self, observer_state: ShipState, red_truth: ShipState,
                t_obs: float) -> Detection | None:
        result = super().measure(self.observer_index, observer_state, red_truth, t_obs)
        return None if result is None else Detection(result)


class TargetEstimator(ConstantVelocityTargetEstimator):
    """Environment-facing estimator API."""

    def update_detection(self, detection: Detection | TargetMeasurement) -> TargetEstimate:
        measurement = detection.measurement if isinstance(detection, Detection) else detection
        return self.update(measurement)


class TargetMessageBus(TargetObservationNetwork):
    """Seconds-based facade for the step-based deterministic message network."""

    def advance(self, now: float) -> dict[int, list[TargetMessage]]:
        step = int(round(float(now) / self.dt))
        return {receiver: self.receive(receiver, step) for receiver in range(self.agent_count)}

    def broadcast(self, sender_index: int, estimate: TargetEstimate,
                  t_send: float) -> int:
        step = int(round(float(t_send) / self.dt))
        if abs(step * self.dt - float(t_send)) > 1e-6:
            raise ValueError("t_send must lie on the configured communication step grid")
        return self.send(sender_index, estimate, step)

    def latest(self, receiver_index: int, now: float,
               local_estimate: TargetEstimate | None = None) -> TargetEstimate:
        return self.receiver_estimate(receiver_index, now, local_estimate)

