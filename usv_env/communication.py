from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .types import ShipState
from .target_tracking import TargetMessage, TargetMessageBus


@dataclass(frozen=True)
class Packet:
    sender_index: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    yaw_rate: float
    surfaced: bool
    send_step: int
    delivery_step: int


class CommunicationNetwork:
    """Directed peer-state links; ablations never alter the target-message bus."""

    def __init__(self, agent_count: int, config: Mapping[str, Any], seed: int) -> None:
        self.agent_count, self.config = int(agent_count), dict(config)
        self.peer_mode = str(self.config.get("peer_mode", "normal"))
        if self.peer_mode not in ("normal", "disabled", "delayed", "shuffled"):
            raise ValueError("peer_mode must be normal, disabled, delayed or shuffled")
        self.peer_extra_delay_steps = self.config.get("peer_extra_delay_steps", 10)
        if type(self.peer_extra_delay_steps) is not int or self.peer_extra_delay_steps < 0:
            raise ValueError("peer_extra_delay_steps must be a nonnegative integer")
        for key in ("enabled", "initial_sync"):
            if type(self.config.get(key, True)) is not bool:
                raise ValueError(f"communication.{key} must be boolean")
        self.enabled = self.config.get("enabled", True)
        delay = self.config.get("delay_steps", (0, 3))
        if len(delay) != 2 or any(type(x) is not int or x < 0 for x in delay) or delay[0] > delay[1]:
            raise ValueError("delay_steps must be ordered nonnegative integers")
        age_cap = self.config.get("age_cap_steps", 10)
        if type(age_cap) is not int or age_cap <= 0: raise ValueError("age_cap_steps must be positive integer")
        loss = self.config.get("target_loss_range", (0., .1))
        if len(loss) != 2 or not all(np.isfinite(loss)) or not 0 <= loss[0] <= loss[1] < 1:
            raise ValueError("target_loss_range must satisfy 0 <= lo <= hi < 1")
        self.rng = np.random.default_rng(int(seed))
        self.target_loss = float(self.rng.uniform(*self.config.get("target_loss_range", (0.0, 0.10))))
        self.model = str(self.config.get("loss_model", "gilbert_elliott"))
        self.delay_range = tuple(map(int, self.config.get("delay_steps", (0, 3))))
        self.age_cap_steps = int(self.config.get("age_cap_steps", 10))
        self.bad_to_good = float(self.config.get("bad_to_good", 0.5))
        if not np.isfinite(self.bad_to_good) or not 0 < self.bad_to_good <= 1:
            raise ValueError("bad_to_good must be in (0,1]")
        if self.model == "gilbert_elliott" and loss[1] * self.bad_to_good / (1-loss[1]) > 1:
            raise ValueError("requested GE stationary loss implies p_GB > 1")
        if not self.enabled:
            self.target_loss = 0.
            self.delay_range = (0, 0)
        if self.target_loss <= 0.0:
            self.good_to_bad = 0.0
        elif self.target_loss >= 1.0:
            self.good_to_bad = 1.0
        else:
            self.good_to_bad = self.target_loss * self.bad_to_good / (1.0 - self.target_loss)
        self.bad = np.zeros((agent_count, agent_count), dtype=bool)
        self.links = tuple((r, s) for r in range(agent_count) for s in range(agent_count) if r != s)
        self.active_mask = np.ones(self.agent_count, dtype=bool)
        self.queues = {link: [] for link in self.links}
        self.cache: dict[tuple[int, int], Packet] = {}
        self.fresh = np.zeros((agent_count, agent_count), dtype=bool)
        self.sent = self.dropped = self.delivered = self.out_of_order = 0
        self.delays: list[int] = []
        self.ages: list[int] = []
        self.current_burst = np.zeros((agent_count, agent_count), dtype=np.int64)
        self.bursts: list[int] = []

    @staticmethod
    def _packet(sender: int, state: ShipState, send: int, delivery: int) -> Packet:
        return Packet(sender, state.pos.copy(), state.velocity.copy(), state.heading, state.yaw_rate,
                      state.surfaced, int(send), int(delivery))

    def _set_active_mask(self, active_mask) -> None:
        mask = np.asarray(active_mask)
        if mask.shape != (self.agent_count,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("peer active_mask must be a binary agent_count vector")
        self.active_mask = mask.astype(bool, copy=True)
        for link in self.links:
            if not all(self.active_mask[i] for i in link):
                self.cache.pop(link, None)
                self.queues[link].clear()

    def reset(self, states: list[ShipState], step: int = 0, active_mask=None) -> None:
        self._set_active_mask(np.ones(self.agent_count) if active_mask is None else active_mask)
        self.cache.clear(); self.fresh.fill(False)
        self.bad.fill(False)
        self.current_burst.fill(0)
        self.sent = self.dropped = self.delivered = self.out_of_order = 0
        self.delays.clear(); self.ages.clear(); self.bursts.clear()
        for receiver, sender in self.links:
            self.queues[(receiver, sender)].clear()
            if (self.enabled and all(self.active_mask[i] for i in (receiver, sender))
                    and self.peer_mode != "disabled" and self.config.get("initial_sync", False)):
                self.cache[(receiver, sender)] = self._packet(sender, states[sender], step, step)

    def _is_drop(self, receiver: int, sender: int) -> bool:
        if not self.enabled: return False
        if self.model == "independent_bernoulli":
            return bool(self.rng.random() < self.target_loss)
        if self.model != "gilbert_elliott":
            raise ValueError("unknown communication loss model")
        drop = bool(self.bad[receiver, sender])
        transition = self.bad_to_good if drop else self.good_to_bad
        if self.rng.random() < transition:
            self.bad[receiver, sender] = not drop
        return drop

    def step(self, states: list[ShipState], step: int, active_mask=None) -> None:
        if active_mask is not None:
            self._set_active_mask(active_mask)
        self.fresh.fill(False)
        if not self.enabled or self.peer_mode == "disabled":
            self.cache.clear()
            return
        for receiver, sender in self.links:
            if not (self.active_mask[receiver] and self.active_mask[sender]):
                continue
            self.sent += 1
            if self._is_drop(receiver, sender):
                self.dropped += 1; self.current_burst[receiver, sender] += 1; continue
            burst = int(self.current_burst[receiver, sender])
            if burst: self.bursts.append(burst); self.current_burst[receiver, sender] = 0
            delay = int(self.rng.integers(self.delay_range[0], self.delay_range[1] + 1))
            if self.peer_mode == "delayed" or (self.peer_mode == "shuffled" and step % 2 == 0):
                delay += self.peer_extra_delay_steps
            self.delays.append(delay)
            self.queues[(receiver, sender)].append(self._packet(sender, states[sender], step, step + delay))
        for link, queue in self.queues.items():
            due = sorted((p for p in queue if p.delivery_step <= step), key=lambda p: p.send_step)
            queue[:] = [p for p in queue if p.delivery_step > step]
            for packet in due:
                if link not in self.cache or packet.send_step > self.cache[link].send_step:
                    self.cache[link] = packet; self.fresh[link] = True; self.delivered += 1
                else:
                    self.out_of_order += 1
        self.ages.extend(max(step-p.send_step, 0) for p in self.cache.values())

    def peer_state(self, receiver: int, sender: int, template: ShipState) -> ShipState:
        packet = self.cache.get((receiver, sender))
        if packet is None:
            return ShipState(np.zeros(2, np.float32), 0., 0., 0., template.unit_type, False, 0.)
        c, s = np.cos(packet.heading), np.sin(packet.heading)
        surge = float(c * packet.velocity[0] + s * packet.velocity[1])
        sway = float(-s * packet.velocity[0] + c * packet.velocity[1])
        return ShipState(packet.position.copy(), surge, packet.heading, packet.yaw_rate,
                         template.unit_type, packet.surfaced, sway)

    def link_features(self, receiver: int, sender: int, step: int) -> tuple[float, float]:
        packet = self.cache.get((receiver, sender))
        if packet is None: return 0., 1.
        age = max(step - packet.send_step, 0)
        return float(self.fresh[receiver, sender]), float(np.clip(age / self.age_cap_steps, 0.0, 1.0))

    def statistics(self) -> dict[str, float | str]:
        bursts = self.bursts + [int(x) for x in self.current_burst.flat if x > 0]
        return {"peer_mode": self.peer_mode, "out_of_order_rejected": float(self.out_of_order),
                "configured_target_loss": self.target_loss,
                "realized_packet_loss": self.dropped / self.sent if self.sent else 0.0,
                "delay_mean_steps": float(np.mean(self.delays)) if self.delays else 0.0,
                "delay_p95_steps": float(np.percentile(self.delays, 95)) if self.delays else 0.0,
                "delay_max_steps": float(max(self.delays, default=0)),
                "message_age_mean_steps": float(np.mean(self.ages)) if self.ages else 0.0,
                "message_age_p95_steps": float(np.percentile(self.ages, 95)) if self.ages else 0.0,
                "message_age_max_steps": float(max(self.ages, default=0)),
                "fresh_message_fraction": self.delivered / self.sent if self.sent else 0.0,
                "burst_count": float(len(bursts)), "mean_burst_length": float(np.mean(bursts)) if bursts else 0.0,
                "max_burst_length": float(max(bursts, default=0))}
