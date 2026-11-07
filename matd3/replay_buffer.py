from __future__ import annotations

import math
import sys

import numpy as np


ROLE_ACTIVE_DIM = 2
PHASE_COUNT = 4
_PHASE_BITS = 2
_MIB = 1024**2
TERMINAL_REASON_IDS = {
    "": 0,
    "success": 1,
    "collision": 2,
    "out_of_bounds": 3,
    "breakthrough": 4,
    "reef_collision": 5,
    "timeout": 6,
    "scenario_anomaly": 7,
    "numerical_error": 8,
}
TERMINAL_REASON_NAMES = {
    value: key for key, value in TERMINAL_REASON_IDS.items()
}


class ReplayBuffer:
    """Preallocated NumPy replay with n-step episode-safe bootstrapping."""


    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        seed: int = 0,
        n_step_return: int = 1,
        gamma: float = 0.99,
        downstream_gamma: float | None = None,
        prioritized: bool = False,
        alpha: float = 0.6,
        priority_eps: float = 1.0e-6,
        uniform_fraction: float = 0.0,
        surface_count: int = 2,
    ) -> None:
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.surface_count = int(surface_count)
        if not 0 < self.surface_count < self.num_agents:
            raise ValueError("surface_count must leave at least one agent in each role")
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_step_return = int(n_step_return)
        self.gamma = float(gamma)
        self.downstream_gamma = float(
            self.gamma if downstream_gamma is None else downstream_gamma
        )
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)
        self.uniform_fraction = float(uniform_fraction)
        if min(
            self.capacity,
            self.num_agents,
            self.obs_dim,
            self.state_dim,
            self.action_dim,
        ) <= 0:
            raise ValueError("replay dimensions and capacity must be positive")
        if self.num_agents > 8 - _PHASE_BITS:
            raise ValueError("packed replay activity supports at most six agents")
        if not 1 <= self.n_step_return <= np.iinfo(np.uint8).max:
            raise ValueError("n_step_return must be in [1, 255]")
        if not np.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        if not np.isfinite(self.downstream_gamma) or not 0.0 < self.downstream_gamma <= 1.0:
            raise ValueError("downstream_gamma must be finite and in (0, 1]")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not np.isfinite(self.priority_eps) or self.priority_eps <= 0.0:
            raise ValueError("priority_eps must be finite and positive")
        if not np.isfinite(self.uniform_fraction) or not 0.0 <= self.uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be finite and in [0, 1]")

        self.obs = np.zeros(
            (self.capacity, self.num_agents, self.obs_dim), dtype=np.float32
        )
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(
            (self.capacity, self.num_agents, self.action_dim), dtype=np.float32
        )
        self.nominal_actions = np.zeros_like(self.actions)
        # Context describes the exact state at each endpoint. N-step reduction
        # uses the first current context and the last REAL next context.
        self.context_dim = 4 * self.num_agents + 2
        self.current_context = np.zeros((self.capacity, self.context_dim), np.float32)
        self.next_context = np.zeros_like(self.current_context)
        self.next_phase_ids = np.zeros((self.capacity, 1), np.uint8)
        self.agent_rewards = np.zeros(
            (self.capacity, self.num_agents), dtype=np.float32
        )
        self.downstream_rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.capacity, self.num_agents, self.obs_dim), dtype=np.float32
        )
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.terminated = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.truncated = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.timeouts = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.n_steps = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.terminal_reason_ids = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.surface_active = np.zeros(
            (self.capacity, self.surface_count), dtype=np.uint8
        )
        # Phase (low two bits) and the per-agent activity mask share one byte
        # while both masks remain explicit through sample().
        self.phase_ids = np.zeros((self.capacity, 1), dtype=np.uint8)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))
        self._pending: list[dict[str, object]] = []
        self.max_raw_priority = 1.0
        self._tree_leaf_count = 1 << max(0, (self.capacity - 1).bit_length())
        self.raw_priorities = (
            np.zeros(self.capacity, dtype=np.float64) if self.prioritized else None
        )
        self.sum_tree = (
            np.zeros(2 * self._tree_leaf_count, dtype=np.float64)
            if self.prioritized
            else None
        )
        self.min_tree = (
            np.full(2 * self._tree_leaf_count, np.inf, dtype=np.float64)
            if self.prioritized
            else None
        )


    def __len__(self) -> int:
        return self.size


    @property
    def storage_arrays(self) -> dict[str, np.ndarray]:
        return {
            "obs": self.obs,
            "states": self.states,
            "actions": self.actions,
            "nominal_actions": self.nominal_actions,
            "current_context": self.current_context,
            "next_context": self.next_context,
            "next_phase_ids": self.next_phase_ids,
            "agent_rewards": self.agent_rewards,
            "downstream_rewards": self.downstream_rewards,
            "next_obs": self.next_obs,
            "next_states": self.next_states,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "timeouts": self.timeouts,
            "n_steps": self.n_steps,
            "terminal_reason_ids": self.terminal_reason_ids,
            "surface_active": self.surface_active,
            "phase_ids": self.phase_ids,
        }


    @property
    def agent_active_mask(self) -> np.ndarray:
        """Decoded per-transition agent mask (the resident form is bit-packed)."""

        return self._unpack_phase_activity(self.phase_ids)[1]


    @property
    def role_active_mask(self) -> np.ndarray:
        """Decoded per-transition role mask derived from agent activity."""

        return self._role_masks_from_agents(self.agent_active_mask)


    @property
    def array_payload_bytes(self) -> int:
        return int(sum(array.nbytes for array in self.storage_arrays.values()))


    @property
    def python_object_overhead_bytes(self) -> int:
        array_headers = sum(
            max(0, sys.getsizeof(array) - int(array.nbytes))
            for array in self.storage_arrays.values()
        )
        scalar_headers = sum(
            sys.getsizeof(getattr(self, name))
            for name in (
                "capacity",
                "num_agents",
                "obs_dim",
                "state_dim",
                "action_dim",
                "n_step_return",
                "gamma",
                "downstream_gamma",
                "index",
                "size",
            )
        )
        return int(
            sys.getsizeof(self)
            + sys.getsizeof(self.__dict__)
            + sys.getsizeof(self.rng)
            + array_headers
            + scalar_headers
        )


    @property
    def estimated_bytes(self) -> int:
        return self.array_payload_bytes + self.python_object_overhead_bytes


    @property
    def bytes_per_transition(self) -> int:
        return self.array_payload_bytes // self.capacity


    @property
    def theoretical_memory_mb(self) -> float:
        return self.estimated_bytes / _MIB


    def memory_budget_report(
        self,
        memory_budget_mb: float,
        warn_memory_ratio: float,
        available_memory_bytes: int | None = None,
    ) -> dict[str, int | float | bool | None]:
        budget_mb = float(memory_budget_mb)
        warning_ratio = float(warn_memory_ratio)
        if not np.isfinite(budget_mb) or budget_mb <= 0.0:
            raise ValueError("memory_budget_mb must be finite and positive")
        if not np.isfinite(warning_ratio) or not 0.0 < warning_ratio <= 1.0:
            raise ValueError("warn_memory_ratio must be in (0, 1]")
        if available_memory_bytes is not None and int(available_memory_bytes) < 0:
            raise ValueError("available_memory_bytes cannot be negative")
        estimated_mb = self.theoretical_memory_mb
        available_mb = (
            None
            if available_memory_bytes is None
            else int(available_memory_bytes) / _MIB
        )
        return {
            "capacity": self.capacity,
            "bytes_per_transition": self.bytes_per_transition,
            "array_payload_bytes": self.array_payload_bytes,
            "python_object_overhead_bytes": self.python_object_overhead_bytes,
            "estimated_bytes": self.estimated_bytes,
            "theoretical_memory_mb": estimated_mb,
            "memory_budget_mb": budget_mb,
            "warn_memory_ratio": warning_ratio,
            "warning_threshold_mb": budget_mb * warning_ratio,
            "within_budget": bool(estimated_mb <= budget_mb),
            "over_budget": bool(estimated_mb > budget_mb),
            "warning": bool(estimated_mb >= budget_mb * warning_ratio),
            "available_memory_mb": available_mb,
            "fits_available_memory": (
                None if available_mb is None else bool(estimated_mb <= available_mb)
            ),
            "python_object_overhead_included": True,
        }


    def _add_transition(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        agent_reward: np.ndarray,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        *,
        terminated: bool,
        truncated: bool,
        timeout: bool,
        terminal_reason: str | None = None,
        surface_active: bool | np.ndarray | None = None,
        role_active_mask: np.ndarray | None = None,
        agent_active_mask: np.ndarray | None = None,
        phase_id: int = 0,
        downstream_reward: float | None = None,
        current_context: dict | None = None,
        next_context: dict | None = None,
        nominal_action: np.ndarray | None = None,
    ) -> None:
        transition = self._prepare_transition(
            obs=obs,
            state=state,
            action=action,
            agent_reward=agent_reward,
            next_obs=next_obs,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
            timeout=timeout,
            terminal_reason=terminal_reason,
            surface_active=surface_active,
            role_active_mask=role_active_mask,
            agent_active_mask=agent_active_mask,
            phase_id=phase_id,
            downstream_reward=downstream_reward,
        )
        current, phase = self._prepare_context(current_context, phase_id)
        following, next_phase = self._prepare_context(next_context, phase_id)
        if terminal_reason in ("collision", "reef_collision", "out_of_bounds", "scenario_anomaly", "numerical_error"):
            current[-2] = 0.0
        transition.update(current_context=current, next_context=following,
                          phase_id=phase, next_phase_id=next_phase,
                          nominal_action=self._array(action if nominal_action is None else nominal_action,
                              (self.num_agents, self.action_dim), "nominal_action").copy())
        self._pending.append(transition)
        if len(self._pending) >= self.n_step_return:
            self._commit_window(self._pending[: self.n_step_return])
            self._pending.pop(0)
        if bool(transition["terminated"]) or bool(transition["truncated"]):
            while self._pending:
                self._commit_window(self._pending)
                self._pending.pop(0)


    def _prepare_transition(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        agent_reward: np.ndarray,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        timeout: bool,
        terminal_reason: str | None,
        surface_active: bool | np.ndarray | None,
        role_active_mask: np.ndarray | None,
        agent_active_mask: np.ndarray | None,
        phase_id: int,
        downstream_reward: float | None,
    ) -> dict[str, object]:
        obs_value = self._array(obs, (self.num_agents, self.obs_dim), "obs")
        state_value = self._array(state, (self.state_dim,), "state")
        action_value = self._array(
            action, (self.num_agents, self.action_dim), "action"
        )
        reward_value = self._array(
            agent_reward, (self.num_agents,), "agent_reward"
        )
        next_obs_value = self._array(
            next_obs, (self.num_agents, self.obs_dim), "next_obs"
        )
        next_state_value = self._array(
            next_state, (self.state_dim,), "next_state"
        )
        terminated_value = self._binary_scalar(terminated, "terminated")
        truncated_value = self._binary_scalar(truncated, "truncated")
        timeout_value = self._binary_scalar(timeout, "timeout")
        reason = "" if terminal_reason is None else str(terminal_reason)
        if reason not in TERMINAL_REASON_IDS:
            raise ValueError(f"unknown terminal_reason: {reason}")
        terminal_reason_value = np.uint8(TERMINAL_REASON_IDS[reason])
        (
            surface_active_value,
            role_active_value,
            agent_active_value,
        ) = self._activity_masks(
            surface_active=surface_active,
            role_active_mask=role_active_mask,
            agent_active_mask=agent_active_mask,
        )
        reward_value = reward_value * agent_active_value.astype(np.float32)
        action_value = action_value * agent_active_value[:, None].astype(np.float32)
        if downstream_reward is None:
            submarine_mask = agent_active_value[self.surface_count:].astype(np.float32)
            downstream_value = float(
                np.sum(reward_value[self.surface_count:] * submarine_mask)
                / max(float(np.sum(submarine_mask)), 1.0)
            )
        else:
            downstream_value = float(downstream_reward)
        if not math.isfinite(downstream_value):
            raise ValueError("downstream_reward must be finite")
        phase_value = int(phase_id)
        if not 0 <= phase_value < PHASE_COUNT:
            raise ValueError(f"phase_id must be in [0, {PHASE_COUNT - 1}]")
        if terminated_value and truncated_value:
            raise ValueError("terminated and truncated cannot both be true")
        if timeout_value and not (terminated_value or truncated_value):
            raise ValueError("timeout requires an episode boundary")
        if reason == "timeout" and not timeout_value:
            raise ValueError("terminal_reason=timeout requires timeout=true")
        if reason and reason != "timeout" and not terminated_value:
            raise ValueError(
                "non-timeout terminal_reason requires terminated=true"
            )

        return {
            "obs": obs_value.copy(),
            "state": state_value.copy(),
            "action": action_value.copy(),
            "agent_reward": reward_value.copy(),
            "downstream_reward": downstream_value,
            "next_obs": next_obs_value.copy(),
            "next_state": next_state_value.copy(),
            "terminated": terminated_value,
            "truncated": truncated_value,
            "timeout": timeout_value,
            "terminal_reason_id": terminal_reason_value,
            "surface_active": surface_active_value.copy(),
            "agent_active_mask": agent_active_value.copy(),
            "phase_id": phase_value,
        }


    def _commit_window(self, window: list[dict[str, object]]) -> None:
        if not window or len(window) > self.n_step_return:
            raise ValueError("n-step replay window has an invalid length")
        first = window[0]
        last = window[-1]
        reward_value = np.zeros(self.num_agents, dtype=np.float64)
        downstream_value = 0.0
        for step, transition in enumerate(window):
            reward_value += (self.gamma**step) * np.asarray(
                transition["agent_reward"], dtype=np.float64
            )
            downstream_value += (self.downstream_gamma**step) * float(
                transition["downstream_reward"]
            )

        row = self.index
        self.obs[row] = np.asarray(first["obs"], dtype=np.float32)
        self.states[row] = np.asarray(first["state"], dtype=np.float32)
        self.actions[row] = np.asarray(first["action"], dtype=np.float32)
        self.nominal_actions[row] = first["nominal_action"]
        self.current_context[row] = first["current_context"]
        self.next_context[row] = last["next_context"]
        self.next_phase_ids[row, 0] = last["next_phase_id"]
        self.agent_rewards[row] = reward_value.astype(np.float32)
        self.downstream_rewards[row, 0] = np.float32(downstream_value)
        self.next_obs[row] = np.asarray(last["next_obs"], dtype=np.float32)
        self.next_states[row] = np.asarray(last["next_state"], dtype=np.float32)
        self.terminated[row, 0] = np.uint8(last["terminated"])
        self.truncated[row, 0] = np.uint8(last["truncated"])
        self.timeouts[row, 0] = np.uint8(last["timeout"])
        self.n_steps[row, 0] = np.uint8(len(window))
        self.terminal_reason_ids[row, 0] = np.uint8(last["terminal_reason_id"])
        self.surface_active[row] = np.asarray(
            first["surface_active"], dtype=np.uint8
        )
        self.phase_ids[row, 0] = self._pack_phase_activity(
            int(first["phase_id"]),
            np.asarray(first["agent_active_mask"], dtype=np.uint8),
        )
        if self.prioritized:
            assert self.raw_priorities is not None
            self.raw_priorities[row] = self.max_raw_priority
            self._set_priority_mass(
                row, (self.max_raw_priority + self.priority_eps) ** self.alpha
            )
        self.index = (row + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size: int, *, beta: float = 0.0) -> dict[str, np.ndarray]:
        if self.size <= 0:
            raise ValueError("replay buffer is empty")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        beta_value = float(beta)
        if not np.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
            raise ValueError("beta must be finite and in [0, 1]")
        if self.prioritized:
            indices, probabilities, is_weights = self._prioritized_indices(
                batch_size, beta_value
            )
        else:
            # Base parity: preserve the exact legacy RNG call and do not draw
            # any other replay randomness in the uniform branch.
            indices = self.rng.integers(0, self.size, size=batch_size)
            probabilities = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
            is_weights = np.ones(batch_size, dtype=np.float32)
        return self._sample_rows(indices, probabilities, is_weights)


    def sample_uniform(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample an unbiased batch even when the resident replay uses PER."""

        if self.size <= 0:
            raise ValueError("replay buffer is empty")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.rng.integers(0, self.size, size=batch_size)
        probabilities = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
        is_weights = np.ones(batch_size, dtype=np.float32)
        return self._sample_rows(indices, probabilities, is_weights)


    def _sample_rows(
        self,
        indices: np.ndarray,
        probabilities: np.ndarray,
        is_weights: np.ndarray,
    ) -> dict[str, np.ndarray]:
        terminated = self.terminated[indices]
        truncated = self.truncated[indices]
        timeouts = self.timeouts[indices]
        n_steps = self.n_steps[indices]
        reason_ids = self.terminal_reason_ids[indices]
        phase_ids, agent_active = self._unpack_phase_activity(self.phase_ids[indices])
        role_active = self._role_masks_from_agents(agent_active)
        bootstrap_masks = self.compute_bootstrap_masks(
            terminated,
            truncated,
            timeouts,
        )
        return {
            "obs": self.obs[indices],
            "states": self.states[indices],
            "actions": self.actions[indices],
            "nominal_actions": self.nominal_actions[indices],
            "current_context": self.current_context[indices],
            "next_context": self.next_context[indices],
            "next_phase_ids": self.next_phase_ids[indices],
            "agent_rewards": self.agent_rewards[indices],
            "downstream_rewards": self.downstream_rewards[indices],
            "next_obs": self.next_obs[indices],
            "next_states": self.next_states[indices],
            "terminated": terminated,
            "truncated": truncated,
            "timeouts": timeouts,
            "n_steps": n_steps,
            "terminal_reason_ids": reason_ids,
            "terminal_reasons": np.asarray(
                [TERMINAL_REASON_NAMES[int(value)] for value in reason_ids[:, 0]],
                dtype=f"<U{max(map(len, TERMINAL_REASON_IDS))}",
            ).reshape(-1, 1),
            "surface_active": self.surface_active[indices],
            "role_active_mask": role_active,
            "agent_active_mask": agent_active,
            "phase_ids": phase_ids,
            "bootstrap_masks": bootstrap_masks,
            "bootstrap_discounts": self.compute_bootstrap_discounts(
                n_steps,
                bootstrap_masks,
                self.gamma,
            ),
            "downstream_bootstrap_discounts": self.compute_bootstrap_discounts(
                n_steps,
                bootstrap_masks,
                self.downstream_gamma,
            ),
            "indices": np.asarray(indices, dtype=np.int64),
            "sampling_probabilities": probabilities.reshape(-1, 1),
            "is_weights": is_weights.reshape(-1, 1),
        }


    def _set_priority_mass(self, index: int, mass: float) -> None:
        if not self.prioritized or self.sum_tree is None or self.min_tree is None:
            raise RuntimeError("prioritized replay is disabled")
        if not math.isfinite(mass) or mass <= 0.0:
            raise RuntimeError("priority tree mass must be finite and positive")
        node = self._tree_leaf_count + int(index)
        self.sum_tree[node] = mass
        self.min_tree[node] = mass
        node //= 2
        while node:
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.min_tree[node] = min(self.min_tree[2 * node], self.min_tree[2 * node + 1])
            if not math.isfinite(float(self.sum_tree[node])):
                raise RuntimeError("sum-tree mass contains NaN/Inf")
            node //= 2


    def _prefix_index(self, value: float) -> int:
        assert self.sum_tree is not None
        node = 1
        while node < self._tree_leaf_count:
            left = 2 * node
            if value < self.sum_tree[left]:
                node = left
            else:
                value -= self.sum_tree[left]
                node = left + 1
        index = node - self._tree_leaf_count
        if not 0 <= index < self.size:
            raise RuntimeError("sum-tree prefix search selected an invalid replay row")
        return index


    def _prioritized_indices(
        self, batch_size: int, beta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.sum_tree is not None and self.min_tree is not None
        total_mass = float(self.sum_tree[1])
        min_mass = float(self.min_tree[1])
        if not math.isfinite(total_mass) or total_mass <= 0.0:
            raise RuntimeError("sum-tree total mass must be finite and positive")
        if not math.isfinite(min_mass) or min_mass <= 0.0:
            raise RuntimeError("min-tree valid mass must be finite and positive")
        uniform_count = int(round(batch_size * self.uniform_fraction))
        uniform_count = min(max(uniform_count, 0), batch_size)
        priority_count = batch_size - uniform_count
        uniform_indices = self.rng.integers(0, self.size, size=uniform_count)
        draws = self.rng.random(priority_count) * total_mass
        priority_indices = np.asarray(
            [self._prefix_index(float(x)) for x in draws], dtype=np.int64
        )
        indices = np.concatenate((uniform_indices, priority_indices)).astype(
            np.int64, copy=False
        )
        if batch_size > 1:
            indices = indices[self.rng.permutation(batch_size)]
        leaf = self._tree_leaf_count + indices
        prioritized_probabilities = self.sum_tree[leaf] / total_mass
        probabilities = (
            self.uniform_fraction / self.size
            + (1.0 - self.uniform_fraction) * prioritized_probabilities
        )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
            raise RuntimeError("sampling probabilities must be finite and positive")
        if beta == 0.0:
            weights = np.ones(batch_size, dtype=np.float64)
        else:
            raw = np.power(self.size * probabilities, -beta)
            p_min = (
                self.uniform_fraction / self.size
                + (1.0 - self.uniform_fraction) * (min_mass / total_mass)
            )
            global_max = float((self.size * p_min) ** (-beta))
            weights = raw / (global_max + self.priority_eps)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or np.any(weights > 1.0 + 1e-12):
            raise RuntimeError("importance-sampling weights must be finite and in (0, 1]")
        return indices, probabilities.astype(np.float32), weights.astype(np.float32)


    def update_priorities(
        self, indices: np.ndarray, raw_priorities: np.ndarray
    ) -> None:
        if not self.prioritized or self.raw_priorities is None:
            raise RuntimeError("prioritized replay is disabled")
        index_values = np.asarray(indices, dtype=np.int64).reshape(-1)
        priority_values = np.asarray(raw_priorities, dtype=np.float64).reshape(-1)
        if index_values.shape != priority_values.shape or index_values.size == 0:
            raise ValueError("indices and raw_priorities must be non-empty matching vectors")
        if np.any(index_values < 0) or np.any(index_values >= self.size):
            raise ValueError("priority index is outside the valid replay range")
        if not np.all(np.isfinite(priority_values)) or np.any(priority_values < 0.0):
            raise RuntimeError("raw priorities must be finite and non-negative")
        aggregated: dict[int, float] = {}
        for index, priority in zip(index_values.tolist(), priority_values.tolist()):
            aggregated[index] = max(aggregated.get(index, 0.0), priority)
        for index in sorted(aggregated):
            priority = aggregated[index]
            mass = (priority + self.priority_eps) ** self.alpha
            if not math.isfinite(mass) or mass <= 0.0:
                raise RuntimeError("priority mass must be finite and positive")
            self.raw_priorities[index] = priority
            self._set_priority_mass(index, mass)
        self.max_raw_priority = max(self.max_raw_priority, max(aggregated.values()))


    @staticmethod
    def compute_bootstrap_masks(
        terminated: np.ndarray,
        truncated: np.ndarray,
        timeouts: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(terminated)
        truncated_values = np.asarray(truncated)
        timeout_values = np.asarray(timeouts)
        if (
            values.shape != truncated_values.shape
            or values.shape != timeout_values.shape
        ):
            raise ValueError(
                "terminated, truncated, and timeouts must have matching shapes"
            )
        if values.dtype.kind not in "buif" or np.any((values != 0) & (values != 1)):
            raise ValueError("terminated must contain only 0/1")
        if truncated_values.dtype.kind not in "buif" or np.any(
            (truncated_values != 0) & (truncated_values != 1)
        ):
            raise ValueError("truncated must contain only 0/1")
        if timeout_values.dtype.kind not in "buif" or np.any(
            (timeout_values != 0) & (timeout_values != 1)
        ):
            raise ValueError("timeouts must contain only 0/1")
        if np.any((timeout_values == 1) & (truncated_values == 0) & (values == 0)):
            raise ValueError("timeout requires an episode boundary")
        # Environment timeouts end the episode and reset from an unsupported
        # episode_remaining=0 observation, so they must not bootstrap. Other
        # external truncations may still retain the value of their final state.
        episode_ending = np.maximum(
            values.astype(np.float32),
            timeout_values.astype(np.float32),
        )
        return (1.0 - episode_ending).astype(np.float32, copy=False)


    @staticmethod
    def compute_bootstrap_discounts(
        n_steps: np.ndarray,
        bootstrap_masks: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        step_values = np.asarray(n_steps)
        mask_values = np.asarray(bootstrap_masks)
        gamma_value = float(gamma)
        if step_values.shape != mask_values.shape:
            raise ValueError("n_steps and bootstrap_masks must have matching shapes")
        if step_values.dtype.kind not in "buif" or np.any(
            (step_values < 1) | (step_values > np.iinfo(np.uint8).max)
        ):
            raise ValueError("n_steps must contain integers in [1, 255]")
        if np.any(step_values != np.floor(step_values)):
            raise ValueError("n_steps must contain integers in [1, 255]")
        if mask_values.dtype.kind not in "buif" or np.any(
            (mask_values != 0) & (mask_values != 1)
        ):
            raise ValueError("bootstrap_masks must contain only 0/1")
        if not np.isfinite(gamma_value) or not 0.0 < gamma_value <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        discounts = np.power(
            gamma_value,
            step_values.astype(np.float64),
        )
        return (discounts * mask_values.astype(np.float64)).astype(np.float32)


    def clear(self) -> None:
        for array in self.storage_arrays.values():
            array.fill(0)
        if self.raw_priorities is not None:
            self.raw_priorities.fill(0.0)
        if self.sum_tree is not None:
            self.sum_tree.fill(0.0)
        if self.min_tree is not None:
            self.min_tree.fill(np.inf)
        self.max_raw_priority = 1.0
        self.index = 0
        self.size = 0
        self._pending.clear()


    def _prepare_context(self, context, fallback_phase=0):
        """Missing metadata is explicit and cannot become an I3 candidate."""
        result = np.zeros(self.context_dim, np.float32)
        if context is None:
            return result, int(fallback_phase)
        required = ("information_valid", "information_age", "information_variance", "safety_margin", "quality", "phase_id")
        if not isinstance(context, dict) or any(key not in context for key in required):
            raise ValueError("learning context requires phase, information, safety and quality metadata")
        phase = int(context["phase_id"])
        if not 0 <= phase < PHASE_COUNT:
            raise ValueError("learning context phase_id outside supported phases")
        for block, name in enumerate(required[:4]):
            values = self._array(context[name], (self.num_agents,), name)
            if name == "information_valid" and not np.isin(values, (0., 1.)).all():
                raise ValueError("information_valid must be binary")
            if name in ("information_age", "information_variance") and np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative")
            result[block*self.num_agents:(block+1)*self.num_agents] = values
        quality = float(context["quality"])
        if not np.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("history quality must be in [0,1]")
        result[-2:] = quality, 1.0
        return result, phase


    @staticmethod
    def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        ReplayBuffer._reject_tensor(value, name)
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN/Inf")
        return array


    @staticmethod
    def _binary_scalar(value: object, name: str) -> np.uint8:
        ReplayBuffer._reject_tensor(value, name)
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "buif":
            raise ValueError(f"{name} must be a bool or 0/1 scalar")
        scalar = array.item()
        if not np.isfinite(scalar) or scalar not in (0, 1, False, True):
            raise ValueError(f"{name} must be a bool or 0/1 scalar")
        return np.uint8(bool(scalar))


    def _surface_mask(self, value: object) -> np.ndarray:
        ReplayBuffer._reject_tensor(value, "surface_active")
        array = np.asarray(value)
        if array.shape == ():
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("surface_active scalar must be bool")
            return np.full(self.surface_count, bool(value), dtype=np.uint8)
        if array.shape != (self.surface_count,):
            raise ValueError(f"surface_active must be bool or a shape=({self.surface_count},) binary mask")
        if array.dtype.kind not in "buif" or not np.all(np.isfinite(array)):
            raise ValueError("surface_active must be binary")
        if np.any((array != 0) & (array != 1)):
            raise ValueError("surface_active must contain only 0/1")
        return array.astype(np.uint8, copy=False)


    def _activity_masks(
        self,
        *,
        surface_active: object | None,
        role_active_mask: object | None,
        agent_active_mask: object | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.num_agents < self.surface_count:
            raise ValueError("agent activity masks do not match configured surface count")

        explicit_surface = (
            None if surface_active is None else self._surface_mask(surface_active)
        )
        explicit_role = (
            None
            if role_active_mask is None
            else self._binary_mask(
                role_active_mask, (ROLE_ACTIVE_DIM,), "role_active_mask"
            )
        )
        explicit_agent = (
            None
            if agent_active_mask is None
            else self._binary_mask(
                agent_active_mask, (self.num_agents,), "agent_active_mask"
            )
        )

        if explicit_agent is None:
            surface = (
                np.ones(self.surface_count, dtype=np.uint8)
                if explicit_surface is None
                else explicit_surface.copy()
            )
            submarine_active = (
                True if explicit_role is None else bool(explicit_role[1])
            )
            agent = np.concatenate(
                (
                    surface,
                    np.full(
                        self.num_agents - self.surface_count,
                        submarine_active,
                        dtype=np.uint8,
                    ),
                )
            )
        else:
            agent = explicit_agent.copy()
            surface = agent[:self.surface_count].copy()
            if explicit_surface is not None and not np.array_equal(
                surface, explicit_surface
            ):
                raise ValueError(
                    "surface_active and agent_active_mask disagree for surface agents"
                )

        derived_role = np.asarray(
            (
                bool(np.any(agent[:self.surface_count])),
                bool(np.any(agent[self.surface_count:])),
            ),
            dtype=np.uint8,
        )
        if explicit_role is not None and not np.array_equal(
            derived_role, explicit_role
        ):
            raise ValueError(
                "role_active_mask must exactly match the per-agent activity mask"
            )
        return surface, derived_role, agent


    def _pack_phase_activity(
        self, phase_id: int, agent_active_mask: np.ndarray
    ) -> np.uint8:
        packed = int(phase_id)
        for index, active in enumerate(agent_active_mask):
            packed |= int(bool(active)) << (_PHASE_BITS + index)
        return np.uint8(packed)


    def _unpack_phase_activity(
        self, packed_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        packed = np.asarray(packed_values, dtype=np.uint8).reshape(-1, 1)
        phases = (packed & np.uint8(PHASE_COUNT - 1)).astype(np.uint8, copy=False)
        agents = np.concatenate(
            [
                ((packed >> np.uint8(_PHASE_BITS + index)) & np.uint8(1))
                for index in range(self.num_agents)
            ],
            axis=1,
        ).astype(np.uint8, copy=False)
        return phases, agents


    def _role_masks_from_agents(self, agent_masks: np.ndarray) -> np.ndarray:
        values = np.asarray(agent_masks, dtype=np.uint8)
        return np.stack(
            (
                np.any(values[:, :self.surface_count], axis=1),
                np.any(values[:, self.surface_count:], axis=1),
            ),
            axis=1,
        ).astype(np.uint8)


    @staticmethod
    def _binary_mask(
        value: object, shape: tuple[int, ...], name: str
    ) -> np.ndarray:
        ReplayBuffer._reject_tensor(value, name)
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if array.dtype.kind not in "buif" or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be binary")
        if np.any((array != 0) & (array != 1)):
            raise ValueError(f"{name} must contain only 0/1")
        return array.astype(np.uint8, copy=False)


    @staticmethod
    def _reject_tensor(value: object, name: str) -> None:
        if type(value).__module__.split(".", 1)[0] == "torch":
            raise TypeError(f"{name} must be converted to NumPy before replay insertion")


    def add(self, *args, **kwargs):
        if kwargs.get("timeout") and (not kwargs.get("terminated") or kwargs.get("truncated")):
            raise ValueError("task deadline must terminate, never truncate")
        if kwargs.get("truncated") and (kwargs.get("terminal_reason") or kwargs.get("timeout")):
            raise ValueError("external cut cannot contain a task event")
        return self._add_transition(*args, **kwargs)


    def external_reset(self):
        """Close pending prefixes at the last REAL next state; never cross a reset."""
        if not self._pending: return
        self._pending[-1]["truncated"]=True
        while self._pending:
            self._commit_window(self._pending)
            self._pending.pop(0)


