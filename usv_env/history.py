"""Named receiver-local history; target fields contain estimates, never truth."""
from dataclasses import dataclass
from collections import deque
from functools import cached_property, lru_cache
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class ObservationSchema:
    agent_count: int = 5
    version: str = "five_agent_local_estimates_capture_v3"
    base_fields: tuple = ("active", "surface", "submarine", "surge", "sway", "yaw_rate", "heading_sin", "heading_cos",
        "red_x", "red_y", "red_vx", "red_vy", "red_heading_sin", "red_heading_cos")
    peer_fields: tuple = ("active", "surface", "surfaced", "x", "y", "vx", "vy", "fresh", "age_seconds")
    context_fields: tuple = ("boundary_left", "boundary_right", "boundary_bottom", "boundary_top",
        "reef_0_x", "reef_0_y", "reef_0_clearance", "reef_1_x", "reef_1_y", "reef_1_clearance",
        "reef_2_x", "reef_2_y", "reef_2_clearance", "protected_x", "protected_y", "protected_radius",
        "previous_surge", "previous_yaw", "target_valid", "target_age_seconds", "target_position_variance",
        "target_velocity_variance", "target_confidence", "own_detection", "remaining_time", "capture_confirmed")

    @cached_property
    def fields(self):
        peers = self.agent_count - 1
        return (self.base_fields + tuple(f"peer_{i}.{name}" for i in range(peers) for name in self.peer_fields)
                + self.context_fields + ("time_seconds",)
                + tuple(f"receiver_{i}" for i in range(self.agent_count))
                + tuple(f"peer_{i}.message_valid" for i in range(peers))
                + tuple(f"peer_{i}.sender_id" for i in range(peers)))

    @cached_property
    def field_indices(self): return {name: i for i, name in enumerate(self.fields)}

    def index(self, name):
        try: return self.field_indices[name]
        except KeyError: raise ValueError(f'{name!r} is not in observation schema') from None
    @property
    def base_dim(self): return len(self.fields)
    @property
    def sensor_dim(self): return len(self.base_fields) + (self.agent_count-1)*len(self.peer_fields) + len(self.context_fields)
    @property
    def frame_dim(self): return self.base_dim + 1
    @property
    def hash(self): return hashlib.sha256(json.dumps([self.version, self.fields, "frame_valid"]).encode()).hexdigest()
    def peer(self, index):
        start = self.index(f"peer_{index}.active")
        return slice(start, start + len(self.peer_fields))


@lru_cache(maxsize=5)
def schema_for_counts(agent_count=5):
    if not 2 <= int(agent_count) <= 6: raise ValueError("supported agent counts: 2..6")
    return ObservationSchema(agent_count=int(agent_count))


SCHEMA = schema_for_counts()
BASE_OBS, FRAME_DIM, SENSOR_DIM = SCHEMA.base_dim, SCHEMA.frame_dim, SCHEMA.sensor_dim


class HistoryEnv:
    """Local frame adapter; current production uses one frame without memory.

    The historical class name and explicit multi-frame utility remain for
    schema/safety diagnostics. Current training config rejects history > 1.
    """
    def __init__(self, env, history_length=1, episode_limit=None):
        self.env, self.history_length = env, int(history_length)
        self.episode_limit = episode_limit
        self.frames = deque(maxlen=self.history_length)
        self.schema = schema_for_counts(env.num_friendly)
        self.observation_dim = self.schema.frame_dim * self.history_length

    def __getattr__(self, name):
        env = self.__dict__.get("env")
        if env is None: raise AttributeError(name)
        return getattr(env, name)

    @property
    def observation_schema(self): return self.schema

    def _push(self, original):
        schema, count = self.schema, self.env.num_friendly
        frame = np.zeros((count, schema.frame_dim), np.float32)
        frame[:, :schema.sensor_dim] = original
        frame[:, schema.index("time_seconds")] = self.step_count * self.dt
        for receiver in range(count):
            frame[receiver, schema.index(f"receiver_{receiver}")] = 1.
            for block, sender in enumerate(i for i in range(count) if i != receiver):
                packet = self.communication.cache.get((receiver, sender))
                frame[receiver, schema.index(f"peer_{block}.sender_id")] = sender / max(count - 1, 1)
                frame[receiver, schema.index(f"peer_{block}.message_valid")] = float(packet is not None)
                if packet is None: frame[receiver, schema.peer(block)] = 0.
                else: frame[receiver, schema.index(f"peer_{block}.age_seconds")] = max(0, self.step_count - packet.send_step) * self.dt
        frame[:, -1] = 1.
        if not np.isfinite(frame).all(): raise ValueError("nonfinite observation")
        self.frames.append(frame)
        return self.current_observation()

    def current_observation(self):
        return np.stack(tuple(self.frames), axis=1).reshape(self.env.num_friendly, -1).copy()

    def reset(self, *args, **kwargs):
        obs, state = self.env.reset(*args, **kwargs)
        self.frames.clear()
        self.frames.extend(np.zeros((self.env.num_friendly, self.schema.frame_dim), np.float32) for _ in range(self.history_length - 1))
        return self._push(obs), state

    def step(self, actions):
        obs, state, reward, done, info = self.env.step(actions)
        history = self._push(obs)
        if not done and self.episode_limit is not None and self.step_count >= self.episode_limit:
            done = True
            info.terminated, info.truncated, info.timeout = False, True, False
            info.terminal_reason = ""
            info.reward_terms.update(terminated=False, truncated=True, timeout=False, bootstrap_mask=1.)
            self.env._episode_done = True
        if done:
            info.terminal_observation, info.terminal_state = history.copy(), state.copy()
        return history, state, reward, done, info

    def fork(self):
        from copy import deepcopy
        return deepcopy(self)

    def close(self): self.env.close()
