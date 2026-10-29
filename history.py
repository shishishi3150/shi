"""Named, versioned receiver-local inputs. No peer slot truth is reconstructed."""
from dataclasses import dataclass
from collections import deque
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class ObservationSchema:
    version: str = "4.3-usv-front-blocking-v3"
    # Historical sensor channels retain their physical meaning and ordering.
    base_fields: tuple = ("active", "surface", "submarine", "surge", "sway", "yaw_rate", "heading_sin", "heading_cos",
        "target_x", "target_y", "target_distance", "desired_vx", "desired_vy", "slot_0", "slot_1", "slot_2",
        "red_x", "red_y", "red_vx", "red_vy", "red_heading_sin", "red_heading_cos")
    peer_fields: tuple = ("active", "surface", "surfaced", "x", "y", "vx", "vy", "fresh", "age_seconds")
    context_fields: tuple = ("boundary_left", "boundary_right", "boundary_bottom", "boundary_top", "island_x", "island_y",
        "island_clearance", "previous_surge", "previous_yaw", "static_target", "compaction_enabled", "formation_progress",
        "shadow_outcome_reserved", "local_gate_estimate", "remaining_time")

    @property
    def fields(self):
        return (self.base_fields + tuple(f"peer_{i}.{n}" for i in range(4) for n in self.peer_fields)
                + self.context_fields + ("time_seconds",)
                + tuple(f"receiver_{i}" for i in range(5))
                + tuple(f"peer_{i}.message_valid" for i in range(4))
                + tuple(f"peer_{i}.sender_id" for i in range(4)))

    def index(self, name): return self.fields.index(name)
    @property
    def base_dim(self): return len(self.fields)
    @property
    def frame_dim(self): return self.base_dim + 1
    @property
    def hash(self):
        return hashlib.sha256(json.dumps([self.version, self.fields, "frame_valid", "age_seconds",
                                          "time relative to first valid frame in seconds",
                                          "USV target=angular bisector at annular radial midpoint; desired velocity=Red ground velocity + clipped correction"]).encode()).hexdigest()
    def peer(self, i):
        start = self.index(f"peer_{i}.active")
        return slice(start, start + len(self.peer_fields))


SCHEMA = ObservationSchema()
BASE_OBS, FRAME_DIM = SCHEMA.base_dim, SCHEMA.frame_dim
SENSOR_DIM = len(SCHEMA.base_fields)+4*len(SCHEMA.peer_fields)+len(SCHEMA.context_fields)


class HistoryEnv:
    def __init__(self, env, history_length=8, episode_limit=None):
        self.env, self.history_length = env, int(history_length)
        self.episode_limit = episode_limit
        self.frames = deque(maxlen=self.history_length)
        self.observation_dim = FRAME_DIM * self.history_length

    def __getattr__(self, name):
        env = self.__dict__.get("env")
        if env is None: raise AttributeError(name)
        return getattr(env, name)

    @property
    def observation_schema(self): return SCHEMA

    def _push(self, original):
        frame = np.zeros((5, FRAME_DIM), np.float32)
        frame[:, :len(SCHEMA.base_fields) + 4 * len(SCHEMA.peer_fields) + len(SCHEMA.context_fields)] = original
        frame[:2, SCHEMA.index("shadow_outcome_reserved")] = 0.
        frame[:, SCHEMA.index("time_seconds")] = self.step_count * self.dt
        for receiver in range(5):
            frame[receiver, SCHEMA.index(f"receiver_{receiver}")] = 1.
            for block, sender in enumerate(i for i in range(5) if i != receiver):
                packet = self.communication.cache.get((receiver, sender))
                frame[receiver, SCHEMA.index(f"peer_{block}.sender_id")] = sender / 4.
                frame[receiver, SCHEMA.index(f"peer_{block}.message_valid")] = float(packet is not None)
                if packet is None:
                    frame[receiver, SCHEMA.peer(block)] = 0.
                else:
                    frame[receiver, SCHEMA.index(f"peer_{block}.age_seconds")] = max(0, self.step_count - packet.send_step) * self.dt
        frame[:, -1] = 1.
        if not np.isfinite(frame).all(): raise ValueError("nonfinite observation")
        self.frames.append(frame)
        return self.current_observation()

    def current_observation(self): return np.stack(tuple(self.frames), axis=1).reshape(5, -1).copy()

    def reset(self, *args, **kwargs):
        obs, state = self.env.reset(*args, **kwargs)
        self.frames.clear()
        self.frames.extend(np.zeros((5, FRAME_DIM), np.float32) for _ in range(self.history_length - 1))
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
            info.terminal_observation = history.copy()
            info.terminal_state = state.copy()
        return history, state, reward, done, info

    def fork(self):
        from copy import deepcopy
        return deepcopy(self)

    def close(self): pass
