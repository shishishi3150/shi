"""Atomic actor-only policy storage. Replay, optimizer and simulator state are never saved."""
from copy import deepcopy
from pathlib import Path
import os
import random
import uuid

import numpy as np
import torch

from usv_env.config import TASK_CONTRACT, VERSION, validate, refresh_hashes
from usv_env.history import schema_for_counts
from .networks import make_actor

POLICY_FORMAT = "shi_actor_policy"


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not allowed in a policy")
        return {"__ndarray__": torch.from_numpy(np.ascontiguousarray(value).copy().view(np.uint8).reshape(-1)),
                "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_encode(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported policy value: {type(value)}")


def _decode(value):
    if isinstance(value, dict):
        if "__ndarray__" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError("object arrays are not allowed in a policy")
            return np.frombuffer(value["__ndarray__"].cpu().numpy().tobytes(), dtype=dtype).reshape(value["shape"]).copy()
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(item) for item in value)
    return value


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(_encode(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read(path):
    """Read and validate the small policy payload; historical task formats fail explicitly."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("format") != POLICY_FORMAT:
        raise ValueError("not a current actor-only policy; historical models require their original source")
    payload = _decode(raw)
    if payload.get("task_contract") != TASK_CONTRACT or payload.get("schema") != VERSION:
        raise ValueError("historical policy task contract or observation schema differs from the current five-vessel task")
    expected = {"format", "schema", "task_contract", "runtime_config", "actors", "limits", "cruise", "global_env_steps", "metadata"}
    if set(payload) != expected:
        raise ValueError("policy payload contains unsupported training or simulator state")
    config = payload["runtime_config"]
    if (config["env"]["num_surface_blue"], config["env"]["num_submarines"]) != (2, 3):
        raise ValueError("historical policy agent counts differ: current task requires 2 USV + 3 AUV and 10 joint action dimensions")
    validate(config)
    stored_hashes = deepcopy(config["hashes"])
    if refresh_hashes(config) != stored_hashes:
        raise ValueError("task/reward/schema/architecture hash mismatch")
    return payload


def global_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_global_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_policy(agent, path, metadata=None):
    actors = [{key: value.detach().cpu() for key, value in actor.state_dict().items()} for actor in agent.actors]
    atomic_save({"format": POLICY_FORMAT, "schema": VERSION, "task_contract": TASK_CONTRACT,
                 "runtime_config": deepcopy(agent.runtime_config), "actors": actors,
                 "limits": agent.limits, "cruise": agent.cruise, "global_env_steps": agent.global_env_steps,
                 "metadata": metadata or {}}, path)


class InferencePolicy:
    def __init__(self, payload, device="cpu"):
        self.runtime_config = deepcopy(payload["runtime_config"])
        config = self.runtime_config
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_surface = int(config["env"]["num_surface_blue"])
        self.num_agents = self.num_surface + int(config["env"]["num_submarines"])
        self.obs_dim = config["revision"]["history_length"] * schema_for_counts(self.num_agents).frame_dim
        self.global_env_steps = int(payload["global_env_steps"])
        self.limits, self.cruise = np.asarray(payload["limits"], np.float32), np.asarray(payload["cruise"], np.float32)
        if self.limits.shape != (self.num_agents, 2) or self.cruise.shape != self.limits.shape:
            raise ValueError("policy physical action dimensions differ from the saved task")
        if not np.isfinite(self.limits).all() or not np.isfinite(self.cruise).all() or np.any(self.limits <= 0):
            raise ValueError("policy physical action values must be finite with positive limits")
        if len(payload["actors"]) != 2:
            raise ValueError("policy requires one shared actor for each friendly role")
        self.actors = [make_actor(config, self.limits[index], self.cruise[index], role).to(self.device)
                       for index, role in ((0, "surface"), (self.num_surface, "submarine"))]
        for actor, state in zip(self.actors, payload["actors"]):
            try:
                actor.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise ValueError("historical or incompatible policy network dimensions differ from the current five-vessel schema") from error
            actor.eval()
        self.surface_actor, self.submarine_actor = self.actors

    def act(self, obs, explore=False):
        if explore:
            raise ValueError("deployment policy has no exploration")
        observation = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if observation.shape != (self.num_agents, self.obs_dim) or not torch.isfinite(observation).all():
            raise ValueError("policy observation differs from the saved current-frame schema")
        with torch.no_grad():
            actions = torch.cat((self.actors[0](observation[:self.num_surface]), self.actors[1](observation[self.num_surface:])), 0)
        return np.clip(actions.cpu().numpy(), -self.limits, self.limits).astype(np.float32)


def load_policy(path, device="cpu"):
    payload = read(path)
    return InferencePolicy(payload, device), payload
