"""Restricted tensor/plain-data checkpoint format, including exact simulator state."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import os
import random
import uuid
import numpy as np
import torch
from usv_env.config import validate, refresh_hashes
from usv_env.history import SCHEMA, FRAME_DIM
from .networks import make_actor

POLICY_FORMAT="shi_v43_policy"
CHECKPOINT_FORMAT="shi_v43_training"


def registry():
    from usv_env.environment import PursuitEvasionEnv
    from usv_env.history import HistoryEnv
    from usv_env.types import ShipState, ObservationSchema, EnvInfo
    from usv_env.task import FinalTask, TaskEnvironment, TaskSuccess, TaskTiming
    from usv_env.dynamics import USVDynamics, AUVDynamics3DOF
    from usv_env.communication import CommunicationNetwork, Packet
    from usv_env.waves import RegularLinearWave, WaveEpisode
    result = {c.__name__:c for c in (PursuitEvasionEnv,HistoryEnv,ShipState,ObservationSchema,EnvInfo,
        FinalTask,TaskEnvironment,TaskSuccess,TaskTiming,USVDynamics,AUVDynamics3DOF,CommunicationNetwork,Packet,
        RegularLinearWave,WaveEpisode)}
    result["V43Environment"] = PursuitEvasionEnv  # Existing safe-format checkpoint record tag.
    return result


def pack(obj):
    if isinstance(obj,slice): return {"__slice__":[obj.start,obj.stop,obj.step]}
    if isinstance(obj,np.ndarray):
        # Bytes avoid torch serialization's dtype limitations (e.g. NumPy uint32 RNG state).
        return {"__array__":torch.from_numpy(np.ascontiguousarray(obj).copy().view(np.uint8).reshape(-1)),
                "dtype":obj.dtype.str,"shape":list(obj.shape)}
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,np.random.Generator): return {"__generator__":pack(obj.bit_generator.state)}
    if isinstance(obj,deque): return {"__deque__":[pack(x) for x in obj],"maxlen":obj.maxlen}
    if isinstance(obj,dict): return {k:pack(v) for k,v in obj.items()}
    if isinstance(obj,tuple): return tuple(pack(v) for v in obj)
    if isinstance(obj,list): return [pack(v) for v in obj]
    if obj is None or isinstance(obj,(str,int,float,bool,torch.Tensor)): return obj
    name=type(obj).__name__
    if name in registry() and type(obj) is registry()[name]:
        return {"__record__":name,"fields":pack(vars(obj))}
    raise TypeError(f"unsupported checkpoint state type: {type(obj)}")


def unpack(obj):
    if isinstance(obj,dict):
        if "__slice__" in obj: return slice(*obj["__slice__"])
        if "__array__" in obj:
            dtype=np.dtype(obj["dtype"])
            if dtype.hasobject: raise ValueError("object dtype forbidden")
            return np.frombuffer(obj["__array__"].cpu().numpy().tobytes(),dtype=dtype).reshape(obj["shape"]).copy()
        if "__generator__" in obj:
            gen=np.random.default_rng(); gen.bit_generator.state=unpack(obj["__generator__"]); return gen
        if "__deque__" in obj: return deque((unpack(x) for x in obj["__deque__"]),maxlen=obj["maxlen"])
        if "__record__" in obj:
            cls=registry().get(obj["__record__"])
            if cls is None: raise ValueError("unknown simulator record")
            result=object.__new__(cls); vars(result).update(unpack(obj["fields"])); return result
        return {k:unpack(v) for k,v in obj.items()}
    if isinstance(obj,list): return [unpack(v) for v in obj]
    if isinstance(obj,tuple): return tuple(unpack(v) for v in obj)
    return obj


def atomic_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        torch.save(pack(payload),temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


def read(path,kind=None):
    return decode(torch.load(path,map_location="cpu",weights_only=True),kind)


def decode(raw,kind=None):
    payload=unpack(raw)
    if payload.get("format") not in (POLICY_FORMAT,CHECKPOINT_FORMAT): raise ValueError("not a v4.3 checkpoint")
    if kind and payload["format"]!=kind: raise ValueError("lightweight policy is not a resumable training checkpoint")
    cfg=payload["runtime_config"]
    validate(cfg)
    stored=deepcopy(cfg["hashes"])
    if refresh_hashes(cfg)!=stored: raise ValueError("task/reward/schema/architecture hash mismatch")
    return payload


def policy_from_payload(payload):
    if payload["format"]==CHECKPOINT_FORMAT:
        return {"format":POLICY_FORMAT,"runtime_config":payload["runtime_config"],
                "actors":payload["agent"]["actors"],"limits":payload["limits"],"cruise":payload["cruise"],
                "global_env_steps":payload["agent"]["counters"]["global_env_steps"]}
    return payload


def global_rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_global_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def save_policy(agent,path,metadata=None):
    atomic_save({"format":POLICY_FORMAT,"schema":43,"runtime_config":deepcopy(agent.runtime_config),
                 "actors":[a.state_dict() for a in agent.actors],"limits":agent.limits,"cruise":agent.cruise,
                 "global_env_steps":agent.global_env_steps,"metadata":metadata or {}},path)


class InferencePolicy:
    def __init__(self,payload,device="cpu"):
        self.runtime_config=deepcopy(payload["runtime_config"])
        cfg=self.runtime_config
        if device in (None,"auto"): device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=torch.device(device)
        self.obs_dim=cfg["revision"]["history_length"]*FRAME_DIM
        self.global_env_steps=payload["global_env_steps"]
        self.limits,self.cruise=payload["limits"],payload["cruise"]
        self.actors=[make_actor(cfg,self.limits[i],self.cruise[i],role).to(self.device)
                     for i,role in ((0,"surface"),(2,"submarine"))]
        for actor,state in zip(self.actors,payload["actors"]): actor.load_state_dict(state,strict=True); actor.eval()
        self.surface_actor,self.submarine_actor=self.actors

    def act(self,obs,explore=False):
        if explore: raise ValueError("deployment policy has no exploration")
        o=torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        with torch.no_grad(): result=torch.cat((self.actors[0](o[:2]),self.actors[1](o[2:])),0).cpu().numpy()
        return np.clip(result,-self.limits,self.limits).astype(np.float32)


def load_policy(path,device="cpu"):
    payload=policy_from_payload(read(path))
    return InferencePolicy(payload,device),payload
