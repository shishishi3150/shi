from __future__ import annotations


import math


from collections.abc import Callable


import torch


from torch import nn


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    normalized = str(name).strip().lower()
    factories: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }
    if normalized not in factories:
        raise ValueError(f"unsupported network activation: {name}")
    return factories[normalized]


def _orthogonal_linear(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain=float(gain))
    nn.init.zeros_(linear.bias)


def mlp(
    input_dim: int,
    hidden_sizes: list[int],
    output_dim: int,
    *,
    activation: str = "relu",
    output_gain: float = 1.0,
    output_bias: float = 0.0,
) -> nn.Sequential:
    activation_factory = _activation_factory(activation)
    layers: list[nn.Module] = []
    last = int(input_dim)
    for hidden in hidden_sizes:
        linear = nn.Linear(last, int(hidden))
        _orthogonal_linear(linear, math.sqrt(2.0))
        layers.extend((linear, activation_factory()))
        last = int(hidden)
    output = nn.Linear(last, int(output_dim))
    _orthogonal_linear(output, output_gain)
    nn.init.constant_(output.bias, float(output_bias))
    layers.append(output)
    return nn.Sequential(*layers)


"""Common actor factory and independent history-conditioned multihead critics."""
import torch
from torch import nn
from usv_env.history import SCHEMA, BASE_OBS, FRAME_DIM


class HistoryEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.length = cfg["history_length"]
        self.mode, self.h = cfg["encoder"], cfg["node_hidden"]
        self.age_inputs, self.age_gate = cfg["age_inputs"], cfg["age_gate"]
        self.gru_hidden = cfg["gru_hidden"]
        if self.mode == "gru":
            self.gru = nn.GRUCell(BASE_OBS, self.gru_hidden)
            self.output_dim = BASE_OBS + self.gru_hidden
        elif self.mode == "mlp": self.output_dim = BASE_OBS
        else:
            self.control_dim = 17  # nine received fields, five sender IDs, node kind, reliability, physical clock
            self.node_init = nn.Linear(self.control_dim, self.h)
            self.self_field = nn.Linear(self.h, self.h)
            self.message_field = nn.Linear(self.h, self.h, bias=False)
            if self.mode == "gncde": self.vector_field = nn.Linear(self.h, self.h * self.control_dim)
            if self.age_gate: self.age_rate = nn.Parameter(torch.tensor(1.))
            else: self.register_buffer("age_rate", torch.tensor(1.))
            self.output_dim = 4*self.h + BASE_OBS + 8

    def _nodes(self, frame, clock):
        batch = frame.shape[0]
        own = torch.cat((frame[:, :8], frame.new_zeros(batch,1)), -1)
        target = torch.cat((frame[:, SCHEMA.index("red_x"):SCHEMA.index("red_heading_cos")+1], frame.new_zeros(batch,3)), -1)
        peers = torch.stack([frame[:, SCHEMA.peer(i)] for i in range(4)], 1)
        raw = torch.cat((own[:,None], target[:,None], peers), 1)
        valid = torch.stack([frame[:, SCHEMA.index(f"peer_{i}.message_valid")] for i in range(4)], -1)
        active = torch.cat((frame[:, :1], frame[:, :1], valid * peers[:,:,0]), -1)
        ids = frame.new_zeros(batch,6,5)
        ids[:,0] = frame[:, SCHEMA.index("receiver_0"):SCHEMA.index("receiver_4")+1]
        for i in range(4):
            idx = (frame[:, SCHEMA.index(f"peer_{i}.sender_id")]*4).round().long().clamp(0,4)
            ids[:,i+2] = torch.nn.functional.one_hot(idx,5).to(frame.dtype)
        ages = torch.cat((frame.new_zeros(batch,2), peers[:,:,-1]), -1)
        reliability = active * (torch.exp(-torch.nn.functional.softplus(self.age_rate)*ages) if self.age_gate else 1.)
        kind = frame.new_zeros(batch,6,1); kind[:,1] = 1. # Explicit target node with its own historical path.
        x = torch.cat((raw, ids, kind, reliability[:,:,None], clock[:,None,None].expand(-1,6,1)), -1)
        adj = torch.eye(6,device=frame.device,dtype=frame.dtype)[None].repeat(batch,1,1)
        adj[:,0,1:] = reliability[:,1:]; adj[:,1:,0] = reliability[:,1:]
        adj *= active[:,:,None]*active[:,None,:]
        adj = adj / adj.sum(-1,keepdim=True).clamp_min(1e-8)
        return x, adj, reliability, peers, ages, active

    def _field(self,z,adj):
        f=torch.tanh(self.self_field(z)+self.message_field(adj@z))
        return torch.tanh(self.vector_field(f)).reshape(-1,6,self.h,self.control_dim)

    def forward(self, obs):
        frames=obs.reshape(-1,self.length,FRAME_DIM).clone()
        if not self.age_inputs:
            for i in range(4):
                frames[:,:,SCHEMA.index(f"peer_{i}.age_seconds")]=0.
                frames[:,:,SCHEMA.index(f"peer_{i}.fresh")]=0.
        valid=frames[:,:,-1]>.5
        times=frames[:,:,SCHEMA.index("time_seconds")]
        first=valid.to(torch.int64).argmax(1)
        origin=times.gather(1,first[:,None])
        frames[:,:,SCHEMA.index("time_seconds")]=torch.where(valid,times-origin,0.)
        latest=frames[:,-1,:BASE_OBS]
        if self.mode=="mlp": return latest
        if self.mode=="gru":
            z=obs.new_zeros(obs.shape[0],self.gru_hidden)
            for k in range(self.length):
                new=self.gru(frames[:,k,:BASE_OBS],z)
                z=torch.where(valid[:,k,None],new,z)
            return torch.cat((z,latest),-1)
        z=obs.new_zeros(obs.shape[0],6,self.h)
        prev=obs.new_zeros(obs.shape[0],6,self.control_dim)
        previous_adj=obs.new_zeros(obs.shape[0],6,6)
        started=torch.zeros(obs.shape[0],dtype=torch.bool,device=obs.device)
        for k in range(self.length):
            x,adj,weights,peers,ages,active=self._nodes(frames[:,k,:BASE_OBS],frames[:,k,SCHEMA.index("time_seconds")])
            first=valid[:,k]&~started
            z=torch.where(first[:,None,None],torch.tanh(self.node_init(x)),z)
            if self.mode=="static_graph":
                init=torch.tanh(self.node_init(x))
                new=torch.tanh(self.self_field(init)+self.message_field(adj@init))
            else:
                dx=x-prev
                k1=torch.einsum("bnhc,bnc->bnh",self._field(z,previous_adj),dx)
                k2=torch.einsum("bnhc,bnc->bnh",self._field(z+k1,adj),dx)
                new=z+.5*(k1+k2)
            z=torch.where((valid[:,k]&started)[:,None,None],new,z)
            prev=torch.where(valid[:,k,None,None],x,prev)
            previous_adj=torch.where(valid[:,k,None,None],adj,previous_adj)
            started |= valid[:,k]
        pools,stats=[],[]
        for surface in (True,False):
            role=peers[:,:,1] if surface else 1-peers[:,:,1]
            w=weights[:,2:]*role
            count=(active[:,2:]*role).sum(1,keepdim=True)
            pools.append((z[:,2:]*w[:,:,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1e-8))
            stats.extend((w.sum(1,keepdim=True)/4, count/4,
                          (ages[:,2:]*active[:,2:]*role).sum(1,keepdim=True)/count.clamp_min(1.),
                          (peers[:,:,7]*active[:,2:]*role).sum(1,keepdim=True)/count.clamp_min(1.)))
        return torch.cat((z[:,0],z[:,1],*pools,latest,*stats),-1)


class Actor(nn.Module):
    def __init__(self,cfg,limits,cruise,hyper=False):
        super().__init__()
        rev,tr=cfg["revision"],cfg["training"]
        self.encoder=HistoryEncoder(rev)
        self.net=mlp(self.encoder.output_dim,tr["hidden_sizes"],2,activation=tr["activation"],output_gain=tr["actor_output_gain"])
        self.register_buffer("action_limit",torch.as_tensor(limits,dtype=torch.float32))
        with torch.no_grad(): self.net[-1].bias.copy_(torch.atanh(torch.as_tensor(cruise,dtype=torch.float32)))
        self.hyper_enabled=hyper
        if hyper:
            hc=rev["i3"]
            hidden=[i for i,x in enumerate(self.net) if isinstance(x,nn.Linear)][:-1]
            self.adapter_index=hidden[1]
            layer=self.net[self.adapter_index]
            self.rank,self.gain=hc["rank"],hc["residual_gain"]
            self.basis_u=nn.Parameter(torch.randn(layer.out_features,self.rank))
            self.basis_v=nn.Parameter(torch.randn(layer.in_features,self.rank))
            self.hyper=mlp(self.encoder.output_dim,[hc["hyper_hidden"]],self.rank,activation="tanh",output_gain=.01)
            nn.init.zeros_(self.hyper[-1].weight); nn.init.zeros_(self.hyper[-1].bias)

    def normalized(self,obs):
        x=self.encoder(obs)
        if not self.hyper_enabled: return torch.tanh(self.net(x))
        z=torch.tanh(self.hyper(x))
        return self._adapted(x,z)

    def coefficients(self,obs,detach_encoder=False):
        return torch.tanh(self.coefficient_logits(obs,detach_encoder))

    def coefficient_logits(self,obs,detach_encoder=False):
        if not self.hyper_enabled: raise ValueError("parameter teacher requires the hypernetwork")
        x=self.encoder(obs)
        return self.hyper(x.detach() if detach_encoder else x)

    def forward_with_coefficients(self,obs,z):
        """Keep coefficients fixed while every action still reacts to new observations."""
        if not self.hyper_enabled: raise ValueError("parameter teacher requires the hypernetwork")
        if z.shape != (obs.shape[0],self.rank): raise ValueError("invalid coefficient shape")
        return self._adapted(self.encoder(obs),z)*self.action_limit

    def _adapted(self,x,z):
        u=nn.functional.normalize(self.basis_u,dim=0,eps=1e-8)
        v=nn.functional.normalize(self.basis_v,dim=0,eps=1e-8)
        for i,layer in enumerate(self.net):
            x=layer(x)+((x@v)*z)@u.T*(self.gain/self.rank) if i==self.adapter_index else layer(x)
        return torch.tanh(x)

    def forward(self,obs): return self.normalized(obs)*self.action_limit


class Critic(nn.Module):
    def __init__(self,cfg,state_dim,limits,heads):
        super().__init__()
        self.length=cfg["revision"]["history_length"]
        self.h=cfg["revision"]["critic_gru_hidden"]
        self.gru=nn.GRUCell(BASE_OBS,self.h)
        self.net=mlp(state_dim+5*self.h+10,cfg["training"]["hidden_sizes"],heads,
                     activation=cfg["training"]["activation"],output_gain=cfg["training"]["critic_output_gain"])
        self.register_buffer("limits",torch.as_tensor(limits,dtype=torch.float32))

    def forward(self,state,actions,history):
        batch=history.shape[0]
        frames=history.reshape(batch*5,self.length,FRAME_DIM).clone()
        valid=frames[:,:,-1]>.5
        times=frames[:,:,SCHEMA.index("time_seconds")]
        origin=times.gather(1,valid.long().argmax(1)[:,None])
        frames[:,:,SCHEMA.index("time_seconds")]=torch.where(valid,times-origin,0.)
        z=history.new_zeros(batch*5,self.h)
        for k in range(self.length):
            new=self.gru(frames[:,k,:BASE_OBS],z)
            z=torch.where(valid[:,k,None],new,z)
        return self.net(torch.cat((state,z.reshape(batch,-1),(actions/self.limits).reshape(batch,-1)),-1))


def make_actor(cfg,limits,cruise,role):
    # A named RNG scope prevents optional hyper parameters from shifting any common initialization.
    from usv_env.config import named_seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(named_seed(cfg["training"]["seed"],"actor:"+role))
        return Actor(cfg,limits,cruise,hyper=role=="surface" and cfg["revision"]["hyper_enabled"])
