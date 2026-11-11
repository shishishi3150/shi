"""Current-frame actors and independent centralized feedforward twin critics."""
from collections.abc import Callable
import math

import torch
from torch import nn

from usv_env.history import schema_for_counts


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    factories = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
    normalized = str(name).strip().lower()
    if normalized not in factories:
        raise ValueError(f"unsupported network activation: {name}")
    return factories[normalized]


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int, *,
        activation: str = "relu", output_gain: float = 1.0, output_bias: float = 0.0) -> nn.Sequential:
    activation_factory = _activation_factory(activation)
    layers = []
    last = int(input_dim)
    for hidden in hidden_sizes:
        linear = nn.Linear(last, int(hidden))
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, activation_factory()))
        last = int(hidden)
    output = nn.Linear(last, int(output_dim))
    nn.init.orthogonal_(output.weight, gain=float(output_gain))
    nn.init.constant_(output.bias, float(output_bias))
    layers.append(output)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, cfg, limits, cruise):
        super().__init__()
        training = cfg["training"]
        count = int(cfg["env"]["num_surface_blue"]) + int(cfg["env"]["num_submarines"])
        self.schema = schema_for_counts(count)
        self.net = mlp(self.schema.base_dim, training["hidden_sizes"], 2,
                       activation=training["activation"], output_gain=training["actor_output_gain"])
        self.register_buffer("action_limit", torch.as_tensor(limits, dtype=torch.float32).clone())
        with torch.no_grad():
            normalized_cruise = torch.as_tensor(cruise, dtype=torch.float32)
            if normalized_cruise.shape != (2,) or not (normalized_cruise.abs() < 1).all():
                raise ValueError("cruise action must have two normalized values strictly within actuator bounds")
            self.net[-1].bias.copy_(torch.atanh(normalized_cruise))

    def normalized(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != self.schema.frame_dim:
            raise ValueError("actor requires one current local observation frame")
        # The final frame-valid marker is transport metadata, not a feature.
        return torch.tanh(self.net(obs[:, :self.schema.base_dim]))

    def forward(self, obs):
        return self.normalized(obs) * self.action_limit


class Critic(nn.Module):
    def __init__(self, cfg, state_dim, limits, heads):
        super().__init__()
        self.num_agents = int(cfg["env"]["num_surface_blue"]) + int(cfg["env"]["num_submarines"])
        self.state_dim = int(state_dim)
        training = cfg["training"]
        self.net = mlp(self.state_dim + 2 * self.num_agents,
                       training["hidden_sizes"], heads, activation=training["activation"],
                       output_gain=training["critic_output_gain"], output_bias=training.get("critic_output_bias", 0.0))
        self.register_buffer("limits", torch.as_tensor(limits, dtype=torch.float32).clone())

    def forward(self, state, actions, observation=None):
        # observation is accepted for the common training/oracle interface;
        # the centralized MLP conditions only on current state and joint action.
        batch = state.shape[0]
        if actions.shape != (batch, self.num_agents, 2) or state.shape != (batch, self.state_dim):
            raise ValueError("critic requires the full friendly joint action and matching global state")
        inputs = torch.cat((state, (actions / self.limits).reshape(batch, -1)), -1)
        return self.net(inputs)


def make_actor(cfg, limits, cruise, role):
    from usv_env.config import named_seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(named_seed(cfg["training"]["seed"], "actor:" + role))
        return Actor(cfg, limits, cruise)
