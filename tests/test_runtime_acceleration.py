"""Acceleration keeps input semantics, autograd and public I3 diagnostics."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from matd3.agent import Agent
from matd3.history_enhancement import I3_COUNT_FIELDS, I3_DISTRIBUTION_FIELDS
from matd3.networks import Actor
from scripts.train import make_env
from usv_env.config import assemble
from usv_env.history import SCHEMA


def observations(requires_grad=False):
    value = torch.zeros((4, SCHEMA.frame_dim))
    value[:, -1] = 1.
    value[:, SCHEMA.index("time_seconds")] = 5.3
    return value.requires_grad_(requires_grad)


def test_current_frame_actor_does_not_mutate_input_or_retain_a_history():
    cfg = assemble(device='cpu')
    actor = Actor(cfg, [2.2, 1.5], [0., 0.])
    obs = observations()
    original = obs.clone()
    first = actor(obs)
    actor(observations()+.1)
    torch.testing.assert_close(actor(obs), first, rtol=0, atol=0)
    torch.testing.assert_close(obs, original, rtol=0, atol=0)


def test_current_frame_actor_observes_mutation_without_stale_features():
    actor = Actor(assemble(device='cpu'), [2.2, 1.5], [0., 0.])
    obs = observations()
    first = actor(obs)
    obs[0, SCHEMA.index('surge')] = 7.
    changed = actor(obs)
    assert not torch.equal(first[0], changed[0])
    torch.testing.assert_close(first[1:], changed[1:], rtol=0, atol=0)


def test_differentiable_current_frame_inputs_have_fresh_autograd_graphs():
    actor = Actor(assemble(device='cpu'), [2.2, 1.5], [0., 0.])
    obs = observations(requires_grad=True)
    left, right = actor(obs), actor(obs)
    assert left.grad_fn is not right.grad_fn
    (left.sum()+right.sum()).backward()
    assert torch.isfinite(obs.grad).all() and obs.grad[:, :SCHEMA.base_dim].abs().sum() > 0


@pytest.mark.parametrize("variant", ["conservative", "hard_max", "random_history", "no_state_matching"])
@pytest.mark.parametrize("terminal_candidates", [False, True])
def test_no_bootstrappable_i3_candidate_preserves_full_confidence_and_fallback(monkeypatch, variant, terminal_candidates):
    torch.set_num_threads(1)
    cfg = assemble("configs/fast.yaml", "Full", 19, "cpu")
    cfg["training"].update(hidden_sizes=[8, 8])
    cfg["replay_buffer"]["capacity"] = 8
    cfg["i3"].update(variant=variant, candidates=2, search_pool=4)
    env = make_env(cfg, 19)
    obs, state = env.reset(seed=19)
    agent = Agent(env, cfg)
    history = agent.tensor(np.repeat(obs[None], 2, 0))
    states = agent.tensor(np.repeat(state[None], 2, 0))
    actions = agent.tensor(np.zeros((2, 5, 2), np.float32))
    valid = np.full((2, 2), terminal_candidates, bool)
    found = dict(valid=valid, indices=np.array([[1, 2], [3, 4]]),
        actions=np.arange(40, dtype=np.float32).reshape(2, 2, 5, 2),
        distance=np.full((2, 2), .3, np.float32), quality=np.full((2, 2), .7, np.float32),
        realized_return=np.ones((2, 2), np.float32), outcome_confidence=np.full((2, 2), .8, np.float32),
        phase_match_fraction=.25, funnel={key: 0 for key in I3_COUNT_FIELDS},
        distributions={key: None for key in I3_DISTRIBUTION_FIELDS})
    monkeypatch.setattr(agent.history, "retrieve", lambda *args, **kwargs: deepcopy(found))
    sample = dict(next_states=np.repeat(state[None], 2, 0), next_context=np.zeros((2, 22), np.float32),
        next_phase_ids=np.zeros((2, 1), np.uint8), agent_active_mask=np.ones((2, 5), np.float32),
        bootstrap_masks=np.full((2, 1), 0. if terminal_candidates else 1., np.float32),
        bootstrap_discounts=np.ones((2, 1), np.float32), next_obs=np.repeat(obs[None], 2, 0))
    with torch.no_grad():
        pairs = [[critic(states, actions, history) for critic in pair] for pair in agent.critic_targets]
        expected = torch.cat([torch.minimum(*pair) for pair in pairs], -1)
        actual, metrics = agent._target_values(sample, states, history, actions, torch.ones((2, 5)))
    torch.testing.assert_close(actual, expected+torch.zeros_like(expected), rtol=0, atol=0)
    similarity = np.exp(-found["distance"]**2/cfg["i3"]["similarity_temperature"]**2)
    if variant in ("random_history", "no_state_matching"):
        similarity[:] = 1.
    np.testing.assert_array_equal(agent.last_i3["confidence"], similarity*found["quality"])
    assert agent.last_i3["confidence"].shape == (2, 2)
    np.testing.assert_array_equal(agent.last_i3["indices"], found["indices"][:, 0])
    np.testing.assert_array_equal(agent.last_i3["actions"], found["actions"][:, 0])
    assert not agent.last_i3["valid"].any() and not agent.last_i3["accepted"].any()
    assert metrics["i3_terminal_rejected"] == (4 if terminal_candidates else 0)
    assert metrics["i3_target_enhancement"] == metrics["i3_valid_candidates"] == 0.
    assert metrics["i3_enhancement_weight"] == (1. if variant == "hard_max" else cfg["i3"]["enhancement_weight"])
    env.close()
