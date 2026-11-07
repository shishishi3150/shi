"""Real role-shared learning and actor-only storage for 2 USV + 3 AUV."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from matd3.agent import Agent
from matd3.checkpoint import load_policy, save_policy
from matd3.replay_buffer import ReplayBuffer
from usv_env.config import assemble, refresh_hashes
from usv_env.history import schema_for_counts


def test_replay_two_surface_role_masks_and_downstream_reward():
    replay = ReplayBuffer(8, 5, 4, 10, 2, surface_count=2)
    obs = np.zeros((5, 4), np.float32)
    state = np.zeros(10, np.float32)
    reward = np.arange(5, dtype=np.float32)
    replay.add(obs, state, np.zeros((5, 2), np.float32), reward, obs, state,
               terminated=False, truncated=False, timeout=False,
               surface_active=np.ones(2), agent_active_mask=np.ones(5))
    sample = replay.sample_uniform(1)
    assert sample["actions"].shape == (1, 5, 2)
    assert sample["surface_active"].shape == (1, 2)
    assert sample["agent_active_mask"].shape == (1, 5)
    assert sample["downstream_rewards"][0, 0] == 3.0


def test_five_agent_real_td3_update_and_light_policy_roundtrip(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = assemble(experiment="Base", device="cpu")
    cfg["training"].update(batch_size=2, n_step_return=1, hidden_sizes=[16, 16])
    cfg["replay_buffer"]["capacity"] = 16
    refresh_hashes(cfg)
    schema = schema_for_counts(5)
    obs_dim = schema.frame_dim * cfg["revision"]["history_length"]
    state_dim = cfg["observation_schema"]["critic_dim"]
    env = SimpleNamespace(num_surface_blue=2, num_submarines=3,
                          observation_dim=obs_dim, critic_state_dim=state_dim,
                          agent_action_limits=np.array([[2.2, 1.5]] * 2 + [[3.035, 1.5]] * 3, np.float32))
    # This fixture isolates five-agent learning from separately tested physics.
    monkeypatch.setattr("matd3.agent.nominal_actions", lambda _: np.zeros((5, 2), np.float32))
    agent = Agent(env, cfg)
    rng = np.random.default_rng(183)
    history = rng.normal(0, 0.1, (5, cfg["revision"]["history_length"], schema.frame_dim)).astype(np.float32)
    history[:, :, -1] = 1.0
    history[:, :, schema.index("time_seconds")] = np.arange(cfg["revision"]["history_length"])[None] * 0.15
    obs = history.reshape(5, -1)
    state = rng.normal(0, 0.1, state_dim).astype(np.float32)
    for _ in range(6):
        executed_action = agent.act(obs)
        info = SimpleNamespace(terminated=False, truncated=False, timeout=False, terminal_reason="")
        agent.add_transition(obs, state, executed_action, np.arange(5, dtype=np.float32) / 5,
                             obs, state, info)
    actions = agent.joint_actions(agent.tensor(obs)[None])
    assert actions.shape == (1, 5, 2)
    assert all(q(agent.tensor(state)[None], actions, agent.tensor(obs)[None]).shape == (1, heads)
               for heads, pair in zip((2, 3), agent.critics) for q in pair)
    actor_before = [p.detach().clone() for p in agent.surface_actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critics[0][0].parameters()]
    first = agent.update()
    assert first["usv_actor_loss"] is None
    assert all(torch.equal(a, b) for a, b in zip(actor_before, agent.surface_actor.parameters()))
    target_inputs = []
    handle = agent.critic_targets[0][0].register_forward_pre_hook(lambda module, args: target_inputs.append(args[1].detach().cpu()))
    second = agent.update()
    handle.remove()
    assert np.isfinite(second["usv_actor_loss"])
    assert np.isfinite(second["auv_actor_loss"])
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.surface_actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critics[0][0].parameters()))
    assert np.max(np.abs(agent.last_target_noise)) <= cfg["training"]["target_noise_clip"]
    assert target_inputs[0].shape == (2, 5, 2)
    assert torch.all(target_inputs[0].abs() <= torch.as_tensor(env.agent_action_limits))
    with pytest.raises(ValueError, match="observation"):
        agent.act(obs[:4])
    with pytest.raises(ValueError, match="full friendly joint action"):
        agent.critics[0][0](agent.tensor(state)[None], actions[:, :4], agent.tensor(obs)[None])
    changed = obs.copy()
    changed[2, schema.index("surge")] += 0.7
    original_action, changed_action = agent.act(obs), agent.act(changed)
    np.testing.assert_array_equal(original_action[[0, 1, 3, 4]], changed_action[[0, 1, 3, 4]])
    assert not np.array_equal(original_action[2], changed_action[2])
    joint = actions.detach().requires_grad_(True)
    q = agent.critics[0][0](agent.tensor(state)[None], joint, agent.tensor(obs)[None])
    derivative = torch.autograd.grad(q.sum(), joint)[0]
    assert torch.all(derivative.abs().sum(dim=2) > 0)  # Both roles condition on every complete 10D action.
    path = tmp_path / "latest.pt"
    save_policy(agent, path)
    restored, payload = load_policy(path)
    np.testing.assert_array_equal(restored.act(obs), agent.act(obs))
    assert "replay" not in payload and "critics" not in payload and "optimizers" not in payload
    assert path.stat().st_size < 2_000_000
    from scripts.export_policy import main as export_policy
    exported_path = tmp_path / "deployment.pt"
    export_policy(["--checkpoint", str(path), "--output", str(exported_path)])
    exported, _ = load_policy(exported_path)
    np.testing.assert_array_equal(exported.act(obs), agent.act(obs))


def test_historical_full_checkpoint_is_rejected(tmp_path):
    old = tmp_path / "old.pt"
    torch.save({"format": "shi_v43_training"}, old)
    with pytest.raises(ValueError, match="historical"):
        load_policy(old)


@pytest.mark.parametrize("historical",["schema","six_agents","network"])
def test_historical_actor_policy_is_rejected_without_altering_source(tmp_path,monkeypatch,historical):
    from matd3.checkpoint import atomic_save,read
    from usv_env.environment import PursuitEvasionEnv
    from usv_env.history import HistoryEnv
    torch.set_num_threads(1)
    cfg=assemble(device="cpu")
    cfg["training"]["hidden_sizes"]=[16,16]; cfg["replay_buffer"]["capacity"]=4
    refresh_hashes(cfg)
    agent=Agent(HistoryEnv(PursuitEvasionEnv(cfg)),cfg)
    current=tmp_path/"latest.pt"; save_policy(agent,current)
    payload=read(current)
    if historical=="schema": payload["schema"]="reef_interception"
    elif historical=="six_agents": payload["runtime_config"]["env"]["num_surface_blue"]=3
    else: payload["actors"][0]["encoder.gru.weight_ih"]=torch.zeros((144,1))
    old=tmp_path/"historical.pt"; atomic_save(payload,old); before=old.read_bytes()
    with pytest.raises(ValueError,match="historical"):
        load_policy(old)
    assert old.read_bytes()==before
