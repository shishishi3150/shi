"""All mechanisms share a current-frame feedforward backbone."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from matd3.agent import Agent
from scripts.train import make_env
from usv_env.config import PRESETS, assemble, refresh_hashes, validate
from usv_env.history import SCHEMA


@pytest.mark.parametrize('name', PRESETS)
def test_clean_presets_use_one_frame_and_no_recurrent_components(name):
    cfg = assemble('configs/fast.yaml', name, device='cpu')
    assert cfg['revision']['encoder'] == 'mlp'
    assert cfg['revision']['history_length'] == 1
    assert not {'gru_hidden', 'critic_gru_hidden'} & set(cfg['revision'])
    assert cfg['observation_schema']['actor_dim'] == SCHEMA.frame_dim
    base = assemble('configs/fast.yaml', 'Base', device='cpu')
    assert cfg['training'] == base['training']
    assert cfg['hashes']['architecture'] == base['hashes']['architecture']
    assert cfg['hashes']['task'] == base['hashes']['task']
    assert cfg['hashes']['reward'] == base['hashes']['reward']


def test_real_actor_and_critic_are_direct_feedforward_networks():
    torch.set_num_threads(1)
    cfg = assemble('configs/fast.yaml', 'Base', device='cpu')
    cfg['replay_buffer']['capacity'] = 8
    env = make_env(cfg, 17)
    observation, state = env.reset(seed=17)
    agent = Agent(env, cfg)
    modules = agent.actors + [q for pair in agent.critics for q in pair]
    assert not any(isinstance(m, (nn.GRU, nn.GRUCell, nn.LSTM, nn.LSTMCell, nn.RNN))
                   for module in modules for m in module.modules())
    assert observation.shape == (5, SCHEMA.frame_dim)
    actor = agent.actors[0]
    obs = agent.tensor(observation[:2])
    torch.testing.assert_close(actor.normalized(obs), torch.tanh(actor.net(obs[:, :SCHEMA.base_dim])))
    actions = agent.joint_actions(agent.tensor(observation)[None])
    critic = agent.critics[0][0]
    global_state = agent.tensor(state)[None]
    inputs = torch.cat((global_state, (actions/critic.limits).flatten(1)), -1)
    expected = critic.net(inputs)
    torch.testing.assert_close(critic(global_state, actions, agent.tensor(observation)[None]), expected)
    assert critic.net[0].in_features == len(state)+10
    assert actor.net[0].in_features == SCHEMA.base_dim
    env.close()


@pytest.mark.parametrize('patch', [dict(encoder='gru'), dict(history_length=8), dict(gru_hidden=48)])
def test_stale_recurrent_config_fails_explicitly(patch):
    cfg = assemble(device='cpu')
    cfg['revision'].update(patch)
    with pytest.raises(ValueError, match='(?i)(single|frame|mlp|recurrent|gru)'):
        validate(cfg)


def test_only_backbone_identity_changes_without_task_or_reward_change():
    cfg = assemble(device='cpu')
    previous = deepcopy(cfg['hashes'])
    cfg['revision']['encoder'] = 'gru'
    cfg['revision'].update(history_length=8, gru_hidden=48, critic_gru_hidden=32)
    refresh_hashes(cfg)
    assert cfg['hashes']['architecture'] != previous['architecture']
    assert cfg['hashes']['task'] == previous['task']
    assert cfg['hashes']['reward'] == previous['reward']


def test_current_frame_adapter_does_not_retain_earlier_observations():
    cfg = assemble(device='cpu')
    env = make_env(cfg, 13)
    observation, _ = env.reset(seed=13)
    env.step(np.zeros((5, 2), np.float32))
    current = env.current_observation()
    assert current.shape == observation.shape == (5, SCHEMA.frame_dim)
    assert len(env.frames) == 1
    np.testing.assert_array_equal(current[:, SCHEMA.index('time_seconds')], np.full(5, .15, np.float32))
    env.close()
