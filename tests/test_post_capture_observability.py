"""Capture latch and effective deadline must be explicit policy information."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from matd3.checkpoint import POLICY_FORMAT, atomic_save, read
from scripts.train import make_env
from usv_env.config import TASK_CONTRACT, assemble
from usv_env.history import SCHEMA
from usv_env.observation import build_actor_observation, build_critic_state


def paired_649():
    cfg = assemble(experiment='Base', seed=515, device='cpu')
    env = make_env(cfg, 515)
    env.reset(seed=515)
    env.env.step_count = 649
    captured = deepcopy(env)
    captured.env.capture_success = True
    captured.env.first_capture_step = 321
    return env, captured


def test_capture_latch_and_effective_remaining_time_are_explicit_to_each_actor():
    uncaptured, captured = paired_649()
    capture_index = SCHEMA.index('capture_confirmed')
    remaining_index = SCHEMA.index('remaining_time')
    assert capture_index == remaining_index+1
    assert capture_index < SCHEMA.sensor_dim
    for receiver in range(5):
        before = build_actor_observation(uncaptured.env, receiver)
        after = build_actor_observation(captured.env, receiver)
        assert before[capture_index] == 0.
        assert after[capture_index] == 1.
        assert before[remaining_index] == pytest.approx(251/900)
        assert after[remaining_index] == pytest.approx(1/650)
        # Local measurements, geometry and actions are unchanged by the flag.
        unchanged = np.ones(len(before), bool)
        unchanged[[capture_index, remaining_index]] = False
        np.testing.assert_array_equal(before[unchanged], after[unchanged])


def test_critic_explicit_capture_latch_prevents_identical_650_terminal_states():
    uncaptured, captured = paired_649()
    before = build_critic_state(uncaptured.env)
    after = build_critic_state(captured.env)
    assert before.shape == after.shape == (uncaptured.critic_state_dim,)
    assert before[-1] == 0. and after[-1] == 1.
    assert not np.array_equal(before, after)
    # Actual terminal behavior still differs; the flag must explain it.
    action = np.zeros((5, 2), np.float32)
    *_, first_done, first_info = uncaptured.step(action)
    *_, second_done, second_info = captured.step(action)
    assert not first_done and not first_info.terminal_reason
    assert second_done and second_info.success and second_info.terminal_reason == 'success'


def test_history_packing_keeps_capture_and_deadline_in_last_valid_receiver_frame():
    _, env = paired_649()
    schema = env.observation_schema
    original = np.stack([build_actor_observation(env.env, receiver) for receiver in range(5)])
    history = env._push(original)
    frames = history.reshape(5, env.history_length, schema.frame_dim)
    assert history.shape == (5, schema.frame_dim*env.history_length)
    np.testing.assert_array_equal(frames[:, -1, schema.index('capture_confirmed')], np.ones(5))
    np.testing.assert_allclose(frames[:, -1, schema.index('remaining_time')], np.full(5, 1/650))
    np.testing.assert_array_equal(frames[:, -1, -1], np.ones(5))
    assert np.all(frames[:, :-2, -1] == 0.)


def test_current_reader_rejects_legacy_capture_unaware_checkpoint_before_network_load(tmp_path):
    cfg = assemble(experiment='Base', device='cpu')
    legacy = replace(SCHEMA, version='five_agent_local_estimates_no_slots_v2',
        context_fields=tuple(field for field in SCHEMA.context_fields if field != 'capture_confirmed'))
    cfg['observation_schema'].update(version=legacy.version, fields=list(legacy.fields),
        base_obs=legacy.base_dim, frame_dim=legacy.frame_dim, valid_frame_index=legacy.base_dim,
        actor_dim=legacy.frame_dim*cfg['revision']['history_length'], critic_dim=98)
    cfg['hashes']['observation'] = legacy.hash
    cfg['schema']['observation_version'] = 'force_control_local_information'
    payload = dict(format=POLICY_FORMAT, schema='force_control_local_information',
        task_contract=TASK_CONTRACT, runtime_config=cfg, actors=[{}, {}],
        limits=np.ones((5, 2), np.float32), cruise=np.zeros((5, 2), np.float32),
        global_env_steps=0, metadata={})
    path = tmp_path/'legacy_actor.pt'
    atomic_save(payload, path)
    with pytest.raises(ValueError, match='(?i)(historical|observation|schema|incompatible|hash)'):
        read(path)
