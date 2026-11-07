"""Cloud probes inspect real kernels without launching a training run."""
from copy import deepcopy
import json

import pytest
import torch

from scripts import cloud_preflight as probe
from usv_env.config import assemble
from usv_env.history import schema_for_counts


def tiny_configuration():
    cfg = assemble('configs/fast.yaml', experiment='Full', device='cpu')
    cfg['training'].update(batch_size=4, hidden_sizes=[8], torch_num_threads=1)
    return cfg


def test_replay_estimate_and_dimensions_follow_current_schema():
    cfg = assemble('configs/fast.yaml', experiment='Full', device='cpu')
    before = deepcopy(cfg)
    report = probe.configuration_report(cfg)
    assert cfg == before
    assert report['frame_dim'] == schema_for_counts(5).frame_dim
    assert report['actor_observation_dim'] == schema_for_counts(5).frame_dim
    assert report['critic_state_dim'] == cfg['observation_schema']['critic_dim']
    assert report['i3_search_pool'] == 256 and report['i3_candidates'] == 4
    assert report['warmup_env_steps'] == 10000 and report['post_capture_total_end_step'] == 650
    assert report['replay_array_bytes_five_runs'] == report['replay_array_bytes']*5
    assert report['replay_array_bytes'] < 1024**3


def test_actual_cpu_forward_backward_and_settings_restored():
    before = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    report = probe.device_check(tiny_configuration(), 'cpu')
    for key in ('actor_mlp_forward_backward', 'critic_mlp_forward_backward', 'mlp_forward_backward'):
        assert report[key] is True
    assert before == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_actual_cuda_actor_and_critic_mlp_backward():
    report = probe.device_check(tiny_configuration(), 'cuda')
    assert report['requested_device'] == 'cuda'
    assert report['actor_mlp_forward_backward'] and report['critic_mlp_forward_backward']
    assert report['history_length'] == 1 and report['encoder'] == 'mlp'
    assert report['total_vram_bytes'] > 0


@pytest.mark.parametrize('content,expected', [('max 100000', None), ('300000 100000', 3.), ('broken', None)])
def test_v2_cpu_quota_is_optional_and_uses_cpu_units(tmp_path, content, expected):
    (tmp_path/'cpu.max').write_text(content)
    assert probe.cpu_quota(tmp_path) == expected


def test_v1_cpu_quota_and_missing_files(tmp_path):
    assert probe.cpu_quota(tmp_path) is None
    folder = tmp_path/'cpu'
    folder.mkdir()
    (folder/'cpu.cfs_quota_us').write_text('150000')
    (folder/'cpu.cfs_period_us').write_text('100000')
    assert probe.cpu_quota(tmp_path) == 1.5
    (folder/'cpu.cfs_quota_us').write_text('-1')
    assert probe.cpu_quota(tmp_path) is None


def test_cli_failure_reports_compatibility_without_training(monkeypatch, capsys):
    def unavailable(cfg, device):
        raise RuntimeError('unsupported GPU architecture')
    monkeypatch.setattr(probe, 'device_check', unavailable)
    assert probe.main(['--device', 'cuda']) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['read_only'] and not report['training_launched']
    assert 'CUDA 12.8' in report['suggestion']
    assert len(report['launch_commands_bash']) == 5
    for method, command in zip(probe.METHODS, report['launch_commands_bash']):
        assert f'--experiment {method} ' in command
        assert 'NUMBA_NUM_THREADS=3' in command
        assert '--config configs/fast.yaml' in command


def test_thread_warnings_do_not_change_environment(monkeypatch, capsys):
    for key in probe.THREAD_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(probe, 'device_check', lambda cfg, device: {'cpu': True})
    assert probe.main(['--device', 'cpu']) == 0
    report = json.loads(capsys.readouterr().out)
    assert any('CPU-only' in warning for warning in report['warnings'])
    assert all(probe.os.environ.get(key) is None for key in probe.THREAD_ENV)
