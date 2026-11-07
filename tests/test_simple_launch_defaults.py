"""The short nohup entry point carries the selected formal launch settings."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import train
from scripts.visualize import ROOT, _checkpoint_path, parse_args
from usv_env.config import assemble


@pytest.mark.parametrize('method', ['Base', 'w_o_I1', 'w_o_I2', 'w_o_I3', 'Full'])
def test_short_episode_command_resolves_selected_formal_defaults(method, capsys):
    train.main(['--experiment', method, '--seed', '0', '--episodes', '3000',
        '--device', 'auto', '--prepare-only'])
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg['i3']['search_pool'] == 256 and cfg['i3']['candidates'] == 4
    assert cfg['provenance']['task_file'] == 'configs/fast.yaml'
    assert Path(cfg['run_dir']).as_posix() == f'outputs/positive_reward/{method}/seed0'
    assert cfg['diagnostics']['train_trace_every'] == 0
    assert cfg['diagnostics']['update_every'] == 1000
    assert cfg['diagnostics']['compression'] == 'gzip'
    assert cfg['diagnostics']['export_csv'] is False
    assert cfg['training_budget'] == dict(unit='episodes', requested=3000,
        maximum_online_steps=3000*900)
    expected = assemble(ROOT/'configs/fast.yaml', method, 0, 'auto')
    for key in ('training', 'i1', 'i3', 'task', 'env', 'ship', 'reward', 'mechanisms'):
        assert cfg[key] == expected[key], key


def test_explicit_config_and_runs_root_still_override_defaults(tmp_path, capsys):
    train.main(['--config', str(ROOT/'configs/default.yaml'), '--experiment', 'Full',
        '--episodes', '3000', '--runs-root', str(tmp_path/'chosen'),
        '--train-trace-every', '2', '--prepare-only'])
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg['i3']['search_pool'] == 512 and cfg['i3']['candidates'] == 8
    assert (ROOT/cfg['run_dir']).resolve() == (tmp_path/'chosen/Full/seed0').resolve()
    assert cfg['diagnostics']['train_trace_every'] == 2


def test_visualize_defaults_match_new_full_training_directory():
    checkpoint = _checkpoint_path({}, parse_args([]))
    assert checkpoint == ROOT/'outputs/positive_reward/Full/seed0/latest.pt'


def test_direct_script_sets_cpu_pools_before_numerical_imports():
    script = '''import json, os, runpy, sys
sys.argv = ['scripts/train.py', '--experiment', 'Full', '--episodes', '3000', '--device', 'cpu', '--prepare-only']
runpy.run_path('scripts/train.py', run_name='__main__')
import numba, torch
keys = ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
print('THREAD_CHECK=' + json.dumps(dict(environment={k:os.environ[k] for k in keys},
    numba_threads=numba.get_num_threads(), torch_threads=torch.get_num_threads())))
'''
    env = dict(os.environ, NUMBA_NUM_THREADS='7', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
        env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    marker = next(row for row in result.stdout.splitlines() if row.startswith('THREAD_CHECK='))
    actual = json.loads(marker.split('=', 1)[1])
    assert actual['environment'] == dict(NUMBA_NUM_THREADS='3', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    assert actual['numba_threads'] == 3 and actual['torch_threads'] == 1
