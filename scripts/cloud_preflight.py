"""Read-only cloud checks; no simulator, training run or full Replay allocation.

Example: NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 python -B -m scripts.cloud_preflight --device cuda
"""
import argparse
import json
import os
from pathlib import Path
import shlex

# Match scripts.train's deterministic CUDA contract in this process only.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numba
import torch
from matd3.networks import Actor, Critic
from matd3.replay_buffer import ReplayBuffer
from usv_env.config import PRESETS, assemble
from usv_env.history import schema_for_counts

METHODS = ('Full', 'Base', 'w_o_I1', 'w_o_I2', 'w_o_I3')
THREAD_ENV = ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def cpu_quota(root=Path('/sys/fs/cgroup')):
    """Best-effort mounted cgroup quota; absent files are normal off Linux."""
    try:
        values = (root/'cpu.max').read_text().split()
        return None if values[0] == 'max' else int(values[0])/int(values[1])
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    for folder in ('cpu', 'cpu,cpuacct', ''):
        try:
            quota = int((root/folder/'cpu.cfs_quota_us').read_text())
            period = int((root/folder/'cpu.cfs_period_us').read_text())
            return None if quota < 0 else quota/period
        except (OSError, ValueError, ZeroDivisionError):
            continue
    return None


def configuration_report(cfg):
    count = cfg['env']['num_surface_blue']+cfg['env']['num_submarines']
    schema = schema_for_counts(count)
    obs_dim = schema.frame_dim*cfg['revision']['history_length']
    # A single row, not capacity rows. This follows the real storage schema.
    row = ReplayBuffer(1, count, obs_dim, cfg['observation_schema']['critic_dim'], 2,
        surface_count=cfg['env']['num_surface_blue'])
    return dict(experiment=cfg['revision']['experiment'], task_hash=cfg['hashes']['task'],
        frame_dim=schema.frame_dim, actor_observation_dim=obs_dim,
        critic_state_dim=cfg['observation_schema']['critic_dim'], batch_size=cfg['training']['batch_size'],
        warmup_env_steps=cfg['training']['random_warmup_env_steps'],
        actor_policy_delay=cfg['training']['policy_delay'],
        critic_schedule='one update per post-warmup environment step in scripts.train',
        post_capture_total_end_step=cfg['task']['post_capture_end_step'],
        uncaptured_max_steps=cfg['env']['max_steps'], dt_seconds=cfg['env']['dt'],
        i3_search_pool=cfg['i3']['search_pool'], i3_candidates=cfg['i3']['candidates'],
        replay_capacity=cfg['replay_buffer']['capacity'],
        replay_array_bytes=row.array_payload_bytes*cfg['replay_buffer']['capacity'],
        replay_array_bytes_five_runs=row.array_payload_bytes*cfg['replay_buffer']['capacity']*5)


def _check_gradients(module, inputs, output):
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('nonfinite forward output')
    output.square().mean().backward()
    gradients = [x.grad for x in inputs]+[p.grad for p in module.parameters()]
    if any(x is None or not bool(torch.isfinite(x).all()) for x in gradients):
        raise FloatingPointError('missing or nonfinite backward gradient')


def device_check(cfg, device):
    """Exercise the actual current-frame actor and centralized critic MLPs."""
    selected = torch.device(device)
    details = dict(requested_device=str(selected), torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if selected.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this PyTorch environment')
        details.update(gpu=torch.cuda.get_device_name(selected),
            compute_capability=list(torch.cuda.get_device_capability(selected)),
            compiled_cuda_architectures=torch.cuda.get_arch_list())
    schema = schema_for_counts(cfg['env']['num_surface_blue']+cfg['env']['num_submarines'])
    batch = cfg['training']['batch_size']
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(cfg['training']['torch_num_threads'])
        torch.use_deterministic_algorithms(cfg['training']['deterministic_algorithms'])
        actor = Actor(cfg, [2.2, 1.5], [0., 0.]).to(selected)
        obs = torch.randn(batch, schema.frame_dim, device=selected)
        obs[:, -1] = 1.
        _check_gradients(actor, [], actor(obs))
        count = cfg['env']['num_surface_blue']+cfg['env']['num_submarines']
        limits = [[2.2, 1.5]]*cfg['env']['num_surface_blue']+[[3.035, 1.5]]*cfg['env']['num_submarines']
        critic = Critic(cfg, cfg['observation_schema']['critic_dim'], limits, 2).to(selected)
        state = torch.randn(batch, critic.state_dim, device=selected, requires_grad=True)
        actions = torch.randn(batch, count, 2, device=selected, requires_grad=True)
        _check_gradients(critic, [state, actions], critic(state, actions))
        if selected.type == 'cuda':
            torch.cuda.synchronize(selected)
            free, total = torch.cuda.mem_get_info(selected)
            details.update(free_vram_after_probe_bytes=free, total_vram_bytes=total)
        details.update(actor_mlp_forward_backward=True, critic_mlp_forward_backward=True,
            mlp_forward_backward=True, encoder='mlp', history_length=1,
            note='Compatibility probe only; not a training throughput or five-process VRAM estimate.')
        return details
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn)


def launch_commands(config):
    prefix = 'NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
    return [f'{prefix} python -B -m scripts.train --config {shlex.quote(str(config))} '
        f'--experiment {method} --episodes 1000 --seed 0 --device cuda '
        '--runs-root outputs/positive_reward --train-trace-every 0 --diagnostic-every 1000'
        for method in METHODS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fast.yaml')
    parser.add_argument('--experiment', choices=PRESETS, default='Full')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    cfg = assemble(args.config, args.experiment, 0, args.device)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    report = dict(ok=False, read_only=True, training_launched=False,
        configuration=configuration_report(cfg), cpu=dict(logical_count=os.cpu_count(),
            allowed_cpu_ids=affinity, mounted_cgroup_quota_cpus=cpu_quota(),
            thread_environment={key: os.environ.get(key) for key in THREAD_ENV},
            configured_torch_cpu_threads=cfg['training']['torch_num_threads'],
            numba_maximum_threads=numba.config.NUMBA_NUM_THREADS,
            numba_active_threads=numba.get_num_threads(),
            effective_i1_batch_threads=min(8, numba.get_num_threads())),
        warnings=[], launch_commands_bash=launch_commands(args.config))
    if report['cpu']['effective_i1_batch_threads'] != 3:
        report['warnings'].append('For the proposed three-thread budget, set NUMBA_NUM_THREADS=3 before Python starts.')
    if any(os.environ.get(key) != '1' for key in THREAD_ENV[1:]):
        report['warnings'].append('Set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 before Python starts to bound other CPU pools.')
    expected = dict(batch_size=256, warmup_env_steps=10000, actor_policy_delay=2,
        post_capture_total_end_step=650, i3_search_pool=256, i3_candidates=4)
    report['configuration_checks'] = {key: report['configuration'][key] == value for key, value in expected.items()}
    if not all(report['configuration_checks'].values()):
        report['warnings'].append('This configuration differs from the selected formal 256/4, batch256, warmup10000, delay2, total650 budget. Compare experiments using the same declared configuration.')
    report['warnings'].append('Five commands share one GPU; 15 CPU worker threads do not reserve nine cores or guarantee speed/VRAM. Test concurrency after warmup before long runs.')
    if args.device == 'cpu':
        report['warnings'].append('CPU-only probe does not validate the rented CUDA device.')
    try:
        report['device_check'] = device_check(cfg, args.device)
        report['ok'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['suggestion'] = 'For RTX 5060 Ti, use a Blackwell-compatible PyTorch CUDA 12.8 image and rerun. nvidia-smi CUDA version alone does not validate PyTorch; nothing was installed.'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
