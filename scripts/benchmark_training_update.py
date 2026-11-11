"""Paired old/new update timings on identical actual Replay and training settings.

Loads only the user's own pre-change source backup. No formal training run or
checkpoint is written. CUDA timings include synchronization and both policy phases.
"""
import os
if __name__ == '__main__':
    os.environ.update(NUMBA_NUM_THREADS='3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import types
import zipfile
import numpy as np
import torch
import numba
from matd3.agent import Agent
from scripts.train import make_env
from scripts.experiment_data import plain
from usv_env.config import ROOT, assemble, code_hash
from usv_env.red_policy import RuleController


def references(backup):
    with zipfile.ZipFile(backup) as archive:
        def module(package, name, path, replacements=()):
            result = types.ModuleType(package+'.'+name)
            result.__package__ = package
            source = archive.read(path).decode('utf-8-sig')
            for old, new in replacements:
                source = source.replace(old, new)
            sys.modules[result.__name__] = result
            exec(compile(source, '<pre-speed/'+path+'>', 'exec'), result.__dict__)
            return result
        vo = module('usv_env', '_speed_reference_vo', 'usv_env/vo_guidance.py')
        history = module('matd3', '_speed_reference_history', 'matd3/history_enhancement.py')
        agent = module('matd3', '_speed_reference_agent', 'matd3/agent.py', [
            ('from usv_env.vo_guidance import project_from_state',
             'project_from_state = _reference_projection')])
        agent._reference_projection = vo.project_from_state
        agent.JointHistory = history.JointHistory
    return agent.Agent


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def digest(agent):
    modules = agent.actors+agent.actor_targets+[q for pair in agent.critics+agent.critic_targets for q in pair]
    return hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes()
        for module in modules for value in module.state_dict().values())).hexdigest()


def collect():
    cfg = assemble('configs/fast.yaml', 'Full', 5, 'cpu')
    env = make_env(cfg, 5, variant='fixed')
    controller = RuleController(); obs, state = env.reset(seed=5)
    rows = []; environment = []
    try:
        while True:
            context = env.learning_context()
            action = controller.actions_for_env(env, ideal=False)
            started = time.perf_counter()
            no, ns, reward, done, info = env.step(action)
            if env.step_count > 20:
                environment.append(time.perf_counter()-started)
            rows.append(dict(obs=obs.copy(), state=state.copy(), action=env.last_executed_actions.copy(),
                reward=reward.copy(), next_obs=no.copy(), next_state=ns.copy(),
                info=types.SimpleNamespace(terminated=info.terminated,truncated=info.truncated,
                    timeout=info.timeout,terminal_reason=info.terminal_reason),
                current_context=deepcopy(context), next_context=deepcopy(env.learning_context()),
                nominal_actions=env.last_nominal_actions.copy()))
            obs, state = no, ns
            if done:
                return rows, dict(steps=env.step_count, first_capture_step=info.first_capture_step,
                    final_reason=info.terminal_reason, mean_environment_ms=1000*float(np.mean(environment)))
    finally:
        env.close()


def stats(samples):
    a = 1000*np.asarray(samples)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
        p10_ms=float(np.percentile(a,10)), p90_ms=float(np.percentile(a,90)), samples_ms=a.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-backup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', choices=('cpu','cuda'), default=['cpu','cuda'])
    parser.add_argument('--pairs', type=int, default=40)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('use a new output file')
    if args.pairs < 10:
        parser.error('at least ten paired samples required')
    torch.set_num_threads(1); numba.set_num_threads(min(3,numba.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    source = code_hash(); ReferenceAgent = references(args.reference_backup)
    rows, rollout = collect()
    groups = []
    for device in args.devices:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        for experiment in ('Base','Full'):
            cfg = assemble('configs/fast.yaml', experiment, 5, device)
            jobs = []
            for label, cls in (('before',ReferenceAgent),('after',Agent)):
                env = make_env(cfg,5,variant='fixed'); agent = cls(env,cfg)
                for row in rows:
                    agent.add_transition(row['obs'],row['state'],row['action'],row['reward'],
                        row['next_obs'],row['next_state'],row['info'],
                        current_context=row['current_context'], next_context=row['next_context'],
                        nominal_actions=row['nominal_actions'])
                jobs.append(dict(label=label,agent=agent,env=env,times=[],critic=[],actor=[],metrics=[]))
            assert digest(jobs[0]['agent']) == digest(jobs[1]['agent'])
            for _ in range(4):
                for job in jobs:
                    job['agent'].update(); job['agent'].update()
                synchronize(device)
            for sample in range(args.pairs):
                for job in (jobs if sample%2 == 0 else list(reversed(jobs))):
                    pair = []
                    for _ in range(2):
                        synchronize(device); started = time.perf_counter()
                        metrics = job['agent'].update()
                        synchronize(device); elapsed = time.perf_counter()-started
                        pair.append(elapsed)
                        job['actor' if job['agent'].update_steps%2 == 0 else 'critic'].append(elapsed)
                        job['metrics'].append(metrics)
                    job['times'].append(sum(pair)/2)
                left, right = (job['agent'] for job in jobs)
                np.testing.assert_array_equal(left.last_sample_indices,right.last_sample_indices)
                np.testing.assert_array_equal(left.last_target_noise,right.last_target_noise)
                if left.last_i3 is not None:
                    for name in left.last_i3:
                        np.testing.assert_array_equal(left.last_i3[name],right.last_i3[name],err_msg=name)
                assert digest(left) == digest(right), 'updated parameters differ'
                for old, new in zip(jobs[0]['metrics'][-2:],jobs[1]['metrics'][-2:]):
                    for name in old:
                        if name.endswith('_time_s'):
                            continue
                        assert old[name] == new[name], (name,old[name],new[name])
            before, after = (stats(job['times']) for job in jobs)
            result = dict(experiment=experiment,device=device,before=before,after=after,
                elapsed_time_reduction_percent=100*(1-after['mean_ms']/before['mean_ms']),
                speedup=before['mean_ms']/after['mean_ms'],
                critic_only={job['label']:stats(job['critic']) for job in jobs},
                actor_phase={job['label']:stats(job['actor']) for job in jobs},
                bitwise_parameters_equal=True, sampling_noise_i3_and_metrics_equal=True,
                parameter_sha256=digest(jobs[0]['agent']),
                dimensions=dict(batch_size=cfg['training']['batch_size'],hidden_sizes=cfg['training']['hidden_sizes'],
                    n_step=cfg['training']['n_step_return'],policy_delay=cfg['training']['policy_delay'],
                    i3=cfg['i3'],replay_capacity=cfg['replay_buffer']['capacity']),
                nested_after_ms={name:1000*float(np.mean([m.get(name,0.) or 0. for m in jobs[1]['metrics']]))
                    for name in ('i1_target_vo_time_s','i1_actor_vo_time_s','i3_vo_screen_time_s')})
            groups.append(result)
            for job in jobs:
                job['env'].close()
            print(json.dumps({key:result[key] for key in
                ('experiment','device','elapsed_time_reduction_percent','bitwise_parameters_equal')}
                | dict(before_ms=before['mean_ms'],after_ms=after['mean_ms'])),flush=True)
    assert source == code_hash(), 'source changed during benchmark'
    report = plain(dict(source_code_hash=source,reference_backup=str(args.reference_backup.resolve()),
        reference_sha256=hashlib.sha256(args.reference_backup.read_bytes()).hexdigest(),
        hardware=dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_threads=torch.get_num_threads(),numba_threads=numba.get_num_threads()),
        config='configs/fast.yaml',paired_samples=args.pairs,updates_per_sample=2,
        common_real_replay_rows=len(rows),rollout=rollout,groups=groups,
        scope='Same-process alternating before/after timings; actual common Replay; unchanged settings. '
              'Finite update equivalence, not a learned-policy convergence test. Replay contains one real episode, '
              'not the long-run 20000-row phase distribution. JIT/startup/collection/evaluation excluded from update time.'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
