"""Reproducible physical reward controls; never used as demonstration Replay."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from usv_env.config import assemble, code_hash
from usv_env.environment import PursuitEvasionEnv
from usv_env.red_policy import RuleController, _reference_action
from usv_env.reward import REWARD_COMPONENTS
from scripts.experiment_data import EpisodeMetrics, atomic_json, plain


def run(mode, seed, root):
    experiment = ('Base' if mode == 'reef_impact' else
                  'w_o_I1' if mode == 'rules_local_no_i1' else 'Full')
    cfg = assemble(experiment=experiment, seed=seed, device='cpu')
    source = code_hash(); env = PursuitEvasionEnv(cfg, seed)
    env.reset(seed=seed); controller = RuleController()
    metrics = EpisodeMetrics(env); rows = []; started = time.perf_counter()
    process = terminal = absolute = 0.
    try:
        while True:
            if mode in ('stationary', 'reef_impact'):
                actions = np.zeros((5,2), np.float32)
                if mode == 'reef_impact':
                    actions[0] = _reference_action(env.blue_states[0],
                        np.asarray(env.reefs[2]['center']), .4, env.blue_dynamics)
            else:
                actions = controller.actions(env.blue_states, env.submarine_states, None,
                    env.protected_center, env.reefs, env.world_size, env.blue_dynamics,
                    env.submarine_dynamics, env.step_count*env.dt,
                    target_estimates=env.target_estimates, active_mask=env.agent_active_mask())
                if mode == 'one_auv_idle': actions[4] = 0.
            _, _, reward, done, info = env.step(actions)
            metrics.observe(env, reward, info)
            parts = sum(np.asarray(info.reward_terms[k]) for k in REWARD_COMPONENTS)
            final = np.asarray(info.reward_terms['terminal_reward'])
            np.testing.assert_allclose(reward, parts+final, atol=1e-6)
            scheme = cfg['reward']['scheme']
            if scheme in ('cooperative_positive_v1', 'cooperative_positive_v2'):
                bound = cfg['reward']['positive']['step_budget']
                if info.capture_just_confirmed:
                    bound += cfg['reward']['positive']['capture_bonus']
                if (np.any(reward < 0) or np.any(parts < 0) or np.any(parts > bound+1e-8)
                        or (info.terminal_reason and not info.success and np.any(reward))):
                    raise ArithmeticError('positive reward or zero-failure budget violated')
            elif scheme == 'cooperative_cost_v2' and (np.any(reward > 0) or np.any(parts < -.1-1e-8)):
                raise ArithmeticError('negative reward budget violated')
            process += float(parts.mean()); terminal += float(final.mean())
            absolute += float(np.mean(np.abs(parts)))
            rows.append(plain(dict(step=env.step_count, time_s=env.step_count*env.dt,
                rewards=reward, process_reward=float(parts.mean()), terminal_reward=float(final.mean()),
                auv_distance_quality=info.reward_terms['auv_distance_quality'][2],
                auv_gap_quality=info.reward_terms['auv_gap_quality'][2],
                auv_hull_quality=info.reward_terms['auv_hull_quality'][2],
                hold_steps=env.hold_steps, terminal_reason=info.terminal_reason)))
            if done: break
        summary = plain(metrics.summary(env, info))
        summary.update(mode=mode, seed=seed, experiment=experiment,
            source_code_hash=source, reward_hash=cfg['hashes']['reward'],
            physical_rollout=True, learned_policy=False,
            process_return=process, terminal_return=terminal,
            terminal_abs_fraction=abs(terminal)/(absolute+abs(terminal)) if absolute+abs(terminal) else 0.,
            success_step_reward=reward.tolist() if info.success else None,
            wall_time_s=time.perf_counter()-started)
        lower, upper = ((0., cfg['reward']['positive']['episode_budget'])
            if cfg['reward']['scheme'] in ('cooperative_positive_v1', 'cooperative_positive_v2') else (-100., 0.))
        if not lower-1e-6 <= summary['team_return'] <= upper+1e-6:
            raise ArithmeticError('episode reward budget violated')
        if source != code_hash(): raise RuntimeError('production source changed during rollout')
        dest = root/f'{mode}_seed{seed}'; dest.mkdir()
        atomic_json(dest/'summary.json', summary)
        atomic_json(dest/'trace.json', rows)
        print(json.dumps({k:summary[k] for k in ('mode','seed','terminal_reason','episode_steps',
            'team_return','discounted_team_return','terminal_abs_fraction')}, allow_nan=False), flush=True)
        return summary
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[5])
    parser.add_argument('--modes', nargs='+', choices=('rules_local','rules_local_no_i1','stationary','reef_impact','one_auv_idle'),
                        default=['rules_local','stationary','reef_impact','one_auv_idle'])
    args = parser.parse_args(argv); root = Path(args.output)
    if root.exists() and any(root.iterdir()): parser.error('refusing to overwrite audit output')
    root.mkdir(parents=True, exist_ok=True)
    summaries = [run(mode, seed, root) for seed in args.seeds for mode in args.modes]
    atomic_json(root/'summary.json', summaries)
    return 0


if __name__ == '__main__': main()
