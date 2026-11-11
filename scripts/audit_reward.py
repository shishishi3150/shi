"""Physical reward diagnostics, with raw and discounted per-component ledgers."""
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from usv_env.config import assemble, code_hash
from usv_env.environment import PursuitEvasionEnv
from usv_env.red_policy import RuleController, _reference_action
from usv_env.reward import REWARD_COMPONENTS, EVENT_COMPONENTS, PAID_COMPONENTS
from scripts.experiment_data import EpisodeMetrics, atomic_json, plain


def run(mode, seed, root, *, config='configs/fast.yaml', limit=None):
    experiment = 'Base' if mode == 'reef_impact' else 'Full'
    cfg = assemble(config,experiment=experiment,seed=seed,device='cpu')
    source = code_hash(); env = PursuitEvasionEnv(cfg,seed); env.reset(seed=seed)
    controller = RuleController(); rng = np.random.default_rng(seed)
    metrics = EpisodeMetrics(env); started = time.perf_counter()
    dest = Path(root)/f'{mode}_seed{seed}'
    dest.mkdir()
    process = np.zeros(5); events = np.zeros(5); process_abs = np.zeros(5)
    discounted_process = np.zeros(5); discounted_events = np.zeros(5)
    discounted_abs = np.zeros(5); first_valid = [None]*5
    with gzip.open(dest/'trace.jsonl.gz','wt',encoding='utf-8') as stream:
        try:
            while True:
                if mode in ('stationary','reef_impact'):
                    actions = np.zeros((5,2),np.float32)
                    if mode == 'reef_impact':
                        actions[0] = _reference_action(env.blue_states[0],
                            np.asarray(env.reefs[2]['center']),.4,env.blue_dynamics)
                elif mode == 'random':
                    actions = rng.uniform(-1,1,(5,2)).astype(np.float32)*env.agent_action_limits
                else:
                    actions = controller.actions_for_env(env,ideal=mode=='rules_ideal')
                    if mode == 'one_auv_idle': actions[4] = 0.
                _,_,reward,done,info = env.step(actions)
                metrics.observe(env,reward,info)
                dense = sum(np.asarray(info.reward_terms[k]) for k in REWARD_COMPONENTS)
                event = sum(np.asarray(info.reward_terms[k]) for k in EVENT_COMPONENTS)
                np.testing.assert_allclose(reward,dense+event,atol=1e-5)
                discount = cfg['training']['gamma']**(env.step_count-1)
                process += dense; events += event; process_abs += np.abs(dense)
                discounted_process += discount*dense; discounted_events += discount*event
                discounted_abs += discount*np.abs(dense)
                for i,estimate in enumerate(env.target_estimates):
                    if estimate.valid and first_valid[i] is None: first_valid[i] = env.step_count
                stream.write(json.dumps(plain(dict(step=env.step_count,reward=reward,
                    components={key:info.reward_terms[key] for key in PAID_COMPONENTS},
                    valid=[e.valid for e in env.target_estimates],hold_steps=env.hold_steps,
                    capture=info.capture_just_confirmed,terminal=info.terminal_reason)),
                    allow_nan=False)+'\n')
                if done or (limit and env.step_count >= limit): break
            summary = plain(metrics.summary(env,info))
            cap = np.asarray(summary['reward_component_sums']['auv_capture'])
            dcap = np.asarray(summary['reward_component_discounted_sums']['auv_capture'])
            summary.update(mode=mode,seed=seed,experiment=experiment,source_code_hash=source,
                reward_hash=cfg['hashes']['reward'],physical_rollout=True,learned_policy=False,
                collector_limit=limit,process_return=process.tolist(),event_return=events.tolist(),
                discounted_process_return=discounted_process.tolist(),
                discounted_event_return=discounted_events.tolist(),
                process_abs_return=process_abs.tolist(),discounted_process_abs_return=discounted_abs.tolist(),
                capture_to_process_abs_ratio=np.divide(np.abs(cap),process_abs,
                    out=np.zeros(5),where=process_abs>0).tolist(),
                discounted_capture_to_process_abs_ratio=np.divide(np.abs(dcap),discounted_abs,
                    out=np.zeros(5),where=discounted_abs>0).tolist(),
                first_target_valid_step=first_valid,first_local_detection_step=env.first_detection_steps,
                wall_time_s=time.perf_counter()-started)
            np.testing.assert_allclose(process+events,metrics.returns,atol=1e-4)
            np.testing.assert_allclose(discounted_process+discounted_events,metrics.discounted_returns,atol=1e-4)
            c=cfg['reward']['engineering_initial']
            assert np.all(cap <= c['capture']) and np.all(cap[:2] == 0.)
            if source != code_hash(): raise RuntimeError('production source changed during rollout')
            atomic_json(dest/'summary.json',summary)
            print(json.dumps({key:summary[key] for key in ('mode','seed','terminal_reason',
                'episode_steps','team_return','discounted_team_return','first_capture_step')},
                allow_nan=False),flush=True)
            return summary
        finally:
            env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--config',default='configs/fast.yaml')
    p.add_argument('--seeds',nargs='+',type=int,default=[3,5])
    p.add_argument('--modes',nargs='+',choices=('rules_local','rules_ideal','stationary','random','reef_impact','one_auv_idle'),
                   default=['rules_local','stationary','random','reef_impact','one_auv_idle'])
    p.add_argument('--limit',type=int)
    args = p.parse_args(argv); root = Path(args.output)
    if root.exists() and any(root.iterdir()): p.error('refusing to overwrite audit output')
    root.mkdir(parents=True,exist_ok=True)
    rows = [run(mode,seed,root,config=args.config,limit=args.limit)
            for seed in args.seeds for mode in args.modes]
    atomic_json(root/'summary.json',rows)
    return 0


if __name__ == '__main__': main()
