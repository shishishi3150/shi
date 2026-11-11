"""Actual default-size learning validation; never starts formal training."""
from pathlib import Path
import json
import numpy as np
import torch
from matd3.agent import Agent
from scripts.train import make_env
from usv_env.config import assemble, code_hash
from scripts.experiment_data import plain


def _parameters(module):
    return torch.cat([p.detach().cpu().reshape(-1) for p in module.parameters()]).clone()


def run_device(output, device, collection_steps=262, update_count=2):
    torch.set_num_threads(1)
    cfg = assemble('configs/fast.yaml', 'Full', 5, device)
    env = make_env(cfg, 5); agent = Agent(env, cfg)
    obs, state = env.reset(seed=5)
    actors = [_parameters(actor) for actor in agent.actors]
    targets = [_parameters(actor) for actor in agent.actor_targets]
    resets = 0
    try:
        for index in range(collection_steps):
            context = env.learning_context(); mask = env.agent_active_mask()
            next_obs, next_state, reward, done, info = env.step(agent.random_actions())
            agent.add_transition(obs, state, env.last_executed_actions.copy(), reward,
                next_obs, next_state, info, mask, env.agent_active_mask(),
                current_context=context, next_context=env.learning_context(),
                nominal_actions=env.last_nominal_actions.copy())
            obs, state = next_obs, next_state
            if done:
                resets += 1; obs, state = env.reset(seed=5+resets)
        observed = []; updates = []; loss_calls = 0
        original_loss = torch.nn.functional.smooth_l1_loss
        def observe_loss(prediction, target, *args, **kwargs):
            nonlocal loss_calls
            if loss_calls % 2 == 0:
                values = target.detach().cpu().numpy()
                observed.append(dict(target=dict(shape=list(values.shape),
                    finite=bool(np.isfinite(values).all()), minimum=float(values.min()), maximum=float(values.max()))))
            loss_calls += 1
            return original_loss(prediction, target, *args, **kwargs)
        torch.nn.functional.smooth_l1_loss = observe_loss
        try:
            for _ in range(update_count):
                metrics = agent.update()
                if metrics is None: raise AssertionError('not enough committed real Replay rows')
                changes = {role+'_actor': dict(l2=float(torch.linalg.vector_norm(_parameters(agent.actors[i])-actors[i])))
                    for i, role in enumerate(('usv', 'auv'))}
                updates.append(dict(metrics=metrics, parameter_changes_from_initial=changes))
        finally:
            torch.nn.functional.smooth_l1_loss = original_loss
        assert all(row['target']['finite'] for row in observed)
        assert np.isfinite(agent.replay.agent_rewards[:len(agent.replay)]).all()
        for role in ('usv', 'auv'):
            assert updates[-1]['metrics'][role+'_critic_grad_norm'] > 0
            assert updates[-1]['metrics'][role+'_actor_grad_norm'] > 0
            assert updates[-1]['parameter_changes_from_initial'][role+'_actor']['l2'] > 0
        for module in [*agent.actors, *agent.actor_targets, *[q for pair in agent.critics for q in pair],
                       *[q for pair in agent.critic_targets for q in pair]]:
            assert all(torch.isfinite(p).all() for p in module.parameters())
        assert any(not torch.equal(_parameters(actor), target) for actor, target in zip(agent.actor_targets, targets))
        report = plain(dict(passed=True, scope='Finite numerical learning, not learned-policy success.',
            source_code_hash=code_hash(), config_hashes=cfg['hashes'], device=str(agent.device),
            dimensions=dict(batch=cfg['training']['batch_size'], hidden_sizes=cfg['training']['hidden_sizes'],
                n_step_return=cfg['training']['n_step_return'], gamma=cfg['training']['gamma'],
                i3_search_pool=cfg['i3']['search_pool'], i3_candidates=cfg['i3']['candidates']),
            collection=dict(real_transitions=collection_steps, committed_replay_rows=len(agent.replay), episode_resets=resets),
            critic_update_rounds=agent.update_steps, actor_updates=agent.actor_updates,
            actual_bellman_targets=observed, updates=updates,
            replay_reward_quantiles=np.percentile(agent.replay.agent_rewards[:len(agent.replay)], [0, 10, 50, 90, 100], axis=0),
            i3_return_scale=cfg['i3']['return_scale']))
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return report
    finally:
        env.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu', 'cuda'])
    args = parser.parse_args(argv)
    root = Path(args.output)
    if root.exists(): parser.error('use a new output directory')
    root.mkdir(parents=True)
    for device in args.devices:
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA unavailable: skipped', flush=True)
            continue
        report = run_device(root/device, device)
        print(json.dumps(dict(device=device, passed=report['passed'],
            critic_updates=report['critic_update_rounds'], actor_updates=report['actor_updates'])), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
