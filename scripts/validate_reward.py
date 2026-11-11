"""Real physical reward checks; independent of disposable documentation files."""
from pathlib import Path
import json
import math
import numpy as np
from usv_env.config import assemble, refresh_hashes
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from usv_env.red_policy import _reference_action
from usv_env.reward import PAID_COMPONENTS
from scripts.experiment_data import EpisodeMetrics, plain


def diagnostic_config():
    """Explicit constructed diagnostic: translating triangle, real dynamics/sensors."""
    cfg = assemble('configs/fast.yaml', 'Base', 5, 'cpu')
    cfg['counterfactual']['enabled'] = False
    cfg['wave']['height_range_m'] = [0., 0.]
    cfg['sensors'].update(range_sigma_m=1.e-5, bearing_sigma_rad=1.e-6,
        minimum_detection_probability=1.)
    center = np.array([10., 27.])
    cfg['scenario'].update(red_start=center.tolist(), red_heading=0.,
        blue_starts=[[35., 45.], [40., 45.]], blue_headings=[0., 0.],
        submarine_starts=[(center+3.5*np.array([math.cos(a), math.sin(a)])).tolist()
            for a in (0., 2*math.pi/3, 4*math.pi/3)], submarine_headings=[0., 0., 0.],
        initial_submarine_speed=.4, initial_surface_speed=0.)
    refresh_hashes(cfg)
    return cfg


def capture_environment(start_step=0):
    env = PursuitEvasionEnv(diagnostic_config(), 5)
    env.step_count = start_step
    return env


def translation_actions(env, brake=False):
    actions = np.zeros((5, 2), np.float32)
    for j, ship in enumerate(env.submarine_states):
        actions[j+2] = ([-env.agent_action_limits[j+2, 0], 0.] if brake else
            _reference_action(ship, ship.pos+np.array([100., 0.]), .4, env.submarine_dynamics))
    return actions


def rollout(name, env, action_fn, *, output_dir, collect_replay=False, learn_after_rollout=False):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    current = HistoryEnv(env, 1)
    obs, state = current.reset(seed=env.seed)
    if collect_replay:
        from matd3.agent import Agent
        agent = Agent(current, env.config)
    else:
        agent = None
    metrics = EpisodeMetrics(env); rows = []
    for _ in range(env.max_steps):
        context = current.learning_context(); mask = current.agent_active_mask()
        next_obs, next_state, reward, done, info = current.step(action_fn(current))
        metrics.observe(current, reward, info)
        if agent is not None:
            agent.add_transition(obs, state, current.last_executed_actions.copy(), reward,
                next_obs, next_state, info, mask, current.agent_active_mask(),
                current_context=context, next_context=current.learning_context(),
                nominal_actions=current.last_nominal_actions.copy())
        rows.append(dict(step=env.step_count, reward=reward.tolist(),
            bootstrap_mask=info.reward_terms['bootstrap_mask'], phase=info.phase,
            capture_just_confirmed=info.capture_just_confirmed,
            components={key: info.reward_terms[key] for key in PAID_COMPONENTS}))
        obs, state = next_obs, next_state
        if done: break
    summary = metrics.summary(env, info)
    result = dict(name=name, end_step=env.step_count, final_reason=info.terminal_reason,
        first_capture_step=info.first_capture_step,
        capture_event_count=sum(row['capture_just_confirmed'] for row in rows),
        per_agent_return=metrics.returns,
        per_agent_discounted_return=metrics.discounted_returns,
        components={key: dict(cumulative=value,
            discounted=metrics.component_discounted_sums[key])
            for key, value in metrics.component_sums.items()}, summary=summary)
    if agent is not None:
        phases = agent.replay._unpack_phase_activity(agent.replay.phase_ids[:len(agent.replay)])[0]
        result['replay'] = dict(transitions=len(agent.replay),
            post_capture_phase_transitions=int(np.count_nonzero(phases[:, 0] == 3)),
            all_rewards_finite=bool(np.isfinite(agent.replay.agent_rewards[:len(agent.replay)]).all()))
        if info.first_capture_step is not None and len(rows) > info.first_capture_step+env.config['training']['n_step_return']:
            index = info.first_capture_step+1
            expected = sum(env.config['training']['gamma']**k*np.array(rows[index+k]['reward'])
                for k in range(env.config['training']['n_step_return']))
            np.testing.assert_allclose(agent.replay.agent_rewards[index], expected, rtol=1e-6, atol=1e-5)
            result['replay']['post_capture_nstep_reward_verified'] = True
        if learn_after_rollout:
            import torch
            before = [[p.detach().clone() for p in actor.parameters()] for actor in agent.actors]
            updates = [agent.update(), agent.update()]
            assert all(update is not None for update in updates)
            changed = [any(not torch.equal(a, b) for a, b in zip(old, actor.parameters()))
                for old, actor in zip(before, agent.actors)]
            assert all(changed)
            result['replay']['post_rollout_learning'] = dict(updates=updates,
                usv_actor_changed=changed[0], auv_actor_changed=changed[1])
    result = plain(result)
    (output_dir/(name+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result, rows


def main(argv=None):
    from scripts.validate_sustained_reward import main as current_main
    return current_main(argv)


if __name__ == '__main__':main()
