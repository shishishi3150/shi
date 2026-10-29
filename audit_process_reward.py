"""Finite USV-only interventions with one common AUV source per comparison.

Frozen learned AUV actors receive each branch's actual history observations.
The separate diagnostic-AUV family is not learned-policy performance. No
diagnostic transition enters replay and no environment rule is changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import pickle

import numpy as np
import torch

from matd3.checkpoint import InferencePolicy, policy_from_payload, unpack
from usv_env.config import assemble, refresh_hashes
from usv_env.environment import PursuitEvasionEnv
from usv_env.geometry import wrap_angle
from usv_env.history import HistoryEnv
from usv_env.reward import boundary_clearance, capsule_clearance, qualities, surface_reference

SEEDS = list(range(920001, 920011))
POLICY_MODES = ('learned_usv', 'reference', 'same_side', 'station_keep', 'reverse_outside',
                'role_inner_hold', 'role_edge_sweep', 'no_apf_position_effect')
DIAGNOSTIC_MODES = tuple(m for m in POLICY_MODES if m not in ('learned_usv', 'no_apf_position_effect'))


def frozen_policy(path):
    """Audit-only inference across reward contracts; no old replay may be loaded."""
    data = Path(path).read_bytes()
    payload = policy_from_payload(unpack(torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)))
    saved = payload['runtime_config']; stored = deepcopy(saved['hashes'])
    if refresh_hashes(saved) != stored:
        raise ValueError('frozen policy has inconsistent stored hashes')
    current = assemble(experiment=saved['revision']['experiment'])
    if any(stored[k] != current['hashes'][k] for k in ('task', 'observation', 'architecture')):
        raise ValueError('frozen policy task/observation/architecture is incompatible')
    policy = InferencePolicy(payload, 'cpu')
    for actor in policy.actors:
        for parameter in actor.parameters(): parameter.requires_grad_(False)
    policy.audit_identity = dict(path=Path(path).as_posix(), sha256=hashlib.sha256(data).hexdigest(),
        experiment=saved['revision']['experiment'], mask=saved['revision']['mask'],
        steps=payload['global_env_steps'], observation_hash=stored['observation'], reward_hash=stored['reward'])
    return policy


def new_env(seed):
    h = HistoryEnv(PursuitEvasionEnv(assemble(experiment='Base')), 8)
    h.set_evaluation_variant('randomized'); h.reset(seed=seed)
    return h


def state_hash(h):
    # Includes physical, wave, communication, random-generator and history states.
    return hashlib.sha256(pickle.dumps(h, protocol=5)).hexdigest()


def steer(e, i, velocity, reverse=False):
    ship = e.friendly_states()[i]; dynamics = e.blue_dynamics if i < 2 else e.submarine_dynamics
    speed = float(np.linalg.norm(velocity))
    heading = math.atan2(velocity[1], velocity[0]) if speed > 1e-8 else ship.heading
    if reverse: heading = float(wrap_angle(heading+math.pi))
    angle = float(wrap_angle(heading-ship.heading))
    surge = (-1 if reverse else 1)*speed*max(0., math.cos(angle))
    return dynamics.reference_action(ship,
        desired_yaw_rate=float(np.clip(1.2*angle, -dynamics.max_yaw_rate, dynamics.max_yaw_rate)),
        desired_surge_acceleration=.8*(surge-ship.speed))


def auv_actions(h, source, policy):
    if source == 'stationary':
        # Feasible no-task-progress control, used for timeout/landing coverage.
        # This is an explicit all-agent intervention, not a learned AUV policy.
        return np.array([steer(h.env, i, np.zeros(2)) for i in range(2, 5)], dtype=np.float32)
    if source == 'frozen_policy':
        if policy is None: raise ValueError('a compatible frozen AUV policy is required')
        return policy.act(h.current_observation())[2:].copy()
    if source != 'diagnostic': raise ValueError('unknown AUV source')
    desired = h.env._assigned_guidance_targets()[2]
    return np.array([steer(h.env, i, desired[i]) for i in range(2, 5)], dtype=np.float32)


def diagnostic_actions(h, mode, anchors, *, auv_source, policy=None, kind=None, branch_start=None):
    """AUV source is mandatory and independent of the USV intervention."""
    e = h.env; action = np.zeros((5, 2), dtype=np.float32)
    action[2:] = auv_actions(h, auv_source, policy)
    if mode == 'learned_usv':
        if policy is None: raise ValueError('learned USV comparison requires a frozen policy')
        action[:2] = policy.act(h.current_observation())[:2]
    else:
        for i, ship in enumerate(e.blue_states):
            reference = surface_reference(e, i)
            if mode == 'station_keep':
                requested = .08*(anchors[i]-ship.pos)
            elif mode == 'same_side':
                requested = e.red_state.velocity+.08*(e.red_state.pos+7*surface_reference(e, 0)['axis']-ship.pos)
            elif mode == 'reverse_outside':
                away = ship.pos-reference['target']
                requested = .3*away/max(np.linalg.norm(away), 1e-12)
            elif mode in ('role_inner_hold', 'role_edge_sweep'):
                # Audit controls use the EXISTING sectors; learning geometry is unchanged.
                radius = e.reward_cfg['surface']['inner'] if mode == 'role_inner_hold' else 7.
                angle = 0. if mode == 'role_inner_hold' else math.radians(20)*(1 if (h.step_count//120+i) % 2 == 0 else -1)
                axis = reference['axis']; c, s = math.cos(angle), math.sin(angle)
                direction = np.array([c*axis[0]-s*axis[1], s*axis[0]+c*axis[1]])
                requested = e.red_state.velocity+.08*(e.red_state.pos+radius*direction-ship.pos)
            else:
                requested = reference['velocity']
            requested *= min(1., .4/max(np.linalg.norm(requested), 1e-12))
            action[i] = steer(e, i, requested, reverse=mode == 'reverse_outside')
        if mode == 'early_failure':
            if kind == 'collision':
                for i in range(2):
                    towards = e.blue_states[1-i].pos-e.blue_states[i].pos
                    action[i] = steer(e, i, .4*towards/max(np.linalg.norm(towards), 1e-12))
            elif kind == 'out_of_bounds':
                ship = e.blue_states[0]
                distances = [ship.pos[0], e.world_size[0]-ship.pos[0], ship.pos[1], e.world_size[1]-ship.pos[1]]
                action[0] = steer(e, 0, .4*np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])[np.argmin(distances)])
            else: raise ValueError('early failure needs a physical failure kind')
        if mode == 'brake_reverse_recover' and h.step_count-branch_start < 100:
            for i in (range(2) if kind == 'collision' else (0,)):
                ship = e.blue_states[i]
                action[i] = e.blue_dynamics.reference_action(ship, desired_yaw_rate=0.,
                    desired_surge_acceleration=.8*(-.3-ship.speed))
    return np.clip(action, -e.agent_action_limits, e.agent_action_limits)


def aggregate(rows):
    rows = [r for r in rows if 'team_return' in r]
    task, risk, event = [sum(r[k] for r in rows) for k in ('A_task', 'A_risk', 'A_event')]
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows), A_task=task, A_risk=risk, A_event=event,
        dense_share=(task+risk)/(task+risk+event) if task+risk+event else None,
        risk_fraction_of_dense=risk/(task+risk) if task+risk else None,
        mean_signed_process=float(np.mean([r['team_return']-r['event_return'] for r in rows])) if rows else None)


def run(h, seed, mode, policy, anchors, auv_source, *, kind=None, trace_file=None, usv_tape=None):
    initial_hash = state_hash(h); start = h.step_count; e = h.env
    initial = qualities(e); initial_radius = e.formation_radius
    initial_red_clearance = float(np.linalg.norm(e.red_state.pos-e.island_center)-e.island_radius-e.landing_buffer)
    if mode == 'no_apf_position_effect':
        if usv_tape is None: raise ValueError('APF intervention requires the pre-frozen reference USV tape')
        e.surface_apf_enabled = False  # Leaves all five bodies, activity masks and collisions intact.
    total = np.zeros(5); discounted = np.zeros(5); signed = {}; absolute = {}
    task = risk = event_abs = event_return = 0.; records = []; tape = []
    entry = start if h.compaction_started else None
    while True:
        observations = h.current_observation()
        action = diagnostic_actions(h, mode, anchors, auv_source=auv_source, policy=policy,
                                    kind=kind, branch_start=start)
        if usv_tape is not None: action[:2] = usv_tape[h.step_count-start]
        _, _, reward, done, info = h.step(action)
        tape.append(action[:2].copy()); q = qualities(e); terms = info.reward_terms
        t = h.step_count-start-1; total += reward
        discounted += e.config['training']['gamma']**t*reward
        for role, parts in (('usv', terms['usv_contributions']), ('auv', terms['auv_contributions'])):
            for key, value in parts.items():
                full_key = role+'.'+key
                signed[full_key] = signed.get(full_key, np.zeros_like(value))+value
                absolute[full_key] = absolute.get(full_key, np.zeros_like(value))+np.abs(value)
                if key in ('safety', 'spacing'): risk += float(np.abs(value).sum())/5
                else: task += float(np.abs(value).sum())/5
        event_return += terms['team_terminal_reward']; event_abs += abs(terms['team_terminal_reward'])
        assert abs(float(reward.mean())-terms['canonical_team_dense_reward']-terms['team_terminal_reward']) < 1e-6
        towards = e.island_center-e.red_state.pos; distance = float(np.linalg.norm(towards))
        closing = float(e.red_state.velocity @ (towards/max(distance, 1e-12)))
        if h.compaction_started and entry is None: entry = h.step_count
        records.append(dict(step=h.step_count, error=q['error'].tolist(), heading_error=q['heading_error'].tolist(),
            velocity_error=q['velocity_error'].tolist(), yaw_rate=[s.yaw_rate for s in e.blue_states],
            motion=q['motion_error'].tolist(), E=q['E'].tolist(), safe=q['safe'], q_clear=q['q_clear'].tolist(),
            risk=q['risk'].tolist(), m_clear=q['m_clear'].tolist(),
            role_quality=(1.+terms['usv_contributions']['tracking'] /
                          (h.dt*e.reward_cfg['surface']['tracking_cost_per_second']*e.reward_cfg['learning_scale'])).tolist(),
            surge=[s.speed for s in e.blue_states], red_position=e.red_state.pos.tolist(),
            red_island_clearance=distance-e.island_radius-e.landing_buffer, red_closing_speed=closing,
            compaction_started=bool(h.compaction_started), shrink_progress=q['xi'], shrink_steps=int(h.shrink_steps),
            actions=action.tolist(), rewards=reward.tolist(), event=terms['team_terminal_reward'],
            auv_observation_hash=hashlib.sha256(observations[2:].tobytes()).hexdigest(),
            usv_terms={k: v.tolist() for k, v in terms['usv_contributions'].items()},
            auv_terms={k: v.tolist() for k, v in terms['auv_contributions'].items()}))
        if done: break
    count = h.step_count-start
    row = dict(seed=seed, mode=mode, kind=kind, auv_source=auv_source,
        policy_identity=policy.audit_identity if policy is not None and auv_source == 'frozen_policy' else None,
        initial_state_hash=initial_hash, start_step=start, steps=count, total_episode_steps=h.step_count,
        seconds=count*h.dt, success=bool(info.success), outcome=info.terminal_reason,
        team_return=float(total.mean()), discounted_team=float(discounted.mean()),
        usv_return=float(total[:2].mean()), discounted_usv=float(discounted[:2].mean()),
        returns=total.tolist(), discounted=discounted.tolist(), event_return=event_return,
        initial_error=initial['error'].tolist(), final_error=q['error'].tolist(),
        mean_error=np.mean([r['error'] for r in records], axis=0).tolist(),
        mean_heading_error=np.mean([r['heading_error'] for r in records], axis=0).tolist(),
        mean_velocity_error=np.mean([r['velocity_error'] for r in records], axis=0).tolist(),
        mean_abs_yaw_rate=np.mean(np.abs([r['yaw_rate'] for r in records]), axis=0).tolist(),
        mean_role_quality=np.mean([r['role_quality'] for r in records], axis=0).tolist(),
        reverse_fraction=np.mean(np.array([r['surge'] for r in records]) < -.01, axis=0).tolist(),
        initial_red_island_clearance=initial_red_clearance, final_red_island_clearance=records[-1]['red_island_clearance'],
        mean_red_closing_speed=float(np.mean([r['red_closing_speed'] for r in records])),
        red_toward_island_fraction=float(np.mean([r['red_closing_speed'] > 0. for r in records])),
        initial_radius=initial_radius, final_radius=e.formation_radius, shrink_progress=q['xi'],
        compaction_first_observed_step=entry, observed_compaction_seconds=(h.step_count-entry)*h.dt if entry is not None else None,
        signed={k: v.tolist() for k, v in signed.items()}, absolute={k: v.tolist() for k, v in absolute.items()},
        A_task=task, A_risk=risk, A_event=event_abs, active_mask=e.agent_active_mask().tolist(),
        apf_position_effect=mode != 'no_apf_position_effect', trace=Path(trace_file).name if trace_file else None)
    if trace_file:
        with gzip.open(trace_file, 'wt', encoding='utf-8') as stream: json.dump(records, stream, separators=(',', ':'))
    return row, tape


def audit_seed(seed, out, policy_path, sources):
    torch.set_num_threads(1)
    policy = frozen_policy(policy_path) if 'frozen_policy' in sources else None
    initial = new_env(seed); anchors = np.array([s.pos.copy() for s in initial.blue_states])
    initial_hash = state_hash(initial); rows = []; unavailable = []
    for source in sources:
        modes = (('reference', 'station_keep') if source == 'stationary' else
                 POLICY_MODES if source == 'frozen_policy' else DIAGNOSTIC_MODES)
        frozen_tape = None
        for mode in modes:
            branch = initial.fork(); assert state_hash(branch) == initial_hash
            row, tape = run(branch, seed, mode, policy, anchors, source,
                trace_file=Path(out)/f'{seed}_{source}_{mode}.json.gz',
                usv_tape=frozen_tape if mode == 'no_apf_position_effect' else None)
            if mode == 'reference':
                frozen_tape = tape+[tape[-1].copy() for _ in range(initial.max_steps-len(tape))]
            rows.append(row)
        for kind in (() if source == 'stationary' else ('collision', 'out_of_bounds')):
            h = initial.fork()
            prefix = []
            def margin():
                return (capsule_clearance(h._capsule(h.blue_states[0]), h._capsule(h.blue_states[1])) if kind == 'collision'
                        else boundary_clearance(h._capsule(h.blue_states[0]), h.world_size))
            while not h._episode_done and h.step_count < 600 and margin() >= 3.:
                action = diagnostic_actions(h, 'early_failure', anchors, auv_source=source, policy=policy, kind=kind)
                observations = h.current_observation()
                _, _, reward, done, info = h.step(action)
                prefix.append(dict(step=h.step_count, actions=action.tolist(), rewards=reward.tolist(),
                    margin=float(margin()), outcome=info.terminal_reason,
                    auv_observation_hash=hashlib.sha256(observations[2:].tobytes()).hexdigest()))
            with gzip.open(Path(out)/f'{seed}_{source}_{kind}_prefix.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(dict(initial_state_hash=initial_hash, branch_state_hash=state_hash(h),
                               records=prefix), stream, separators=(',', ':'))
            if h._episode_done or margin() >= 3.:
                unavailable.append(dict(seed=seed, auv_source=source, kind=kind, status='NOT_REACHED', prefix_steps=h.step_count))
                continue
            shared = state_hash(h)
            for mode in ('early_failure', 'brake_reverse_recover', 'reference'):
                branch = h.fork(); assert state_hash(branch) == shared
                row, _ = run(branch, seed, mode, policy, anchors, source, kind=kind,
                    trace_file=Path(out)/f'{seed}_{source}_{kind}_{mode}.json.gz')
                assert row['initial_state_hash'] == shared
                rows.append(row)
    Path(out, f'seed{seed}.json').write_text(json.dumps(dict(rows=rows, unavailable=unavailable), indent=2), encoding='utf-8')
    return dict(seed=seed, trajectories=len(rows), unavailable=unavailable,
        reset=[(r['auv_source'], r['mode'], r['outcome'], r['steps'], r['discounted_usv'], r['discounted_team']) for r in rows if r['kind'] is None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/usv_negative_tracking_audit/rollouts')
    parser.add_argument('--policy', default='docs/usv_negative_tracking_audit/frozen_policy.pt')
    parser.add_argument('--auv-source', choices=('both', 'frozen_policy', 'diagnostic'), default='both')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--stationary-auv', action='store_true', help='also test physically stopped AUV controls')
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    sources = ['frozen_policy', 'diagnostic'] if args.auv_source == 'both' else [args.auv_source]
    if args.stationary_auv:
        sources.append('stationary')
    identity = None; policy_status = 'NOT_RUN: not selected'
    if 'frozen_policy' in sources:
        try:
            identity = frozen_policy(args.policy).audit_identity; policy_status = 'COMPATIBLE_FROZEN_POLICY'
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            sources.remove('frozen_policy'); policy_status = 'NOT_RUN: '+str(error)
    code = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('usv_env', 'matd3', 'scripts') for p in Path(folder).glob('*.py')}
    seeds = args.seeds
    runtime = dict(seeds=seeds, gamma=assemble()['training']['gamma'], sources=sources, policy_status=policy_status, policy=identity,
        reward=assemble()['reward'], frozen_code=code, training_steps=0, replay_insertions=0)
    (out/'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(audit_seed, seed, out.as_posix(), args.policy, sources) for seed in seeds]):
            print(json.dumps(future.result()), flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in code.items())
    rows = [r for seed in seeds for r in json.loads((out/f'seed{seed}.json').read_text())['rows']]
    (out/'summary.json').write_text(json.dumps(dict(runtime=runtime, all=aggregate(rows),
        sources={source: aggregate([r for r in rows if r['auv_source'] == source]) for source in sources},
        ranking='See explicit per-seed report; dense_share is not an acceptance gate'), indent=2), encoding='utf-8')
    print(json.dumps(dict(completed_seeds=len(seeds), trajectories=len(rows), source_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
