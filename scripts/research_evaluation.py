"""Same held-out scenes and outcome schema for internal and external policies."""
from copy import deepcopy
import time
import numpy as np
from usv_env.config import refresh_hashes, digest
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from matd3.checkpoint import global_rng_state,restore_global_rng
from scripts.experiment_data import EpisodeMetrics,policy_hash

SELECTION_METRIC = 'success_then_safety_then_capture_then_discounted_return'


def validation_key(result):
    """Outcomes take precedence over accumulated earnings in policy selection."""
    return (result['success_rate'], -result['safety_failure_rate'],
            result['capture_rate'], result['discounted_return_value'])


def conditions(suite):
    nominal=('nominal',None)
    delays=[(f'delay_{d}_loss10',dict(delay_steps=[d,d],target_loss_range=[.1,.1],loss_rate=.1)) for d in (0,3,10,20,40)]
    losses=[(f'loss_{p}_delay0to3',dict(delay_steps=[0,3],target_loss_range=[p/100,p/100],loss_rate=p/100)) for p in (0,5,10,20,30)]
    if suite=='collaboration':
        return [nominal,('only_usvs',{'_active_mask':[1,1,0,0,0]}),('only_auvs',{'_active_mask':[0,0,1,1,1]}),('without_usv1',{'_active_mask':[0,1,1,1,1]}),
                ('without_auv1',{'_active_mask':[1,1,0,1,1]}),('no_peer_state',{'peer_mode':'disabled'}),
                ('delayed_peer_state',{'peer_mode':'delayed'}),('reordered_peer_state',{'peer_mode':'shuffled'})]
    if suite=='nominal': return [nominal]
    if suite=='screen': return [nominal,delays[-1],losses[-1]]
    if suite=='communication':
        return [nominal,*[(f'delay_{delay}_loss{loss}',dict(delay_steps=[delay,delay],
                target_loss_range=[loss/100,loss/100],loss_rate=loss/100))
                for delay in (0,3,10,20,40) for loss in (0,5,10,20,30)]]
    raise ValueError('unknown evaluation suite')


def make_env(cfg,seed,variant='randomized'):
    env=PursuitEvasionEnv(cfg,seed); env.set_evaluation_variant(variant)
    return HistoryEnv(env,cfg['revision']['history_length'])


def evaluate_policy(policy,cfg,episodes,base_seed,variant='randomized',*,suite='nominal',writer=None,
                    checkpoint_id='',checkpoint_steps=None,planned_steps=None,split='validation',trace_episodes=1):
    if episodes<1 or trace_episodes<0: raise ValueError('invalid evaluation episode count')
    saved_rng=global_rng_state(); all_rows=[]; summaries=[]
    modes=[(a,a.training) for a in getattr(policy,'actors',[])]
    content_sha=policy_hash(policy) if modes else None
    env=None
    try:
        for a,_ in modes: a.eval()
        for condition,override in conditions(suite):
            local=deepcopy(cfg)
            override=deepcopy(override or {}); active_mask=override.pop('_active_mask',None)
            local['communication'].update(override)
            refresh_hashes(local)
            context=dict(checkpoint_id=checkpoint_id,checkpoint_env_steps=checkpoint_steps,
                policy_sha256=content_sha,
                planned_env_steps=planned_steps,split=split,variant=variant,condition=condition,
                evaluation_task_hash=local['hashes']['task'],evaluation_communication=local['communication'],
                active_mask=active_mask,condition_hash=digest(dict(variant=variant,communication=local['communication'],active_mask=active_mask)))
            env=make_env(local,base_seed,variant)
            if active_mask is not None: env.set_active_agents(active_mask)
            rows=[]; start=time.perf_counter(); inference_s=0.
            for episode_index,seed in enumerate(range(base_seed,base_seed+episodes)):
                obs,_=env.reset(seed=seed)
                reset=getattr(policy,'reset',None)
                if callable(reset): reset(seed=seed)
                measurement=EpisodeMetrics(env); trajectory=[]
                identity=dict(**context,eval_seed=seed,evaluation_episode=episode_index+1)
                capture=writer is not None and episode_index<trace_episodes
                while True:
                    tick=time.perf_counter(); action=np.asarray(policy.act(obs,explore=False),np.float32)
                    inference_s+=time.perf_counter()-tick
                    limits=np.asarray(env.agent_action_limits)
                    if action.shape!=limits.shape or not np.isfinite(action).all() or (np.abs(action)>limits+1e-6).any():
                        raise ValueError(f'policy must return finite physical actions {limits.shape} within env limits')
                    obs,_,reward,done,info=env.step(action)
                    row=measurement.observe(env,reward,info,capture)
                    if capture:
                        trajectory.append(row)
                        if len(trajectory)>=128 or done:
                            writer.append('evaluation_steps',dict(**identity,steps=trajectory)); trajectory=[]
                    if done: break
                row={**identity,**measurement.summary(env,info)}; rows.append(row); all_rows.append(row)
                if writer: writer.append('evaluation_episodes',row)
            summary={**context,'episodes':episodes,'episode_steps_total':sum(r['episode_steps'] for r in rows),
                'outcome_counts':{reason:sum(row['terminal_reason']==reason for row in rows)
                                  for reason in sorted({row['terminal_reason'] for row in rows})},
                'success_rate':float(np.mean([r['success'] for r in rows])),
                'eval_return':float(np.mean([r['team_return'] for r in rows])),
                'eval_discounted_team_return':float(np.mean([r['discounted_team_return'] for r in rows])),
                'eval_wall_time_s':time.perf_counter()-start,'inference_wall_time_s':inference_s}
            captured=[r for r in rows if r.get('capture_success',False)]
            capture_count=len(captured)
            completed=sum(bool(r['success']) for r in captured)
            withdrawn=sum(bool(r.get('withdrawal_cleared_entry',False)) for r in captured)
            # Plain rates use every evaluated episode. Conditional rates use
            # confirmed captures only; no captures means an undefined rate.
            # Clearing the entry is a latched event, not a safe-ending verdict.
            summary.update(capture_count=capture_count,capture_rate=capture_count/episodes,
                safety_failure_rate=sum(r['terminal_reason'] in ('collision','reef_collision','out_of_bounds')
                    for r in rows)/episodes,
                post_capture_success_count=completed,post_capture_success_rate=completed/episodes,
                post_capture_success_rate_given_capture=completed/capture_count if capture_count else None,
                withdrawal_cleared_entry_count=withdrawn,withdrawal_cleared_entry_rate=withdrawn/episodes,
                withdrawal_cleared_entry_rate_given_capture=withdrawn/capture_count if capture_count else None)
            # Capture timing and post-capture length describe captured episodes,
            # including those that later fail; never impute noncaptures as zero.
            for field in ('first_capture_step','first_capture_time_s','post_capture_steps'):
                valid=[r[field] for r in captured if r.get(field) is not None]
                summary[field]=float(np.mean(valid)) if valid else None
                summary[field+'_n']=len(valid)
            # Each row is an episode, not a new training seed. Preserve effective n.
            fields=['follow_stage_entered','follow_worst_mean_m','follow_worst_max_m','follow_any_lost_fraction',
                    'all_worst_mean_m','red_goal_progress_m','protected_zone_clearance_min_m','episode_seconds',
                    'collision','breakthrough','reef_collision','out_of_bounds','timeout','scenario_anomaly','numerical_error',
                    'realized_packet_loss','message_age_mean_seconds','first_discovery_seconds','joint_hold_max_seconds','capture_hold_max_seconds']
            fields += [f'auv{i}_{suffix}' for i in (1,2,3) for suffix in ('follow_mean_m','lost_fraction')]
            for field in fields:
                valid=[r[field] for r in rows if r.get(field) is not None]
                summary[field]=float(np.mean(valid)) if valid else None
                summary[field+'_n']=len(valid)
            if writer: writer.append('evaluation_summary',summary)
            summaries.append(summary); env.close(); env=None
        if modes and policy_hash(policy)!=content_sha: raise RuntimeError('frozen evaluation changed actor parameters')
        nominal=summaries[0]
        return dict(success_rate=nominal['success_rate'],return_value=nominal['eval_return'],
                    discounted_return_value=nominal['eval_discounted_team_return'],
                    safety_failure_rate=nominal['safety_failure_rate'],capture_rate=nominal['capture_rate'],
                    rows=all_rows,summaries=summaries)
    finally:
        if env is not None: env.close()
        for actor,mode in modes: actor.train(mode)
        restore_global_rng(saved_rng)
