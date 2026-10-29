"""Same held-out scenes and outcome schema for internal and external policies."""
from copy import deepcopy
import time
import numpy as np
import torch
from usv_env.config import refresh_hashes, digest
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from matd3.checkpoint import global_rng_state,restore_global_rng
from scripts.experiment_data import EpisodeMetrics


def conditions(suite):
    nominal=('nominal',None)
    delays=[(f'delay_{d}_loss10',dict(delay_steps=[d,d],target_loss_range=[.1,.1])) for d in (0,1,3,6)]
    losses=[(f'loss_{p}_delay0to3',dict(delay_steps=[0,3],target_loss_range=[p/100,p/100])) for p in (0,10,20,30)]
    if suite=='nominal': return [nominal]
    if suite=='screen': return [nominal,delays[-1],losses[-1]]
    if suite=='communication': return [nominal,*delays,*losses]
    raise ValueError('unknown evaluation suite')


def make_env(cfg,seed,variant='randomized'):
    env=PursuitEvasionEnv(cfg,seed); env.set_evaluation_variant(variant)
    return HistoryEnv(env,cfg['revision']['history_length'])


def evaluate_policy(policy,cfg,episodes,base_seed,variant='randomized',*,suite='nominal',writer=None,
                    checkpoint_id='',checkpoint_steps=None,planned_steps=None,split='validation',trace_episodes=1):
    if episodes<1 or trace_episodes<0: raise ValueError('invalid evaluation episode count')
    saved_rng=global_rng_state(); all_rows=[]; summaries=[]
    modes=[(a,a.training) for a in getattr(policy,'actors',[])]
    env=None
    try:
        for a,_ in modes: a.eval()
        for condition,override in conditions(suite):
            local=deepcopy(cfg)
            if override: local['communication'].update(override)
            refresh_hashes(local)
            context=dict(checkpoint_id=checkpoint_id,checkpoint_env_steps=checkpoint_steps,
                planned_env_steps=planned_steps,split=split,variant=variant,condition=condition,
                evaluation_task_hash=local['hashes']['task'],evaluation_communication=local['communication'],
                condition_hash=digest(dict(variant=variant,communication=local['communication'])))
            env=make_env(local,base_seed,variant); rows=[]; start=time.perf_counter(); inference_s=0.
            for episode_index,seed in enumerate(range(base_seed,base_seed+episodes)):
                obs,_=env.reset(seed=seed)
                reset=getattr(policy,'reset',None)
                if callable(reset): reset(seed=seed)
                measurement=EpisodeMetrics(env); trajectory=[]
                identity=dict(**context,eval_seed=seed,evaluation_episode=episode_index+1)
                capture=writer is not None and episode_index<trace_episodes
                while True:
                    coeff=None
                    if capture and getattr(getattr(policy,'surface_actor',None),'hyper_enabled',False):
                        with torch.no_grad():
                            tensor=torch.as_tensor(obs[:2],dtype=torch.float32,device=policy.device)
                            coeff=policy.surface_actor.coefficients(tensor).cpu().numpy().tolist()
                    tick=time.perf_counter(); action=np.asarray(policy.act(obs,explore=False),np.float32)
                    inference_s+=time.perf_counter()-tick
                    limits=np.asarray(env.agent_action_limits)
                    if action.shape!=limits.shape or not np.isfinite(action).all() or (np.abs(action)>limits+1e-6).any():
                        raise ValueError('policy must return finite physical actions [5,2] within env limits')
                    obs,_,reward,done,info=env.step(action)
                    row=measurement.observe(env,reward,info,capture)
                    if capture:
                        row['usv_coefficients_before_action']=coeff
                        trajectory.append(row)
                        if len(trajectory)>=128 or done:
                            writer.append('evaluation_steps',dict(**identity,steps=trajectory)); trajectory=[]
                    if done: break
                row={**identity,**measurement.summary(env,info)}; rows.append(row); all_rows.append(row)
                if writer: writer.append('evaluation_episodes',row)
            summary={**context,'episodes':episodes,'episode_steps_total':sum(r['episode_steps'] for r in rows),
                'success_rate':float(np.mean([r['success'] for r in rows])),
                'eval_return':float(np.mean([r['team_return'] for r in rows])),
                'eval_wall_time_s':time.perf_counter()-start,'inference_wall_time_s':inference_s}
            # Each row is an episode, not a new training seed. Preserve effective n.
            fields=['follow_stage_entered','follow_worst_mean_m','follow_worst_max_m','follow_any_lost_fraction',
                    'all_worst_mean_m','red_island_progress_m','landing_clearance_min_m','episode_seconds',
                    'collision','landing','out_of_bounds','timeout','realized_packet_loss','message_age_mean_steps']
            fields += [f'auv{i}_{suffix}' for i in (1,2,3) for suffix in ('follow_mean_m','lost_fraction')]
            for field in fields:
                valid=[r[field] for r in rows if r[field] is not None]
                summary[field]=float(np.mean(valid)) if valid else None
                summary[field+'_n']=len(valid)
            if writer: writer.append('evaluation_summary',summary)
            summaries.append(summary); env.close(); env=None
        nominal=summaries[0]
        return dict(success_rate=nominal['success_rate'],return_value=nominal['eval_return'],
                    rows=all_rows,summaries=summaries)
    finally:
        if env is not None: env.close()
        for actor,mode in modes: actor.train(mode)
        restore_global_rng(saved_rng)
