"""Train MATD3 with bounded diagnostics and actor-only latest/best policies."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Apply the standalone launch budget before importing numerical runtimes.
# Imports by evaluation/tests keep their caller's thread configuration.
if __name__ == '__main__':
    os.environ.update(NUMBA_NUM_THREADS='3', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import sys
from pathlib import Path
_PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(_PROJECT_ROOT))
import argparse
from copy import deepcopy
import json
import signal
import threading
import time
import numpy as np
import torch
import yaml
from usv_env.config import assemble,validate,refresh_hashes,named_seed,run_directory,code_hash,ROOT,PRESETS
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from matd3.agent import Agent
from matd3.history_enhancement import I3_COUNT_FIELDS
from matd3.checkpoint import load_policy,save_policy
from scripts.csv_logging import CsvLog,TRAIN_FIELDS,UPDATE_FIELDS
from scripts.experiment_data import ExperimentWriter,EpisodeMetrics,file_hash,policy_hash,SCHEMA as DATA_SCHEMA
from scripts.research_evaluation import evaluate_policy,validation_key,SELECTION_METRIC
from scripts.experiment_identity import validate_learning_capacity


def make_env(cfg,seed,variant='training',episode_limit=None):
    env=PursuitEvasionEnv(cfg,seed); env.set_evaluation_variant(variant)
    return HistoryEnv(env,cfg['revision']['history_length'],episode_limit)


def evaluate(agent,cfg,episodes,base_seed,variant='randomized'):
    result=evaluate_policy(agent,cfg,episodes,base_seed,variant,trace_episodes=0)
    return {'success_rate':result['success_rate'],'return':result['return_value'],'rows':result['rows']}


def write_config(cfg,path):
    temporary=path.with_suffix('.yaml.tmp')
    temporary.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    os.replace(temporary,path)


def _warm_start(agent,path,device):
    policy,_=load_policy(path,device)
    for key in ('task','reward','observation','architecture'):
        if policy.runtime_config['hashes'][key]!=agent.runtime_config['hashes'][key]:
            raise ValueError(f'warm-start {key} is incompatible with this run')
    if len(policy.actors)!=len(agent.actors): raise ValueError('warm-start actor count differs')
    if not np.array_equal(policy.limits, agent.limits):
        raise ValueError('warm-start physical action limits differ from the target task')
    for source,actor,target in zip(policy.actors,agent.actors,agent.actor_targets):
        # Transfer learned weights only. Actuator buffers belong to the target
        # physical task and must never be imported from a source checkpoint.
        actor.net.load_state_dict(source.net.state_dict(),strict=True)
        target.net.load_state_dict(source.net.state_dict(),strict=True)
    return dict(kind='actor_only_warm_start_new_training',source=str(Path(path).resolve()),
        source_sha256=file_hash(path),source_online_steps=policy.global_env_steps,
        restored_replay=False,restored_optimizers=False,new_online_steps_start=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/fast.yaml')); p.add_argument('--experiment',choices=PRESETS,default='Base')
    p.add_argument('--seed',type=int,default=0)
    budget=p.add_mutually_exclusive_group()
    budget.add_argument('--steps',type=int,default=100000,help='new online interactions including warmup (default: 100000)')
    budget.add_argument('--episodes',type=int,help='stop after this many completed training episodes, including warmup episodes')
    p.add_argument('--device',choices=('auto','cpu','cuda'))
    destination=p.add_mutually_exclusive_group(); destination.add_argument('--runs-root'); destination.add_argument('--output')
    p.add_argument('--resume',help='unsupported: full replay/simulator persistence is disabled')
    p.add_argument('--warm-start',help='copy actor weights into a new run with new optimizer/replay state')
    p.add_argument('--eval-every',type=int); p.add_argument('--eval-episodes',type=int)
    p.add_argument('--prepare-only',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-episode-limit',type=int,default=32); p.add_argument('--debug-diagnostics',action='store_true')
    p.add_argument('--eval-suite',choices=('nominal','screen','communication','collaboration'),default='nominal')
    p.add_argument('--diagnostic-every',type=int,default=1000)
    p.add_argument('--train-trace-every',type=int,default=0,help='full training trajectory every N episodes; 0 disables large traces')
    p.add_argument('--export-diagnostics-csv',action='store_true',help='also export compressed diagnostic logs as duplicate CSV tables')
    p.add_argument('--eval-trace-episodes',type=int,default=1)
    args=p.parse_args(argv)
    if args.resume: p.error('exact resume is disabled: replay/optimizer/simulator checkpoints are not saved; use --warm-start latest.pt with a new run directory')
    if args.steps<1 or args.seed<0 or args.smoke_episode_limit<0: p.error('invalid steps, seed or collector limit')
    if args.episodes is not None and args.episodes<1: p.error('--episodes must be positive')
    for value in (args.eval_every,args.diagnostic_every,args.train_trace_every,args.eval_trace_episodes):
        if value is not None and value<0: p.error('intervals must be nonnegative')
    if args.eval_episodes is not None and args.eval_episodes<1: p.error('eval-episodes must be positive')
    cfg=assemble(args.config,args.experiment,args.seed,args.device); cfg['replay_buffer']['save_in_checkpoint']=False
    runs_root=Path(args.runs_root or cfg['outputs']['root']).expanduser()
    runs_root=(runs_root if runs_root.is_absolute() else ROOT/runs_root).resolve()
    root=Path(args.output).resolve() if args.output else run_directory(runs_root,args.experiment,args.seed)
    if args.smoke:
        if not args.output: p.error('--smoke requires --output to separate smoke data from formal runs')
        if root.is_relative_to(runs_root): p.error('smoke output must be outside formal runs-root')
        cfg['purpose']='implementation_smoke'
        cfg['training'].update(batch_size=8,random_warmup_env_steps=8,hidden_sizes=[32,32])
        cfg['replay_buffer']['capacity']=128
    if args.episodes is not None:
        args.steps=args.episodes*cfg['env']['max_steps']
    try: validate_learning_capacity(cfg)
    except ValueError as error: p.error(str(error))
    cfg['training_budget']=dict(unit='episodes' if args.episodes is not None else 'steps',
        requested=args.episodes if args.episodes is not None else args.steps,
        maximum_online_steps=args.steps)
    if not args.smoke and not args.prepare_only and args.steps<=cfg['training']['random_warmup_env_steps']:
        p.error(f"--steps must exceed random_warmup_env_steps={cfg['training']['random_warmup_env_steps']}; "
                "this budget would perform no gradient updates. Increase --steps, or use --smoke with "
                "a separate --output directory for implementation checks")
    cfg['diagnostics']=dict(schema=DATA_SCHEMA,eval_suite=args.eval_suite,train_trace_every=args.train_trace_every,
        update_every=args.diagnostic_every,eval_trace_episodes=args.eval_trace_episodes,
        compression='gzip',export_csv=args.export_diagnostics_csv)
    cfg['outputs']['root']=os.path.relpath(runs_root,ROOT); cfg['run_dir']=os.path.relpath(root,ROOT)
    cfg.setdefault('evaluation',dict(interval_steps=10000,episodes=5,base_seed=1000000))
    if args.eval_every is not None: cfg['evaluation']['interval_steps']=args.eval_every
    if args.eval_episodes is not None: cfg['evaluation']['episodes']=args.eval_episodes
    cfg['provenance']['code_hash']=code_hash(); refresh_hashes(cfg); validate(cfg)
    if args.prepare_only: print(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False)); return
    if root.exists() and any(root.iterdir()): p.error(f'refusing to overwrite nonempty run directory: {root}')
    torch.set_num_threads(cfg['training']['torch_num_threads'])
    torch.use_deterministic_algorithms(cfg['training']['deterministic_algorithms'],warn_only=False)
    env=make_env(cfg,args.seed,episode_limit=(args.smoke_episode_limit or None) if args.smoke else None)
    agent=Agent(env,cfg); initialization=dict(kind='new_training',new_online_steps_start=0)
    if args.warm_start: initialization=_warm_start(agent,args.warm_start,args.device or 'cpu')
    cfg['initialization']=initialization; agent.runtime_config=deepcopy(cfg)
    root.mkdir(parents=True,exist_ok=True); write_config(cfg,root/'resolved_config.yaml')
    train_log=CsvLog(root/'train.csv',TRAIN_FIELDS); update_log=CsvLog(root/'updates.csv',UPDATE_FIELDS)
    data=ExperimentWriter(root,cfg); start=time.perf_counter(); completed=last_eval=eval_steps=0
    best=(-1.,-1.,-float('inf'),-1.,-float('inf')); stop=[False]; previous_handler=None
    if threading.current_thread() is threading.main_thread():
        previous_handler=signal.getsignal(signal.SIGINT)
        def request_stop(signum,frame):
            stop[0]=True; print('Stop requested; saving actor-only latest.pt after this step',flush=True)
        signal.signal(signal.SIGINT,request_stop)
    print(json.dumps(dict(replay_capacity=agent.replay.capacity,replay_ram_bytes=agent.replay.array_payload_bytes,
        replay_saved=False,observation_dim=env.observation_dim,agents=env.num_friendly,
        requested_device=cfg['training']['device'],actual_device=str(agent.device),
        gpu=torch.cuda.get_device_name(agent.device) if agent.device.type=='cuda' else None)),flush=True)
    window=[]; measurement=None; trace=[]; seed=None; active=False
    diagnostic_totals={key:0 for key in I3_COUNT_FIELDS}
    diagnostic_totals.update(i1_target_vo_time_s=0.,i1_actor_vo_time_s=0.,i1_environment_vo_time_s=0.)
    diagnostic_totals.update(i3_vo_screen_time_s=0.,vo_trigger_count=0.,vo_unresolved_count=0.)
    runtime_seconds={key:0. for key in ('action','environment','replay','metrics_and_trace',
                                      'update','evaluation','checkpoint')}
    def wall(): return time.perf_counter()-start
    def runtime_snapshot():
        elapsed=wall()
        return dict(wall_time_s=elapsed,actual_device=str(agent.device),
            online_steps_per_second=agent.global_env_steps/max(elapsed,1e-12),
            phase_seconds=dict(runtime_seconds),
            other_seconds=max(0.,elapsed-sum(runtime_seconds.values())),
            update_count=agent.update_steps,
            update_ms=1000*runtime_seconds['update']/max(agent.update_steps,1),
            environment_ms=1000*runtime_seconds['environment']/max(agent.global_env_steps,1))
    def write_manifest(status):
        data.manifest(agent.global_env_steps,0,dict(status=status,
            completed_episodes=completed,partial_episode_steps=env.step_count if active else 0,
            wall_time_s=wall(),requested_episodes=args.episodes,evaluation_sim_steps=eval_steps,
            initialization=initialization,latest_policy_sha256=policy_hash(agent),
            best_validation_key=list(best) if best[0]>=0 else None,diagnostic_totals=diagnostic_totals,
            best_validation_metric=SELECTION_METRIC,runtime=runtime_snapshot(),
            update_count_semantics='I3 count columns are sums per update window; other update columns are means'))
    def flush_updates():
        if not window: return
        row={field:None for field in UPDATE_FIELDS}
        for field in UPDATE_FIELDS[2:]:
            if field in ("teacher_calls","teacher_accepted","teacher_sim_steps","wall_time_s"): continue
            values=[float(m[field]) for m in window if m.get(field) is not None]
            row[field]=float(np.sum(values) if field in I3_COUNT_FIELDS else np.mean(values)) if values else None
        row.update(env_steps=agent.global_env_steps,update_step=agent.update_steps,teacher_calls=0,
            teacher_accepted=0,teacher_sim_steps=0,wall_time_s=wall())
        update_log.append(row); window.clear()
    write_manifest('running')
    try:
        while (agent.global_env_steps<args.steps and
               (args.episodes is None or completed<args.episodes) and not stop[0]):
            if not active:
                seed=named_seed(args.seed,'training_scene',completed); obs,state=env.reset(seed=seed)
                current_context=env.learning_context()
                measurement=EpisodeMetrics(env); total=np.zeros(env.num_friendly)
                discounted_total=np.zeros(env.num_friendly); discounted_process=discounted_terminal=0.
                process=terminal=absolute=0.; held=None; hold_age=0; trace=[]; active=True
            tick=time.perf_counter()
            warmup=agent.global_env_steps<cfg['training']['random_warmup_env_steps']
            if warmup:
                if held is None or hold_age>=cfg['training']['random_action_hold_steps']: held=agent.random_actions(); hold_age=0
                action=held.copy(); hold_age+=1
            else: action=agent.act(obs,explore=True)
            runtime_seconds['action']+=time.perf_counter()-tick
            tick=time.perf_counter()
            mask=env.agent_active_mask(); next_obs,next_state,reward,done,info=env.step(action)
            runtime_seconds['environment']+=time.perf_counter()-tick
            tick=time.perf_counter()
            next_context=env.learning_context()
            agent.add_transition(obs,state,env.last_executed_actions.copy(),reward,next_obs,next_state,info,mask,env.agent_active_mask(),current_context=current_context,next_context=next_context,nominal_actions=env.last_nominal_actions.copy())
            # No environment mutation occurs between these interactions.
            # Reuse s_{t+1}'s metadata as the next transition's s_t metadata.
            current_context=next_context
            runtime_seconds['replay']+=time.perf_counter()-tick
            tick=time.perf_counter()
            capture=args.train_trace_every>0 and completed%args.train_trace_every==0
            record=measurement.observe(env,reward,info,capture)
            diagnostic_totals['i1_environment_vo_time_s']+=float(info.safety_diagnostics.get('vo_time_s',0.))
            diagnostic_totals['vo_trigger_count']+=int(np.count_nonzero(info.safety_diagnostics.get('triggered',[])))
            diagnostic_totals['vo_unresolved_count']+=int(np.count_nonzero(info.safety_diagnostics.get('infeasible',[])))
            identity=dict(episode_index=completed+1,scene_seed=seed,env_steps=agent.global_env_steps)
            if capture:
                trace.append(record)
                if len(trace)>=128 or done: data.append('training_steps',dict(**identity,steps=trace)); trace=[]
            if args.diagnostic_every and agent.global_env_steps%args.diagnostic_every==0:
                data.append('reward_components',dict(**identity,step=env.step_count,terms=info.reward_terms))
            runtime_seconds['metrics_and_trace']+=time.perf_counter()-tick
            if not warmup:
                tick=time.perf_counter()
                metrics=agent.update()
                runtime_seconds['update']+=time.perf_counter()-tick
                if metrics is not None:
                    for key in diagnostic_totals:
                        if key != 'i1_environment_vo_time_s': diagnostic_totals[key]+=metrics.get(key,0.) or 0.
                    window.append(metrics)
                    if args.diagnostic_every and agent.update_steps%args.diagnostic_every==0:
                        data.append('updates',dict(**identity,update_step=agent.update_steps,metrics=metrics))
                    if len(window)>=100: flush_updates()
            obs,state=next_obs,next_state; total+=reward
            terminal_step=float(info.reward_terms.get('team_terminal_reward',0.))
            dense_step=float(info.reward_terms.get('canonical_team_dense_reward',np.mean(reward)-terminal_step))
            process+=dense_step; terminal+=terminal_step; absolute+=abs(dense_step)
            reward_gamma=cfg['training']['gamma']; discount=reward_gamma**(env.step_count-1)
            discounted_total+=discount*np.asarray(reward,dtype=np.float64)
            discounted_process+=discount*dense_step; discounted_terminal+=discount*terminal_step
            if done:
                completed+=1; agent.episodes_completed=completed; active=False
                data.append('training_episodes',dict(**identity,**measurement.summary(env,info)))
                row=dict(episode=completed,env_steps=agent.global_env_steps,episode_steps=env.step_count,
                    team_return=float(total.mean()),usv_return=float(total[:env.num_surface_blue].mean()),
                    auv_return=float(total[env.num_surface_blue:].mean()),process_return=process,terminal_return=terminal,
                    discounted_team_return=float(discounted_total.mean()),
                    discounted_usv_return=float(discounted_total[:env.num_surface_blue].mean()),
                    discounted_auv_return=float(discounted_total[env.num_surface_blue:].mean()),
                    discounted_process_return=discounted_process,discounted_terminal_return=discounted_terminal,
                    reward_gamma=reward_gamma,
                    process_abs_sum=absolute,success=info.success,terminal_reason=info.terminal_reason or 'collector_cut',
                    capture_success=info.capture_success,first_capture_step=info.first_capture_step,
                    first_capture_time_s=info.first_capture_time_s,post_capture_steps=info.post_capture_steps,
                    withdrawal_cleared_entry=info.withdrawal_cleared_entry,
                    shrink_progress=None,wall_time_s=wall(),eval_success_rate=None,eval_return=None)
                summary=measurement.summary(env,info)
                from scripts.csv_logging import MISSION_TRAIN_FIELDS
                for key in MISSION_TRAIN_FIELDS:
                    value=summary[key]
                    row[key]=float(np.mean(value)) if isinstance(value,(list,tuple,np.ndarray)) else value
                for key,values in measurement.component_sums.items():
                    row['component_'+key]=float(np.mean(values))
                    row['discounted_component_'+key]=float(np.mean(measurement.component_discounted_sums[key]))
                if not np.isclose(row['team_return'],process+terminal,rtol=1e-5,atol=1e-5): raise ArithmeticError('episode reward ledger does not balance')
                if not np.isclose(row['discounted_team_return'],discounted_process+discounted_terminal,rtol=1e-5,atol=1e-5):
                    raise ArithmeticError('discounted episode reward ledger does not balance')
                interval=cfg['evaluation']['interval_steps']
                if interval and agent.global_env_steps-last_eval>=interval:
                    tick=time.perf_counter()
                    sha=policy_hash(agent)
                    result=evaluate_policy(agent,cfg,cfg['evaluation']['episodes'],cfg['evaluation']['base_seed'],
                        suite=args.eval_suite,writer=data,checkpoint_id=sha,checkpoint_steps=agent.global_env_steps,
                        planned_steps=last_eval+interval,trace_episodes=args.eval_trace_episodes)
                    if policy_hash(agent)!=sha: raise RuntimeError('evaluation changed policy parameters')
                    runtime_seconds['evaluation']+=time.perf_counter()-tick
                    key=validation_key(result)
                    if key>best:
                        best=key; save_policy(agent,root/'best.pt',dict(selection_split='validation',policy_sha256=sha,
                            selection_metric=SELECTION_METRIC,selection_key=list(key)))
                    eval_steps+=sum(r['episode_steps_total'] for r in result['summaries'])
                    row.update(eval_success_rate=result['success_rate'],eval_return=result['return_value']); last_eval=agent.global_env_steps
                train_log.append(row); print(json.dumps(row,allow_nan=False),flush=True)
                tick=time.perf_counter()
                save_policy(agent,root/'latest.pt',dict(online_steps=agent.global_env_steps,policy_sha256=policy_hash(agent)))
                runtime_seconds['checkpoint']+=time.perf_counter()-tick
            if agent.global_env_steps%1000==0:
                progress=dict(kind='runtime',env_steps=agent.global_env_steps,completed_episodes=completed,
                              **runtime_snapshot())
                data.append('runtime',progress)
                write_manifest('running')
                print(json.dumps(progress,allow_nan=False),flush=True)
        if trace: data.append('training_steps',dict(episode_index=completed+1,scene_seed=seed,env_steps=agent.global_env_steps,steps=trace))
        flush_updates(); tick=time.perf_counter()
        save_policy(agent,root/'latest.pt',dict(online_steps=agent.global_env_steps,mid_episode=active,policy_sha256=policy_hash(agent)))
        runtime_seconds['checkpoint']+=time.perf_counter()-tick
        if args.export_diagnostics_csv: data.export()
        data.append('runtime',dict(env_steps=agent.global_env_steps,completed_episodes=completed,**runtime_snapshot()))
        write_manifest('stopped' if stop[0] else 'complete')
        print(f'Saved lightweight policies and logs in {root}; new online steps={agent.global_env_steps}',flush=True)
    finally:
        env.close()
        if previous_handler is not None: signal.signal(signal.SIGINT,previous_handler)


if __name__=='__main__': main()
