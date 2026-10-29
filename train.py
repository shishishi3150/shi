"""One v4.3 training entry point. Preparation is shared; online steps are exact."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
import argparse
import hashlib
import uuid
from copy import deepcopy
import json
import os
import signal
import shutil
import threading
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from usv_env.config import assemble, validate, refresh_hashes, named_seed, run_directory, code_hash, ROOT, PRESETS
from usv_env.calibration import prepare
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from matd3.agent import Agent
from matd3.parameter_teacher import ParameterTeacher
from matd3.checkpoint import (read,save_policy,global_rng_state,restore_global_rng,CHECKPOINT_FORMAT)
from scripts.csv_logging import CsvLog,UpdateWindow,TRAIN_FIELDS,UPDATE_FIELDS
from scripts.experiment_data import ExperimentWriter,EpisodeMetrics,strip_branch_traces,SCHEMA as DATA_SCHEMA
from scripts.research_evaluation import evaluate_policy


def make_env(cfg,seed,variant="training",episode_limit=None):
    env=PursuitEvasionEnv(cfg,seed); env.set_evaluation_variant(variant)
    return HistoryEnv(env,cfg["revision"]["history_length"],episode_limit)


def evaluate(agent,cfg,episodes,base_seed,variant="randomized"):
    rng=global_rng_state(); rows=[]
    try:
        env=make_env(cfg,base_seed,variant)
        for seed in range(base_seed,base_seed+episodes):
            obs,_=env.reset(seed=seed); total=0.
            while True:
                obs,_,r,done,info=env.step(agent.act(obs,explore=False)); total+=float(r.mean())
                if done: break
            rows.append(dict(seed=seed,condition=variant,success=int(info.success),terminal_reason=info.terminal_reason,
                             episode_steps=env.step_count,team_return=total))
        return {"success_rate":float(np.mean([r["success"] for r in rows])),
                "return":float(np.mean([r["team_return"] for r in rows])),"rows":rows}
    finally: restore_global_rng(rng)


def write_config(cfg,path):
    temporary=path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    os.replace(temporary,path)


def teacher_log_commit(path):
    data=path.read_bytes()
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def restore_teacher_log(path,commit):
    data=path.read_bytes(); size=commit["bytes"]
    if len(data)<size or hashlib.sha256(data[:size]).hexdigest()!=commit["sha256"]:
        raise ValueError("teacher log committed prefix missing/corrupt")
    if len(data)>size:
        archive=path.parent/"orphaned_attempt"/uuid.uuid4().hex; archive.mkdir(parents=True)
        (archive/"teacher.jsonl").write_bytes(data[size:])
        path.write_bytes(data[:size])


def main(argv=None):
    p=argparse.ArgumentParser(description="shi v4.3 shared calibration, MH training and full resume")
    p.add_argument("--config",default=None)
    p.add_argument("--experiment",choices=PRESETS,default=None)
    p.add_argument("--seed",type=int,default=None)
    p.add_argument("--steps",type=int,default=1000000,help="total target online steps, including warmup")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default=None)
    destination=p.add_mutually_exclusive_group()
    destination.add_argument("--runs-root",default=None)
    destination.add_argument("--output",default=None,help="complete run_dir; mutually exclusive with --runs-root")
    p.add_argument("--resume",default=None,help="full checkpoint.pt only; restores exact run_dir and frozen c_A")
    p.add_argument("--eval-every",type=int,default=None)
    p.add_argument("--eval-episodes",type=int,default=None)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--smoke",action="store_true",help="small implementation test using existing shared calibration")
    p.add_argument("--smoke-episode-limit",type=int,default=32,help="external collector cut; zero disables it")
    p.add_argument("--debug-diagnostics",action="store_true")
    p.add_argument('--eval-suite',choices=('nominal','screen','communication'),default=None,
                   help='default screen: nominal plus delay/loss stress; best selection uses nominal only')
    p.add_argument('--diagnostic-every',type=int,default=None,help='I2 update sampling interval; default 1000; 0 disables')
    p.add_argument('--train-trace-every',type=int,default=None,help='trace first and every Nth episode; default 100; 0 disables')
    p.add_argument('--eval-trace-episodes',type=int,default=None,help='first N episodes per condition; default 1')
    p.add_argument('--teacher-trace-every',type=int,default=None,help='trace first and every Nth search; default 20; 0 disables')
    args=p.parse_args(argv)
    if args.steps<1 or args.smoke_episode_limit<0: p.error("steps must be positive; smoke limit nonnegative")
    if args.eval_every is not None and args.eval_every<0: p.error("eval-every must be nonnegative")
    if args.eval_episodes is not None and args.eval_episodes<1: p.error("eval-episodes must be positive")
    options=(args.diagnostic_every,args.train_trace_every,args.eval_trace_episodes,args.teacher_trace_every)
    if any(x is not None and x<0 for x in options): p.error('diagnostic intervals must be nonnegative')
    if args.resume and any((args.config,args.experiment,args.seed is not None,args.runs_root,args.output,args.prepare_only,args.smoke,
                            args.eval_every is not None,args.eval_episodes is not None,args.eval_suite,
                            any(x is not None for x in options))):
        p.error("resume restores saved configuration/run identity; only --steps and --device may change")
    saved=None
    if args.resume:
        saved=read(args.resume,CHECKPOINT_FORMAT)
        if 'experiment_data' not in saved: p.error('old checkpoint lacks resumable metric state; start a new run or use the original release')
        cfg=saved["runtime_config"]
        if args.device: cfg["training"]["device"]=args.device
        root=(ROOT / saved["run_dir"]).resolve()
        if Path(args.resume).resolve().parent!=root.resolve(): p.error("resume checkpoint is outside its recorded run_dir")
    else:
        cfg=assemble(args.config,args.experiment or "Base",args.seed or 0,args.device)
        runs_root=Path(args.runs_root or ROOT/"outputs").resolve()
        root=Path(args.output).resolve() if args.output else run_directory(runs_root,cfg["revision"]["experiment"],cfg["training"]["seed"])
        if not args.prepare_only and root.exists() and any(root.iterdir()):
            p.error(f"nonempty run_dir refuses overwrite: {root}; use --resume or a new seed/runs-root")
        cfg,report=prepare(cfg,runs_root,require_cache=args.smoke)
        print(json.dumps({"calibration":report["status"],"experiment":cfg["revision"]["experiment"],
                          "mask":cfg["revision"]["mask"],"c_A":report["c_A"],"Q75":report["Q75"],
                          "shared_probe_steps":report["actual_environment_steps"]}),flush=True)
        if args.prepare_only: return
        if args.smoke:
            if not args.output: p.error("--smoke requires an explicit --output outside formal experiment directories")
            if root.is_relative_to(runs_root): p.error("smoke output must be outside formal runs-root")
            cfg["purpose"]="implementation_smoke"
            cfg["training"].update(batch_size=8,random_warmup_env_steps=8,hidden_sizes=[32,32])
            cfg["replay_buffer"]["capacity"]=128
            cfg["collector_limit"]=args.smoke_episode_limit or None
        if args.eval_every is not None: cfg["evaluation"]["interval_steps"]=args.eval_every
        if args.eval_episodes is not None: cfg["evaluation"]["episodes"]=args.eval_episodes
        cfg["outputs"]["root"]=Path(os.path.relpath(runs_root,ROOT)).as_posix()
        cfg["run_dir"]=Path(os.path.relpath(root,ROOT)).as_posix()
        cfg["debug_diagnostics"]=args.debug_diagnostics
        cfg['diagnostics']=dict(schema=DATA_SCHEMA,eval_suite=args.eval_suite or ('nominal' if args.smoke else 'screen'),
            i2_update_every=1000 if args.diagnostic_every is None else args.diagnostic_every,
            train_trace_every=100 if args.train_trace_every is None else args.train_trace_every,
            eval_trace_episodes=1 if args.eval_trace_episodes is None else args.eval_trace_episodes,
            teacher_trace_every=20 if args.teacher_trace_every is None else args.teacher_trace_every)
        refresh_hashes(cfg)
    validate(cfg)
    current_code_hash=code_hash()
    if saved and cfg["provenance"].get("code_hash")!=current_code_hash:
        p.error("exact resume requires the saved source revision; use a new run after code changes")
    cfg["provenance"]["code_hash"]=current_code_hash
    torch.set_num_threads(cfg["training"]["torch_num_threads"])
    torch.use_deterministic_algorithms(cfg["training"]["deterministic_algorithms"],warn_only=False)
    root.mkdir(parents=True,exist_ok=True)
    teacher_log=root/"teacher.jsonl"
    if saved: restore_teacher_log(teacher_log,saved["logs"]["teacher"])
    else: teacher_log.touch(exist_ok=False)
    if cfg.get("debug_diagnostics"): (root/"debug").mkdir(exist_ok=True)
    env=make_env(cfg,cfg["training"]["seed"],episode_limit=cfg.get("collector_limit"))
    print(json.dumps({"replay_estimate_bytes":cfg["replay_buffer"]["capacity"]*(2*5*env.observation_dim*4+2*env.critic_state_dim*4+100),
                      "observation_dim":env.observation_dim,"device":cfg["training"]["device"]}),flush=True)
    agent=Agent(env,cfg); teacher=ParameterTeacher(cfg)
    diagnostics=cfg['diagnostics']; agent.diagnostic_every=diagnostics['i2_update_every']
    data=ExperimentWriter(root,cfg,saved['experiment_data'] if saved else None)
    print(json.dumps({"actual_replay_payload_bytes":agent.replay.array_payload_bytes,
                      "replay_capacity":agent.replay.capacity}),flush=True)
    runner={"episode_index":0,"completed_episodes":0,"last_eval_step":0,"best_validation_key":[-1.,-1e30],
            'best_validation_checkpoint':None,
            "wall_time_s":0.,"active_episode":False,
            "episode_return":np.zeros(5),"process_return":0.,"terminal_return":0.,"process_abs_sum":0.,
            "obs":None,"state":None,"held":None,"hold_age":0,
            'measurement':None,'trace_buffer':[],'scene_seed':None,
            'teacher_wall_time_s':0.,'eval_wall_time_s':0.,'evaluation_sim_steps':0}
    if saved:
        agent.load_state_dict(saved["agent"]); teacher.load_state_dict(saved["teacher"])
        runner=saved["runner"]; env=saved["environment"]
        train_log=CsvLog.recover(root/"train.csv",saved["logs"]["train"],TRAIN_FIELDS)
        update_log=CsvLog.recover(root/"updates.csv",saved["logs"]["updates"],UPDATE_FIELDS)
        window=UpdateWindow(saved["window"])
        selected=runner['best_validation_checkpoint']
        best=root/'best.pt'
        if best.exists():
            archive=root/'orphaned_attempt'/uuid.uuid4().hex
            archive.mkdir(parents=True); best.replace(archive/'best.pt')
        if selected: shutil.copyfile(root/selected,best)
        restore_global_rng(saved["rng"])  # after all constructor initialization
    else:
        train_log=CsvLog(root/"train.csv",TRAIN_FIELDS); update_log=CsvLog(root/"updates.csv",UPDATE_FIELDS)
        window=UpdateWindow()
    write_config(cfg,root/"resolved_config.yaml")
    measurement=EpisodeMetrics(env,runner['measurement']) if runner['measurement'] is not None else None
    data.manifest(agent.global_env_steps,teacher.sim_steps,dict(status='running'))
    start=time.perf_counter(); elapsed=runner["wall_time_s"]
    def wall(): return elapsed+time.perf_counter()-start
    def finalize_logs():
        runner["wall_time_s"]=wall()
        runner['measurement']=measurement.state_dict() if measurement is not None else None
        data.export()
        data.manifest(agent.global_env_steps,teacher.sim_steps,dict(
            wall_time_s=runner['wall_time_s'],teacher_wall_time_s=runner['teacher_wall_time_s'],
            evaluation_wall_time_s=runner['eval_wall_time_s'],evaluation_sim_steps=runner['evaluation_sim_steps'],
            calibration_sim_steps=cfg['calibration'].get('shared_environment_steps',0)))
    stop_requested=[False]
    previous_handler=None
    if threading.current_thread() is threading.main_thread():
        previous_handler=signal.getsignal(signal.SIGINT)
        def request_stop(signum,frame):
            stop_requested[0]=True
            print("Stop requested; finishing the current valid step and exporting logs/policy",flush=True)
        signal.signal(signal.SIGINT,request_stop)
    try:
        while agent.global_env_steps<args.steps and not stop_requested[0]:
            if not runner["active_episode"]:
                seed=named_seed(cfg["training"]["seed"],"training_scene",runner["episode_index"])
                obs,state=env.reset(seed=seed)
                runner.update(active_episode=True,obs=obs,state=state,episode_return=np.zeros(5),process_return=0.,
                              terminal_return=0.,process_abs_sum=0.,held=None,hold_age=0,scene_seed=seed,trace_buffer=[])
                measurement=EpisodeMetrics(env)
            obs,state=runner["obs"],runner["state"]
            warmup=agent.global_env_steps<cfg["training"]["random_warmup_env_steps"]
            if warmup:
                if runner["held"] is None or runner["hold_age"]>=cfg["training"]["random_action_hold_steps"]:
                    runner["held"]=agent.random_actions(); runner["hold_age"]=0
                action=runner["held"].copy(); runner["hold_age"]+=1
            else: action=agent.act(obs,explore=True)
            current_mask=env.agent_active_mask()
            next_obs,next_state,reward,done,info=env.step(action)
            agent.add_transition(obs,state,env.last_executed_actions.copy(),reward,next_obs,next_state,info,
                                 current_mask,env.agent_active_mask())
            trace_interval=diagnostics['train_trace_every']
            capture=trace_interval>0 and runner['episode_index']%trace_interval==0
            trace=measurement.observe(env,reward,info,capture)
            episode_identity=dict(episode_index=runner['episode_index']+1,scene_seed=runner['scene_seed'],
                                  env_steps=agent.global_env_steps)
            if capture:
                runner['trace_buffer'].append(trace)
                if len(runner['trace_buffer'])>=128 or done:
                    data.append('training_steps',dict(**episode_identity,steps=runner['trace_buffer']))
                    runner['trace_buffer']=[]
            if not warmup:
                if not done and teacher.due(agent):
                    before_calls=teacher.calls; teacher_tick=time.perf_counter()
                    interval=diagnostics['teacher_trace_every']
                    teacher.capture_trace=interval>0 and before_calls%interval==0
                    label,record=teacher.propose(env,agent)
                    teacher_seconds=time.perf_counter()-teacher_tick
                    runner['teacher_wall_time_s']+=teacher_seconds
                    label_id=f"{data.identity['run_id']}:{agent.global_env_steps}:{agent.actor_updates}"
                    if label is not None:
                        agent.guidance.add(label["observations"],label["coefficients"],agent.global_env_steps,agent.actor_updates,
                                           label["quality"],label["improvement"],label_id=label_id)
                    branches=strip_branch_traces(record)
                    record.update(**data.identity,**episode_identity,actor_version=agent.actor_updates,
                        episode_step=env.step_count,label_id=label_id if label is not None else None,
                        search_started=teacher.calls>before_calls,teacher_wall_time_s=teacher_seconds,
                        teacher_calls=teacher.calls,teacher_accepted=teacher.accepted,teacher_sim_steps=teacher.sim_steps,
                        search_steps=teacher.pc['horizon_steps'],verify_steps=teacher.pc['verify_steps'],dt_s=env.dt)
                    if branches:
                        data.append('teacher_branches',dict(**episode_identity,actor_version=agent.actor_updates,
                                    search_id=label_id,label_id=record['label_id'],branches=branches))
                    with (root/"teacher.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(record,allow_nan=False)+"\n")
                metrics=agent.update()
                if metrics is not None:
                    for diagnostic in agent.i2_diagnostics:
                        data.append('i2_updates',{**diagnostic,'i2_grad_ratio':metrics['i2_grad_ratio']})
                    if agent.guidance_event is not None: data.append('guidance_events',agent.guidance_event)
                    window.add(metrics)
                    if window.count>=100: window.flush(update_log,agent,teacher,wall())
            runner["obs"],runner["state"]=next_obs,next_state
            runner["episode_return"]+=reward
            for key in ("process_return","terminal_return","process_abs_sum"):
                source={"process_return":"canonical_team_dense_reward","terminal_return":"team_terminal_reward","process_abs_sum":"process_abs_sum"}[key]
                runner[key]+=info.reward_terms[source]
            if done:
                data.append('training_episodes',{**episode_identity,**measurement.summary(env,info)})
                runner["active_episode"]=False; runner["episode_index"]+=1
                if info.terminated:
                    runner["completed_episodes"]+=1; agent.episodes_completed=runner["completed_episodes"]
                    er=runner["episode_return"]
                    row=dict(episode=runner["completed_episodes"],env_steps=agent.global_env_steps,episode_steps=env.step_count,
                        team_return=float(er.mean()),usv_return=float(er[:2].mean()),auv_return=float(er[2:].mean()),
                        process_return=runner["process_return"],terminal_return=runner["terminal_return"],process_abs_sum=runner["process_abs_sum"],
                        success=info.success,terminal_reason=info.terminal_reason,shrink_progress=info.reward_terms["shrink_progress"],
                        wall_time_s=wall(),eval_success_rate=None,eval_return=None)
                    if not np.isclose(row["team_return"],row["process_return"]+row["terminal_return"],rtol=1e-6,atol=1e-5):
                        raise ArithmeticError("episode reward ledger does not balance")
                    evaluation=cfg["evaluation"]
                    if evaluation["interval_steps"] and agent.global_env_steps-runner["last_eval_step"]>=evaluation["interval_steps"]:
                        policy_path=root/'validation_checkpoints'/f'step_{agent.global_env_steps}.pt'
                        save_policy(agent,policy_path,{'selection_split':'validation'})
                        data.register_checkpoint(policy_path)
                        result=evaluate_policy(agent,cfg,evaluation['episodes'],evaluation['base_seed'],
                            suite=diagnostics['eval_suite'],writer=data,
                            checkpoint_id=policy_path.relative_to(root).as_posix(),checkpoint_steps=agent.global_env_steps,
                            planned_steps=runner['last_eval_step']+evaluation['interval_steps'],
                            trace_episodes=diagnostics['eval_trace_episodes'])
                        result['return']=result['return_value']
                        runner['eval_wall_time_s']+=sum(r['eval_wall_time_s'] for r in result['summaries'])
                        runner['evaluation_sim_steps']+=sum(r['episode_steps_total'] for r in result['summaries'])
                        data.export()
                        row.update(eval_success_rate=result["success_rate"],eval_return=result["return"],wall_time_s=wall())
                        key=[result["success_rate"],result["return"]]
                        if tuple(key)>tuple(runner["best_validation_key"]):
                            runner["best_validation_key"]=key
                            runner['best_validation_checkpoint']=policy_path.relative_to(root).as_posix()
                            shutil.copyfile(policy_path,root/'best.pt')
                        runner["last_eval_step"]=agent.global_env_steps
                    train_log.append(row)
                    print(json.dumps(row,allow_nan=False),flush=True)
                    save_policy(agent,root/"latest.pt",{"completed_episode":runner["completed_episodes"]})
        window.flush(update_log,agent,teacher,wall())
        finalize_logs()
        save_policy(agent,root/"latest.pt",{"online_steps":agent.global_env_steps,"mid_episode":runner["active_episode"]})
        print(f"Saved policy and logs in {root}; online steps={agent.global_env_steps}",flush=True)
    finally:
        if previous_handler is not None: signal.signal(signal.SIGINT,previous_handler)



if __name__=="__main__": main()
