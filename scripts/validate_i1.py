"""Reproduce the fixed-Actor reef failure and audit I1 braking time.

This is a safety regression, never evidence of learning/convergence. It does
not replace the saved Actor, alter the scene, or persist replay/checkpoints.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse
from copy import deepcopy
import gzip
import json
import time
import numpy as np
import torch

from matd3.checkpoint import load_policy
from scripts.experiment_data import EpisodeMetrics, atomic_json, plain, policy_hash, file_hash
from usv_env.config import assemble, refresh_hashes, code_hash
from usv_env.environment import PursuitEvasionEnv
from usv_env.geometry import capsule_segment, capsule_circle_clearance
from usv_env.history import HistoryEnv
from usv_env.reward import boundary_clearance
from usv_env.safety_projection import _Predictor, _dimensions
from usv_env.types import ShipState


def braking_envelope(cfg):
    predictor=_Predictor(cfg); rows=[]
    for sub,model in enumerate(predictor.models):
        for surge in (model.min_speed,0.,model.max_speed):
            for sway in (-model.max_sway_speed,0.,model.max_sway_speed):
                for yaw in (-model.max_yaw_rate,0.,model.max_yaw_rate):
                    own=ShipState(np.array([10.,10.],np.float32),surge,0.,yaw,
                                  'submarine' if sub else 'surface',not sub,sway)
                    origin=own.pos.copy(); stopped=False; distance=0.
                    for k in range(cfg['i1']['backup_steps']):
                        own=model.step(own,predictor.braking_action(own))
                        distance=max(distance,float(np.linalg.norm(own.pos-origin)))
                        stopped=(own.ground_speed<=cfg['i1']['stop_speed_mps']
                                 and abs(own.yaw_rate)<=cfg['i1']['stop_yaw_rate_rps'])
                        if stopped: break
                    rows.append(dict(role='auv' if sub else 'usv',surge=surge,sway=sway,yaw=yaw,
                        stopped=stopped,seconds=(k+1)*cfg['env']['dt'],max_displacement_m=distance))
    return rows


def replay(policy,cfg,seed,trace_path):
    env=HistoryEnv(PursuitEvasionEnv(cfg,seed),cfg['revision']['history_length'])
    env.set_evaluation_variant('fixed'); obs,_=env.reset(seed=seed)
    measurement=EpisodeMetrics(env); minimum=float('inf'); first_intervention=None
    first_infeasible=None; records=[]; start=time.perf_counter()
    try:
        with gzip.open(trace_path,'wt',encoding='utf-8') as trace:
            while True:
                action=policy.act(obs,explore=False)
                obs,_,reward,done,info=env.step(action)
                record=measurement.observe(env,reward,info,True)
                trace.write(json.dumps(plain(record),allow_nan=False,separators=(',',':'))+'\n')
                records.append(plain(dict(step=env.step_count,action=env.last_executed_actions,
                    positions=[s.pos for s in env.friendly_states()],diagnostics=info.safety_diagnostics)))
                diag=info.safety_diagnostics
                if first_intervention is None and any(diag.get('intervention',[])): first_intervention=env.step_count
                if first_infeasible is None and any(diag.get('infeasible',[])): first_infeasible=env.step_count
                for own in env.friendly_states():
                    length,width=_dimensions(cfg,own.unit_type=='surface')
                    capsule=capsule_segment(own.pos,own.heading,length,width)
                    clearance=boundary_clearance(capsule,env.world_size)
                    for reef in cfg['reefs']:
                        clearance=min(clearance,capsule_circle_clearance(capsule,np.asarray(reef['center']),
                            reef['radius']+cfg['depth']['reef_shallow_buffer_m']))
                    minimum=min(minimum,clearance)
                if done: break
        return dict(summary=measurement.summary(env,info),wall_time_s=time.perf_counter()-start,
                    minimum_poststep_static_clearance_m=minimum,first_intervention_step=first_intervention,
                    first_infeasible_step=first_infeasible),records
    finally: env.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(ROOT/'outputs/Full/seed101/latest.pt'))
    p.add_argument('--output',default=str(ROOT/'validation/five_agent_acceptance/i1_optimization/safety_validation.json'))
    args=p.parse_args(argv); output=Path(args.output)
    if output.exists(): p.error('refusing to overwrite an existing diagnostic')
    output.parent.mkdir(parents=True,exist_ok=True)
    trace_root=output.parent/output.stem
    trace_root.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    policy,_=load_policy(args.checkpoint,'cpu'); original_hash=policy_hash(policy)
    cfg=deepcopy(policy.runtime_config)
    cfg['communication']['peer_mode']='disabled'
    settings=assemble(experiment='Full',device='cpu')['i1']
    configs={}
    for name in ('short_window','recovery'):
        current=deepcopy(cfg)
        current['i1']=deepcopy(settings)
        if name=='short_window': current['i1'].update(horizon_steps=8,backup_steps=0)
        refresh_hashes(current); configs[name]=current
    results={}; traces={}
    for name,current in configs.items():
        results[name],traces[name]=replay(policy,current,2000200,trace_root/(name+'_replay.jsonl.gz'))
        results[name]['settings']=current['i1']
        print(name,results[name]['summary']['terminal_reason'],results[name]['summary']['episode_steps'],flush=True)
    first_difference=next((a['step'] for a,b in zip(traces['short_window'],traces['recovery'])
                           if not np.array_equal(a['action'],b['action'])),None)
    assert policy_hash(policy)==original_hash,'safety regression changed frozen Actor'
    atomic_json(output,dict(scope='One frozen 300-step Actor, one fixed held-out failure seed; not a success-rate study.',
        source_code_hash=code_hash(),checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=file_hash(args.checkpoint),actor_parameters_unchanged=True,
        eval_seed=2000200,condition='no_peer_state',first_changed_action_step=first_difference,
        braking_envelope=braking_envelope(configs['recovery']),results=results,
        limits='Zero-wave/CV safety predictor; actual environment retains waves. Unknown vessels, future acceleration, finite candidates and near-rest terminal tolerance prevent a formal safety guarantee.'))
    print(output)


if __name__=='__main__': main()
