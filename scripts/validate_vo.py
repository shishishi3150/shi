"""Bounded fixed/random VO acceptance with real swept-capsule physics.

All failures (including blind encounters and initially overlapping fixtures)
are retained. This is a safety regression, not learned mission performance.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from usv_env.config import assemble,code_hash
from usv_env.environment import PursuitEvasionEnv
from usv_env.types import ShipState
from usv_env.safety import path_contact
from usv_env.geometry import capsule_circle_clearance
from usv_env.reward import boundary_clearance,capsule_clearance
from usv_env.vo_guidance import VOGuidance


def vessel(x,y,heading=0.,speed=.3,role='surface',sway=0.):
    return ShipState(np.array([x,y],np.float32),speed,heading,0.,role,role=='surface',sway)


def cases(random_count):
    fixed=[('reef_usv',[vessel(24.,28.1)],{}),
        ('reef_auv',[vessel(24.,28.1,role='submarine')],{}),
        ('boundary_usv',[vessel(1.8,10.,math.pi,.4)],{}),
        ('boundary_auv',[vessel(1.8,10.,math.pi,.4,'submarine')],{}),
        ('head_on_usv',[vessel(6.,27.),vessel(14.,27.,math.pi)],{}),
        ('crossing_usv',[vessel(6.,27.),vessel(10.,23.,math.pi/2)],{}),
        ('three_auv',[vessel(10.+4.*math.cos(t),27.+4.*math.sin(t),t+math.pi,.25,'submarine') for t in (0.,2*math.pi/3,4*math.pi/3)],{}),
        ('reef_channel',[vessel(23.5,32.,.2,.3)],{}),
        ('red_encounter',[vessel(6.,27.),vessel(14.,27.,math.pi)],{'uncontrolled_last':True}),
        ('head_on_auv',[vessel(6.,27.,role='submarine'),vessel(14.,27.,math.pi,role='submarine')],{}),
        ('vertical_separation',[vessel(10.,27.),vessel(10.,27.,role='submarine')],{}),
        ('near_zero',[vessel(24.,28.1,speed=1.e-12)],{}),
        ('infeasible_overlap',[vessel(19.6,38.,speed=0.)],{'initially_infeasible':True}),
        ('sideslip_usv',[vessel(24.,28.1,sway=.08)],{}),
        ('sideslip_auv',[vessel(24.,28.1,role='submarine',sway=.08)],{}),
        ('pi_wrap',[vessel(2.2,27.,math.pi-.001,.35,sway=.01)],{}),
        ('delay_loss',[vessel(6.,27.),vessel(14.,27.,math.pi)],{'delay':3,'loss':.1}),
        ('blind_red',[vessel(6.,27.),vessel(14.,27.,math.pi)],{'uncontrolled_last':True,'blind':True})]
    rng=np.random.default_rng(3375023)
    cfg=assemble(device='cpu');env=PursuitEvasionEnv(cfg)
    for i in range(random_count):
        while True:
            role='surface' if i%2==0 else 'submarine'
            own=vessel(*rng.uniform(3.,52.,2),heading=rng.uniform(-math.pi,math.pi),speed=rng.uniform(.05,.4),role=role,sway=rng.uniform(-.06,.06))
            cap=env._capsule(own)
            if min(capsule_circle_clearance(cap,np.array(r['center']),r['radius']+cfg['depth']['reef_shallow_buffer_m']) for r in cfg['reefs'])>1.5:break
        fixed.append((f'random_{i:03d}',[own],{'seed':3375023+i}))
    env.close();return fixed


def run_case(cfg,guidance,name,initial,settings,steps,trace_path):
    env=PursuitEvasionEnv(cfg);states=deepcopy(initial);history=[deepcopy(states)];cache={}
    n=len(states);controlled=n-int(settings.get('uncontrolled_last',False));previous=np.zeros(n)
    rng=np.random.default_rng(settings.get('seed',432));minimum=1.e6;interventions=np.zeros(controlled,int)
    unresolved=np.zeros(controlled,int);triggered=np.zeros(controlled,int);vo_seconds=0.;reason='';pair='';finite=True
    count=0
    with gzip.open(trace_path,'xt',encoding='utf-8') as trace:
        for step in range(steps):
            commands=[];diagnostics=[]
            for i,own in enumerate(states):
                model=guidance.models[int(own.unit_type!='surface')];model.record_trajectory=True
                speed=.3 if own.unit_type=='surface' else .4
                command=np.array([(model.damping_matrix(np.array([speed,0.,0.]))@np.array([speed,0.,0.]))[0],0.],np.float32)
                if i<controlled:
                    obstacles=[]
                    for j in range(n):
                        if i==j or (settings.get('blind') and j==n-1):continue
                        if step>=settings.get('delay',0) and rng.random()>=settings.get('loss',0.):
                            cache[i,j]=deepcopy(history[max(0,step-settings.get('delay',0))][j])
                        if (i,j) in cache:
                            peer=cache[i,j]
                            obstacles.append(dict(position=peer.pos,velocity=peer.velocity,surface=peer.unit_type=='surface',uncertainty=cfg['i1']['uncertainty_margin_m']))
                    started=__import__('time').perf_counter()
                    command,diag=guidance.project_one(own,obstacles,command,np.array([model.max_surge_force,model.max_yaw_moment]),previous_turn=previous[i])
                    vo_seconds+=__import__('time').perf_counter()-started
                    previous[i]=command[1];interventions[i]+=diag['intervention'];unresolved[i]+=diag['infeasible'];triggered[i]+=diag.get('triggered',diag['intervention'])
                    diagnostics.append(diag)
                commands.append(command)
            following=[guidance.models[int(own.unit_type!='surface')].step(own,command) for own,command in zip(states,commands)]
            for i,(old,new) in enumerate(zip(states,following)):
                cap=env._capsule(new);minimum=min(minimum,boundary_clearance(cap,env.world_size))
                if path_contact(env,old,new):reason='out_of_bounds';pair=str(i)
                for reef in cfg['reefs']:
                    center=np.array(reef['center']);radius=reef['radius']+cfg['depth']['reef_shallow_buffer_m']
                    minimum=min(minimum,capsule_circle_clearance(cap,center,radius))
                    if path_contact(env,old,new,obstacle=(center,radius)):reason='reef_collision';pair=f'{i}-reef-{reef["name"]}'
                for j in range(i+1,n):
                    if not env._can_collide(new,following[j]):continue
                    minimum=min(minimum,capsule_clearance(cap,env._capsule(following[j])))
                    if path_contact(env,old,new,states[j],following[j]):reason='collision';pair=f'{i}-{j}'
                finite &= bool(np.isfinite(new.pos).all() and np.isfinite(new.body_velocity).all() and new.speed>=0.)
            count=step+1
            trace.write(json.dumps(dict(step=count,states=[dict(pos=s.pos.tolist(),heading=s.heading,u=s.speed,v=s.sway_speed,r=s.yaw_rate) for s in following],
                actions=[a.tolist() for a in commands],diagnostics=diagnostics,contact=reason),allow_nan=False,default=lambda x:x.item())+'\n')
            states=following;history.append(deepcopy(states))
            if reason or not finite:break
    env.close()
    return dict(name=name,steps=count,contact=reason,collision_pair=pair,finite=finite,
        minimum_actual_clearance_m=minimum,interventions=interventions.tolist(),unresolved=unresolved.tolist(),
        triggered=triggered.tolist(),intervention_rate=float(interventions.sum()/max(count*controlled,1)),vo_time_s=vo_seconds,
        information='no target information' if settings.get('blind') else 'explicit delivered held packets; ideal at zero delay/loss',
        fixture=settings,trace=str(trace_path),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())


def theorem_audit(cfg,g):
    rows=[]
    for role,model in zip(('usv','auv'),g.models):
        accelerations=[]
        for u in np.linspace(0.,model.max_speed,6):
            for v in (-model.max_sway_speed,0.,model.max_sway_speed):
                for r in (-model.max_yaw_rate,0.,model.max_yaw_rate):
                    for f in (-model.max_surge_force,model.max_surge_force):
                        accelerations.append(abs(model.body_acceleration(np.array([u,v,r]),np.array([f,model.max_yaw_moment]))[0]))
        rows.append(dict(role=role,minimum_surge_mps=model.min_speed,maximum_surge_mps=model.max_speed,
            red_maximum_surge_mps=cfg['ship']['max_speed_red'],strictly_faster_at_all_operating_speeds=False,
            sampled_maximum_abs_surge_acceleration_mps2=max(accelerations),
            full_theorem_applicable=False))
    return dict(paper_doi='10.1109/TCST.2024.3375023',roles=rows,
        reasons=['V.4 positive minimum surge is violated by stationary starts and emergency braking.',
            'V.8 requires Umin strictly above obstacle speed: USV maximum .40 equals Red .40; AUV may start at zero and meets other AUVs capped at .50.',
            'V.7, V.9, V.10 and bounds (57),(72),(73) are not certified for saturated reference_action, nonlinear drag, wave loads and delayed/noisy observations.',
            'Course adapter is bounded force tracking using the existing dynamics; it is not the exact feedback-linearizing controller (48).',
            'Finite sampled tests are empirical evidence only; unsafe/infeasible and unknown obstacles remain explicit.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--random-cases',type=int,default=24);p.add_argument('--steps',type=int,default=180)
    args=p.parse_args(argv);out=Path(args.output)
    if out.exists():p.error('refusing to overwrite validation evidence')
    out.mkdir(parents=True);cfg=assemble(experiment='Full',device='cpu')
    g=VOGuidance(cfg)
    results=[]
    for name,initial,settings in cases(args.random_cases):
        row=run_case(cfg,g,name,initial,settings,args.steps,out/(name+'.jsonl.gz'));results.append(row)
        print(name,row['contact'] or 'clear',row['minimum_actual_clearance_m'],flush=True)
    report=dict(method=cfg['i1']['method'],source_code_hash=code_hash(),config=cfg,
        scope='Fixed and seeded random force-policy regressions using real unchanged dynamics and continuous swept-capsule contacts; no formal learning.',
        all_samples_retained=True,results=results,
        contacts={k:sum(r['contact']==k for r in results) for k in ('collision','reef_collision','out_of_bounds')},
        finite=all(r['finite'] for r in results),theory=theorem_audit(cfg,g))
    with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report['contacts']))


if __name__=='__main__':main()
