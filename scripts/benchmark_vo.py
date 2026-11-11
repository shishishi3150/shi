"""Bounded paired CPU benchmark with immutable shared fixtures and raw samples.

Default batch, network, n-step return and update schedule are unchanged.
Each update sample contains both delayed-policy phases, amortized per update.
No checkpoints, long training or formal success-rate claims are produced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

if __name__=='__main__':
    os.environ.update(NUMBA_NUM_THREADS='3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import numba
import psutil
from matd3.agent import Agent
from scripts.train import make_env
from usv_env.config import assemble,code_hash,refresh_hashes
from usv_env.vo_guidance import project_from_state as project


def timed(function,repeats,cycle=1):
    for _ in range(4):
        for _ in range(cycle):function()
    process=psutil.Process();cpu_before=sum(process.cpu_times()[:2]);start_all=time.perf_counter()
    values=[];rss=[]
    for _ in range(repeats):
        start=time.perf_counter()
        for _ in range(cycle):function()
        values.append((time.perf_counter()-start)/cycle);rss.append(process.memory_info().rss)
    elapsed=time.perf_counter()-start_all;cpu=sum(process.cpu_times()[:2])-cpu_before
    return dict(median_s=float(np.median(values)),mean_s=float(np.mean(values)),min_s=min(values),samples_s=values,
        calls_per_sample=cycle,cpu_seconds=cpu,process_cpu_percent=100.*cpu/elapsed,peak_sampled_rss_bytes=max(rss))


def fixture(path,count=256):
    if path.exists():return dict(np.load(path,allow_pickle=False))
    cfg=assemble(experiment='Full',device='cpu');env=make_env(cfg,20261107)
    rng=np.random.default_rng(20261107);states=[];observations=[]
    for i in range(count):
        obs,state=env.reset(seed=i)
        if i%2:
            own=env.blue_states[0];own.pos=np.array([25.8+i*.001,28.],np.float32)
            own.heading=0.;own.speed=.4;own.sway_speed=.03;own.yaw_rate=.02
            raw,state=env.env._observations();obs=env._push(raw)
        states.append(state);observations.append(obs)
    result=dict(states=np.asarray(states),obs=np.asarray(observations),
        actions=(rng.uniform(-1,1,(count,5,2))*env.agent_action_limits).astype(np.float32))
    env.close();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**result)
    return result


def compare(report,baseline):
    old=json.loads(Path(baseline).read_text(encoding='utf-8'))
    for k in ('fixture_sha256','platform','processor','python','torch','device','threads','numba_thread_budget','batch_size'):
        if report[k]!=old[k]:raise ValueError('incomparable benchmark setting: '+k)
    rows=[]
    for a,b in zip(old['groups'],report['groups']):
        if any(a[k]!=b[k] for k in ('experiment','task_hash','reward_hash')):raise ValueError('incomparable group')
    for a,b in zip(old['filters'],report['filters']):
        rows.append(dict(batch=b['batch'],fresh_speedup=a['fresh_actions']['median_s']/b['fresh_actions']['median_s'],
            repeated_speedup=a['identical_input']['median_s']/b['identical_input']['median_s']))
    return dict(baseline=str(Path(baseline).resolve()),filters=rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--fixture',required=True)
    p.add_argument('--baseline');p.add_argument('--repeats',type=int,default=20)
    args=p.parse_args(argv);output=Path(args.output)
    if output.exists():p.error('refusing to overwrite benchmark evidence')
    if args.repeats<4:p.error('at least four paired samples required')
    torch.set_num_threads(1);numba.set_num_threads(min(3,numba.get_num_threads()));torch.use_deterministic_algorithms(True)
    system_cpu_before=psutil.cpu_percent(interval=.1)
    data=fixture(Path(args.fixture));cfg=assemble(experiment='Full',device='cpu')
    started=time.perf_counter();project(cfg,data['states'][:1],data['obs'][:1],data['actions'][:1]);startup=time.perf_counter()-started
    filters=[]
    for batch in (1,32,256):
        arrays=[data[k][:batch] for k in ('states','obs','actions')]
        first_call=time.perf_counter();expected,diag=project(cfg,*arrays);first_call=time.perf_counter()-first_call
        row=dict(batch=batch,first_call_s=first_call,identical_input=timed(lambda:project(cfg,*arrays),args.repeats),
            reject_only=timed(lambda:project(cfg,*arrays,reject_only=True,decision_only=True),args.repeats),
            intervention_rate=float(np.mean(diag['intervention'])),infeasible_rate=float(np.mean(diag['infeasible'])),
            actions_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
        counter=[0]
        def fresh():
            counter[0]+=1;a=arrays[2].copy();a[:,:,1]+=np.float32(counter[0]*1.e-5)
            return project(cfg,arrays[0],arrays[1],a)
        row['fresh_actions']=timed(fresh,args.repeats);filters.append(row)
    jobs=[]
    def make_job(name):
        current=assemble(experiment=name,device='cpu');current['replay_buffer']['capacity']=512;refresh_hashes(current)
        env=make_env(current,20261107);obs,state=env.reset(seed=20261107);agent=Agent(env,current)
        for i in range(300):
            j=i%len(data['states'])
            agent.replay.add(data['obs'][j],data['states'][j],data['actions'][j],np.full(5,.015,np.float32),
                data['obs'][j],data['states'][j],terminated=False,truncated=False,timeout=False,
                current_context=env.learning_context(),next_context=env.learning_context())
        metric_rows=[];env_rows=[]
        def update():
            m=agent.update();metric_rows.append(m);return m
        def step():
            nonlocal obs,state
            context=env.learning_context();a=agent.act(obs);started=time.perf_counter()
            obs2,state2,reward,done,info=env.step(a);env_seconds=time.perf_counter()-started
            env_rows.append(dict(environment_s=env_seconds,filter_s=float(info.safety_diagnostics.get('vo_time_s',0.))))
            agent.add_transition(obs,state,env.last_executed_actions.copy(),reward,obs2,state2,info,
                current_context=context,next_context=env.learning_context())
            update();obs,state=env.reset(seed=20261107) if done else (obs2,state2)
        return dict(name=name,cfg=current,env=env,update_fn=update,step_fn=step,metrics=metric_rows,env_rows=env_rows)
    for name in ('Base','I1_only','w_o_I1','Full'):jobs.append(make_job(name))
    # Alternate ABBA order each sample so clock/thermal drift does not
    # systematically favour one mechanism. Every block has BOTH policy phases.
    def interleaved(key):
        process=psutil.Process();samples=[[] for _ in jobs];cpu=[0.]*len(jobs);wall=[0.]*len(jobs);rss=[0]*len(jobs)
        for warm in range(4):
            for job in jobs:
                for _ in range(2):job[key]()
        # Environment diagnostics must exclude JIT and all other warm-up work,
        # just as the wall-clock samples do.
        if key=='step_fn':
            for job in jobs:job['env_rows'].clear()
        for sample in range(args.repeats):
            order=range(len(jobs)) if sample%2==0 else reversed(range(len(jobs)))
            for j in order:
                start_cpu=sum(process.cpu_times()[:2]);start=time.perf_counter()
                for _ in range(2):jobs[j][key]()
                elapsed=time.perf_counter()-start
                samples[j].append(elapsed/2.);wall[j]+=elapsed;cpu[j]+=sum(process.cpu_times()[:2])-start_cpu
                rss[j]=max(rss[j],process.memory_info().rss)
        return [dict(median_s=float(np.median(v)),mean_s=float(np.mean(v)),min_s=min(v),samples_s=v,
            calls_per_sample=2,cpu_seconds=cpu[j],process_cpu_percent=100.*cpu[j]/wall[j],peak_sampled_rss_bytes=rss[j]) for j,v in enumerate(samples)]
    updates=interleaved('update_fn');steps=interleaved('step_fn');groups=[]
    for job,update_timing,step_timing in zip(jobs,updates,steps):
        name=job['name'];current=job['cfg'];metric_rows=job['metrics'];env_rows=job['env_rows']
        def component(key):return float(np.mean([m.get(key,0.) or 0. for m in metric_rows]))
        groups.append(dict(experiment=name,update=update_timing,training_step=step_timing,
            environment_mean_s=float(np.mean([r['environment_s'] for r in env_rows])),
            environment_filter_mean_s=float(np.mean([r['filter_s'] for r in env_rows])),
            filter_component_mean_s=dict(target=component('i1_target_vo_time_s'),actor=component('i1_actor_vo_time_s'),i3=component('i3_vo_screen_time_s')),
            i3_retrieved_candidates=max(m['i3_retrieved_candidates'] for m in metric_rows),
            task_hash=current['hashes']['task'],reward_hash=current['hashes']['reward'],
            training_settings=current['training'],i3_settings=current['i3']))
        job['env'].close();print(name,round(update_timing['median_s']*1000,3),'ms/update',flush=True)
    by={r['experiment']:r for r in groups};increments=[]
    for off,on in (('Base','I1_only'),('w_o_I1','Full')):
        increments.append(dict(off=off,on=on,
            update_fraction=by[on]['update']['median_s']/by[off]['update']['median_s']-1.,
            training_step_fraction=by[on]['training_step']['median_s']/by[off]['training_step']['median_s']-1.,
            environment_fraction=by[on]['environment_mean_s']/by[off]['environment_mean_s']-1.))
    process=psutil.Process();info=process.memory_info()
    report=dict(method=cfg['i1']['method'],source_code_hash=code_hash(),fixture_sha256=hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        settings=cfg['i1'],platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,numba=numba.__version__,device='cpu',threads=1,numba_thread_budget=numba.get_num_threads(),batch_size=256,
        startup_s=startup,filters=filters,groups=groups,increments=increments,
        memory=dict(rss_bytes=info.rss,peak_working_set_bytes=getattr(info,'peak_wset',info.rss)),
        cpu=dict(process_cpu_s=sum(process.cpu_times()[:2]),logical_cpus=psutil.cpu_count(),
            system_cpu_percent_before=system_cpu_before,system_cpu_percent_after=psutil.cpu_percent(interval=.1)),
        measurement_order='Interleaved ABBA; two calls per block, equal policy-delay phases',scope='Synthetic fixed Replay and short real environment steps, both policy-delay phases; not convergence, GPU throughput or mission success.')
    if args.baseline:report['comparison']=compare(report,args.baseline)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),increments=increments),indent=2))


if __name__=='__main__':main()
