"""Bounded physical diagnostics, not training or a learned-policy claim.

The ideal-information reference is used ONLY to test task feasibility. Its
initial relative offsets never enter a reward, Actor, Replay observation or
production controller. No optimizer, curriculum, or red-script change is used.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from scripts.validate_reward import diagnostic_config,translation_actions
from scripts.experiment_data import EpisodeMetrics,plain
from usv_env.config import assemble,refresh_hashes
from usv_env.environment import PursuitEvasionEnv
from usv_env.red_policy import _reference_action,RuleController


def physical_case(mode,seed=5,collect=True):
    cfg=diagnostic_config();cfg['scenario']['blue_starts']=[[15.,24.],[15.,30.]]
    refresh_hashes(cfg);env=PursuitEvasionEnv(cfg,seed);metrics=EpisodeMetrics(env)
    offsets=[s.pos-env.red_state.pos for s in env.submarine_states];trajectory=[]
    try:
        for _ in range(env.max_steps):
            actions=np.zeros((5,2),np.float32)
            for i,ship in enumerate(env.friendly_states()):
                if i<2 and not env.capture_success:continue
                if mode=='scatter' and i>=2 and env.capture_success:
                    actions[i]=[-env.agent_action_limits[i,0],0.];continue
                off=np.array([5.,-3. if i==0 else 3.]) if i<2 else offsets[i-2]
                velocity=env.red_state.velocity+.65*(env.red_state.pos+off-ship.pos)
                velocity*=min(1.,env.max_speed_by_type[ship.unit_type]/max(np.linalg.norm(velocity),1e-9))
                actions[i]=_reference_action(ship,ship.pos+5*velocity,float(np.linalg.norm(velocity)),
                    env.blue_dynamics if i<2 else env.submarine_dynamics)
            _,_,reward,done,info=env.step(actions)
            metrics.observe(env,reward,info)
            if collect:trajectory.append(dict(step=env.step_count,positions=[s.pos.copy() for s in env.friendly_states()],
                red=env.red_state.pos.copy(),strict=bool(info.stage_success_gates and all(info.stage_success_gates.values())),
                captured=info.capture_success,quality=info.objective_quality))
            if done:break
        summary=plain(metrics.summary(env,info));summary['diagnostic_information']='ideal_truth_reference_for_feasibility_only'
        return summary,plain(trajectory)
    finally:env.close()


def local_case(experiment,seed=5,steps=96):
    cfg=assemble('configs/fast.yaml',experiment,seed,'cpu');env=PursuitEvasionEnv(cfg,seed);controller=RuleController();m=EpisodeMetrics(env)
    try:
        for _ in range(steps):
            *_,r,done,info=env.step(controller.actions_for_env(env));m.observe(env,r,info)
            assert np.isfinite(r).all()
            if done:break
        return plain(m.summary(env,info))
    finally:env.close()


def plot_cases(cases,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    colors=['#1764aa','#438ccc','#199b72','#4bae88','#91bb61']
    for ax,(name,(summary,rows)) in zip(axes,cases.items()):
        points=np.asarray([r['positions'] for r in rows]);red=np.asarray([r['red'] for r in rows])
        for j,color in enumerate(colors):ax.plot(points[:,j,0],points[:,j,1],color=color,label=('USV' if j<2 else 'AUV')+str(j+1 if j<2 else j-1))
        ax.plot(red[:,0],red[:,1],color='#c33b3b',lw=2,label='Red')
        capture=summary['first_capture_step']
        for step,marker,label in ((capture,'o','Capture / retreat'),(650,'s','End')):
            if step and step<=len(rows):ax.scatter(*red[step-1],marker=marker,color='black',s=40,label=label)
        reentries=[i for i in range(1,len(rows)) if rows[i]['captured'] and rows[i]['strict'] and not rows[i-1]['strict']]
        if reentries:ax.scatter(red[reentries,0],red[reentries,1],marker='^',color='#9952a3',label='Re-encircled')
        ax.set(title=f"{name}: capture ratio {summary['post_capture_encirclement_ratio']:.1%}",xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2);fig.savefig(path,dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args(argv)
    root=Path(args.output)
    if root.exists():p.error('use a new output directory')
    root.mkdir(parents=True)
    cases={name:physical_case(name) for name in ('sustained','scatter')}
    good,bad=(cases[n][0] for n in ('sustained','scatter'))
    assert good['full_mission_success']==1,good
    assert bad['full_mission_success']==0 and bad['capture_triggered']==1,bad
    assert np.all(np.asarray(good['per_agent_return'])>np.asarray(bad['per_agent_return']))
    local={name:local_case(name) for name in ('Base','Full')}
    result=dict(physical={n:s for n,(s,t) in cases.items()},local_short_diagnostics=local,
        feasibility='constructed real dynamics with ideal-information upper-bound controller; not proof of global/learned-policy feasibility',
        production_algorithms_unchanged=True)
    (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for n,(s,t) in cases.items():(root/(n+'_trajectory.json')).write_text(json.dumps(t,allow_nan=False),encoding='utf-8')
    plot_cases(cases,root/'trajectories.png')
    print(json.dumps({n:{k:s[k] for k in ('first_capture_step','full_mission_success','post_capture_encirclement_ratio','maximum_loss_duration','usv_safe_follow_fraction','episode_return')} for n,(s,t) in cases.items()},ensure_ascii=False))

if __name__=='__main__':main()
