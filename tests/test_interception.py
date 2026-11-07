"""Acceptance of the five-agent task and its physical information boundaries."""
from copy import deepcopy
import math
import numpy as np
import pytest
from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from usv_env.observation import build_actor_observation,build_critic_state
from usv_env.geometry import capsule_circle_clearance,capsule_inside_rectangle
from usv_env.task import TaskState


def env(seed=0): return PursuitEvasionEnv(assemble(device='cpu'),seed)


def test_fixed_deployment_geometry_and_hulls():
    e=env()
    np.testing.assert_array_equal([s.pos for s in e.blue_states],[[38,22],[41.5,22]])
    np.testing.assert_array_equal([s.pos for s in e.submarine_states],[[12,36],[29,13],[40,36]])
    np.testing.assert_array_equal(e.red_state.pos,[11,30])
    for ship in e.friendly_states()+[e.red_state]:
        assert capsule_inside_rectangle(e._capsule(ship),e.world_size)
        assert min(capsule_circle_clearance(e._capsule(ship),np.asarray(r['center']),r['radius']+.4) for r in e.reefs)>0
    assert e._contacts(e.friendly_states()+[e.red_state],e.friendly_states()+[e.red_state])==('','')


def test_actor_cannot_see_undetected_target_and_critic_can():
    e=env(); before=np.stack([build_actor_observation(e,i) for i in range(5)]); state=build_critic_state(e)
    assert not any(x.valid for x in e.target_estimates)
    assert all(d is None for d in e.step_detections)
    for i in range(5):
        assert before[i,e.observation_schema.index('target_valid')]==0
        assert before[i,e.observation_schema.index('target_valid')]==0
    e.red_state.pos+=np.array([3.,1.]); e.red_state.speed=.12
    np.testing.assert_array_equal(before,np.stack([build_actor_observation(e,i) for i in range(5)]))
    assert not np.array_equal(state,build_critic_state(e))


def test_no_initial_target_or_peer_truth_sync():
    e=env(); assert not e.communication.cache
    assert e.target_communication.sent==0
    e._sense_and_communicate()
    assert not any(x.valid for x in e.target_estimates)


def test_depth_separation_and_role_hulls():
    e=env(); assert not e._can_collide(e.blue_states[0],e.submarine_states[0])
    assert e._can_collide(e.blue_states[0],e.blue_states[1])
    assert e._can_collide(e.submarine_states[0],e.submarine_states[1])
    assert e.surface_length!=e.submarine_length


@pytest.mark.parametrize('seed',[0,1,2])
def test_random_actions_five_shapes_finite_and_deterministic(seed):
    e=HistoryEnv(env(seed),8); f=HistoryEnv(env(seed),8)
    obs,state=e.reset(seed=seed); other,_=f.reset(seed=seed)
    assert obs.shape==(5,e.observation_dim) and state.shape==(e.critic_state_dim,)
    np.testing.assert_array_equal(obs,other)
    rng=np.random.default_rng(seed)
    for _ in range(24):
        action=rng.uniform(-1,1,(5,2)).astype(np.float32)*e.agent_action_limits
        obs,state,reward,done,info=e.step(action); o,s,r,d,i=f.step(action)
        assert all(np.isfinite(x).all() for x in (obs,state,reward)) and reward.shape==(5,)
        np.testing.assert_array_equal(obs,o); np.testing.assert_array_equal(reward,r)
        assert done==d
        if done: break


def fixture_gates(e):
    return dict(config=e.config['task'],dt=.15,red_x=25.,engaged_usvs=2,shadow_gap=1.,progress_deficit=.5,
                observers=3,angle_gap=2.1,informed_usvs=2,route_available=True,red_intent=True,safe=True,counterfactual_valid=True,
                capture_distances=[3.,4.,4.5],capture_hull_margin=1.)


def test_success_requires_continuous_three_seconds_of_capture():
    e=env(); task=TaskState(); values=fixture_gates(e)
    for _ in range(19): assert not task.advance(**values)
    assert task.advance(**values) and task.hold_steps*.15>=3.


@pytest.mark.parametrize('key,value', [('observers',2),('angle_gap',math.pi),
    ('route_available',False),('safe',False),('capture_distances',[3.,4.,6.]),('capture_hull_margin',0.)])
def test_false_success_rejected_and_hold_resets(key,value):
    e=env(); task=TaskState(); values=fixture_gates(e)
    for _ in range(19): task.advance(**values)
    values[key]=value
    assert not task.advance(**values) and task.hold_steps==0


def test_static_state_does_not_collect_arrival_progress():
    from usv_env.reward import snapshot,evaluate
    e=env(); previous=snapshot(e)
    reward,terms=evaluate(previous,e)
    assert terms['blocking']==[0.]*5 and terms['search']==[0.]*5
    assert np.all(reward>=0) and terms['capture_bonus']==[0.]*5


def test_collision_terminal_cannot_pay_positive_task_or_terminal_reward():
    from usv_env.reward import snapshot,evaluate
    e=env(); previous=snapshot(e)
    r,terms=evaluate(previous,e,'collision')
    assert not np.any(r)
    assert terms['team_terminal_reward'] == 0


def test_unknown_discovery_has_no_bonus():
    from usv_env.reward import snapshot,evaluate
    e=env(); r,terms=evaluate(snapshot(e),e)
    assert terms['discovery']==[0.]*5 and terms['tracking']==[0.]*5


def test_static_valid_target_estimate_does_not_grant_blocking_credit():
    from usv_env.reward import snapshot,evaluate
    from usv_env.target_tracking import TargetEstimate
    e=env()
    e.target_estimates[0]=TargetEstimate(True,np.r_[e.blue_states[0].pos,0.,0.],np.eye(4),0.,0.,1.,3,'3:1')
    r,terms=evaluate(snapshot(e),e)
    assert r[0]>=0 and terms['usv_blocking_reward'][0]==0 and terms['usv_contribution_reward'][0]==0


def test_invalid_actions_are_rejected_and_limits_store_execution():
    e=env()
    with pytest.raises(ValueError): e.step(np.zeros((6,2)))
    with pytest.raises(ValueError): e.step(np.full((5,2),np.nan))
    e.step(e.agent_action_limits*100.)
    np.testing.assert_array_equal(e.last_executed_actions,e.agent_action_limits)


def test_innovation_switches_are_distinct_from_base():
    base=assemble(experiment='Base')['mechanisms']
    assert not any(base.values())
    assert all(assemble(experiment='Full')['mechanisms'].values())


def test_deliberate_early_crash_cannot_escape_a_negative_living_cost():
    e=env(); cfg=e.config; gamma=cfg["training"]["gamma"]
    from usv_env.reward import evaluate
    safe, terms=evaluate(None,e)
    crash,_=evaluate(None,e,'collision')
    assert np.all(sum(gamma**t*safe for t in range(e.max_steps)) > crash)
    assert not np.any(terms['time_cost']) and not np.any(crash)
