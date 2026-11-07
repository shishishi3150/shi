"""I3 uses online complete actions, aligned metadata and conservative n-step targets."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from matd3.agent import Agent
from matd3.history_enhancement import DEFAULTS, JointHistory, conservative_value, semantic_features
from matd3.replay_buffer import ReplayBuffer
from usv_env.config import assemble, refresh_hashes
from usv_env.history import schema_for_counts


def context(phase=1, age=0., quality=1., valid=1.):
    return dict(phase_id=phase, information_valid=np.full(5, valid),
                information_age=np.full(5, age), information_variance=np.ones(5),
                safety_margin=np.full(5, .6), quality=quality)


def state_fixture(cfg):
    state = np.zeros(cfg["observation_schema"]["critic_dim"], np.float32)
    for i, pos in enumerate(cfg["scenario"]["blue_starts"]+cfg["scenario"]["submarine_starts"]):
        state[i*9:i*9+9] = [1., i<2, pos[0]/55., pos[1]/55., 0., 0., 0., 1., 0.]
    state[45:52] = [11./55., 27./55., .4/1.5, 0., 0., 1., 0.]
    return state


def small_agent(monkeypatch, experiment="I3_only"):
    torch.set_num_threads(1)
    cfg = assemble(experiment=experiment, device="cpu")
    cfg["training"].update(batch_size=2, n_step_return=1, hidden_sizes=[16, 16])
    cfg["replay_buffer"]["capacity"] = 16
    cfg["i3"].update(candidates=2, search_pool=16)
    refresh_hashes(cfg)
    schema = schema_for_counts(5)
    obs_dim = schema.frame_dim*cfg["revision"]["history_length"]
    env = SimpleNamespace(num_surface_blue=2, num_submarines=3, observation_dim=obs_dim,
        critic_state_dim=cfg["observation_schema"]["critic_dim"],
        agent_action_limits=np.asarray([[2.2,1.5]]*2+[[3.035,1.5]]*3,np.float32))
    monkeypatch.setattr("matd3.agent.nominal_actions", lambda _:np.zeros((5,2),np.float32))
    agent = Agent(env,cfg)
    frames = np.zeros((5,cfg["revision"]["history_length"],schema.frame_dim),np.float32)
    frames[:,:,-1] = 1.
    return agent, frames.reshape(5,-1), state_fixture(cfg)


def add(replay, obs, state, action, current=None, following=None, **ending):
    flags = dict(terminated=False,truncated=False,timeout=False,terminal_reason="")
    flags.update(ending)
    replay.add(obs,state,action,np.ones(5,np.float32),obs,state,
               current_context=current,next_context=following,**flags)


def test_nstep_metadata_nominal_and_joint_action_are_from_correct_endpoints():
    replay = ReplayBuffer(8,5,3,4,2,n_step_return=3,gamma=.5)
    obs = np.zeros((5,3),np.float32)
    for t in range(3):
        state = np.full(4,t,np.float32)
        replay.add(obs+t,state,np.full((5,2),t,np.float32),np.full(5,t+1,np.float32),
            obs+t+1,state+1,terminated=t==2,truncated=False,timeout=t==2,
            terminal_reason="timeout" if t==2 else "",current_context=context(phase=t%3,age=t),
            next_context=context(phase=min(t+1,2),age=t+1),nominal_action=np.full((5,2),10+t,np.float32))
    sample = replay._sample_rows(np.array([0]),np.ones(1),np.ones(1))
    np.testing.assert_array_equal(sample["actions"][0],np.zeros((5,2)))
    np.testing.assert_array_equal(sample["nominal_actions"][0],np.full((5,2),10))
    np.testing.assert_array_equal(sample["states"][0],np.zeros(4))
    np.testing.assert_array_equal(sample["next_states"][0],np.full(4,3))
    np.testing.assert_array_equal(sample["current_context"][0,5:10],np.zeros(5))
    np.testing.assert_array_equal(sample["next_context"][0,5:10],np.full(5,3))
    np.testing.assert_array_equal(sample["agent_rewards"][0],np.full(5,1+.5*2+.25*3))
    assert sample["n_steps"][0,0]==3 and sample["bootstrap_discounts"][0,0]==0
    assert sample["phase_ids"][0,0]==0 and sample["next_phase_ids"][0,0]==2


def test_external_cut_bootstraps_without_crossing_reset():
    replay=ReplayBuffer(8,5,2,3,2,n_step_return=3,gamma=.5)
    obs=np.zeros((5,2),np.float32); state=np.zeros(3,np.float32)
    add(replay,obs,state,np.zeros((5,2),np.float32),context(),context(age=1.))
    replay.external_reset()
    add(replay,obs+7,state+7,np.ones((5,2),np.float32),context(),context(),terminated=True)
    assert len(replay)==2
    sample=replay._sample_rows(np.array([0,1]),np.ones(2),np.ones(2))
    np.testing.assert_array_equal(sample["n_steps"].ravel(),[1,1])
    np.testing.assert_array_equal(sample["bootstrap_discounts"].ravel(),[.5,0.])
    np.testing.assert_array_equal(sample["next_states"][0],state)


@pytest.mark.parametrize("corruption", ["missing","negative_age","invalid_quality","invalid_mask"])
def test_invalid_learning_metadata_is_rejected(corruption):
    replay=ReplayBuffer(4,5,2,3,2); current=context()
    if corruption=="missing": del current["safety_margin"]
    elif corruption=="negative_age": current["information_age"][0]=-1
    elif corruption=="invalid_quality": current["quality"]=1.1
    else: current["information_valid"][0]=.5
    with pytest.raises(ValueError):
        add(replay,np.zeros((5,2)),np.zeros(3),np.zeros((5,2)),current,context())


def test_unsafe_first_step_quality_and_observed_nstep_terminal_reason_align():
    replay=ReplayBuffer(4,5,2,3,2,n_step_return=2)
    obs=np.zeros((5,2)); state=np.zeros(3)
    add(replay,obs,state,np.zeros((5,2)),context(),context())
    add(replay,obs,state,np.zeros((5,2)),context(),context(),terminated=True,terminal_reason="collision")
    assert replay.current_context[0,-2]==1. and replay.current_context[1,-2]==0.
    np.testing.assert_array_equal(replay.terminal_reason_ids[:2,0],[2,2])


def test_realized_nstep_returns_reduce_quality_without_using_unknown_episode_future(monkeypatch):
    agent,obs,state=small_agent(monkeypatch)
    agent.history.settings["minimum_quality"]=0.
    for reward in (2.,-2.):
        agent.replay.add(obs,state,np.full((5,2),reward/10.),np.full(5,reward),obs,state,
            terminated=False,truncated=False,timeout=False,current_context=context(),next_context=context())
    query,_=agent.replay._prepare_context(context())
    found=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    by_index={int(index):slot for slot,index in enumerate(found["indices"][0]) if index>=0}
    for index,reward in enumerate((2.,-2.)):
        slot=by_index[index]
        assert found["realized_return"][0,slot]==reward
        expected=1./(1.+np.exp(-reward/agent.history.settings["return_scale"]))
        assert found["outcome_confidence"][0,slot]==pytest.approx(expected)
        assert found["quality"][0,slot]==pytest.approx(expected)
    agent.history.settings["minimum_quality"]=.2
    filtered=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    assert filtered["valid"].sum()==1 and filtered["indices"][0,0]==0
    agent.history.settings["return_scale"]=2.
    scaled=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    positive_slot=int(np.flatnonzero(scaled["indices"][0]==0)[0])
    assert scaled["outcome_confidence"][0,positive_slot]==pytest.approx(1./(1.+np.exp(-1.)))


@pytest.mark.parametrize("reason",["collision","reef_collision","out_of_bounds","scenario_anomaly","numerical_error"])
@pytest.mark.parametrize("variant",["conservative","random_history","no_state_matching","no_safety_screening","hard_max"])
def test_fatal_observed_nstep_window_never_supplies_history(monkeypatch,reason,variant):
    agent,obs,state=small_agent(monkeypatch)
    agent.replay.n_step_return=2
    agent.history.settings.update(variant=variant,minimum_quality=0.)
    add(agent.replay,obs,state,np.ones((5,2)),context(),context())
    assert len(agent.replay)==0  # Uncommitted future outcomes are unavailable.
    add(agent.replay,obs,state,np.ones((5,2)),context(),context(),terminated=True,terminal_reason=reason)
    assert len(agent.replay)==2 and agent.replay.current_context[0,-2]==1.
    query,_=agent.replay._prepare_context(context())
    found=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    assert not found["valid"].any() and (found["indices"]==-1).all()


def test_retrieval_preserves_complete_executed_action_and_filters_semantics(monkeypatch):
    agent,obs,state=small_agent(monkeypatch)
    for value,phase,valid,quality in ((.1,1,1,1.),(.2,2,1,1.),(.3,1,0,1.),(.4,1,1,.1)):
        add(agent.replay,obs,state,np.full((5,2),value,np.float32),
            context(phase=phase,valid=valid,quality=quality),context())
    query,_=agent.replay._prepare_context(context())
    result=agent.history.retrieve(agent.replay,state[None],query[None],np.array([[1]]),np.ones((1,5)))
    assert result["valid"].sum()==1 and result["indices"][0,0]==0
    np.testing.assert_array_equal(result["actions"][0,0],agent.replay.actions[0])
    # Semantic groups react separately to geometry, age, variance and safety.
    baseline=semantic_features(state[None],query[None],agent.runtime_config)
    for index in (5,10,15):
        changed=query.copy(); changed[index]+=2.
        assert not np.array_equal(baseline,semantic_features(state[None],changed[None],agent.runtime_config))


def test_missing_metadata_never_becomes_a_candidate(monkeypatch):
    agent,obs,state=small_agent(monkeypatch)
    add(agent.replay,obs,state,np.ones((5,2)))
    query,_=agent.replay._prepare_context(context())
    found=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    assert not found["valid"].any()


def test_conservative_formula_clips_positive_advantage_and_selects_one_joint_row():
    q=torch.tensor([[1.,2.,3.,4.,5.]])
    historical=torch.tensor([[[5.,1.,3.5,4.,4.],[.5,6.,2.,8.,4.]]])
    value,detail=conservative_value(q,historical,torch.tensor([[.8,.5]]),torch.ones((1,2),dtype=torch.bool),DEFAULTS)
    # candidate 0 score=.12; candidate 1 score=.10: all heads use candidate 0.
    assert detail["chosen"].item()==0
    torch.testing.assert_close(value,torch.tensor([[1.4,2.,3.2,4.,5.]]))
    assert (value>=q).all() and (value-q<=.5).all()
    assert (2.+.99**3*value).shape==(1,5)


def test_empty_or_nonpositive_history_is_exact_fallback_and_hardmax_is_distinct():
    q=torch.ones((1,5)); historical=torch.full((1,2,5),20.)
    valid=torch.zeros((1,2),dtype=torch.bool)
    fallback,detail=conservative_value(q,historical,torch.ones((1,2)),valid,DEFAULTS)
    torch.testing.assert_close(fallback,q); assert not detail["accepted"].any()
    conservative,_=conservative_value(q,historical,torch.ones((1,2)),~valid,DEFAULTS)
    hard,_=conservative_value(q,historical,torch.zeros((1,2)),~valid,{**DEFAULTS,"variant":"hard_max"})
    torch.testing.assert_close(conservative,torch.full((1,5),1.5))
    torch.testing.assert_close(hard,torch.full((1,5),20.))
    lower,_=conservative_value(q,torch.zeros_like(historical),torch.ones((1,2)),~valid,DEFAULTS)
    torch.testing.assert_close(lower,q)


class LinearJointValue(nn.Module):
    def __init__(self,heads,offset): super().__init__(); self.heads,self.offset=heads,offset
    def forward(self,state,actions,obs):
        return (actions.flatten(1).mean(1,keepdim=True)+self.offset).expand(-1,self.heads)


@pytest.mark.parametrize("safe,variant,accepted",[(True,"conservative",True),(False,"conservative",False),
                                                  (False,"no_safety_screening",True)])
def test_actual_target_uses_twin_min_current_state_screen_and_confidence(monkeypatch,safe,variant,accepted):
    agent,obs,state=small_agent(monkeypatch)
    agent.history.settings["variant"]=variant
    agent.critic_targets=[[LinearJointValue(2,.2),LinearJointValue(2,0.)],
                          [LinearJointValue(3,.2),LinearJointValue(3,0.)]]
    action=np.full((5,2),.8,np.float32)
    query_state=state.copy(); query_state[2]+=.01
    agent.replay.add(obs,state,action,np.ones(5),obs,query_state,
        terminated=False,truncated=False,timeout=False,current_context=context(),next_context=context())
    sample=agent.replay.sample_uniform(1)
    calls=[]
    native_project = agent._project_tensor
    def project_host_history(states, observations, actions, **kwargs):
        assert isinstance(states, np.ndarray) and isinstance(observations, np.ndarray)
        assert observations.shape[-1] == agent.runtime_config['observation_schema']['frame_dim']
        return native_project(states, observations, actions, **kwargs)
    monkeypatch.setattr(agent, '_project_tensor', project_host_history)
    def screen(cfg,states,observations,actions,reject_only=False,receivers=None,decision_only=False):
        assert reject_only
        calls.append((states.copy(),actions.copy()))
        return actions,dict(safe_mask=np.full(len(actions),safe))
    monkeypatch.setattr("usv_env.safety_projection.project_from_state",screen)
    with torch.no_grad():
        value,metrics=agent._target_values(sample,agent.tensor(query_state)[None],agent.tensor(obs)[None],
                                          torch.zeros((1,5,2)),torch.ones((1,5)))
    assert bool(agent.last_i3["accepted"][0]) is accepted
    if accepted:
        distance=metrics["i3_similarity_distance"]
        assert distance>0.
        expected=.5*.8/1.2*np.exp(-distance**2)/(1.+np.exp(-1./agent.history.settings["return_scale"]))
        torch.testing.assert_close(value,torch.full((1,5),expected),rtol=1e-5,atol=1e-6)
        assert metrics["i3_realized_nstep_return"]==1.
        assert metrics["i3_outcome_confidence"]==pytest.approx(1./(1.+np.exp(-1./agent.history.settings["return_scale"])))
        np.testing.assert_array_equal(agent.last_i3["actions"][0],action)
    else: torch.testing.assert_close(value,torch.zeros((1,5)))
    assert len(calls)==int(variant!="no_safety_screening")
    if calls:
        np.testing.assert_array_equal(calls[0][0][0],query_state)
        assert not np.array_equal(calls[0][0][0],state)
    assert metrics["i3_safety_rejection_fraction"]==int(not safe and variant!="no_safety_screening")


def test_terminal_target_cannot_receive_history_enhancement(monkeypatch):
    agent,obs,state=small_agent(monkeypatch)
    add(agent.replay,obs,state,np.ones((5,2)),context(),context(),terminated=True,timeout=True,terminal_reason="timeout")
    sample=agent.replay.sample_uniform(1)
    def forbidden(*args,**kwargs): raise AssertionError("terminal target must not screen history")
    monkeypatch.setattr("usv_env.safety_projection.project_from_state",forbidden)
    with torch.no_grad():
        value,metrics=agent._target_values(sample,agent.tensor(state)[None],agent.tensor(obs)[None],
                                          torch.zeros((1,5,2)),torch.ones((1,5)))
    assert metrics["i3_target_enhancement"]==0 and not agent.last_i3["accepted"].any()


@pytest.mark.parametrize("variant",["random_history","no_state_matching"])
def test_state_matching_ablations_retrieve_where_distance_gate_rejects(monkeypatch,variant):
    agent,obs,state=small_agent(monkeypatch)
    add(agent.replay,obs,state,np.ones((5,2)),context(),context())
    query,_=agent.replay._prepare_context(context())
    distant=state.copy(); distant[:45].reshape(5,9)[:,2:4]+=10.
    agent.history.settings["max_distance"]=.001
    normal=agent.history.retrieve(agent.replay,distant[None],query[None],[[1]],np.ones((1,5)))
    assert not normal["valid"].any()
    agent.history.settings["variant"]=variant
    ablation=agent.history.retrieve(agent.replay,distant[None],query[None],[[1]],np.ones((1,5)))
    assert ablation["valid"].any()


def test_i1_target_projection_and_actor_straight_through_are_used(monkeypatch):
    agent,obs,state=small_agent(monkeypatch,"I1_only")
    calls=[]
    native_project = agent._project_tensor
    def host_projection(states, observations, actions, **kwargs):
        assert isinstance(states, np.ndarray) and isinstance(observations, np.ndarray)
        return native_project(states, observations, actions, **kwargs)
    monkeypatch.setattr(agent, '_project_tensor', host_projection)
    def projector(cfg,states,observations,actions,reject_only=False,receivers=None,decision_only=False):
        calls.append(actions.copy())
        output=actions.copy()
        output[:,list(range(5) if receivers is None else receivers)]*=.5
        return output,dict(safe_mask=np.ones(len(actions),bool),intervention=np.ones(len(actions)),
                               projection_norm=np.ones(len(actions)),infeasible=np.zeros(len(actions)))
    monkeypatch.setattr("usv_env.safety_projection.project_from_state",projector)
    for _ in range(3): add(agent.replay,obs,state,np.zeros((5,2)),context(),context())
    agent.update(); result=agent.update()
    assert result["i1_target_intervention"]==1. and np.isfinite(result["usv_actor_loss"])
    assert len(calls)==3  # Two target batches plus the reused Actor reference.
    action=torch.ones((1,5,2),requires_grad=True)
    safe,_=native_project(agent.tensor(state)[None],agent.tensor(obs)[None],action,straight_through=True)
    torch.testing.assert_close(safe,action*.5)
    torch.testing.assert_close(torch.autograd.grad(safe.sum(),action)[0],torch.ones_like(action))


def test_base_never_calls_projection_or_history_and_replay_sampling_rng_is_shared(monkeypatch):
    base,obs,state=small_agent(monkeypatch,"Base")
    enhanced,_,_=small_agent(monkeypatch,"I3_only")
    for agent in (base,enhanced):
        for i in range(6): add(agent.replay,obs,state,np.full((5,2),i/10.),context(),context())
    before=[base.replay.sample_uniform(2)["indices"] for _ in range(2)]
    enhanced.history.retrieve(enhanced.replay,state[None],enhanced.replay.current_context[:1],[[1]],np.ones((1,5)))
    after=[enhanced.replay.sample_uniform(2)["indices"] for _ in range(2)]
    for left,right in zip(before,after): np.testing.assert_array_equal(left,right)
    def forbidden(*args,**kwargs): raise AssertionError("Base must not invoke a mechanism")
    monkeypatch.setattr("usv_env.safety_projection.project_from_state",forbidden)
    assert base.history is None
    assert np.isfinite(base.update()["usv_critic_loss"])


def test_real_predictor_rejects_previously_safe_joint_action_in_current_unsafe_state():
    from usv_env.environment import PursuitEvasionEnv
    from usv_env.history import HistoryEnv
    from usv_env.safety_projection import project_from_state
    cfg=assemble(experiment="I3_only",device="cpu")
    cfg["training"].update(hidden_sizes=[16,16],n_step_return=1)
    cfg["replay_buffer"]["capacity"]=4
    cfg["i3"].update(candidates=1,search_pool=4,minimum_quality=0.,max_distance=100.)
    refresh_hashes(cfg)
    env=HistoryEnv(PursuitEvasionEnv(cfg)); obs,state=env.reset(); current=env.learning_context()
    agent=Agent(env,cfg); action=np.full((5,2),.5,np.float32)
    assert project_from_state(cfg,state,obs,action,reject_only=True)[1]["safe_mask"].all()
    # At this different query state the hull already intersects the boundary.
    env.blue_states[0].pos[0]=.055
    raw,query_state=env.env._observations(); query_obs=env._push(raw)
    agent.replay.add(obs,state,action,np.zeros(5),query_obs,query_state,terminated=False,truncated=False,
                     timeout=False,current_context=current,next_context=env.learning_context())
    sample=agent.replay.sample_uniform(1)
    with torch.no_grad():
        value,metrics=agent._target_values(sample,agent.tensor(query_state)[None],agent.tensor(query_obs)[None],
            torch.zeros((1,5,2)),torch.ones((1,5)))
    assert metrics["i3_retrieved_candidates"]==1. and metrics["i3_safety_rejection_fraction"]==1.
    assert metrics["i3_valid_candidates"]==0. and not agent.last_i3["accepted"].any()
    np.testing.assert_array_equal(value.numpy(),agent.last_i3["q_policy"])


def test_funnel_rejections_partition_query_candidate_pairs(monkeypatch):
    agent,obs,state=small_agent(monkeypatch)
    replay=agent.replay
    # Each candidate is rejected at one declared stage, plus two survivors.
    for index in range(9):
        ctx=context()
        if index==1: ctx=context(phase=2)
        if index==3: ctx=context(valid=0.)
        if index==5: ctx=context(quality=0.)
        mask=np.ones(5); mask[0]=int(index!=2)
        flags=dict(terminated=index==4,truncated=False,timeout=False,
                   terminal_reason='collision' if index==4 else '')
        replay.add(obs,state,np.zeros((5,2)),np.ones(5),obs,state,
            current_context=ctx,next_context=context(),agent_active_mask=mask,**flags)
    replay.current_context[0,-1]=0
    replay.states[6,:45].reshape(5,9)[:,2:4]+=100.
    query,_=replay._prepare_context(context())
    found=agent.history.retrieve(replay,state[None],query[None],[[1]],np.ones((1,5)))
    counts=found['funnel']
    for name in ('metadata','phase','active','information','fatal_window','quality','distance'):
        assert counts[f'i3_{name}_rejected']==1
    assert counts['i3_queries']==1 and counts['i3_pool_candidate_pairs']==9
    assert counts['i3_eligible_candidates']==counts['i3_retrieval_selected']==2
    assert sum(counts[f'i3_{name}_rejected'] for name in (
        'metadata','phase','active','information','fatal_window','quality','distance'))+counts['i3_eligible_candidates']==9
    assert found['distributions']['i3_history_return_p50']==1.
    # Query metadata failure accounts for every pair, without double counting.
    query[-1]=0
    invalid=agent.history.retrieve(replay,state[None],query[None],[[1]],np.ones((1,5)))
    assert invalid['funnel']['i3_metadata_rejected']==9 and not invalid['valid'].any()


def test_funnel_reports_topk_cap_and_empty_history(monkeypatch):
    agent,obs,state=small_agent(monkeypatch)
    query,_=agent.replay._prepare_context(context())
    empty=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    assert empty['funnel']['i3_queries_without_history']==1
    for _ in range(4): add(agent.replay,obs,state,np.zeros((5,2)),context(),context())
    found=agent.history.retrieve(agent.replay,state[None],query[None],[[1]],np.ones((1,5)))
    assert found['funnel']['i3_eligible_candidates']==4
    assert found['funnel']['i3_topk_discarded']==2 and found['valid'].sum()==2
