"""Independent force-model equivalence and recovery-horizon regression tests."""
from copy import deepcopy
import math
import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.geometry import capsule_segment, capsule_circle_clearance, wrap_angle
from usv_env.reward import capsule_clearance, boundary_clearance
from usv_env.safety_kernel import model_arrays, step_state, brake_action
from usv_env.safety_projection import _Predictor, _model, _dimensions, _separated
from usv_env.types import ShipState


def reference_clearance(cfg, own, obstacles, action):
    """Object-model oracle, independent of the compiled dynamics/geometry."""
    model=_model(cfg,own.unit_type!='surface')
    surface=own.unit_type=='surface'; length,width=_dimensions(cfg,surface)
    def clearance(pos,heading,t):
        capsule=capsule_segment(pos,heading,length,width)
        value=boundary_clearance(capsule,cfg['env']['world_size'])
        for reef in cfg['reefs']:
            value=min(value,capsule_circle_clearance(capsule,np.asarray(reef['center']),
                reef['radius']+cfg['depth']['reef_shallow_buffer_m']))
        for obstacle in obstacles:
            if obstacle['surface']!=surface and _separated(cfg): continue
            other_length,_=_dimensions(cfg,obstacle['surface'])
            other=capsule_segment(obstacle['position']+t*obstacle['velocity'],0.,other_length,other_length)
            value=min(value,capsule_clearance(capsule,other)-obstacle.get('uncertainty',0.))
        return value
    minimum=clearance(own.pos,own.heading,0.)
    max_speed=max((np.linalg.norm(o['velocity']) for o in obstacles),default=0.)
    current=deepcopy(own); dt=cfg['env']['dt']
    for step in range(cfg['i1']['horizon_steps']):
        current=model.step(current,action); trace=current.integration_trace
        for k,((p0,h0),(p1,h1)) in enumerate(zip(trace[:-1],trace[1:])):
            t0=(step+k/(len(trace)-1))*dt; t1=(step+(k+1)/(len(trace)-1))*dt
            bound=np.linalg.norm(p1-p0)+.5*length*abs(wrap_angle(h1-h0))+max_speed*(t1-t0)
            minimum=min(minimum,min(clearance(p0,h0,t0),clearance(p1,h1,t1))-.5*bound)
    return minimum


@pytest.mark.parametrize('submarine',[False,True])
def test_compiled_integrator_matches_real_model_through_clamps_and_reverse(submarine):
    cfg=assemble(device='cpu'); model=_model(cfg,submarine); arrays=model_arrays(model)
    rng=np.random.default_rng(729)
    for _ in range(48):
        state=ShipState(rng.uniform(1,54,2).astype(np.float32),rng.uniform(model.min_speed,model.max_speed),
            rng.uniform(-math.pi,math.pi),rng.uniform(-.675,.675),'submarine' if submarine else 'surface',
            not submarine,rng.uniform(-.6,.6))
        q=_Predictor._state_array(state)
        for _ in range(12):
            action=rng.uniform(-1,1,2)*[model.max_surge_force,model.max_yaw_moment]
            q,trace=step_state(q,action,*arrays,cfg['env']['dt'],model.internal_substeps)
            state=model.step(state,action)
            np.testing.assert_allclose(q,_Predictor._state_array(state),rtol=0,atol=2e-6)
            expected=np.array([[*pos,heading] for pos,heading in state.integration_trace])
            np.testing.assert_allclose(trace,expected,rtol=0,atol=2e-6)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_compiled_clearance_matches_independent_swept_capsule_oracle(dtype):
    cfg=assemble(device='cpu'); cfg['i1'].update(horizon_steps=8,backup_steps=0)
    predictor=_Predictor(cfg); rng=np.random.default_rng(98)
    for sub in (False,True):
        for _ in range(20):
            # Near-boundary positions force full integration, not far-field shortcut.
            own=ShipState(np.array([1.1,rng.uniform(2,50)],np.float32),rng.uniform(-.1,.4),
                rng.uniform(-math.pi,math.pi),rng.uniform(-.5,.5),'submarine' if sub else 'surface',
                not sub,rng.uniform(-.2,.2))
            obstacles=[dict(position=(own.pos+[1.,.3]).astype(dtype),velocity=np.array([.1,-.03],dtype),
                            surface=not sub,uncertainty=.1)]
            action=rng.uniform(-1,1,2)*[2.2,1.5]
            np.testing.assert_allclose(predictor.assess(own,obstacles,action),
                reference_clearance(cfg,own,obstacles,action),rtol=0,atol=3e-6)


def test_safe_short_window_is_rejected_when_no_safe_braking_tail_remains():
    cfg=assemble(device='cpu'); cfg['i1'].update(horizon_steps=8,backup_steps=0)
    own=ShipState(np.array([1.8,10.],np.float32),.4,math.pi,0.,'surface')
    nominal=np.array([2.2,0.]); limits=np.array([2.2,1.5])
    _,short=_Predictor(cfg).project_one(own,[],nominal,limits,reject_only=True)
    assert short['safe']
    cfg['i1']['backup_steps']=60
    _,checked=_Predictor(cfg).project_one(own,[],nominal,limits,reject_only=True)
    assert not checked['safe'] and checked['minimum_clearance'] < cfg['i1']['margin_m']
    action,rescued=_Predictor(cfg).project_one(own,[],nominal,limits)
    assert rescued['safe'] and rescued['terminal_recoverable'] and action[0]<0.


def test_timeout_of_braking_check_is_not_reported_as_safe():
    cfg=assemble(device='cpu'); cfg['i1'].update(horizon_steps=1,backup_steps=1)
    own=ShipState(np.array([10.,10.],np.float32),.4,0.,0.,'surface')
    _,diag=_Predictor(cfg).project_one(own,[],np.array([2.2,0.]),np.array([2.2,1.5]),reject_only=True)
    assert diag['minimum_clearance']>cfg['i1']['margin_m']
    assert not diag['safe'] and not diag['terminal_recoverable'] and diag['infeasible']


@pytest.mark.parametrize('submarine',[False,True])
def test_configured_braking_budget_covers_sampled_velocity_limit_grid(submarine):
    cfg=assemble(device='cpu'); predictor=_Predictor(cfg)
    index=int(submarine); model=predictor.models[index]; arrays=predictor.model_data[index]
    for u in (model.min_speed,0.,model.max_speed):
        for v in (-model.max_sway_speed,0.,model.max_sway_speed):
            for r in (-model.max_yaw_rate,0.,model.max_yaw_rate):
                q=np.array([10.,10.,u,v,r,0.])
                for _ in range(cfg['i1']['backup_steps']):
                    command=brake_action(q,arrays[0],arrays[1],arrays[3],model.dt)
                    q,_=step_state(q,command,*arrays,model.dt,model.internal_substeps)
                    if np.hypot(q[2],q[3])<=cfg['i1']['stop_speed_mps'] and abs(q[4])<=cfg['i1']['stop_yaw_rate_rps']:
                        break
                else: pytest.fail(f'backup horizon too short for {submarine=}, {u=}, {v=}, {r=}')


def test_subset_projection_reuses_unchanged_peers_exactly():
    from usv_env.environment import PursuitEvasionEnv
    from usv_env.history import HistoryEnv
    from usv_env.safety_projection import project_from_state
    cfg=assemble(experiment='Full',device='cpu')
    env=HistoryEnv(PursuitEvasionEnv(cfg),1); obs,state=env.reset()
    actions=np.tile([2.2,.2],(5,1)).astype(np.float32)
    full,_=project_from_state(cfg,state,obs,actions)
    for i in range(5):
        candidate=full.copy();candidate[i]=actions[i]
        subset,_=project_from_state(cfg,state,obs,candidate,receivers=(i,))
        np.testing.assert_array_equal(subset,full)


def test_actor_projection_reuse_preserves_loss_gradients_and_optimizer_step():
    import torch
    from matd3.agent import Agent, frozen
    from usv_env.environment import PursuitEvasionEnv
    from usv_env.history import HistoryEnv
    torch.set_num_threads(1)
    cfg=assemble(experiment='I1_only',device='cpu')
    cfg['training']['hidden_sizes']=[16,16]; cfg['replay_buffer']['capacity']=8
    env=HistoryEnv(PursuitEvasionEnv(cfg),1); obs,state=env.reset()
    agent=Agent(env,cfg); original=deepcopy(agent)
    obs=agent.tensor(np.stack([obs,obs])); state=agent.tensor(np.stack([state,state]))
    mask=torch.ones((2,5)); mask[1,1]=0.
    with torch.no_grad():
        reference=agent.joint_actions(obs)
        safe_reference,_=agent._project_tensor(state,obs,reference)
    for role in range(2):
        actor=original.actors[role]; role_slice=original.role_slices[role]
        loss=obs.new_zeros(()); count=mask[:,role_slice].sum().clamp_min(1.)
        with frozen(q for pair in original.critics for q in pair):
            for head,i in enumerate(range(role_slice.start,role_slice.stop)):
                candidate=torch.stack([actor(obs[:,j]) if j==i else reference[:,j].detach()
                                       for j in range(5)],1)*mask[:,:,None]
                candidate,_=original._project_tensor(state,obs,candidate,straight_through=True)
                values=original.critics[role][0](state,candidate,obs)
                loss-=(values[:,head]*mask[:,i]).sum()/count
            optimizer=original.actor_optimizers[role]; optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['training']['grad_clip_norm'])
            optimizer.step()
        actual=agent._actor_step(role,obs,state,mask,reference,safe_reference)
        assert actual==pytest.approx(float(loss.detach()),abs=2e-7)
        for expected,updated in zip(actor.parameters(),agent.actors[role].parameters()):
            torch.testing.assert_close(updated.grad,expected.grad,rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(updated,expected,rtol=2e-5,atol=2e-7)
