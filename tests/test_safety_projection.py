"""I1 uses legal receiver data and reports conservative numerical infeasibility."""
from copy import deepcopy
import math

import numpy as np

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv
from usv_env.geometry import capsule_segment
from usv_env.history import HistoryEnv
from usv_env.observation import build_actor_observation, build_critic_state
from usv_env.reward import capsule_clearance
from usv_env import safety_projection as safety
from usv_env.target_tracking import TargetEstimate
from usv_env.types import ShipState


def state(pos, heading=0., speed=0., sway=0., kind="surface"):
    return ShipState(np.asarray(pos,np.float32),speed,heading,0.,kind,kind=="surface",sway)


def config():
    cfg=assemble(device="cpu")
    cfg["communication"].update(delay_steps=[0,0],loss_rate=0.,target_loss_range=[0.,0.])
    return cfg


def receiver_inputs(env):
    raw=np.stack([build_actor_observation(env,i) for i in range(env.num_friendly)])
    obs=HistoryEnv(env,history_length=1)._push(raw)
    return build_critic_state(env),obs


def test_stationary_crosswise_peer_cannot_be_falsely_certified_from_velocity_heading():
    cfg=config(); predictor=safety._Predictor(cfg)
    own=state([10.,10.]); peer=state([10.,10.7],heading=math.pi/2)
    length,width=safety._dimensions(cfg,True)
    actual=capsule_clearance(capsule_segment(own.pos,own.heading,length,width),
                             capsule_segment(peer.pos,peer.heading,length,width))
    assert actual < 0.  # True capsules already overlap.
    # The received zero velocity does not identify peer heading. The filter
    # must cover all possible headings rather than assume heading zero.
    obstacle=dict(position=peer.pos,velocity=peer.velocity,heading=0.,surface=True,uncertainty=.1)
    action,diag=predictor.project_one(own,[obstacle],np.zeros(2),np.array([2.2,1.5]))
    assert not diag["safe"] and diag["infeasible"] and diag["fallback"]
    assert diag["minimum_clearance"] <= actual and diag["slack"] > 0.
    assert np.isfinite(action).all()


def test_batch_reconstructs_same_known_peer_positions_and_velocities_as_deployment(monkeypatch):
    env=PursuitEvasionEnv(config())
    env.blue_states[0]=state([10.,10.],heading=.7,speed=.2,sway=.1)
    env.blue_states[1]=state([12.,12.],heading=-.5,speed=.15,sway=-.07)
    env.communication.step(env.friendly_states(),0)
    env.target_estimates[0]=TargetEstimate(True,np.array([14.,11.,.18,.04]),np.eye(4)*.1,
        0.,0.,.9,2,"2:0")
    calls=[]
    def capture(self,own,obstacles,nominal,limits,reject_only=False):
        calls.append((deepcopy(own),deepcopy(obstacles)))
        return np.asarray(nominal),dict(safe=True,intervention=False,projection_norm=0.,
            infeasible=False,fallback=False,slack=0.,minimum_clearance=1.,candidates=1)
    monkeypatch.setattr(safety._Predictor,"project_one",capture)
    actions=np.zeros((5,2),np.float32)
    deployed,_=safety.SafetyProjector(env).project(actions)
    execution=calls.copy(); calls.clear()
    centralized,obs=receiver_inputs(env)
    trained,_=safety.project_from_state(env.config,centralized,obs,actions)
    np.testing.assert_array_equal(deployed,trained)
    for (own,left),(decoded,right) in zip(execution,calls):
        np.testing.assert_allclose(own.pos,decoded.pos,atol=3e-6)
        np.testing.assert_allclose(own.velocity,decoded.velocity,atol=3e-6)
        assert abs(own.heading-decoded.heading)<1e-6
        assert len(left)==len(right)
        for first,second in zip(left,right):
            np.testing.assert_allclose(first["position"],second["position"],atol=3e-6)
            np.testing.assert_allclose(first["velocity"],second["velocity"],atol=3e-6)
            assert first["surface"] == second["surface"]


def test_unknown_red_and_unreceived_peers_never_cross_i1_information_boundary(monkeypatch):
    env=PursuitEvasionEnv(config())
    assert not env.communication.cache and not any(e.valid for e in env.target_estimates)
    calls=[]
    def capture(self,own,obstacles,nominal,limits,reject_only=False):
        calls.append(deepcopy(obstacles))
        return np.asarray(nominal),dict(safe=True,intervention=False,projection_norm=0.,
            infeasible=False,fallback=False,slack=0.,minimum_clearance=1.,candidates=1)
    monkeypatch.setattr(safety._Predictor,"project_one",capture)
    actions=np.zeros((5,2),np.float32)
    safety.SafetyProjector(env).project(actions)
    assert calls[0] == []
    env.red_state.pos=env.blue_states[0].pos.copy()
    env.shadow_red_state.pos=env.blue_states[0].pos.copy()
    env.blue_states[1].pos=env.blue_states[0].pos.copy()
    calls.clear(); safety.SafetyProjector(env).project(actions)
    assert calls[0] == []
    centralized,obs=receiver_inputs(env)
    calls.clear(); safety.project_from_state(env.config,centralized,obs,actions)
    assert calls[0] == []
    changed=centralized.copy(); changed[9:52]=1234.  # Other four agents and red truth.
    calls.clear(); safety.project_from_state(env.config,changed,obs,actions)
    assert calls[0] == []


def test_infeasible_reef_contact_returns_bounded_fallback_with_positive_slack():
    cfg=config(); predictor=safety._Predictor(cfg)
    own=state(cfg["reefs"][0]["center"])
    limits=np.array([cfg["ship"]["max_surge_force"],cfg["ship"]["max_yaw_moment"]])
    action,diag=predictor.project_one(own,[],np.array([1.,.3]),limits)
    assert diag["infeasible"] and diag["fallback"] and not diag["safe"]
    assert diag["slack"] > 0. and diag["candidates"] > 1
    assert np.isfinite(action).all() and (np.abs(action)<=limits).all()
    _,reject=predictor.project_one(own,[],np.zeros(2),limits,reject_only=True)
    assert not reject["safe"] and reject["infeasible"]


def test_depth_separation_is_preserved_in_predictive_collision_geometry():
    cfg=config(); predictor=safety._Predictor(cfg)
    own=state([10.,10.],kind="submarine")
    obstacle=dict(position=own.pos.copy(),velocity=np.zeros(2),heading=0.,surface=True,uncertainty=.1)
    clear=predictor.assess(own,[],np.zeros(2))
    assert predictor.assess(own,[obstacle],np.zeros(2)) == clear
    unseparated=deepcopy(cfg); unseparated["depth"]["auv_depth_m"]=.6
    assert safety._Predictor(unseparated).assess(own,[obstacle],np.zeros(2)) < 0.


def test_i1_predictor_retains_real_usv_and_auv_force_dynamics():
    env=PursuitEvasionEnv(config()); predictor=safety._Predictor(env.config)
    for submarine,dynamics in ((False,env.blue_dynamics),(True,env.submarine_dynamics)):
        own=state([10.,10.],heading=.3,speed=.15,sway=.02,
                  kind="submarine" if submarine else "surface")
        action=np.array([.6,.2])
        reference=dynamics.step(deepcopy(own),action)
        projected=predictor.models[int(submarine)].step(deepcopy(own),action)
        np.testing.assert_array_equal(reference.pos,projected.pos)
        assert reference.speed == projected.speed and reference.heading == projected.heading
        assert reference.yaw_rate == projected.yaw_rate
def test_actuator_projection_brakes_before_boundary_and_skips_inactive():
    from usv_env.config import assemble
    from usv_env.safety_projection import _Predictor, SafetyProjector
    from usv_env.environment import PursuitEvasionEnv
    from usv_env.types import ShipState
    import numpy as np
    import math
    cfg=assemble(experiment='Full',device='cpu')
    # 1.2 m was only safe over the former 1.2 s window: stopping from .4 m/s
    # needs about .78 m plus the hull/margin. Start while braking is feasible.
    own=ShipState(np.array([1.8,10.],np.float32),.4,math.pi,0.,'surface')
    action,row=_Predictor(cfg).project_one(own,[],np.array([2.2,0.]),np.array([2.2,1.5]))
    assert row['safe'] and row['intervention'] and not row['infeasible']
    assert action[0]<0 and row['minimum_clearance']>=cfg['i1']['margin_m']
    np.testing.assert_array_equal(own.pos,np.array([1.8,10.],np.float32))
    env=PursuitEvasionEnv(cfg)
    env.set_active_agents([0,1,1,1,1]); env.reset()
    env.blue_states[0].pos=np.array([-100.,-100.],np.float32)
    result,diag=SafetyProjector(env).project(np.zeros((5,2),np.float32))
    assert not diag['infeasible'][0] and diag['candidates'][0]==0
    np.testing.assert_array_equal(result[0],[0,0])


def test_assessment_cache_preserves_clearance_projection_and_diagnostics_exactly():
    cfg=config(); cached=safety._Predictor(cfg); reference=safety._Predictor(cfg)
    reference._assessment=reference._detail_uncached
    cases=[(state([1.2,10.],heading=math.pi,speed=.4),[],np.array([2.2,0.])),
           (state([27.5,28.],speed=.1,kind="submarine"),[],np.array([.4,.1])),
           (state([10.,10.]),[dict(position=np.array([10.,10.7]),velocity=np.zeros(2),
               surface=True,uncertainty=.1)],np.zeros(2))]
    for own,obstacles,nominal in cases:
        assert cached.assess(own,obstacles,nominal) == reference.assess(own,obstacles,nominal)
        limits=np.array([2.2,1.5]) if own.unit_type=="surface" else np.array([3.035,1.5])
        actual,diagnostics=cached.project_one(own,obstacles,nominal,limits)
        expected,expected_diagnostics=reference.project_one(own,obstacles,nominal,limits)
        np.testing.assert_array_equal(actual,expected)
        assert diagnostics == expected_diagnostics
    assert cached.assess_cache_info()["hits"] > 0
    assert cached.assess_cache_info()["capacity"] == 8192


def test_assessment_cache_has_exact_input_keys_and_detects_mutations():
    predictor=safety._Predictor(config())
    own=state([10.,10.]); action=np.zeros(2)
    obstacles=[dict(position=np.array([30.,30.]),velocity=np.zeros(2),surface=True,uncertainty=.1),
               dict(position=np.array([35.,35.]),velocity=np.zeros(2),surface=False,uncertainty=.1)]
    predictor.assess(own,obstacles,action)
    predictor.assess(deepcopy(own),deepcopy(obstacles),action.copy())
    assert predictor.assess_cache_info()["hits"] == 1
    variants=[]
    for field in ("speed","sway_speed","heading","yaw_rate"):
        changed=deepcopy(own); setattr(changed,field,np.nextafter(0.,1.))
        variants.append((changed,obstacles,action))
    moved=deepcopy(own); moved.pos[0]=np.nextafter(moved.pos[0],np.float32(11.))
    variants.append((moved,obstacles,action))
    turned=deepcopy(own); turned.unit_type="submarine"
    variants.append((turned,obstacles,action))
    changed=action.copy(); changed[0]=np.nextafter(0.,1.)
    variants.append((own,obstacles,changed))
    for key in ("position","velocity","surface","uncertainty"):
        changed=deepcopy(obstacles)
        if key in ("position","velocity"):
            changed[0][key][0]=np.nextafter(changed[0][key][0],np.inf)
        elif key=="surface": changed[0][key]=False
        else: changed[0][key]=np.nextafter(changed[0][key],np.inf)
        variants.append((own,changed,action))
    # Preserve dtype as well as values: obstacle NumPy arithmetic can round
    # differently in float32 and float64 before capsule construction.
    for key in ("position","velocity"):
        changed=deepcopy(obstacles); changed[0][key]=changed[0][key].astype(np.float32)
        variants.append((own,changed,action))
    changed=deepcopy(own); changed.pos=changed.pos.astype(np.float64)
    variants.append((changed,obstacles,action))
    variants.append((own,obstacles,action.astype(np.float32)))
    variants.append((own,list(reversed(obstacles)),action))
    for inputs in variants:
        predictor.assess(*inputs)
    assert predictor.assess_cache_info()["misses"] == len(variants)+1


def test_assessment_cache_is_bounded_and_evicts_least_recently_used_entry():
    predictor=safety._Predictor(config(),assess_cache_capacity=2)
    own=state([10.,10.])
    for force in (0.,.1,0.,.2,.1):
        predictor.assess(own,[],np.array([force,0.]))
    assert predictor.assess_cache_info() == dict(hits=1,misses=4,evictions=2,size=2,capacity=2)

