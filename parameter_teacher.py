"""Training-only, budgeted low-rank coefficient search with paired validation.

Geometry is measured in metres. Policies, dynamics, references and rewards are
never changed by the teacher. Fixed coefficients are a short-horizon surrogate,
not a guarantee about the deployed, observation-dependent hypernetwork.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from usv_env.config import named_seed
from usv_env.reward import boundary_clearance, capsule_clearance
from usv_env.geometry import capsule_circle_clearance


@dataclass
class ParameterRollout:
    margins: np.ndarray
    outcome: str
    steps: int
    trajectory: list | None = None

    @property
    def failure(self):
        return self.outcome not in ("", "success")

    def record(self):
        result=dict(margins=self.margins.tolist(),outcome=self.outcome,steps=self.steps)
        if self.trajectory is not None: result['trajectory']=self.trajectory
        return result


def physical_margins(env):
    states=env.friendly_states(); red=env.red_state
    clearance=1e3
    for i,ship in enumerate(states):
        if not env.agent_active_mask()[i]: continue
        cap=env._capsule(ship)
        clearance=min(clearance,boundary_clearance(cap,env.world_size))
        if env.island_active:
            clearance=min(clearance,capsule_circle_clearance(cap,env.island_center,env.island_radius))
        if ship.unit_type=="surface" or ship.surfaced:
            clearance=min(clearance,capsule_clearance(cap,env._capsule(red)))
        for j in range(i+1,len(states)):
            if env.agent_active_mask()[j] and env._friendly_pair_can_collide(ship,states[j]):
                clearance=min(clearance,capsule_clearance(cap,env._capsule(states[j])))
    distance=float(np.linalg.norm(red.pos-env.island_center))
    landing=distance-env.island_radius-env.landing_buffer
    velocity=red.velocity.astype(float); speed=float(np.linalg.norm(velocity))
    direction=velocity/max(speed,1e-8)
    along=max(0.,float((env.island_center-red.pos)@direction)) if speed>1e-8 else 0.
    projected=float(np.linalg.norm(red.pos+along*direction-env.island_center))-env.island_radius-env.landing_buffer
    # The full simulator additionally catches swept collisions between endpoints.
    return np.array([landing if env.island_active else 1e3,clearance,
                     boundary_clearance(env._capsule(red),env.world_size),
                     projected if env.island_active else 1e3,distance],float)


def correct_coefficients(z,m,j,tau,cfg,unit,preferred=None):
    """Project a preferred joint move with soft constraints; verify nonlinearly.

    With preferred=None this is the original minimum-norm feasibility repair.
    A nonzero preferred move supplies the previously missing improvement goal.
    """
    z,m,j,tau=[np.asarray(x,float) for x in (z,m,j,tau)]
    if not all(np.isfinite(x).all() for x in (z,m,j,tau)) or j.shape!=(len(m),len(z)):
        raise ValueError("invalid local correction")
    m,j,tau=m/unit,j/unit,tau/unit
    rho=cfg["slack_weight"]; radius=cfg["trust_radius"]
    lower=np.maximum(-radius,-1-z); upper=np.minimum(radius,1-z)
    preferred=np.zeros_like(z) if preferred is None else np.asarray(preferred,float)
    if preferred.shape!=z.shape or not np.isfinite(preferred).all():
        raise ValueError("invalid preferred coefficient move")
    lipschitz=1+2*rho*np.linalg.norm(j,2)**2
    d=np.zeros_like(z)
    for iteration in range(cfg["solver_steps"]):
        gradient=d-preferred-2*rho*j.T@np.maximum(tau-m-j@d,0.)
        new=np.clip(d-gradient/lipschitz,lower,upper)
        residual=float(np.linalg.norm(new-d)*lipschitz); d=new
        if residual<=cfg["solver_tolerance"]: break
    return z+d,dict(solver_residual=residual,solver_iterations=iteration+1,
                    solver_converged=residual<=cfg["solver_tolerance"])


class ParameterTeacher:
    margin_names=["auv1_window","auv2_window","auv3_window","auv1_end","auv2_end","auv3_end",
                  "red_landing_window","team_clearance_window","red_boundary_window",
                  "red_projected_island_clearance_end","red_island_distance_end"]

    def __init__(self,cfg):
        if cfg["revision"].get("teacher_method")!="paired_parameter_objective_v2":
            raise ValueError("v2 teacher requires a new run/config; old policies remain readable for inference")
        self.cfg=deepcopy(cfg); self.hc=cfg["revision"]["i3"]
        self.pc=cfg["revision"]["parameter_teacher"]
        self.enabled=cfg["revision"]["teacher_enabled"]
        self.next_due_online_step=self.hc["warmup_steps"]
        self.calls=self.accepted=self.sim_steps=self.reserved=0
        self.rng=np.random.default_rng(named_seed(cfg["training"]["seed"],"parameter_teacher_future"))
        self.unit=cfg["scene_geometry"]["scale"]
        self.capture_trace=False

    def due(self,agent):
        return (self.enabled and agent.global_env_steps>=self.next_due_online_step
                and (agent.update_steps+1)%self.cfg["training"]["policy_delay"]==0
                and len(agent.replay)>=self.cfg["training"]["batch_size"])

    def rollout(self,snapshot,agent,z,horizon,future_seed=None):
        branch=snapshot.fork(); branch.episode_limit=None
        if future_seed is not None: branch.communication.rng=np.random.default_rng(future_seed)
        z=torch.as_tensor(z.reshape(2,self.hc["rank"]),dtype=torch.float32,device=agent.device)
        margins=np.full(11,1e3); outcome=""; count=0; trajectory=[] if self.capture_trace else None
        try:
            with torch.no_grad():
                for _ in range(horizon):
                    obs=torch.as_tensor(branch.current_observation(),dtype=torch.float32,device=agent.device)
                    actions=torch.cat((agent.surface_actor.forward_with_coefficients(obs[:2],z),
                                       agent.submarine_actor(obs[2:])),0).cpu().numpy()
                    before=branch.step_count
                    try: _,_,_,done,info=branch.step(actions)
                    finally: self.sim_steps+=max(0,branch.step_count-before)
                    count+=1
                    errors=np.linalg.norm(np.array([s.pos for s in branch.submarine_states])-branch.submarine_target_slots,axis=1)
                    tracking=branch.encirclement_cfg["shrink_tracking"]["max_slot_error_max"]-errors
                    margins[:3]=np.minimum(margins[:3],tracking); margins[3:6]=tracking
                    physical=physical_margins(branch)
                    margins[6:9]=np.minimum(margins[6:9],physical[:3]); margins[9:]=physical[3:]
                    if trajectory is not None:
                        trajectory.append(dict(step=count,time_s=count*branch.dt,errors_m=errors.tolist(),
                            tracking_margins_m=tracking.tolist(),physical_margins_m=physical.tolist(),actions=actions.tolist(),
                            phase=info.phase,terminal_reason=info.terminal_reason))
                    if done:
                        outcome=info.terminal_reason; break
            if not np.isfinite(margins).all(): raise FloatingPointError("nonfinite parameter margins")
            return ParameterRollout(margins,outcome,count,trajectory)
        finally: branch.close()

    def thresholds(self,reference):
        tau=np.minimum(reference.margins,0.)
        tau[:9]-=self.pc["margin_tolerance_m"]*self.unit
        tau[9:]=reference.margins[9:]-self.pc["progress_tolerance_m"]*self.unit
        return tau

    def improvement(self,trial,current,reference):
        if trial.failure or current.failure or reference.failure: return False,0.,"terminal_branch"
        if (current.outcome=="success" or reference.outcome=="success") and trial.outcome!="success":
            return False,0.,"success_regression"
        tau=self.thresholds(reference)
        old=np.maximum(tau-current.margins,0.); new=np.maximum(tau-trial.margins,0.)
        eps=self.pc["numerical_tolerance_m"]*self.unit
        required=self.pc["improvement_tolerance_m"]*self.unit
        # Absolute target progress may not deteriorate outside the declared tolerance.
        if np.any(trial.margins[9:] < current.margins[9:]-self.pc["progress_tolerance_m"]*self.unit):
            return False,0.,"target_progress_regression"
        # Apply the current-relative guard in BOTH repair and improvement mode.
        # Previously the repair return bypassed this protection for other AUVs.
        if np.any(trial.margins[:9]<current.margins[:9]-self.pc["margin_tolerance_m"]*self.unit):
            return False,0.,"tracking_or_safety_regression"
        if old.max()>eps:
            gain=float(np.linalg.norm(old)-np.linalg.norm(new))
            # A deficit smaller than the absolute improvement target must still
            # be repairable: require a substantial relative reduction, above
            # the numerical floor, never an impossible larger-than-deficit gain.
            repair_required=max(eps,min(required,.5*float(np.linalg.norm(old))))
            return bool(np.all(new<=old+eps) and gain>repair_required),gain,"repair_margin_deficit"
        if new.max()>eps: return False,0.,"new_margin_deficit"
        gain=float(min(trial.margins[3:6])-min(current.margins[3:6])
                   +.25*sum(trial.margins[9:]-current.margins[9:]))
        return gain>required,gain,"improve_worst_follower_and_target"

    def search_candidates(self,z,current,reference,jac,axis_candidates):
        """Joint goal/repair directions, ranked by the local model only.

        Predicted gains never create a label: every candidate still has to pass
        the short rollout and every longer paired communication-future rollout.
        """
        m=current.margins; tau=self.thresholds(reference)
        current_floor=m.copy()
        current_floor[:9]-=self.pc["margin_tolerance_m"]*self.unit
        current_floor[9:]-=self.pc["progress_tolerance_m"]*self.unit
        guard=np.maximum(tau,current_floor)
        deficit=np.maximum(tau-m,0.)
        weakest=int(np.argmin(m[3:6]))+3
        follower=jac[weakest]
        objective=follower+.25*jac[9:].sum(axis=0)
        directions=[("joint_task",objective),("worst_follower",follower)]
        if np.max(deficit)>self.pc["numerical_tolerance_m"]*self.unit:
            directions.insert(0,("deficit_repair",jac.T@deficit))
        proposals=[]; diagnostics=[]
        for name,g in directions:
            norm=float(np.max(np.abs(g)))
            if norm<=1e-12: continue
            preferred=self.pc["trust_radius"]*g/norm
            p,solver=correct_coefficients(z,m,jac,guard,self.pc,self.unit,preferred)
            diagnostics.append(dict(source=name,**solver))
            if solver["solver_converged"]:
                for fraction in (1.,.5):
                    proposals.append((name+f"_{fraction:g}",z+fraction*(p-z)))
        repair,solver=correct_coefficients(z,m,jac,guard,self.pc,self.unit)
        diagnostics.append(dict(source="minimum_norm_repair",**solver))
        if solver["solver_converged"]: proposals.append(("minimum_norm_repair",repair))
        proposals.extend(("axis_probe",p) for _,p in axis_candidates)
        ranked=[]
        for source,p in proposals:
            if np.linalg.norm(p-z)<1e-7 or any(np.allclose(p,x[2],atol=1e-8,rtol=0) for x in ranked): continue
            predicted=ParameterRollout(m+jac@(p-z),current.outcome,current.steps)
            ok,gain,reason=self.improvement(predicted,current,reference)
            ranked.append((int(ok),float(gain),p,source,reason))
        ranked.sort(key=lambda x:(x[0],x[1]),reverse=True)
        chosen=ranked[:self.pc["max_backtracks"]]
        return [x[2] for x in chosen],dict(
            solvers=diagnostics,objective_gradient_norm=float(np.linalg.norm(objective)),
            coefficient_saturation_fraction=float(np.mean(np.abs(z)>.99)),
            proposed_candidates=[dict(source=x[3],predicted_pass=bool(x[0]),predicted_gain_m=x[1],
                                      predicted_reason=x[4],coefficients=x[2].tolist()) for x in chosen])

    def propose(self,env,agent):
        record=dict(status="not_due",simulator_steps=0,margin_names=self.margin_names,
                    teacher_method="paired_parameter_objective_v2")
        if not self.enabled: record["status"]="disabled"; return None,record
        if agent.global_env_steps<self.next_due_online_step: return None,record
        while self.next_due_online_step<=agent.global_env_steps: self.next_due_online_step+=self.hc["interval_steps"]
        if env._episode_done or env.max_steps-env.step_count<self.pc["verify_steps"]:
            record["status"]="insufficient_horizon"; return None,record
        if not np.all(env.agent_active_mask()): record["status"]="requires_full_team"; return None,record
        if not agent.surface_actor.hyper_enabled: raise ValueError("parameter teacher requires hypernetwork")
        search,verify=self.pc["horizon_steps"],self.pc["verify_steps"]
        communication=env.config["communication"]
        extra=communication["enabled"] and (max(communication["target_loss_range"])>0 or communication["delay_steps"][0]!=communication["delay_steps"][1])
        dimension=2*self.hc["rank"]; tries=self.pc["max_backtracks"]
        worst=(2+2*dimension+tries)*search+(2+tries)*verify*(2 if extra else 1)
        budget=int(self.hc["budget_ratio"]*agent.global_env_steps)
        if self.sim_steps+worst>budget:
            record.update(status="budget_skip",remaining_budget=budget-self.sim_steps,worst_case_steps=worst); return None,record
        self.calls+=1; self.reserved=worst; start=self.sim_steps
        record.update(status="rejected",worst_case_steps=worst)
        try:
            obs=env.current_observation()[:2].copy()
            with torch.no_grad():
                z=agent.surface_actor.coefficients(torch.as_tensor(obs,dtype=torch.float32,device=agent.device)).cpu().numpy().astype(float).reshape(-1)
            reference=self.rollout(env,agent,np.zeros_like(z),search)
            current=self.rollout(env,agent,z,search)
            record.update(reference=reference.record(),candidate=current.record(),coefficients=z.tolist())
            if reference.failure or current.failure: record["status"]="terminal_reference_or_candidate"; return None,record
            jac=np.zeros((11,dimension)); candidates=[]; rows=[]
            for k in range(dimension):
                plus,minus=z.copy(),z.copy()
                delta=min(self.pc["finite_difference"],self.pc["trust_radius"])
                plus[k]=min(1.,plus[k]+delta); minus[k]=max(-1.,minus[k]-delta)
                rp=self.rollout(env,agent,plus,search); rm=self.rollout(env,agent,minus,search)
                rows.extend([dict(coefficients=plus.tolist(),**rp.record()),dict(coefficients=minus.tolist(),**rm.record())])
                # A bad probe is evidence against that direction, not against
                # every other coordinate. Never differentiate terminal rewards
                # or extrapolate from a failed branch.
                if not rp.failure and not rm.failure:
                    jac[:,k]=(rp.margins-rm.margins)/(plus[k]-minus[k])
                elif not rp.failure and plus[k]-z[k]>1e-8:
                    jac[:,k]=(rp.margins-current.margins)/(plus[k]-z[k])
                elif not rm.failure and z[k]-minus[k]>1e-8:
                    jac[:,k]=(current.margins-rm.margins)/(z[k]-minus[k])
                for coefficients,result in ((plus,rp),(minus,rm)):
                    ok,gain,_=self.improvement(result,current,reference)
                    if ok: candidates.append((gain,coefficients))
            record.update(sensitivity_branches=rows,jacobian_norm=float(np.linalg.norm(jac)),
                          failed_sensitivity_branches=sum(x["outcome"] not in ("","success") for x in rows))
            trials,search_diagnostics=self.search_candidates(z,current,reference,jac,candidates)
            record.update(search_diagnostics)
            if not trials: record["status"]="no_reliable_improvement"; return None,record
            seeds=[None]+([int(self.rng.integers(0,2**32))] if extra else [])
            # Longer paired revalidation, including an independent communication
            # future when stochastic. Current already-delivered packets stay intact.
            bases=[(seed,self.rollout(env,agent,np.zeros_like(z),verify,seed),self.rollout(env,agent,z,verify,seed)) for seed in seeds]
            record["verification_attempts"]=[]
            for trial in trials:
                short=self.rollout(env,agent,trial,search)
                ok,gain,reason=self.improvement(short,current,reference)
                attempt=dict(coefficients=trial.tolist(),short=short.record(),reason=reason,
                             passed_short=bool(ok),short_gain_m=gain,pairs=[])
                record["verification_attempts"].append(attempt)
                if not ok: continue
                gains=[gain]
                for seed,ref,base in bases:
                    result=self.rollout(env,agent,trial,verify,seed)
                    ok,value,reason=self.improvement(result,base,ref)
                    attempt["pairs"].append(dict(seed=seed,reference=ref.record(),baseline=base.record(),trial=result.record(),passed=bool(ok),gain_m=value,reason=reason))
                    if not ok: break
                    gains.append(value)
                if not ok: continue
                self.accepted+=1
                record.update(status="accepted",improvement=min(gains),coefficient_change_norm=float(np.linalg.norm(trial-z)))
                return dict(observations=obs,coefficients=trial.reshape(2,self.hc["rank"]).astype(np.float32),quality=1.,improvement=min(gains)),record
            record["status"]="nonlinear_verification_rejected"; return None,record
        except (FloatingPointError,ValueError,RuntimeError) as exc:
            record.update(status="branch_error",error=type(exc).__name__+": "+str(exc)); return None,record
        finally:
            record["simulator_steps"]=self.sim_steps-start; self.reserved=0
            if record["simulator_steps"]>worst or self.sim_steps>budget: raise RuntimeError("parameter teacher exceeded budget")

    def state_dict(self):
        return dict(method="paired_parameter_objective_v2",next_due_online_step=self.next_due_online_step,
                    calls=self.calls,accepted=self.accepted,sim_steps=self.sim_steps,rng=deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self,state):
        if state.get("method")!="paired_parameter_objective_v2": raise ValueError("old teacher state is incompatible; start a new run")
        for key in ("next_due_online_step","calls","accepted","sim_steps"): setattr(self,key,int(state[key]))
        self.rng.bit_generator.state=state["rng"]
