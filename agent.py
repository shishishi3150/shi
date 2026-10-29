"""Common MH TD3, bounded I2 routing and parameter-label I3 in a single Adam step."""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .networks import make_actor, Critic
from .replay_buffer import ReplayBuffer
from usv_env.config import validate, named_seed
from usv_env.calibration import nominal_actions, mixture_action


@contextmanager
def frozen(modules):
    modules=tuple(modules)
    flags=[(p,p.requires_grad) for m in modules for p in m.parameters()]
    modes=[m.training for m in modules]
    try:
        for p,_ in flags: p.requires_grad_(False)
        for m in modules: m.eval()
        yield
    finally:
        for p,flag in flags: p.requires_grad_(flag)
        for m,mode in zip(modules,modes): m.train(mode)


def gradient_norm(grads):
    return torch.sqrt(sum((g.detach().square().sum() for g in grads)))


def ema(previous, value, decay=.99):
    return float(value) if previous is None else decay*previous+(1-decay)*float(value)




class CoefficientGuidance:
    """Single-use label for the exact actor version used by the paired teacher."""
    def __init__(self,rank): self.rank=rank; self.pending=None; self.last_event=None

    def add(self,obs,coefficients,step,version,quality,improvement,label_id=None):
        o,z=np.asarray(obs,np.float32),np.asarray(coefficients,np.float32)
        if o.ndim!=2 or o.shape[0]!=2 or z.shape!=(2,self.rank) or not np.isfinite(o).all() or not np.isfinite(z).all() or (np.abs(z)>1).any():
            raise ValueError("invalid coefficient label")
        if not np.isfinite(quality) or not 0<quality<=1 or not np.isfinite(improvement):
            raise ValueError("invalid coefficient label quality")
        self.pending=dict(obs=o.copy(),coefficients=z.copy(),step=int(step),version=int(version),quality=float(quality),label_id=label_id)

    def sample(self,step,version):
        item=self.pending; self.pending=None
        self.last_event=None
        if item is None: return None
        valid=item['step']==step and item['version']==version
        self.last_event=dict(label_id=item.get('label_id'),label_step=item['step'],label_actor_version=item['version'],
                             consumer_step=step,consumer_actor_version=version,status='sampled' if valid else 'stale_discarded')
        if not valid: return None
        return item["obs"][None],item["coefficients"][None],np.array([item["quality"]],np.float32)

    def state_dict(self): return dict(kind="coefficient_single_use_v1",pending=deepcopy(self.pending))
    def load_state_dict(self,state):
        if state.get("kind")!="coefficient_single_use_v1": raise ValueError("old action labels are incompatible")
        self.pending=deepcopy(state["pending"])


class Agent:
    def __init__(self,env,cfg):
        validate(cfg)
        if cfg.get("purpose")=="reward_scale_probe": raise ValueError("probe cannot construct an Agent")
        self.runtime_config=deepcopy(cfg)
        tr,rev=cfg["training"],cfg["revision"]
        device=tr["device"]
        if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=torch.device(device)
        self.obs_dim,self.state_dim=env.observation_dim,env.critic_state_dim
        self.limits=np.asarray(env.agent_action_limits,np.float32)
        self.cruise=nominal_actions(env)
        self.surface_actor=make_actor(cfg,self.limits[0],self.cruise[0],"surface").to(self.device)
        self.submarine_actor=make_actor(cfg,self.limits[2],self.cruise[2],"submarine").to(self.device)
        self.actors=[self.surface_actor,self.submarine_actor]
        self.actor_targets=[deepcopy(a).requires_grad_(False) for a in self.actors]
        self.critics=[]
        for role,heads in (("surface",2),("submarine",3)):
            if cfg["critic"]["head_mode"]=="role_mean": heads=1
            pair=[]
            for twin in range(2):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(named_seed(tr["seed"],f"critic:{role}:{twin}"))
                    pair.append(Critic(cfg,self.state_dim,self.limits,heads).to(self.device))
            self.critics.append(pair)
        self.critic_targets=[[deepcopy(q).requires_grad_(False) for q in pair] for pair in self.critics]
        self.actor_optimizers=[torch.optim.Adam(a.parameters(),lr=tr["actor_lr"]) for a in self.actors]
        self.critic_optimizers=[torch.optim.Adam([p for q in pair for p in q.parameters()],lr=tr["critic_lr"]) for pair in self.critics]
        self.replay=ReplayBuffer(cfg["replay_buffer"]["capacity"],5,self.obs_dim,self.state_dim,2,
            seed=named_seed(tr["seed"],"replay"),n_step_return=tr["n_step_return"],gamma=tr["gamma"])
        self.exploration_rng=np.random.default_rng(named_seed(tr["seed"],"exploration"))
        self.target_rng=torch.Generator(device=self.device)
        self.target_rng.manual_seed(named_seed(tr["seed"],"target_noise"))
        self.guidance=CoefficientGuidance(rev["i3"]["rank"])
        self.global_env_steps=self.update_steps=self.actor_updates=self.episodes_completed=0
        self.local_grad_ema=[None,None]; self.guide_reference_ema=None; self.auv_td_ema=None
        self.last_sample_indices=None; self.last_target_noise=None
        self.diagnostic_every=0; self.i2_diagnostics=[]; self.guidance_event=None; self._capture_i2=False

    def tensor(self,value): return torch.as_tensor(value,dtype=torch.float32,device=self.device)

    def sigma(self):
        tr=self.runtime_config["training"]
        fraction=np.clip((self.global_env_steps-tr["random_warmup_env_steps"])/tr["exploration_decay_steps"],0,1)
        return tr["exploration_noise_start"]+(tr["exploration_noise_end"]-tr["exploration_noise_start"])*fraction

    def random_actions(self): return mixture_action(self.exploration_rng,self.cruise)*self.limits

    def joint_actions(self,obs,targets=False):
        actors=self.actor_targets if targets else self.actors
        batch=obs.shape[0]
        return torch.cat((actors[0](obs[:,:2].reshape(-1,self.obs_dim)).reshape(batch,2,2),
                          actors[1](obs[:,2:].reshape(-1,self.obs_dim)).reshape(batch,3,2)),1)

    def act(self,obs,explore=False):
        modes=[a.training for a in self.actors]
        try:
            for a in self.actors: a.eval()
            with torch.no_grad(): result=self.joint_actions(self.tensor(obs)[None])[0].cpu().numpy()
        finally:
            for a,mode in zip(self.actors,modes): a.train(mode)
        if explore: result=result+self.exploration_rng.normal(0,self.sigma(),result.shape)*self.limits
        return np.clip(result,-self.limits,self.limits).astype(np.float32)

    def add_transition(self,obs,state,actions,rewards,next_obs,next_state,info,active=None,next_active=None):
        mask=np.ones(5,np.float32) if active is None else np.asarray(active,np.float32)
        if next_active is not None and not np.array_equal(mask,next_active):
            raise ValueError("v4.3 fixed-activity task does not support dynamic activity changes")
        self.replay.add(obs,state,actions,rewards,next_obs,next_state,terminated=info.terminated,truncated=info.truncated,
                        timeout=info.timeout,terminal_reason=info.terminal_reason,agent_active_mask=mask,surface_active=mask[:2])
        self.global_env_steps+=1

    def _router(self,state):
        threshold=self.runtime_config["task"]["encirclement"]["shrink_tracking"]["max_slot_error_max"]
        scale=max(self.runtime_config["env"]["world_size"])
        # build_critic_state: five 9-field states, then five 4-field task vectors.
        deficits=torch.stack([torch.linalg.vector_norm(state[:,45+4*j:47+4*j]*scale,dim=-1)/threshold for j in range(2,5)],-1).clamp(0,5)
        if self.runtime_config["revision"]["uniform_downstream"]: return torch.full_like(deficits,1/3)
        return (.5/3+.5*torch.softmax(deficits/self.runtime_config["revision"]["i2"]["temperature"],-1)).detach()

    def _local_and_downstream(self,role,obs,state,mask,joint_reference=None):
        actor=self.actors[role]
        with torch.no_grad(): joint=self.joint_actions(obs) if joint_reference is None else joint_reference
        joint=joint*mask[:,:,None]
        indices=range(0,2) if role==0 else range(2,5)
        local,downstream=obs.new_zeros(()),obs.new_zeros(())
        count=sum(mask[:,i].sum() for i in indices).clamp_min(1.)
        pair=self.critics[role]
        i2=self.runtime_config["revision"]["i2"]
        ramp=float(np.clip((self.global_env_steps-i2["warmup_steps"])/i2["ramp_steps"],0,1))
        enabled=role==0 and self.runtime_config["revision"]["i2_enabled"] and ramp>0
        for offset,i in enumerate(indices):
            candidate=torch.stack([actor(obs[:,j]) if j==i else joint[:,j].detach() for j in range(5)],1)
            q=pair[0](state,candidate,obs)
            head=0 if q.shape[1]==1 else offset
            local=local-(q[:,head]*mask[:,i]).sum()/count
            if enabled:
                q1,q2=[q(state,candidate,obs) for q in self.critics[1]]
                w=self._router(state)*mask[:,2:]
                w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
                delta=(w*(q1-q2).abs()).sum(-1)
                confidence=(1/(1+delta/((self.auv_td_ema or 0.)+1e-3))).detach()
                valid=mask[:,i]*(mask[:,2:].sum(-1)>0)
                if self._capture_i2:
                    # Read already-computed tensors only; no new sampling/forward/update.
                    errors=torch.stack([torch.linalg.vector_norm(state[:,45+4*j:47+4*j]*max(self.runtime_config['env']['world_size']),dim=-1) for j in range(2,5)],-1)
                    take=min(8,len(state))
                    self.i2_diagnostics.append(dict(usv_index=i,update_step=self.update_steps,env_steps=self.global_env_steps,
                        source='actual_replay_actor_update',batch_size=len(state),sampled_rows=take,
                        replay_indices=self.last_sample_indices[:take].tolist(),errors_m=errors[:take].detach().cpu().tolist(),
                        router_weights=w[:take].detach().cpu().tolist(),confidence=confidence[:take].detach().cpu().tolist(),
                        valid=valid[:take].detach().cpu().tolist(),q1=q1[:take].detach().cpu().tolist(),q2=q2[:take].detach().cpu().tolist(),
                        weights_batch_mean=w.detach().mean(0).cpu().tolist(),confidence_batch_mean=float(confidence.mean()),
                        ramp=ramp,lambda_value=i2['lambda']))
                downstream=downstream-i2["lambda"]*ramp*(confidence*(w*torch.minimum(q1,q2)).sum(-1)*valid).sum()/count
        return local,downstream,enabled

    def _actor_step(self,role,obs,state,mask,joint_reference=None):
        actor=self.actors[role]
        params=tuple(actor.parameters())
        metrics={"actor_loss":None,"i2_grad_ratio":0.,"i3_bc_loss":0. if not self.runtime_config["revision"]["teacher_enabled"] else None,"i3_grad_ratio":0.}
        if mask[:,slice(0,2) if role==0 else slice(2,5)].sum().item()==0:
            return metrics
        with frozen(q for pair in self.critics for q in pair):
            local,downstream,enabled=self._local_and_downstream(role,obs,state,mask,joint_reference)
            lg=torch.autograd.grad(local,params,retain_graph=enabled)
            local_norm=float(gradient_norm(lg))
            self.local_grad_ema[role]=ema(self.local_grad_ema[role],local_norm)
            gradients=[g.detach().clone() for g in lg]
            metrics["actor_loss"]=float(local.detach())
            if enabled:
                dg=torch.autograd.grad(downstream,params)
                norm=float(gradient_norm(dg))
                reference=max(local_norm,self.local_grad_ema[role],1e-8)
                scale=min(1.,self.runtime_config["revision"]["i2"]["gradient_ratio"]*reference/(norm+1e-12))
                gradients=[l+scale*d.detach() for l,d in zip(gradients,dg)]
                metrics["actor_loss"]=float(local.detach()+scale*downstream.detach())
                metrics["i2_grad_ratio"]=scale*norm/reference
            if role==0:
                base_norm=float(gradient_norm(gradients))
                self.guide_reference_ema=ema(self.guide_reference_ema,base_norm)
                hc=self.runtime_config["revision"]["i3"]
                if self.runtime_config["revision"]["teacher_enabled"]:
                    labels=self.guidance.sample(self.global_env_steps,self.actor_updates)
                    self.guidance_event=deepcopy(self.guidance.last_event)
                    if labels is not None:
                        loss=self.guidance_loss(labels)
                        gg=torch.autograd.grad(loss,params,allow_unused=True)
                        gg=tuple(torch.zeros_like(p) if g is None else g for p,g in zip(params,gg))
                        norm=float(gradient_norm(gg))
                        reference=max(base_norm,self.guide_reference_ema,1e-8)
                        scale=min(1.,hc["gradient_ratio"]*reference/(norm+1e-12))
                        gradients=[l+scale*g.detach() for l,g in zip(gradients,gg)]
                        metrics.update(i3_bc_loss=float(loss.detach()),i3_grad_ratio=scale*norm/reference)
                        with torch.no_grad():
                            history,target,_=map(self.tensor,labels)
                            predicted=self.surface_actor.coefficients(history.reshape(-1,self.obs_dim)).reshape_as(target)
                            coefficient_mse=float((predicted-target).square().mean())
                        self.guidance_event.update(coefficient_mse=coefficient_mse,
                                                   guidance_loss=float(loss.detach()),loss_kind="soft_target_logit_bce",
                                                   teacher_gradient_norm=norm,guide_gradient_ratio=scale*norm/reference,
                                                   teacher_gradient_scale=scale)
            opt=self.actor_optimizers[role]; opt.zero_grad(set_to_none=True)
            for p,g in zip(params,gradients):
                if not torch.isfinite(g).all(): raise FloatingPointError("nonfinite actor gradient")
                p.grad=g
            nn.utils.clip_grad_norm_(params,self.runtime_config["training"]["grad_clip_norm"],error_if_nonfinite=True)
            opt.step()
            if role==0 and self.guidance_event is not None:
                if self.guidance_event['status']=='sampled': self.guidance_event['status']='applied'
                self.guidance_event['update_step']=self.update_steps
        return metrics

    def guidance_loss(self,labels):
        """Guide the hypernetwork only; encoder and low-rank bases are detached."""
        history,coefficients,weight=map(self.tensor,labels)
        logits=self.surface_actor.coefficient_logits(history.reshape(-1,self.obs_dim),detach_encoder=True).reshape(-1,2,self.surface_actor.rank)
        # sigmoid(2*h) == (tanh(h)+1)/2. The derivative is z-z_teacher,
        # without MSE's additional (1-z*z) saturation factor. This is soft-target
        # distillation of continuous coefficients, not an action classifier.
        loss=nn.functional.binary_cross_entropy_with_logits(2*logits,(coefficients+1)/2,reduction="none")
        return (weight*loss.mean((1,2))).mean()

    def update(self):
        self.i2_diagnostics=[]; self.guidance_event=None
        self._capture_i2=self.diagnostic_every>0 and (self.update_steps+1)%self.diagnostic_every==0
        tr=self.runtime_config["training"]
        if len(self.replay)<tr["batch_size"]: return None
        sample=self.replay.sample_uniform(tr["batch_size"])
        self.last_sample_indices=sample["indices"].copy()
        obs,state,actions,rewards,next_obs,next_state,discount,mask=[self.tensor(sample[k]) for k in
            ("obs","states","actions","agent_rewards","next_obs","next_states","bootstrap_discounts","agent_active_mask")]
        with torch.no_grad():
            next_action=self.joint_actions(next_obs,targets=True)
            noise=torch.randn(next_action.shape,device=self.device,generator=self.target_rng)*tr["target_policy_noise"]
            noise=noise.clamp(-tr["target_noise_clip"],tr["target_noise_clip"])
            self.last_target_noise=noise.cpu().numpy().copy()
            limits=self.tensor(self.limits)
            next_action=((next_action/limits+noise).clamp(-1,1)*limits)*mask[:,:,None]
        metrics={}
        for role,sl in ((0,slice(0,2)),(1,slice(2,5))):
            active=mask[:,sl]
            if active.sum().item()==0:
                prefix="usv" if role==0 else "auv"
                metrics.update({prefix+"_critic_loss":None,prefix+"_td_abs":None,prefix+"_actor_loss":None})
                continue
            with torch.no_grad():
                qnext=torch.minimum(*[q(next_state,next_action,next_obs) for q in self.critic_targets[role]])
                target=rewards[:,sl]+discount*qnext*active
                if qnext.shape[1]==1:
                    target=(rewards[:,sl]*active).sum(1,keepdim=True)/active.sum(1,keepdim=True).clamp_min(1.)+discount*qnext
                    active=(active.sum(1,keepdim=True)>0).float()
            predicted=[q(state,actions,obs) for q in self.critics[role]]
            error=[p-target for p in predicted]
            loss=sum((nn.functional.smooth_l1_loss(p,target,reduction="none")*active).sum()/active.sum().clamp_min(1.) for p in predicted)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite critic loss")
            optimizer=self.critic_optimizers[role]; optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_([p for q in self.critics[role] for p in q.parameters()],tr["grad_clip_norm"],error_if_nonfinite=True)
            optimizer.step()
            td=float(sum((e.detach().abs()*active).sum() for e in error)/(2*active.sum().clamp_min(1.)))
            prefix="usv" if role==0 else "auv"
            metrics.update({prefix+"_critic_loss":float(loss.detach()),prefix+"_td_abs":td,prefix+"_actor_loss":None})
            if role==1: self.auv_td_ema=ema(self.auv_td_ema,td)
        self.update_steps+=1
        metrics.update(i2_grad_ratio=None,i3_bc_loss=None,i3_grad_ratio=None)
        if self.update_steps%tr["policy_delay"]==0:
            with torch.no_grad(): joint_reference=self.joint_actions(obs)
            for role,prefix in ((0,"usv"),(1,"auv")):
                am=self._actor_step(role,obs,state,mask,joint_reference)
                metrics[prefix+"_actor_loss"]=am.pop("actor_loss")
                if role==0: metrics.update(am)
            self.actor_updates+=1
            with torch.no_grad():
                sources=self.actors+[q for pair in self.critics for q in pair]
                targets=self.actor_targets+[q for pair in self.critic_targets for q in pair]
                for source,target in zip(sources,targets):
                    for p,t in zip(source.parameters(),target.parameters()): t.lerp_(p,tr["tau"])
        return metrics

    def state_dict(self):
        return {"actors":[a.state_dict() for a in self.actors],"actor_targets":[a.state_dict() for a in self.actor_targets],
            "critics":[[q.state_dict() for q in pair] for pair in self.critics],
            "critic_targets":[[q.state_dict() for q in pair] for pair in self.critic_targets],
            "actor_optimizers":[o.state_dict() for o in self.actor_optimizers],
            "critic_optimizers":[o.state_dict() for o in self.critic_optimizers],"replay":self.replay.state_dict(),
            "guidance":self.guidance.state_dict(),"exploration_rng":deepcopy(self.exploration_rng.bit_generator.state),
            "target_rng":self.target_rng.get_state(),"counters":{k:getattr(self,k) for k in
                ("global_env_steps","update_steps","actor_updates","episodes_completed","local_grad_ema","guide_reference_ema","auv_td_ema")}}

    def load_state_dict(self,state):
        for name in ("actors","actor_targets","actor_optimizers","critic_optimizers"):
            for obj,data in zip(getattr(self,name),state[name]): obj.load_state_dict(data)
        for name in ("critics","critic_targets"):
            for pair,stored in zip(getattr(self,name),state[name]):
                for q,data in zip(pair,stored): q.load_state_dict(data,strict=True)
        self.replay.load_state_dict(state["replay"]); self.guidance.load_state_dict(state["guidance"])
        for k,v in state["counters"].items(): setattr(self,k,v)
        self.exploration_rng.bit_generator.state=state["exploration_rng"]
        self.target_rng.set_state(state["target_rng"].cpu())
