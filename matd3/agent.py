"""Role-shared MATD3 with current-frame actors and centralized twin critics."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

import numpy as np
import torch
from torch import nn

from .networks import make_actor, Critic
from .replay_buffer import ReplayBuffer
from .history_enhancement import JointHistory, conservative_value, I3_COUNT_FIELDS, I3_DISTRIBUTION_FIELDS
from usv_env.config import validate, named_seed


_DEFER_METRIC_SCALARS = ContextVar("defer_metric_scalars", default=False)


def _metric_float(value):
    # Detach at the original measurement point. Only the host transfer is
    # delayed; validation branches and nonfinite gradient checks stay eager.
    return value.detach() if _DEFER_METRIC_SCALARS.get() else float(value)


def _batch_metric_scalars(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        token = _DEFER_METRIC_SCALARS.set(True)
        try:
            metrics = function(self, *args, **kwargs)
        finally:
            _DEFER_METRIC_SCALARS.reset(token)
        if metrics is None:
            return None
        groups = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                groups.setdefault((value.device, value.dtype), []).append((key, value))
        for fields in groups.values():
            values = torch.stack([value.reshape(()) for _, value in fields]).cpu().tolist()
            for (key, _), value in zip(fields, values):
                metrics[key] = float(value)
        if isinstance(getattr(self, "last_actor_grad_norm", None), torch.Tensor):
            self.last_actor_grad_norm = float(self.last_actor_grad_norm)
        return metrics
    return wrapped


@contextmanager
def frozen(modules):
    modules = tuple(modules)
    flags = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    modes = [module.training for module in modules]
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        for module in modules:
            module.eval()
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in zip(modules, modes):
            module.train(mode)


def nominal_actions(env):
    """Cruise-force initialization from the unchanged role-specific drag model."""
    count = int(env.num_surface_blue) + int(env.num_submarines)
    surface_count = int(env.num_surface_blue)
    normalized = np.zeros((count, 2), np.float32)
    for role_slice, dynamics, speed in (
        (slice(0, surface_count), env.blue_dynamics, 0.30),
        (slice(surface_count, count), env.submarine_dynamics, 0.40),
    ):
        velocity = np.asarray([speed, 0.0, 0.0])
        equilibrium = dynamics.damping_matrix(velocity) @ velocity
        normalized[role_slice, 0] = equilibrium[0] / env.agent_action_limits[role_slice, 0]
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) >= 1.0):
        raise ValueError("nominal cruise exceeds normalized actuator range")
    return normalized


class Agent:
    def __init__(self, env, cfg):
        validate(cfg)
        self.runtime_config = deepcopy(cfg)
        training = cfg["training"]
        device = training["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.obs_dim, self.state_dim = env.observation_dim, env.critic_state_dim
        self.num_surface = int(env.num_surface_blue)
        self.num_submarines = int(env.num_submarines)
        self.num_agents = self.num_surface + self.num_submarines
        self.role_slices = (slice(0, self.num_surface), slice(self.num_surface, self.num_agents))
        if any(cfg["revision"].get(key, False) for key in ("hyper_enabled", "teacher_enabled")):
            raise ValueError("old gradient routing and parameter teachers are not current-task innovations")
        self.i1_enabled = bool(cfg["mechanisms"]["safety_projection"])
        self.i3_enabled = bool(cfg["mechanisms"]["joint_action_enhancement"])
        self.limits = np.asarray(env.agent_action_limits, np.float32)
        if self.limits.shape != (self.num_agents, 2) or not np.isfinite(self.limits).all() or np.any(self.limits <= 0):
            raise ValueError("agent_action_limits must match all friendly agents and both actuators")
        self.cruise = nominal_actions(env)
        self.surface_actor = make_actor(cfg, self.limits[0], self.cruise[0], "surface").to(self.device)
        self.submarine_actor = make_actor(cfg, self.limits[self.num_surface], self.cruise[self.num_surface], "submarine").to(self.device)
        self.actors = [self.surface_actor, self.submarine_actor]
        self.actor_targets = [deepcopy(actor).requires_grad_(False) for actor in self.actors]
        self.critics = []
        for role, heads in (("surface", self.num_surface), ("submarine", self.num_submarines)):
            pair = []
            for twin in range(2):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(named_seed(training["seed"], f"critic:{role}:{twin}"))
                    pair.append(Critic(cfg, self.state_dim, self.limits, heads).to(self.device))
            self.critics.append(pair)
        self.critic_targets = [[deepcopy(critic).requires_grad_(False) for critic in pair] for pair in self.critics]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=training["actor_lr"]) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam([p for critic in pair for p in critic.parameters()], lr=training["critic_lr"])
                                  for pair in self.critics]
        self.replay = ReplayBuffer(cfg["replay_buffer"]["capacity"], self.num_agents, self.obs_dim, self.state_dim, 2,
                                   seed=named_seed(training["seed"], "replay"), n_step_return=training["n_step_return"],
                                   gamma=training["gamma"], surface_count=self.num_surface)
        self.exploration_rng = np.random.default_rng(named_seed(training["seed"], "exploration"))
        self.target_rng = torch.Generator(device=self.device)
        self.target_rng.manual_seed(named_seed(training["seed"], "target_noise"))
        self.history = JointHistory(cfg, named_seed(training["seed"], "joint_history")) if self.i3_enabled else None
        self.global_env_steps = self.update_steps = self.actor_updates = self.episodes_completed = 0
        self.last_sample_indices = self.last_target_noise = None
        self.last_i3 = None

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def sigma(self):
        training = self.runtime_config["training"]
        fraction = np.clip((self.global_env_steps - training["random_warmup_env_steps"])
                           / training["exploration_decay_steps"], 0, 1)
        return training["exploration_noise_start"] + (training["exploration_noise_end"] - training["exploration_noise_start"]) * fraction

    def random_actions(self):
        if self.exploration_rng.random() < 0.5:
            normalized = self.exploration_rng.uniform(-1, 1, self.cruise.shape)
        else:
            normalized = np.clip(self.cruise + self.exploration_rng.normal(0, 0.20, self.cruise.shape), -1, 1)
        return (normalized * self.limits).astype(np.float32)

    def joint_actions(self, obs, targets=False):
        actors = self.actor_targets if targets else self.actors
        batch = obs.shape[0]
        return torch.cat((actors[0](obs[:, :self.num_surface].reshape(-1, self.obs_dim)).reshape(batch, self.num_surface, 2),
                          actors[1](obs[:, self.num_surface:].reshape(-1, self.obs_dim)).reshape(batch, self.num_submarines, 2)), 1)

    def act(self, obs, explore=False):
        observation = np.asarray(obs, np.float32)
        if observation.shape != (self.num_agents, self.obs_dim) or not np.isfinite(observation).all():
            raise ValueError("actor observation shape or finite-value contract violated")
        modes = [actor.training for actor in self.actors]
        try:
            for actor in self.actors:
                actor.eval()
            with torch.no_grad():
                result = self.joint_actions(self.tensor(observation)[None])[0].cpu().numpy()
        finally:
            for actor, mode in zip(self.actors, modes):
                actor.train(mode)
        if explore:
            result += self.exploration_rng.normal(0, self.sigma(), result.shape) * self.limits
        return np.clip(result, -self.limits, self.limits).astype(np.float32)

    def add_transition(self, obs, state, actions, rewards, next_obs, next_state, info, active=None, next_active=None,
                       current_context=None, next_context=None, nominal_actions=None):
        mask = np.ones(self.num_agents, np.float32) if active is None else np.asarray(active, np.float32)
        if next_active is not None and not np.array_equal(mask, next_active):
            raise ValueError("fixed-activity MATD3 does not support activity changes within a transition")
        self.replay.add(obs, state, actions, rewards, next_obs, next_state,
                        terminated=info.terminated, truncated=info.truncated, timeout=info.timeout,
                        terminal_reason=info.terminal_reason, agent_active_mask=mask,
                        surface_active=mask[:self.num_surface], current_context=current_context,
                        next_context=next_context, nominal_action=nominal_actions)
        self.global_env_steps += 1

    def _project_tensor(self, state, obs, actions, *, straight_through=False, reject_only=False, receivers=None):
        from usv_env.safety_projection import project_from_state
        state_cpu = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state, np.float32)
        obs_cpu = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs, np.float32)
        safe, diagnostics = project_from_state(self.runtime_config, state_cpu,
            obs_cpu, actions.detach().cpu().numpy(), reject_only=reject_only,
            receivers=receivers, decision_only=reject_only)
        safe = self.tensor(safe)
        if safe.shape != actions.shape or not torch.isfinite(safe).all():
            raise ValueError("safety projection violated the joint physical action contract")
        # The discrete predictor is not differentiable. This explicitly uses
        # an identity straight-through approximation at the projected action;
        # Bellman targets and executed-action storage use the actual projection.
        return (actions + (safe-actions).detach() if straight_through else safe), diagnostics

    def _target_values(self, sample, next_state, next_obs, next_action, mask):
        pairs = [[critic(next_state, next_action, next_obs) for critic in pair]
                 for pair in self.critic_targets]
        q_policy = torch.cat([torch.minimum(*pair) for pair in pairs], -1)
        metrics = {"i3_retrieved_candidates": 0., "i3_valid_candidates": 0., "i3_acceptance_rate": 0.,
                   "i3_similarity_distance": None, "i3_phase_match_fraction": 0., "i3_safety_rejection_fraction": 0.,
                   "i3_q_policy": _metric_float(q_policy.mean()), "i3_q_history": None, "i3_advantage": 0.,
                   "i3_confidence": 0., "i3_history_quality": None, "i3_enhancement_weight": 0.,
                   "i3_realized_nstep_return": None, "i3_outcome_confidence": None,
                   "i3_target_enhancement": 0., "i3_queries": 0., "i3_candidate_acceptance_rate": 0.}
        for prefix, pair in zip(("usv", "auv"), pairs):
            metrics[prefix+"_target_q1"] = _metric_float(pair[0].mean())
            metrics[prefix+"_target_q2"] = _metric_float(pair[1].mean())
        metrics.update({key: 0 for key in I3_COUNT_FIELDS})
        metrics.update({key: None for key in I3_DISTRIBUTION_FIELDS})
        self.last_i3 = None
        if not self.i3_enabled:
            return q_policy, metrics
        # Retrieval is exclusively from online training replay. Query endpoints
        # are s_{t+n}, never the current or reset state of the collected batch.
        found = self.history.retrieve(self.replay, sample["next_states"], sample["next_context"],
                                      sample["next_phase_ids"], sample["agent_active_mask"])
        metrics.update(found['funnel']); metrics.update(found['distributions'])
        valid = found["valid"].copy()
        metrics['i3_terminal_rejected'] = int(np.count_nonzero(valid & (sample['bootstrap_masks'][:, 0, None] == 0)))
        valid &= sample["bootstrap_masks"][:, 0, None] > 0
        retrieved = int(valid.sum())
        if not retrieved:
            # No historical action can influence the Bellman target. Avoid
            # building/scoring a CUDA candidate tensor full of masked zeros,
            # but retain exactly the same public metrics and last_i3 record.
            settings = self.history.settings
            similarity = np.exp(-np.minimum(found["distance"], 1e3)**2 / settings["similarity_temperature"]**2)
            if settings["variant"] in ("random_history", "no_state_matching"):
                similarity = np.ones_like(similarity)
            confidence = similarity*found["quality"] / (1.+np.zeros(valid.shape, np.float32)/settings["uncertainty_scale"])
            q_cpu = q_policy.cpu().numpy()
            metrics.update(i3_queries=float(len(sample["next_states"])),
                i3_phase_match_fraction=found["phase_match_fraction"],
                i3_enhancement_weight=1. if settings["variant"]=="hard_max" else settings["enhancement_weight"])
            self.last_i3 = dict(indices=found["indices"][:, 0].copy(), valid=valid,
                accepted=np.zeros(len(valid), bool), actions=found["actions"][:, 0].copy(),
                confidence=confidence, realized_nstep_return=found["realized_return"][:, 0].copy(),
                outcome_confidence=found["outcome_confidence"][:, 0].copy(),
                delta=np.zeros_like(q_cpu), q_policy=q_cpu, q_history=q_cpu.copy())
            return q_policy+torch.zeros_like(q_policy), metrics
        rows, slots = np.where(valid)
        q_history = q_policy[:, None].expand(-1, valid.shape[1], -1).clone()
        uncertainty = np.zeros(valid.shape, np.float32)
        rejected = 0
        if len(rows):
            candidates = self.tensor(found["actions"][rows, slots])
            if self.history.settings["variant"] != "no_safety_screening":
                # These are the same host arrays backing next_state/next_obs.
                # I1 only consumes the last receiver frame: do not expand full
                # observations on GPU and copy them back for each candidate.
                frame_dim = self.runtime_config['observation_schema']['frame_dim']
                frames = sample['next_obs'].reshape(len(next_obs), self.num_agents, -1, frame_dim)[:, :, -1]
                _, safety = self._project_tensor(sample['next_states'][rows], frames[rows],
                    candidates, reject_only=True)
                safe = np.asarray(safety["safe_mask"], dtype=bool).reshape(-1)
                if safe.shape != (len(rows),):
                    raise ValueError("candidate safety screening requires one safe_mask per joint action")
                rejected = int((~safe).sum())
                valid[rows[~safe], slots[~safe]] = False
                candidates = candidates[safe]
                rows, slots = rows[safe], slots[safe]
            if len(rows):
                states, histories = next_state[rows], next_obs[rows]
                history_pairs = [[critic(states, candidates, histories) for critic in pair]
                                 for pair in self.critic_targets]
                q_history[rows, slots] = torch.cat([torch.minimum(*pair) for pair in history_pairs], -1)
                uncertainty[rows, slots] = torch.cat([(a-b).abs() for a,b in history_pairs], -1).mean(-1).cpu().numpy()
        settings = self.history.settings
        similarity = np.exp(-np.minimum(found["distance"], 1e3)**2 / settings["similarity_temperature"]**2)
        if settings["variant"] in ("random_history", "no_state_matching"):
            similarity = np.ones_like(similarity)
        confidence = similarity*found["quality"] / (1.+uncertainty/settings["uncertainty_scale"])
        value, detail = conservative_value(q_policy, q_history, self.tensor(confidence),
                                           torch.as_tensor(valid, device=self.device), settings, mask)
        selected = detail["chosen"].cpu().numpy()
        batch_rows = np.arange(len(selected))
        accepted = detail["accepted"].cpu().numpy()
        metrics.update(i3_safety_rejected=rejected,
            i3_qualified_rows=int(valid.any(1).sum()),
            i3_positive_advantage_missing_rows=int(np.count_nonzero(valid.any(1) & ~accepted)),
            i3_selected_rows=int(accepted.sum()),
            i3_nonzero_enhancement_rows=int((self.tensor(sample['bootstrap_discounts'])*detail['delta']).sum(-1).gt(0).sum()))
        chosen_q = q_history[batch_rows, selected]
        metrics.update(i3_retrieved_candidates=float(retrieved), i3_valid_candidates=float(valid.sum()),
            i3_acceptance_rate=float(accepted.mean()), i3_phase_match_fraction=found["phase_match_fraction"],
            i3_queries=float(len(sample["next_states"])), i3_candidate_acceptance_rate=float(valid.sum())/max(retrieved,1),
            i3_safety_rejection_fraction=rejected/max(retrieved, 1),
            i3_similarity_distance=float(found["distance"][valid].mean()) if valid.any() else None,
            i3_q_history=_metric_float(chosen_q[accepted].mean()) if accepted.any() else None,
            i3_advantage=_metric_float(detail["advantage"][detail["accepted"]].mean()) if accepted.any() else 0.,
            i3_confidence=float(confidence[batch_rows, selected][accepted].mean()) if accepted.any() else 0.,
            i3_history_quality=float(found["quality"][batch_rows,selected][accepted].mean()) if accepted.any() else None,
            i3_realized_nstep_return=float(found["realized_return"][batch_rows,selected][accepted].mean()) if accepted.any() else None,
            i3_outcome_confidence=float(found["outcome_confidence"][batch_rows,selected][accepted].mean()) if accepted.any() else None,
            i3_enhancement_weight=1. if settings["variant"]=="hard_max" else settings["enhancement_weight"],
            i3_target_enhancement=_metric_float((self.tensor(sample["bootstrap_discounts"])*detail["delta"]).mean()))
        self.last_i3 = dict(indices=found["indices"][batch_rows, selected], valid=valid, accepted=accepted,
                            actions=found["actions"][batch_rows, selected], confidence=confidence,
                            realized_nstep_return=found["realized_return"][batch_rows, selected],
                            outcome_confidence=found["outcome_confidence"][batch_rows, selected],
                            delta=detail["delta"].cpu().numpy(), q_policy=q_policy.cpu().numpy(),
                            q_history=chosen_q.cpu().numpy())
        return value, metrics

    def _actor_step(self, role, obs, state, mask, joint_reference, safe_reference=None):
        role_slice = self.role_slices[role]
        if mask[:, role_slice].sum().item() == 0:
            return None
        actor = self.actors[role]
        indices = range(role_slice.start, role_slice.stop)
        loss = obs.new_zeros(())
        count = mask[:, role_slice].sum().clamp_min(1.0)
        projection_seconds=0.
        if self.i1_enabled and safe_reference is None:
            safe_reference, detail=self._project_tensor(state,obs,joint_reference.detach())
            projection_seconds+=float(detail.get('solver_time_s',0.))
        reference=safe_reference if self.i1_enabled else joint_reference
        with frozen(q for pair in self.critics for q in pair):
            critic = self.critics[role][0]
            role_actions = actor(obs[:, role_slice].reshape(-1, self.obs_dim)).reshape(
                len(obs), role_slice.stop-role_slice.start, 2)
            # Current role actions normally equal the joint reference exactly:
            # each role has its own actor and it has not been stepped yet.
            # Check rather than assume this, preserving stale-reference callers.
            reuse_projection = self.i1_enabled and torch.equal(
                role_actions.detach() * mask[:, role_slice, None],
                joint_reference[:, role_slice].detach() * mask[:, role_slice, None])
            for head, agent_index in enumerate(indices):
                # A shared actor still makes one decentralized action per agent.
                # Only that agent's action carries gradients for its own Q head.
                candidate = torch.stack([role_actions[:, head] if j == agent_index else reference[:, j].detach()
                                         for j in range(self.num_agents)], 1) * mask[:, :, None]
                if self.i1_enabled:
                    # I1 predicts each receiver using its own observation, not
                    # peers' candidate actions. Unchanged peer projections are
                    # therefore reusable without changing the loss/gradient.
                    if reuse_projection:
                        projected = reference * mask[:, :, None]
                        candidate = candidate + (projected-candidate).detach()
                    else:
                        candidate, detail = self._project_tensor(state, obs, candidate,
                            straight_through=True,receivers=(agent_index,))
                        projection_seconds+=float(detail.get('solver_time_s',0.))
                values = critic(state, candidate, obs)
                loss -= (values[:, head] * mask[:, agent_index]).sum() / count
            optimizer = self.actor_optimizers[role]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(actor.parameters(), self.runtime_config["training"]["grad_clip_norm"], error_if_nonfinite=True)
            self.last_actor_grad_norm = _metric_float(norm)
            optimizer.step()
        self.last_actor_projection_seconds=projection_seconds
        return _metric_float(loss.detach())

    @_batch_metric_scalars
    def update(self):
        training = self.runtime_config["training"]
        if len(self.replay) < training["batch_size"]:
            return None
        sample = self.replay.sample_uniform(training["batch_size"])
        self.last_sample_indices = sample["indices"].copy()
        obs, state, actions, rewards, next_obs, next_state, discounts, mask = [self.tensor(sample[key]) for key in
            ("obs", "states", "actions", "agent_rewards", "next_obs", "next_states", "bootstrap_discounts", "agent_active_mask")]
        with torch.no_grad():
            next_action = self.joint_actions(next_obs, targets=True)
            noise = torch.randn(next_action.shape, device=self.device, generator=self.target_rng) * training["target_policy_noise"]
            noise = noise.clamp(-training["target_noise_clip"], training["target_noise_clip"])
            self.last_target_noise = noise.cpu().numpy().copy()
            limits = self.tensor(self.limits)
            next_action = ((next_action / limits + noise).clamp(-1, 1) * limits) * mask[:, :, None]
            if self.i1_enabled:
                # I1 runs on CPU. Reuse the identical host Replay values rather
                # than copying the full CUDA history back to its source.
                next_action, projection = self._project_tensor(sample['next_states'], sample['next_obs'], next_action)
                next_action *= mask[:, :, None]
            target_values, metrics = self._target_values(sample, next_state, next_obs, next_action, mask)
            if self.i1_enabled:
                metrics.update(i1_target_intervention=float(np.mean(projection["intervention"])),
                               i1_target_projection_norm=float(np.mean(projection["projection_norm"])),
                               i1_target_infeasible=float(np.mean(projection["infeasible"])),
                               i1_target_solver_time_s=float(projection.get("solver_time_s",0.)))
        metrics["exploration_noise"] = float(self.sigma())
        for role, role_slice in enumerate(self.role_slices):
            active = mask[:, role_slice]
            prefix = "usv" if role == 0 else "auv"
            metrics[prefix + "_actor_loss"] = None
            if active.sum().item() == 0:
                metrics.update({prefix + "_critic_loss": None, prefix + "_td_abs": None})
                continue
            with torch.no_grad():
                target = rewards[:, role_slice] + discounts * target_values[:, role_slice] * active
            predicted = [critic(state, actions, obs) for critic in self.critics[role]]
            metrics[prefix+"_q1"] = _metric_float(predicted[0].detach().mean())
            metrics[prefix+"_q2"] = _metric_float(predicted[1].detach().mean())
            if training.get("use_huber_loss", True):
                losses = [nn.functional.smooth_l1_loss(value, target, reduction="none") for value in predicted]
            else:
                losses = [(value - target).square() for value in predicted]
            loss = sum((value * active).sum() / active.sum().clamp_min(1.0) for value in losses)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite critic loss")
            optimizer = self.critic_optimizers[role]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_([p for critic in self.critics[role] for p in critic.parameters()],
                                           training["grad_clip_norm"], error_if_nonfinite=True)
            metrics[prefix+'_critic_grad_norm'] = _metric_float(norm)
            optimizer.step()
            td_error = sum(((value - target).detach().abs() * active).sum() for value in predicted) / (2 * active.sum().clamp_min(1.0))
            metrics[prefix + "_critic_loss"] = _metric_float(loss.detach())
            metrics[prefix + "_td_abs"] = _metric_float(td_error)
        self.update_steps += 1
        metrics['i1_actor_solver_time_s']=0.
        if self.update_steps % training["policy_delay"] == 0:
            with torch.no_grad():
                joint_reference = self.joint_actions(obs)
                safe_reference=None
                if self.i1_enabled:
                    safe_reference,detail=self._project_tensor(sample['states'],sample['obs'],joint_reference)
                    metrics['i1_actor_solver_time_s']+=float(detail.get('solver_time_s',0.))
            for role, prefix in enumerate(("usv", "auv")):
                self.last_actor_projection_seconds=0.
                self.last_actor_grad_norm=None
                metrics[prefix + "_actor_loss"] = self._actor_step(role, obs, state, mask, joint_reference,safe_reference)
                metrics[prefix+'_actor_grad_norm'] = self.last_actor_grad_norm
                metrics['i1_actor_solver_time_s']+=self.last_actor_projection_seconds
            self.actor_updates += 1
            with torch.no_grad():
                sources = self.actors + [q for pair in self.critics for q in pair]
                targets = self.actor_targets + [q for pair in self.critic_targets for q in pair]
                for source, target in zip(sources, targets):
                    for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                        target_parameter.lerp_(parameter, training["tau"])
        return metrics
