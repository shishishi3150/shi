"""Execute physical scene acceptance diagnostics and plot only recorded data."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

from usv_env.config import assemble, code_hash
from usv_env.environment import PursuitEvasionEnv
from usv_env.geometry import capsule_circle_clearance, capsules_overlap, capsule_inside_rectangle
from usv_env.red_policy import RuleController
from usv_env.reward import REWARD_COMPONENTS
from usv_env.rendering import (BLUE_SHIP_FILL, BLUE_SHIP_2_FILL, RED_SHIP_FILL,
    SHADOW_RED_COLOR, SUBMERGED_SUBMARINE_FILL, ISLAND_FILL, ISLAND_OUTLINE,
    WORLD_BACKGROUND_FILL, GRID_LINE_COLOR, PROTECTED_ZONE_OUTLINE)


def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def _write_json(path, data):
    path.write_text(json.dumps(_plain(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(_plain(v)) if isinstance(v, (list, tuple, dict)) else _plain(v))
                             for k, v in row.items()})


def geometry_audit(config):
    env = PursuitEvasionEnv(config)
    try:
        obs, _ = env.reset(seed=config["scenario"]["seed"])
        states = env.friendly_states() + [env.red_state]
        capsules = [env._capsule(s) for s in states]
        violations, reef_clearances = [], []
        for i, capsule in enumerate(capsules):
            if not capsule_inside_rectangle(capsule, env.world_size): violations.append(f"{i}-boundary")
            for reef in env.reefs:
                distance = capsule_circle_clearance(capsule, np.asarray(reef["center"]),
                    reef["radius"] + config["depth"]["reef_shallow_buffer_m"])
                reef_clearances.append(distance)
                if distance <= 0.: violations.append(f'{i}-reef-{reef["name"]}')
            for j in range(i + 1, len(states)):
                if env._can_collide(states[i], states[j]) and capsules_overlap(capsule, capsules[j]):
                    violations.append(f"{i}-{j}")
        # Perturb truth without running sensors. Unknown
        # Actor input must remain identical; centralized critic may change.
        old_position = env.red_state.pos.copy()
        env.red_state.pos = old_position + np.asarray([.73, -.29], np.float32)
        changed, _ = env._observations()
        env.red_state.pos = old_position
        initial_unknown = not any(e.valid for e in env.target_estimates)
        return dict(no_initial_illegal_overlap=not violations, violations=violations,
                    minimum_reef_contour_clearance_m=min(reef_clearances),
                    initial_target_estimates_unknown=initial_unknown,
                    actor_unknown_input_independent_of_red_truth=bool(np.array_equal(obs, changed)),
                    actor_shape=list(obs.shape), critic_shape=[env.critic_state_dim])
    finally:
        env.close()


def _trace_row(env, mode, seed, reward, info=None):
    now = env.step_count * env.dt
    row = dict(mode=mode, seed=seed, step=env.step_count, time_s=now,
               red_x=float(env.red_state.pos[0]), red_y=float(env.red_state.pos[1]),
               shadow_x=float(env.shadow_red_state.pos[0]), shadow_y=float(env.shadow_red_state.pos[1]),
               actual_distance_to_protection_m=float(np.linalg.norm(env.protected_center - env.red_state.pos)),
               shadow_distance_to_protection_m=float(np.linalg.norm(env.protected_center - env.shadow_red_state.pos)),
               shadow_gap_m=env.counterfactual_gap, progress_deficit_m=env.progress_deficit,
               hold_seconds=env.hold_steps * env.dt, mean_reward=float(np.mean(reward)),
               capture_geometry=env.capture_geometry.copy(),
               target_packets_sent=env.target_communication.sent,
               target_packets_dropped=env.target_communication.dropped,
               target_packets_delivered=env.target_communication.delivered,
               observer_maximum_angle_gap_degrees=math.degrees(env.observer_angle_gap),
               last_red_action=env.last_red_action.tolist(),
               red_behavior=env.red_navigator.mode,
               last_red_reference=env.last_red_reference.tolist(),
               red_positive_propulsion_fraction=(float(np.mean(env._red_propulsion_history))
                   if env._red_propulsion_history else 0.),
               shadow_recent_progress_m=env.shadow_recent_progress,
               nominal_actions=env.last_nominal_actions.tolist(),safe_actions=env.last_safe_actions.tolist(),
               executed_actions=env.last_executed_actions.tolist(),safety_diagnostics=env.safety_diagnostics,
               counterfactual_audit=info.counterfactual_audit if info else {},
               red_incident=env.red_incident,team_new_search_cells=env.team_new_search_cells,
               team_cumulative_search_cells=len(env._team_search_cells),
               terminal_reason=info.terminal_reason if info else "")
    for name in ("auv_participation", "capture_distance", "encirclement", "observation_support", "route_available", "safety"):
        row[f"gate_{name}"] = int(env.task_state.gates.get(name, False))
    for name in ("engagement", "causal_blocking", "information_support", "current_state_counterfactual", "red_breakthrough_intent"):
        row[f"support_{name}"] = int(env.task_state.support_gates.get(name, False))
    for i, state in enumerate(env.friendly_states()):
        row[f"agent_{i}_x"], row[f"agent_{i}_y"] = float(state.pos[0]), float(state.pos[1])
        row[f"agent_{i}_surge_mps"],row[f"agent_{i}_ground_speed_mps"]=float(state.speed),float(state.ground_speed)
        row[f"agent_{i}_sway_mps"],row[f"agent_{i}_yaw_rate_radps"]=float(state.sway_speed),float(state.yaw_rate)
        row[f"agent_{i}_heading_rad"]=float(state.heading)
        estimate = env.target_estimates[i]
        row[f"agent_{i}_target_valid"] = int(estimate.valid)
        row[f"agent_{i}_age_s"] = max(0., now - estimate.t_obs) if estimate.valid else None
        row[f"agent_{i}_position_variance_m2"] = float(np.trace(estimate.covariance[:2, :2])) if estimate.valid else None
        row[f"agent_{i}_tracking_error_m"] = float(np.linalg.norm(estimate.position - env.red_state.pos)) if estimate.valid else None
    for j in range(env.num_submarines): row[f"auv_{j}_detected"] = int(env.sensor_detected[j])
    row['red_surge_mps']=float(env.red_state.speed); row['red_ground_speed_mps']=float(env.red_state.ground_speed)
    row['red_yaw_rate_radps']=float(env.red_state.yaw_rate)
    row['reward_components']={name:info.reward_terms[name] if info else [0.]*env.num_friendly for name in REWARD_COMPONENTS}
    return row


def run_episode(config, mode, seed, record=False):
    config=deepcopy(config)
    peer_modes={'peer_disabled':'disabled','peer_delayed':'delayed','peer_shuffled':'shuffled'}
    if mode in peer_modes: config['communication']['peer_mode']=peer_modes[mode]
    env = PursuitEvasionEnv(config, seed)
    if mode == "no_blue": env.set_active_agents(np.zeros(env.num_friendly, np.float32))
    elif mode=='usv_only': env.set_active_agents([1,1,0,0,0])
    elif mode=='auv_only': env.set_active_agents([0,0,1,1,1])
    elif mode.startswith('without_usv_'):
        mask=np.ones(env.num_friendly); mask[int(mode.rsplit('_',1)[1])]=0; env.set_active_agents(mask)
    elif mode.startswith('without_auv_'):
        mask=np.ones(env.num_friendly); mask[env.num_surface_blue+int(mode.rsplit('_',1)[1])]=0; env.set_active_agents(mask)
    env.reset(seed=seed)
    rng=np.random.default_rng(seed)
    controller = RuleController()
    trace = [_trace_row(env, mode, seed, np.zeros(env.num_friendly))] if record else []
    all_ages, all_errors, event_rows, seen_gates = [], [], [], set()
    finite = True
    maximum_hold, maximum_gap = 0, 0.
    component_values={name:[] for name in REWARD_COMPONENTS}
    phase_rewards={}; episode_return=np.zeros(env.num_friendly); minimum_clearance=math.inf
    intervention_count=np.zeros(env.num_friendly); infeasible_count=np.zeros(env.num_friendly)
    counterfactual_events=[]
    started = time.perf_counter()
    try:
        for _ in range(env.max_steps):
            if mode == "no_blue": actions = np.zeros((env.num_friendly, 2), np.float32)
            elif mode=='stationary': actions=np.zeros((env.num_friendly,2),np.float32)
            elif mode=='random': actions=rng.uniform(-1,1,(env.num_friendly,2)).astype(np.float32)*env.agent_action_limits
            else:
                actions = controller.actions_for_env(env, ideal=mode == "rules_ideal")
                actions=actions*env.agent_active_mask()[:,None]
            obs, state, reward, done, info = env.step(actions)
            episode_return+=reward
            for name in REWARD_COMPONENTS: component_values[name].append(info.reward_terms[name])
            phase_row=phase_rewards.setdefault(info.phase,dict(steps=0,components={name:0. for name in REWARD_COMPONENTS}))
            phase_row['steps']+=1
            for name in REWARD_COMPONENTS: phase_row['components'][name]+=float(np.mean(info.reward_terms[name]))
            minimum_clearance=min(minimum_clearance,min(min(env._clearances(i)) for i in range(env.num_friendly) if env.agent_active_mask()[i]) if np.any(env.agent_active_mask()) else math.inf)
            intervention_count+=np.asarray(env.safety_diagnostics.get('intervention',[0.]*env.num_friendly),float)
            infeasible_count+=np.asarray(env.safety_diagnostics.get('infeasible',[0.]*env.num_friendly),float)
            if info.counterfactual_audit:
                counterfactual_events.append(dict(mode=mode,seed=seed,event='current_state_counterfactual_audit',**info.counterfactual_audit))
            finite &= bool(np.isfinite(obs).all() and np.isfinite(state).all() and np.isfinite(reward).all())
            maximum_hold = max(maximum_hold, env.hold_steps)
            maximum_gap = max(maximum_gap, env.counterfactual_gap)
            now = env.step_count * env.dt
            for estimate in env.target_estimates:
                if estimate.valid:
                    all_ages.append(max(0., now - estimate.t_obs))
                    all_errors.append(float(np.linalg.norm(estimate.position - env.red_state.pos)))
            for name, active in info.stage_success_gates.items():
                if active and name not in seen_gates:
                    seen_gates.add(name)
                    event_rows.append(dict(mode=mode, seed=seed, event=f"first_{name}",
                                           step=env.step_count, time_s=now))
            if record: trace.append(_trace_row(env, mode, seed, reward, info))
            if done: break
        sent = env.target_communication.sent
        absolute_component_total=sum(abs(float(np.mean(np.sum(values,axis=0)))) for values in component_values.values())
        reward_audit={}
        for name,values in component_values.items():
            array=np.asarray(values,float); cumulative=array.sum(0); team=float(cumulative.mean())
            reward_audit[name]=dict(step_team_mean=float(array.mean()),step_agent_min=float(array.min()),step_agent_max=float(array.max()),
                episode_per_agent=cumulative.tolist(),episode_team_mean=team,episode_usv_sum=float(cumulative[:env.num_surface_blue].sum()),
                episode_auv_sum=float(cumulative[env.num_surface_blue:].sum()),absolute_cumulative_fraction=abs(team)/max(absolute_component_total,1e-12))
        summary = dict(mode=mode, seed=seed, episode_steps=env.step_count,
            outcome=info.terminal_reason, success=bool(info.success), finite=finite,
            collision_pair=info.collision_pair, red_reached_protected_zone=bool(info.landing_success),
            first_detection_steps=list(env.first_detection_steps),
            discovered_auv_count=sum(x is not None for x in env.first_detection_steps),
            maximum_hold_seconds=maximum_hold * env.dt,
            final_capture_geometry=env.capture_geometry.copy(),
            final_support_gates=env.task_state.support_gates.copy(),
            maximum_shadow_gap_m=maximum_gap,
            final_shadow_gap_m=env.counterfactual_gap,
            final_progress_deficit_m=env.progress_deficit,
            final_red_positive_propulsion_fraction=(float(np.mean(env._red_propulsion_history))
                if env._red_propulsion_history else 0.),
            final_shadow_recent_progress_m=env.shadow_recent_progress,
            red_recovery_count=env.red_navigator.recovery_count,
            shadow_recovery_count=env.shadow_navigator.recovery_count,
            target_packets_sent=sent, target_packets_dropped=env.target_communication.dropped,
            empirical_target_packet_loss=(env.target_communication.dropped / sent if sent else None),
            mean_information_age_s=float(np.mean(all_ages)) if all_ages else None,
            p95_information_age_s=float(np.percentile(all_ages, 95)) if all_ages else None,
            mean_tracking_error_m=float(np.mean(all_errors)) if all_errors else None,
            final_gates=info.stage_success_gates,
            red_incident=env.red_incident,team_return=float(episode_return.mean()),agent_returns=episode_return.tolist(),
            reward_audit=reward_audit,phase_reward_audit=phase_rewards,
            terminal_discount_by_gamma={str(gamma):float(gamma**(env.step_count-1)*info.reward_terms['team_terminal_reward']) for gamma in (.99,.995,.9985)},
            minimum_blue_clearance_m=minimum_clearance if math.isfinite(minimum_clearance) else None,
            safety_intervention_fraction=(intervention_count/max(env.step_count,1)).tolist(),
            safety_infeasible_fraction=(infeasible_count/max(env.step_count,1)).tolist(),
            counterfactual_simulation_steps=env.counterfactual_simulation_steps,
            counterfactual_wall_time_s=env.counterfactual_wall_time_s,
            counterfactual_audit_count=len(counterfactual_events),
            counterfactual_cooperation_supported_count=sum(e['cooperation_supported'] for e in counterfactual_events),
            wall_time_s=time.perf_counter() - started)
        event_rows.extend(dict(mode=mode, seed=seed, **event) for event in env.events)
        event_rows.extend(counterfactual_events)
        return summary, trace, event_rows
    finally:
        env.close()


def _color(value): return np.asarray(value, dtype=float) / 255.


def _base_map(config, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axis = plt.subplots(figsize=(7.6, 7.2), facecolor=_color(WORLD_BACKGROUND_FILL))
    axis.set(xlim=(0, 55), ylim=(0, 55), aspect="equal", xlabel="X (m)", ylabel="Y (m)", title=title)
    axis.set_xticks(np.arange(0, 56, 5)); axis.set_yticks(np.arange(0, 56, 5))
    axis.grid(color=_color(GRID_LINE_COLOR), linewidth=.8, zorder=0)
    for reef in config["reefs"]:
        axis.add_patch(Circle(reef["center"], reef["radius"], facecolor=_color(ISLAND_FILL),
                             edgecolor=_color(ISLAND_OUTLINE), linewidth=1.4, zorder=3))
        axis.add_patch(Circle(reef["center"], reef["radius"] + config["depth"]["reef_shallow_buffer_m"],
                             fill=False, linestyle="--", edgecolor=_color(ISLAND_OUTLINE), linewidth=.8, zorder=3))
        axis.text(*reef["center"], reef["name"], ha="center", va="center", color="white", weight="bold")
    zone = config["protected_zone"]
    axis.add_patch(Circle(zone["center"], zone["radius"], facecolor=_color(PROTECTED_ZONE_OUTLINE),
                         alpha=.12, edgecolor=_color(PROTECTED_ZONE_OUTLINE), linewidth=1.6, zorder=2))
    axis.text(zone["center"][0], zone["center"][1], "Protected", ha="center", va="center", fontsize=8)
    return fig, axis


def plot_geometry(config, output):
    import matplotlib.pyplot as plt
    fig, axis = _base_map(config, "Standard deployment and shallow-water contours")
    ns=config['env']['num_surface_blue']; na=config['env']['num_submarines']
    colors = [BLUE_SHIP_FILL, BLUE_SHIP_2_FILL][:ns] + [SUBMERGED_SUBMARINE_FILL] * na + [RED_SHIP_FILL]
    points = config["scenario"]["blue_starts"] + config["scenario"]["submarine_starts"] + [config["scenario"]["red_start"]]
    labels=[f'B{i+1}' for i in range(ns)]+[f'Y{i+1}' for i in range(na)]+['Red']
    for label, point, color in zip(labels, points, colors):
        axis.scatter(*point, s=42, color=_color(color), edgecolor="white", linewidth=.5, zorder=5)
        axis.annotate(label, point, xytext=(5, 5), textcoords="offset points", fontsize=9)
    fig.tight_layout(); fig.savefig(output / "scene_geometry.png", dpi=170); plt.close(fig)


def plot_trace(config, mode, trace, output):
    import matplotlib.pyplot as plt
    if not trace: return
    ns=config['env']['num_surface_blue']; na=config['env']['num_submarines']; count=ns+na
    fig, axis = _base_map(config, f"{mode}: seed {trace[0]['seed']} / {trace[-1]['terminal_reason']}")
    axis.plot([r["red_x"] for r in trace], [r["red_y"] for r in trace], color=_color(RED_SHIP_FILL), label="Red", linewidth=1.8)
    axis.plot([r["shadow_x"] for r in trace], [r["shadow_y"] for r in trace], color=_color(SHADOW_RED_COLOR), linestyle="--", label="Shadow red", linewidth=1.4)
    if mode != "no_blue":
        colors = [BLUE_SHIP_FILL, BLUE_SHIP_2_FILL][:ns] + [SUBMERGED_SUBMARINE_FILL] * na
        for i, color in enumerate(colors):
            axis.plot([r[f"agent_{i}_x"] for r in trace], [r[f"agent_{i}_y"] for r in trace], color=_color(color), linewidth=1.1, alpha=.9)
            last = trace[-1]
            axis.annotate(f"{'B' if i < ns else 'Y'}{i + 1 if i < ns else i - ns + 1}",
                          (last[f"agent_{i}_x"], last[f"agent_{i}_y"]), fontsize=8)
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(output / f"{mode}_trajectory.png", dpi=170); plt.close(fig)
    times = [r["time_s"] for r in trace]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for key, label in (("actual_distance_to_protection_m", "Actual"), ("shadow_distance_to_protection_m", "Shadow")):
        axes[0].plot(times, [r[key] for r in trace], label=label)
    axes[0].set_ylabel("Distance to zone center (m)"); axes[0].legend(fontsize=8)
    for i in range(ns, count):
        error = [np.nan if r[f"agent_{i}_tracking_error_m"] is None else r[f"agent_{i}_tracking_error_m"] for r in trace]
        axes[1].plot(times, error, label=f"AUV{i - ns + 1}")
        detected = [r[f"auv_{i - ns}_detected"] for r in trace]
        for row, hit in zip(trace, detected):
            if hit: axes[1].plot(row["time_s"], 0., ".", color="gray", markersize=1.5)
    axes[1].set_ylabel("Estimate error (m)"); axes[1].legend(fontsize=8)
    for i in range(count):
        ages = [np.nan if r[f"agent_{i}_age_s"] is None else r[f"agent_{i}_age_s"] for r in trace]
        axes[2].plot(times, ages, linewidth=.8, label=f"{'USV' if i < ns else 'AUV'}{i + 1 if i < ns else i - ns + 1}")
    axes[2].set_ylabel("Observation age (s)"); axes[2].legend(fontsize=7, ncol=3)
    axes[3].plot(times, [r["shadow_gap_m"] for r in trace], label="Actual-shadow distance gap (m)")
    axes[3].plot(times, [r["hold_seconds"] for r in trace], label="Continuous hold (s)")
    axes[3].set_ylabel("Gap / hold"); axes[3].set_xlabel("Time (s)"); axes[3].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / f"{mode}_diagnostics.png", dpi=170); plt.close(fig)
    if mode!='no_blue':
        fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True)
        for i in range(count):
            label=f"{'USV' if i<ns else 'AUV'}{i+1 if i<ns else i-ns+1}"
            axes[0].plot(times,[r[f'agent_{i}_ground_speed_mps'] for r in trace],label=label)
            axes[1].plot(times,[r[f'agent_{i}_yaw_rate_radps'] for r in trace],label=label)
            axes[2].plot(times,[r['executed_actions'][i][0] for r in trace],label=label)
            axes[3].plot(times,[r['executed_actions'][i][1] for r in trace],label=label)
        for axis,label in zip(axes,['Ground speed (m/s)','Yaw rate (rad/s)','Surge force (N)','Yaw moment (N m)']):
            axis.set_ylabel(label); axis.grid(alpha=.25)
        axes[0].legend(ncol=count,fontsize=7); axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(); fig.savefig(output/f'{mode}_motion.png',dpi=170); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument('--experiment',default='Base')
    parser.add_argument("--red-episodes", type=int, default=20)
    parser.add_argument("--rule-episodes", type=int, default=10)
    parser.add_argument("--ideal-episodes", type=int, default=1)
    parser.add_argument('--random-episodes',type=int,default=0)
    parser.add_argument('--stationary-episodes',type=int,default=0)
    parser.add_argument('--cooperation-ablation-episodes',type=int,default=0)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", default="validation/scene")
    args = parser.parse_args(argv)
    if min(args.red_episodes, args.rule_episodes, args.ideal_episodes, args.random_episodes,
           args.stationary_episodes,args.cooperation_ablation_episodes,args.base_seed) < 0:
        parser.error("episode counts and base seed must be nonnegative")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"refuses nonempty output directory to preserve prior evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = assemble(args.config,experiment=args.experiment,seed=args.base_seed)
    source_hash = code_hash()
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    geometry = geometry_audit(config)
    rows, all_events, traces = [], [], {}
    modes=[('no_blue',args.red_episodes),('rules_local',args.rule_episodes),('rules_ideal',args.ideal_episodes),
           ('random',args.random_episodes),('stationary',args.stationary_episodes)]
    modes += [(mode,args.cooperation_ablation_episodes) for mode in ('usv_only','auv_only','without_usv_0','without_usv_1','without_auv_0',
        'peer_disabled','peer_delayed','peer_shuffled')]
    for mode, count in modes:
        for episode in range(count):
            row, trace, events = run_episode(config, mode, args.base_seed + episode, record=episode == 0)
            rows.append(row); all_events.extend(events)
            if trace: traces[mode] = trace
            print(json.dumps(_plain(row), ensure_ascii=False, allow_nan=False), flush=True)
    _write_csv(output / "episodes.csv", rows)
    _write_json(output / "events.json", all_events)
    for mode, trace in traces.items(): _write_csv(output / f"{mode}_trajectory.csv", trace)
    plot_geometry(config, output)
    for mode, trace in traces.items(): plot_trace(config, mode, trace, output)
    aggregate = {}
    for mode,_ in modes:
        selected = [row for row in rows if row["mode"] == mode]
        counts = Counter(row["outcome"] for row in selected)
        outcomes = {name: counts.get(name, 0) for name in ("success", "breakthrough", "collision", "reef_collision",
                    "out_of_bounds", "timeout", "scenario_anomaly", "numerical_error")}
        aggregate[mode] = dict(episodes=len(selected), outcomes=outcomes,
            success_count=sum(row["success"] for row in selected),
            finite_count=sum(row["finite"] for row in selected),
            all_three_auvs_discovered_count=sum(row["discovered_auv_count"] == 3 for row in selected),
            mean_episode_steps=float(np.mean([row["episode_steps"] for row in selected])) if selected else None)
        aggregate[mode]['red_incident_classification']=dict(Counter(row['red_incident'].get('category','none') for row in selected))
        aggregate[mode]['reward_by_outcome']={}
        for outcome in sorted(set(row['outcome'] for row in selected)):
            episodes=[row for row in selected if row['outcome']==outcome]
            aggregate[mode]['reward_by_outcome'][outcome]={name:float(np.mean([row['reward_audit'][name]['episode_team_mean'] for row in episodes])) for name in REWARD_COMPONENTS}
    report = dict(config_hashes=config["hashes"], source_code_hash=source_hash,
        resolved_config=config, controller_settings=RuleController().config, base_seed=args.base_seed, geometry=geometry,
        aggregate=aggregate, episodes=rows,
        evidence_scope="Recorded physical diagnostics only; ideal controller input is explicitly labelled. No learning-performance claim.",
        limitations=["Success requires three AUVs to enclose the target within the configured distance band continuously; USV blocking is auxiliary.",
                    "Ideal-information diagnostics use truth only in the labelled scripted controller; learned Actor input remains local.",
                    "Reef contours include the configured shallow-water buffer at the fixed AUV operating depth."])
    report['limitations'] += ['Current-state counterfactuals hold the current executed USV force/moment open loop; all/none/leave-one-out share the actual Red/navigation state and wave episode, but this is not closed-loop causal identification.',
        'Search coverage is unique shared 2-D sensor-footprint grid exposure, not a proof that stochastic sensing excluded a target from every visited cell.',
        'Gamma values in terminal_discount_by_gamma are arithmetic reward-horizon diagnostics; they are not a gamma hyperparameter training experiment.',
        'Deadline at 900 steps is the finite task horizon with remaining_time observed; it terminates value bootstrap. External collector cuts are separate truncations.']
    _write_json(output / "report.json", report)
    print(json.dumps(dict(report=str(output / "report.json"), aggregate=aggregate), ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__": main()
