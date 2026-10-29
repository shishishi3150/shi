"""No-learning, stratified shared scale preparation. Never constructs Agent/Replay."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import inspect
import json
import os
import time
import uuid
import numpy as np
from .config import digest, named_seed, refresh_hashes, validate
from .environment import PursuitEvasionEnv
from .reward import original_auv
from . import reward as auv_kernel

PROTOCOL = {"version": 43, "seeds": [910001, 910002, 910003, 910004],
            "modes": ["fixed", "randomized"], "probes": ["zero", "cruise", "mixture"],
            "steps": 128, "hold_steps": 5, "quantile_method": "linear", "budget_factor": .8}
ORDINARY = ("position", "heading", "speed", "alignment", "cooperation", "yaw_rate", "time")


def nominal_actions(env):
    normalized = np.zeros((5, 2), dtype=np.float32)
    for sl, dynamics, speed in ((slice(0, 2), env.blue_dynamics, .30), (slice(2, 5), env.submarine_dynamics, .40)):
        velocity = np.asarray([speed, 0., 0.])
        equilibrium = dynamics.damping_matrix(velocity) @ velocity
        normalized[sl, 0] = equilibrium[0] / env.agent_action_limits[sl, 0]
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) >= 1):
        raise ValueError("nominal cruise exceeds normalized actuator range")
    return normalized


def mixture_action(rng, cruise):
    return (rng.uniform(-1, 1, cruise.shape) if rng.random() < .5 else
            np.clip(cruise + rng.normal(0, .20, cruise.shape), -1, 1)).astype(np.float32)


def dependency_hash(cfg):
    task = {k: deepcopy(cfg[k]) for k in ("env", "ship", "wave", "communication", "island", "red_policy", "final_task", "task", "scenario")}
    task["scenario"].pop("seed", None)
    return digest({"task": task, "weights": cfg["reward"]["weights"], "normalization": cfg["reward"]["normalization"],
        "gamma": cfg["training"]["gamma"], "F": abs(cfg["reward"]["terminal"]["collision"]), "protocol": PROTOCOL,
        "kernel": [inspect.getsource(getattr(auv_kernel,n)) for n in
                   ("_dense_components", "_weighted_dense", "_angular_formation_quality", "_spacing_penalty", "_safety_penalty")],
        "cooperation": inspect.getsource(original_auv),
        "environment_version": "transactional_loose_fixed_heading_v43"})


def coefficient(q75, gamma=.998, failure=12.):
    if not np.isfinite(q75) or q75 <= 1e-8: raise ValueError("calibration FAIL: Q75 is not positive")
    reference = .8 * (1-gamma) * failure
    return min(1., reference/q75), reference


def quantiles(values):
    q = np.quantile(values, [.5, .75, .95], method="linear")
    return dict(zip(("q50", "q75", "q95"), map(float, q)))


def probe(cfg):
    start = time.perf_counter()
    config = deepcopy(cfg)
    config["reward"]["c_A"] = 1.
    config["purpose"] = "reward_scale_probe"
    layers, runs, all_ordinary, dangerous, total, progress = [], [], [], [], [], []
    actual_steps = 0
    for mode in PROTOCOL["modes"]:
        for name in PROTOCOL["probes"]:
            ordinary = []
            for seed in PROTOCOL["seeds"]:
                env = PursuitEvasionEnv(config, seed)
                env.set_evaluation_variant(mode)
                env.reset(seed=seed)
                cruise = nominal_actions(env)
                rng = np.random.default_rng(named_seed(seed, "reward-probe:"+mode+":"+name))
                process_return, event_return = np.zeros(5), 0.
                for step in range(PROTOCOL["steps"]):
                    if name == "zero": action = np.zeros_like(cruise)
                    elif name == "cruise": action = cruise
                    elif step % PROTOCOL["hold_steps"] == 0: action = mixture_action(rng, cruise)
                    _, _, rewards, done, info = env.step(action * env.agent_action_limits)
                    actual_steps += 1
                    parts = info.reward_terms["raw_auv_contributions"]
                    for key in ORDINARY:
                        if not np.isfinite(parts[key]).all() or np.any(parts[key] > 1e-10):
                            raise ValueError(f"calibration FAIL: original {key} sign/nonfinite error")
                    if not done:
                        cost = np.maximum(-sum(parts[k] for k in ORDINARY), 0.)
                        ordinary.extend(cost.tolist())
                        dangerous.extend((-(parts["spacing"]+parts["safety"])).tolist())
                        total.extend(sum(parts.values()).tolist())
                        progress.extend(parts["progress"].tolist())
                    event = info.reward_terms["team_terminal_reward"]
                    process_return += rewards-event
                    event_return += event
                    if done: break
                runs.append({"seed": seed, "mode": mode, "probe": name, "steps": step+1,
                             "process_by_agent_raw_auv": process_return.tolist(), "terminal": event_return,
                             "outcome": info.terminal_reason})
                del env
            if len(ordinary) < 32: raise ValueError(f"calibration FAIL: {mode}/{name} has {len(ordinary)} samples")
            layers.append({"mode": mode, "probe": name, "samples": len(ordinary),
                           "q75": float(np.quantile(ordinary, .75, method="linear"))})
            all_ordinary.extend(ordinary)
    q75 = float(np.median([layer["q75"] for layer in layers]))
    ca, reference = coefficient(q75, cfg["training"]["gamma"], abs(cfg["reward"]["terminal"]["collision"]))
    for run in runs:
        p = run["process_by_agent_raw_auv"]
        run["process_by_agent_final"] = p[:2] + [ca*x for x in p[2:]]
    return {"status": "PASS", "protocol": PROTOCOL, "dependency_hash": dependency_hash(cfg),
        "actual_environment_steps": actual_steps, "actor_updates": 0, "critic_updates": 0, "replay_transitions": 0,
        "layers": layers, "Q75": q75, "c_A": ca, "reference_step_cost": reference, "hit_upper_cap": ca == 1.,
        "ordinary_raw": quantiles(all_ordinary), "ordinary_final": quantiles(np.asarray(all_ordinary)*ca),
        "dangerous_raw": quantiles(dangerous), "dangerous_final": quantiles(np.asarray(dangerous)*ca),
        "progress_raw": quantiles(progress), "total_process_raw": quantiles(total),
        "dangerous_above_reference_fraction": float(np.mean(np.asarray(dangerous)*ca > reference)),
        "runs": runs, "wall_time_s": time.perf_counter()-start,
        "interpretation": "USV unchanged; AUV alone rescaled. Ordinary cost anchor is not a safety or trajectory-ranking guarantee."}


@contextmanager
def exclusive_lock(path, wait_seconds=120):
    deadline = time.monotonic()+wait_seconds
    while True:
        try: fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); break
        except FileExistsError:
            if time.monotonic() >= deadline: raise RuntimeError(f"calibration lock busy: {path}; inspect owner before removing stale lock")
            time.sleep(.1)
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        yield
    finally: Path(path).unlink()


def prepare(cfg, runs_root, *, require_cache=False):
    cfg = deepcopy(cfg)
    if cfg["reward"]["c_A"] != "auto_once":
        # Numeric frozen scale is independent of terminal magnitude. Normal
        # validation and semantic hashes remain mandatory for save/resume.
        validate(cfg)
        refresh_hashes(cfg)
        cfg["calibration"] = {"mode": "frozen_numeric", "shared_environment_steps": 0,
                              "hashes": dict(cfg["hashes"])}
        return cfg, {"status": "FROZEN", "c_A": cfg["reward"]["c_A"], "Q75": None,
                     "actual_environment_steps": 0}
    raise ValueError("auto_once is audit-only; formal startup requires a frozen numeric c_A; use probe explicitly")
