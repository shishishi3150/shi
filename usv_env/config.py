"""Assemble the current task while keeping one config and output boundary."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_CONTRACT = "heterogeneous_reef_interception"
VERSION = "force_control_local_information_capture_mlp_v4"
PRESETS = {"Base": (0,0,0), "Full": (1,1,1), "w_o_I1": (0,1,1), "w_o_I2": (1,0,1), "w_o_I3": (1,1,0),
           "I1_only": (1,0,0), "I2_only": (0,1,0), "I3_only": (0,0,1), "Traditional_CV": (0,0,0),
           "Random_History": (1,1,1), "No_State_Matching": (1,1,1), "No_Safety_Screening": (1,1,1), "Hard_Max": (1,1,1)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def named_seed(seed, namespace, index=0):
    return int(digest([int(seed), namespace, int(index)])[:8], 16)


def strict_merge(base, patch, prefix=""):
    if not isinstance(patch, dict): raise ValueError(f"{prefix} must be a mapping")
    for key, value in patch.items():
        name = f"{prefix}.{key}".strip(".")
        if key not in base: raise ValueError(f"unknown override: {name}")
        if isinstance(base[key], dict): strict_merge(base[key], value, name)
        else: base[key] = deepcopy(value)


def finite_tree(value, path="config"):
    if isinstance(value, dict):
        for k, v in value.items(): finite_tree(v, path + "." + k)
    elif isinstance(value, (list, tuple)):
        for v in value: finite_tree(v, path)
    elif isinstance(value, float) and not math.isfinite(value): raise ValueError(f"{path} contains NaN/Inf")


def assemble(path=None, experiment="Base", seed=0, device=None):
    source = Path(path or ROOT / "configs/default.yaml").resolve()
    spec = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict) or set(spec) - {"template", "components", "overrides", "evaluation"}:
        raise ValueError("unsupported config keys; historical policies require their original source")
    template = (source.parent / spec["template"]).resolve()
    cfg = yaml.safe_load(template.read_text(encoding="utf-8-sig"))
    strict_merge(cfg, spec.get("overrides", {}))
    if experiment not in PRESETS: raise ValueError(f"unknown experiment: {experiment}")
    components = {"history_length": 1}
    strict_merge(components, spec.get("components", {}))
    i1,i2,i3 = PRESETS[experiment]
    cfg["revision"] = dict(version=VERSION, experiment=experiment, mask=f"{i1}{i2}{i3}", encoder="mlp", **components)
    cfg["experiment"] = {"name": experiment}
    cfg["critic"] = {"head_mode": "individual_multihead"}
    cfg["mechanisms"] = {"safety_projection": bool(i1), "age_uncertainty_compensation": bool(i2), "joint_action_enhancement": bool(i3)}
    cfg["information"]["mode"] = "age_uncertainty" if i2 else ("cv" if experiment == "Traditional_CV" else "hold")
    cfg["i3"]["variant"] = {"Random_History":"random_history", "No_State_Matching":"no_state_matching", "No_Safety_Screening":"no_safety_screening", "Hard_Max":"hard_max"}.get(experiment,"conservative")
    cfg["training"].update(seed=int(seed), device=device or cfg["training"].get("device", "auto"))
    cfg["scenario"]["seed"] = int(seed)
    cfg["schema"] = {"observation_version": VERSION, "replay_version": 17, "checkpoint_version": VERSION}
    cfg["evaluation"] = {"interval_steps": 100000, "episodes": 30, "base_seed": 1000000,
                         "test_episodes": 100, "test_base_seed": 2000000}
    strict_merge(cfg["evaluation"], spec.get("evaluation", {}))
    cfg["provenance"] = {"task_file": os.path.relpath(source, ROOT).replace("\\", "/"),
                         "template": os.path.relpath(template, ROOT).replace("\\", "/"), "code_version": VERSION}
    validate(cfg)
    refresh_hashes(cfg)
    return cfg


def validate(cfg, allow_auto=False):
    finite_tree(cfg)
    if cfg["task"]["contract"] != TASK_CONTRACT: raise ValueError("wrong task contract")
    task=cfg['task']
    if task.get('objective')!='auv_encirclement':
        raise ValueError('task objective differs: current task requires AUV encirclement; historical policies require their original source')
    if (not 0 < task['capture_min_radius_m'] < task['capture_max_radius_m']
            or not 0 < task['capture_min_hull_margin_m'] < task['capture_max_radius_m']
            or not 120 <= task['maximum_observer_angle_gap_degrees'] < 180
            or task['minimum_auv_observers']!=3 or task['success_hold_seconds']<=0):
        raise ValueError('invalid three-AUV encirclement criteria')
    env = cfg["env"]
    post_capture_end = task.get('post_capture_end_step', 0)
    if type(post_capture_end) is not int or not 0 <= post_capture_end <= env['max_steps']:
        raise ValueError('task.post_capture_end_step must be an integer in [0, max_steps]')
    if env["num_surface_blue"] != 2 or env["num_submarines"] != 3: raise ValueError("task requires 2 USV + 3 AUV")
    if cfg['ship'].get('allow_reverse') is not False:
        raise ValueError('current task forbids reverse motion for every vessel: ship.allow_reverse must be false; historical tasks require their original source')
    if env["world_size"] != [55., 55.] or env["dt"] != .15 or env["max_steps"] != 900:
        raise ValueError("fixed benchmark requires 55x55 m, dt=.15 s, 900 steps")
    if len(cfg["reefs"]) != 3: raise ValueError("three reef contours required")
    for reef in cfg["reefs"]:
        if len(reef["center"]) != 2 or reef["radius"] <= 0: raise ValueError("invalid reef")
    if len(cfg["scenario"]["blue_starts"]) != 2 or len(cfg["scenario"]["submarine_starts"]) != 3:
        raise ValueError("invalid deployment")
    if cfg["replay_buffer"]["save_in_checkpoint"] is not False: raise ValueError("replay persistence is disabled")
    if cfg["revision"]["experiment"] != cfg["experiment"]["name"]: raise ValueError("experiment identity mismatch")
    for key in ("batch_size", "n_step_return", "policy_delay", "torch_num_threads"):
        if type(cfg["training"][key]) is not int or cfg["training"][key] <= 0: raise ValueError(f"invalid {key}")
    if not 0 < cfg["training"]["gamma"] < 1: raise ValueError("invalid gamma")
    if 'scheme' in cfg['reward']:
        from .reward import validate_settings
        validate_settings(cfg['reward'], cfg['training']['gamma'], cfg['env']['max_steps'])
    revision = cfg['revision']
    if (revision.get('encoder') != 'mlp' or type(revision.get('history_length')) is not int
            or revision['history_length'] != 1
            or {'gru_hidden', 'critic_gru_hidden'} & set(revision)):
        raise ValueError('current backbone requires a single-frame MLP without recurrent/GRU components; historical policies require their original source')
    if cfg["sensors"].get("false_alarm_rate", 0) != 0 or cfg["sensors"].get("acoustic_occlusion", False):
        raise ValueError("false alarms and acoustic occlusion are not implemented")
    from .target_tracking import information_mode
    information_mode(cfg)
    if not 1 <= int(cfg['i1']['horizon_steps']) <= 40 or cfg['i1']['margin_m'] < 0:
        raise ValueError('invalid predictive safety horizon/margin')
    if int(cfg['i1']['candidate_levels']) < 2: raise ValueError('safety needs multiple actuator candidates')
    safety=cfg['i1']
    # Old actor-only policies carry the old predictor config. Missing backup
    # fields preserve that recorded configuration instead of silently upgrading it.
    if type(safety.get('backup_steps',0)) is not int or not 0 <= safety.get('backup_steps',0) <= 240:
        raise ValueError('invalid predictive safety backup horizon')
    for key in ('stop_speed_mps','stop_yaw_rate_rps'):
        if key in safety and not 0 < safety[key] <= .1: raise ValueError(f'invalid i1.{key}')
    if cfg['i3'].get('return_scale',1.)<=0: raise ValueError('i3.return_scale must be positive')
    if cfg['i3']['advantage_clip'] < 0 or not 0 <= cfg['i3']['enhancement_weight'] <= 1:
        raise ValueError('invalid conservative history enhancement bounds')
    from .communication import CommunicationNetwork, TargetMessageBus
    from .scene_config import validate_scene
    validate_scene(cfg)
    CommunicationNetwork(5, cfg["communication"], 0)
    TargetMessageBus(5, cfg["communication"], 0, env["dt"])
    communication = cfg["communication"]
    if any(value != communication["loss_rate"] for value in communication["target_loss_range"]):
        raise ValueError("communication.loss_rate and target_loss_range must specify the same fixed loss rate")


def refresh_hashes(cfg):
    from .history import schema_for_counts
    schema = schema_for_counts(cfg["env"]["num_surface_blue"] + cfg["env"]["num_submarines"])
    from .observation import critic_state_dim
    cfg["observation_schema"] = dict(version=schema.version, fields=list(schema.fields), base_obs=schema.base_dim,
        frame_dim=schema.frame_dim, valid_frame_index=schema.base_dim, actor_dim=schema.frame_dim*cfg["revision"]["history_length"],
        critic_dim=critic_state_dim(5, len(cfg["reefs"])), age_unit="seconds", physical_time_unit="seconds")
    scenario = {k: v for k, v in cfg["scenario"].items() if k != "seed"}
    task = {k: cfg[k] for k in ("task", "env", "ship", "wave", "communication", "sensors", "depth", "reefs", "protected_zone", "red_policy", "counterfactual")}
    task["scenario"] = scenario
    # PBRS depends on training.gamma; its identity must include the discount.
    # Preserve historical hash verification for old CSV/config artifacts.
    reward_identity = dict(parameters=cfg['reward'], gamma=cfg['training']['gamma']) if 'scheme' in cfg['reward'] else cfg['reward']
    cfg["hashes"] = dict(task=digest(task), reward=digest(reward_identity), observation=schema.hash,
        architecture=digest({"backbone": {k: cfg["revision"].get(k) for k in ("encoder","history_length","gru_hidden","critic_gru_hidden")},
            "critic": cfg["critic"], "hidden_sizes": cfg["training"]["hidden_sizes"],
            "activation": cfg["training"]["activation"]}))
    return cfg["hashes"]


def code_hash():
    files = [p for folder in ("usv_env", "matd3", "scripts") for p in sorted((ROOT/folder).glob("*.py"))]
    return digest({str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def run_directory(runs_root, experiment, seed):
    if experiment not in PRESETS or type(seed) is not int or seed < 0: raise ValueError("invalid run identity")
    return Path(runs_root or ROOT / "outputs").expanduser().resolve() / experiment / f"seed{seed}"


load_revision = assemble
