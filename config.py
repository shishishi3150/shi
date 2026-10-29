"""Single assembly and identity boundary for the v4.3 task."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSION = "4.3"
PRESETS = {"Base": (0, 0, 0), "w_o_I1": (0, 1, 1), "w_o_I2": (1, 0, 1),
           "w_o_I3": (1, 1, 0), "Full": (1, 1, 1), "HyperOnly": (1, 1, 1),
           "UniformDownstream": (1, 1, 1),
           "RoleMean-GRU": (0, 0, 0), "Memoryless-MH": (0, 0, 0),
           "StaticGraph": (1, 1, 1), "NoAgeGate": (1, 1, 1), "NoAgeInputs": (1, 1, 1)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def named_seed(seed, namespace, index=0):
    return int(digest([int(seed), namespace, int(index)])[:8], 16)


def strict_merge(base, patch, prefix=""):
    if not isinstance(patch, dict):
        raise ValueError(f"{prefix} must be a mapping")
    for key, value in patch.items():
        path = f"{prefix}.{key}".strip(".")
        if key not in base:
            raise ValueError(f"unknown/unconsumed override: {path}")
        original = base[key]
        if isinstance(original, dict):
            strict_merge(original, value, path)
        else:
            if isinstance(original, bool) and type(value) is not bool:
                raise ValueError(f"{path} requires a boolean")
            if type(original) is int and type(value) is not int:
                raise ValueError(f"{path} requires an integer")
            if type(original) is float and (isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value)):
                raise ValueError(f"{path} requires a finite number")
            if isinstance(original,list):
                if not isinstance(value,list): raise ValueError(f"{path} requires a list")
                if original and all(type(x) is int for x in original) and any(type(x) is not int for x in value):
                    raise ValueError(f"{path} requires integer elements")
            base[key] = deepcopy(value)


def finite_tree(value, path="config"):
    if isinstance(value, dict):
        for k, v in value.items(): finite_tree(v, path + "." + k)
    elif isinstance(value, list):
        for v in value: finite_tree(v, path)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains NaN/Inf")


def default_components():
    return {"history_length": 8, "gru_hidden": 48, "critic_gru_hidden": 32,
            "node_hidden": 24, "encoder": "gncde", "age_gate": True, "age_inputs": True,
            "i2": {"lambda": .2, "warmup_steps": 20000, "ramp_steps": 50000,
                   "temperature": 1., "gradient_ratio": .3, "ema_decay": .99},
            "i3": {"rank": 4, "residual_gain": .5, "hyper_hidden": 32,
                   "warmup_steps": 30000, "interval_steps": 2048,
                   "budget_ratio": .25,
                   "gradient_ratio": .2}}


def parameter_teacher_defaults():
    return {"horizon_steps": 64, "verify_steps": 96, "finite_difference": .25,
            "trust_radius": .50, "solver_steps": 500, "slack_weight": 100.,
            "solver_tolerance": 1e-6, "max_backtracks": 4,
            "margin_tolerance_m": .002, "progress_tolerance_m": .01,
            "improvement_tolerance_m": .002, "numerical_tolerance_m": 1e-5}


def assemble(path=None, experiment="Base", seed=0, device=None):
    path = Path(path or ROOT / "configs/default.yaml").resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict) or spec.get("version") != VERSION:
        raise ValueError("v4.3 requires version: '4.3'; legacy config cannot be resumed")
    allowed = {"version", "template", "scene_side_m", "overrides", "components", "c_A", "evaluation"}
    if set(spec) - allowed: raise ValueError(f"unknown config keys: {set(spec)-allowed}")
    template = (path.parent / spec["template"]).resolve()
    cfg = yaml.safe_load(template.read_text(encoding="utf-8-sig"))
    overrides = deepcopy(spec.get("overrides", {}))
    allowed_sections={"training","replay_buffer","env","scenario","ship","wave","communication","island","red_policy","final_task","task"}
    if set(overrides)-allowed_sections: raise ValueError("override targets a frozen or diagnostic-only section")
    if set(overrides.get("final_task",{})) - {"max_steps","environment"}:
        raise ValueError("final_task success/timing aliases are derived from the single loose geometry contract")
    if set(overrides.get("task",{})) - {"encirclement","active_surface_mask"}:
        raise ValueError("old task success/pressure thresholds are diagnostic-only")
    if "prediction_horizon_seconds" in overrides.get("task",{}).get("encirclement",{}):
        raise ValueError("prediction horizon is derived from dt")
    if set(overrides.get("training",{})) & {"save_interval_episodes","eval_seed_offset"}:
        raise ValueError("unconsumed legacy training override")
    strict_merge(cfg, overrides)
    if cfg["env"]["max_steps"] != cfg["final_task"]["max_steps"]:
        raise ValueError("env.max_steps conflicts with final_task.max_steps")
    components = default_components()
    components["parameter_teacher"] = parameter_teacher_defaults()
    strict_merge(components, spec.get("components", {}), "components")
    if experiment not in PRESETS: raise ValueError(f"unknown experiment: {experiment}")
    bits = tuple(bool(x) for x in PRESETS[experiment])
    components.update(i1_enabled=bits[0], i2_enabled=bits[1],
                      hyper_enabled=bits[2],
                      teacher_enabled=bits[2] and experiment != "HyperOnly",
                      uniform_downstream=experiment == "UniformDownstream")
    if not bits[0]: components["encoder"] = "gru"
    if experiment == "Memoryless-MH": components["encoder"] = "mlp"
    if experiment == "StaticGraph": components["encoder"] = "static_graph"
    if experiment == "NoAgeGate": components["age_gate"] = False
    if experiment == "NoAgeInputs": components.update(age_inputs=False, age_gate=False)
    # Historical fields are required only by the geometry resolver/diagnostics.
    cfg["revision"] = {"i3": {"margin_tolerance_m": .001, "progress_tolerance_m": .05,
                                "acceptance_tolerance": 1e-6}}
    from .scene_config import scale_scene
    side_m = float(spec.get("scene_side_m", 55.))
    cfg = scale_scene(cfg, side_m)
    cfg["task"]["encirclement"]["prediction_horizon_seconds"] = cfg["env"]["dt"]
    gate = cfg["task"]["encirclement"]["shrink_tracking"]
    success = cfg["final_task"]["success"]
    success.update(max_slot_error=gate["max_slot_error_max"], mean_slot_error=gate["mean_slot_error_max"],
                   max_angle_gap_degrees=gate["max_angle_gap_degrees"], require_surface_intervention=False,
                   hold_seconds=0.)
    cfg["final_task"]["timing"]["final_hold_steps"] = 0
    cfg["revision"].update(version=VERSION, experiment=experiment, mask="".join(str(int(x)) for x in bits),
                           **components)
    cfg["revision"]["teacher_method"] = "paired_parameter_objective_v2"
    cfg["critic"] = {"head_mode": "role_mean" if experiment == "RoleMean-GRU" else "individual_multihead"}
    cfg["mechanisms"] = {"i1": bits[0], "i2": bits[1], "i3": {
        "hyper_enabled": components["hyper_enabled"], "teacher_enabled": components["teacher_enabled"]}}
    cfg["experiment"]["name"] = experiment
    cfg["diagnostic_legacy"] = {"not_consumed_by_training": ["innovations", "task.physical_capture",
        "task.surface_pressure.*history*", "final_task.success.mean_velocity_error", "final_task.evaluation",
        "final_task.max_episodes", "final_task.max_environment_steps", "checkpoint", "logging",
        "training.save_interval_episodes", "communication.mean_bad_burst_packets"],
        "surface_reward": cfg["reward"].pop("surface")}
    cfg["reward"].update(formulation="usv_role_tracking_cost_v1", implementation_revision=52,
        c_A=spec.get("c_A", 0.14256914496164447), learning_scale=1.,
        terminal={"success": 5., "landing": -5., "collision": -5., "out_of_bounds": -5., "timeout": -5.},
        surface={"tracking_length_m": 5., "tracking_cost_per_second": .20,
                 "tracking_weights": {"position": .6, "heading": .2, "velocity": .2},
                 "b1_progress_scale": .10, "b2_progress_scale": .10,
                 "risk_cost_per_second": .20, "clearance_scale_m": 2.5,
                 "half_angle_degrees": 20., "k_pos": .08, "max_speed": .4,
                 "motion_softening": 2., "heading_speed_scale": .05,
                 "heading_weight": 1., "velocity_weight": .5,
                 "prediction_seconds": 6., "safe_margin": .6 * cfg["env"]["world_size"][0] / 12.,
                 "inner": 4 * cfg["scene_geometry"]["scale"],
                 "outer": 10 * cfg["scene_geometry"]["scale"], "tail": 5 * cfg["scene_geometry"]["scale"],
                 "warning": 2.5 * cfg["scene_geometry"]["scale"]})
    cfg["training"].update(seed=int(seed), device=device or cfg["training"]["device"])
    cfg["scenario"]["seed"] = int(seed)
    cfg["evaluation"] = {"interval_steps": 100000, "episodes": 30, "base_seed": 1000000,
                         "test_episodes": 100, "test_base_seed": 2000000}
    strict_merge(cfg["evaluation"], spec.get("evaluation", {}), "evaluation")
    cfg["outputs"]["root"] = "outputs"
    cfg["schema"] = {"observation_version": VERSION, "replay_version": 43, "checkpoint_version": 43}
    cfg["provenance"] = {"task_file": Path(os.path.relpath(path, ROOT)).as_posix(),
                          "template": Path(os.path.relpath(template, ROOT)).as_posix(), "overrides": overrides,
                          "geometry_passes": 1, "code_version": VERSION}
    validate(cfg, allow_auto=True)
    refresh_hashes(cfg)
    return cfg


def validate(cfg, allow_auto=False):
    finite_tree(cfg)
    if cfg["revision"]["version"] != VERSION: raise ValueError("wrong task version")
    rev = cfg["revision"]
    # v1 remains readable for inference; training resume also checks source hash.
    if rev.get("teacher_method") not in ("paired_parameter_margin_v1", "paired_parameter_objective_v2"):
        raise ValueError("old action-teacher checkpoint: use the original release or start a new run")
    pc=rev["parameter_teacher"]
    if set(pc)!=set(parameter_teacher_defaults()): raise ValueError("invalid parameter teacher settings")
    for key,value in pc.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f"invalid parameter teacher setting: {key}")
    for key in ("horizon_steps","verify_steps","solver_steps","max_backtracks"):
        if type(pc[key]) is not int: raise ValueError(f"parameter teacher {key} must be integer")
    if pc["verify_steps"]<pc["horizon_steps"] or pc["trust_radius"]>1 or pc["finite_difference"]>1:
        raise ValueError("invalid parameter teacher horizon/trust region")
    if cfg["experiment"]["name"] != rev["experiment"]:
        raise ValueError("experiment display name differs from effective run identity")
    bits = PRESETS[rev["experiment"]]
    if (rev["mask"] != "".join(str(int(x)) for x in bits)
            or rev["i1_enabled"] != bool(bits[0]) or rev["i2_enabled"] != bool(bits[1])
            or rev["hyper_enabled"] != bool(bits[2])
            or rev["teacher_enabled"] != bool(bits[2] and rev["experiment"] != "HyperOnly")
            or cfg["mechanisms"] != {"i1": bool(bits[0]), "i2": bool(bits[1]), "i3": {
                "hyper_enabled": rev["hyper_enabled"], "teacher_enabled": rev["teacher_enabled"]}}):
        raise ValueError("experiment identity and effective switches disagree")
    from .reward import _validate_reward_config
    _validate_reward_config(cfg["reward"])
    if cfg["reward"]["learning_scale"] != 1 or "process_scale" in cfg["reward"]:
        raise ValueError("duplicate reward scaling")
    ca = cfg["reward"]["c_A"]
    if not (allow_auto and ca == "auto_once"):
        if isinstance(ca, bool) or not isinstance(ca, (int, float)) or not math.isfinite(ca) or ca <= 0:
            raise ValueError("c_A must be a frozen finite positive number")
    if cfg["env"]["max_steps"] != cfg["final_task"]["max_steps"]: raise ValueError("conflicting deadline")
    for section, names in ((cfg["training"], ("batch_size", "n_step_return", "policy_delay", "torch_num_threads")),
                           (cfg["revision"], ("history_length", "gru_hidden", "critic_gru_hidden", "node_hidden")),
                           (cfg["env"], ("max_steps",)), (cfg["replay_buffer"], ("capacity",))):
        for name in names:
            if type(section[name]) is not int or section[name] <= 0: raise ValueError(f"invalid integer: {name}")
    if not 2 <= cfg["revision"]["history_length"] <= 32: raise ValueError("invalid history length")
    if cfg["training"]["n_step_return"] not in (1, 3, 7): raise ValueError("n-step must be 1, 3 or 7")
    if not 0 < cfg["training"]["gamma"] < 1: raise ValueError("invalid gamma")
    rev=cfg["revision"]
    for key in ("i1_enabled","i2_enabled","hyper_enabled","teacher_enabled","age_gate","age_inputs"):
        if type(rev[key]) is not bool: raise ValueError(f"{key} must be boolean")
    for section in (rev["i2"],rev["i3"]):
        for key,value in section.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError(f"invalid mechanism number: {key}")
    for key in ("interval_steps","rank","hyper_hidden"):
        if type(rev["i3"][key]) is not int or rev["i3"][key]<=0: raise ValueError(f"invalid I3 integer {key}")
    if rev["i2"]["temperature"]<=0 or rev["i2"]["ramp_steps"]<=0: raise ValueError("invalid I2 divisor")
    if not 0<rev["i2"]["gradient_ratio"]<=.3 or not 0<rev["i3"]["gradient_ratio"]<=.2:
        raise ValueError("auxiliary gradient cap exceeds contract")
    if not 0<rev["i3"]["budget_ratio"]<=.25: raise ValueError("teacher budget exceeds contract")
    if len(cfg["training"]["hidden_sizes"])<2 or min(cfg["training"]["hidden_sizes"])<rev["i3"]["rank"]:
        raise ValueError("hidden layers must support hyper rank")
    for k in ("actor_lr", "critic_lr", "tau", "grad_clip_norm"):
        if cfg["training"][k] <= 0: raise ValueError(f"invalid {k}")
    from .waves import RegularLinearWave
    from .communication import CommunicationNetwork
    RegularLinearWave.sample(cfg["wave"], 0)
    CommunicationNetwork(5, cfg["communication"], 0)


def code_hash():
    files=[p for folder in ("usv_env","matd3","scripts") for p in sorted((ROOT/folder).glob("*.py"))]
    return digest({str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def refresh_hashes(cfg):
    from .history import SCHEMA
    cfg["observation_schema"]={"version":SCHEMA.version,"fields":list(SCHEMA.fields),
        "base_obs":SCHEMA.base_dim,"frame_dim":SCHEMA.frame_dim,"valid_frame_index":SCHEMA.base_dim,
        "actor_dim":SCHEMA.frame_dim*cfg["revision"]["history_length"],"critic_dim":96,
        "age_unit":"seconds","physical_time_unit":"seconds"}
    scenario = {k: v for k, v in cfg["scenario"].items() if k != "seed"}
    task = {k: cfg[k] for k in ("env", "ship", "wave", "communication", "island", "red_policy", "final_task")}
    task.update(scenario=scenario, encirclement=cfg["task"]["encirclement"])
    cfg["hashes"] = {"task": digest(task), "reward": digest(cfg["reward"]), "observation": SCHEMA.hash,
                     "architecture": digest({"revision": {k: v for k,v in cfg["revision"].items()
                         if k not in ("experiment", "mask")}, "critic": cfg["critic"],
                         "hidden_sizes": cfg["training"]["hidden_sizes"]})}
    return cfg["hashes"]


def run_directory(runs_root, experiment, seed):
    if experiment not in PRESETS or type(seed) is not int or seed < 0: raise ValueError("invalid run identity")
    return Path(runs_root or ROOT / "outputs").expanduser().resolve() / experiment / f"seed{seed}"

TASK_CONTRACT = "cooperative_blockade_matd3_v1"

def load_config(path=None):
    return assemble(path)

def load_revision(path=None, experiment="Base", seed=0, device=None):
    return assemble(path, experiment, seed, device)
