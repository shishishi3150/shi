"""Comparable experimental conditions, separate from named mechanism ablations."""
from copy import deepcopy

from usv_env.config import ROOT, assemble


def default_runs_root():
    path = ROOT / assemble()["outputs"]["root"]
    return path.resolve()


def validate_learning_capacity(cfg):
    capacity = cfg["replay_buffer"]["capacity"]
    batch = cfg["training"]["batch_size"]
    if type(capacity) is not int or capacity <= 0 or capacity < batch:
        raise ValueError("Replay capacity must be a positive integer at least batch_size; this configuration cannot learn")


def comparison_contract(cfg, source=None):
    """Keep shared conditions; exclude only declared ablation selectors/seed/device.

    Base/Full and named I3 variants may differ in mechanism flags and variant.
    They may not silently differ in network, safety settings or search budgets.
    """
    training = {key: value for key, value in cfg["training"].items() if key not in ("seed", "device")}
    history = {key: value for key, value in cfg["i3"].items() if key != "variant"}
    initialization = cfg.get("initialization", {"kind": "new_training"})
    initialization = {key: value for key, value in initialization.items() if key != "source"}
    return deepcopy(dict(
        hashes={key: cfg["hashes"][key] for key in ("task", "reward", "observation", "architecture", "i1")},
        history_length=cfg["revision"]["history_length"],
        source=source if source is not None else cfg.get("provenance", {}).get("code_hash"),
        purpose=cfg.get("purpose", "formal_training"), training=training,
        replay=cfg["replay_buffer"], history=history,
        evaluation=cfg.get("evaluation"), budget=cfg.get("training_budget"), initialization=initialization))
