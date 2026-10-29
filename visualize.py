"""Play an actor-only policy using the same local-history environment as training."""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import argparse
from pathlib import Path
import torch
from matd3.checkpoint import load_policy
from scripts.train import make_env
from usv_env.rendering import Renderer
from usv_env.config import ROOT, PRESETS, assemble


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--preview", action="store_true", help="preview the current 55 m task with nominal cruise actions; no learned model")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--max-steps",type=int,default=None,help="optional external playback limit")
    p.add_argument("--experiment", choices=PRESETS, default="Full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--best", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--variant", choices=("fixed", "randomized", "training"), default="fixed")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args(argv)
    if args.episodes < 1:
        p.error("episodes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("max-steps must be positive")
    if args.preview and (args.checkpoint or args.best):
        p.error("preview cannot be combined with checkpoint/best")
    return args


def _run_episode(env, agent, *, seed: int, renderer: Renderer | None, max_steps=None):
    obs, _ = env.reset(seed=seed)
    done = False
    info = None
    if renderer is not None:
        renderer.reset_trails()
        renderer.render(env.render_frame())
    while not done:
        if max_steps is not None and env.step_count>=max_steps: break
        if renderer is not None and not renderer.pump_events():
            return info, True
        obs, _, _, done, info = env.step(agent.act(obs, explore=False))
        if renderer is not None:
            renderer.render(env.render_frame())
    return info, False


def _checkpoint_path(config: dict, args: argparse.Namespace) -> Path:
    root = Path(getattr(args,"runs_root",None) or config["outputs"]["root"])
    if not root.is_absolute():
        root = ROOT / root
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        root
        / args.experiment
        / f"seed{args.seed}"
        / ("best.pt" if args.best else "latest.pt")
    )
    return checkpoint if checkpoint.is_absolute() else ROOT / checkpoint


def _check_scene(cfg, args, path):
    expected = assemble()["env"]["world_size"]
    actual = cfg["env"]["world_size"]
    if actual != expected and args.checkpoint is None:
        raise SystemExit(
            f"Default checkpoint is a historical {actual} m model: {path}\n"
            f"Current task is {expected} m. Train a new model, use --preview for an untrained scene, "
            "or pass --checkpoint explicitly to play this historical model at its saved size."
        )


class _NominalPreview:
    """Fixed cruise control solely for inspecting the current scene."""
    def __init__(self, env):
        from usv_env.calibration import nominal_actions
        self.actions = nominal_actions(env) * env.agent_action_limits

    def act(self, obs, explore=False):
        return self.actions.copy()


def main(argv=None):
    args = parse_args(argv)
    path = None
    if args.preview:
        from usv_env.calibration import prepare
        cfg, _ = prepare(assemble(experiment=args.experiment, seed=args.seed),
                         args.runs_root or ROOT / "outputs", require_cache=True)
    else:
        path = _checkpoint_path({"outputs": {"root": "outputs"}}, args)
        try:
            agent, _ = load_policy(path, args.device)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"{exc}\nUse --preview to inspect the current scene without a trained model.") from exc
        cfg = agent.runtime_config
        _check_scene(cfg, args, path)
    torch.set_num_threads(cfg["training"]["torch_num_threads"])
    env = make_env(cfg, args.seed, args.variant)
    if args.preview:
        agent = _NominalPreview(env)
    renderer = None if args.headless else Renderer(env.world_size)
    mode = "UNTRAINED_PREVIEW" if args.preview else "CHECKPOINT_POLICY"
    print(f"mode={mode}, experiment={cfg['revision']['experiment']}, mask={cfg['revision']['mask']}, checkpoint={path}, scene={cfg['env']['world_size']} metres, "
          f"variant={args.variant}, policy_step={getattr(agent, 'global_env_steps', 'none')}")
    try:
        for episode in range(args.episodes):
            info, stopped = _run_episode(env, agent, seed=args.seed+episode, renderer=renderer,max_steps=args.max_steps)
            if stopped:
                break
            if info is not None:
                print(f"episode={episode+1} success={info.success} reason={info.terminal_reason}")
    finally:
        env.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
