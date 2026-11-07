"""Play an actor-only policy using the same local-history environment as training."""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import argparse
import time
from pathlib import Path
import torch
from matd3.checkpoint import load_policy
from scripts.train import make_env
from usv_env.rendering import Renderer
from usv_env.config import ROOT, PRESETS, assemble


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--preview", action="store_true", help="preview the current 55 m task with zero blue actuator commands")
    p.add_argument("--rules", action="store_true", help="run the current rule controller using legal local estimates; no checkpoint")
    p.add_argument("--speed", type=float, default=None, help="playback speed relative to simulation time (e.g. 1 or 2)")
    p.add_argument("--start-paused", action="store_true", help="wait for Space before starting each episode")
    p.add_argument("--hold-end", action="store_true", help="keep the last frame until Space; Esc closes the window")
    p.add_argument("--show-withdrawal", action="store_true",
                   help="legacy immediate-capture configs only: append a separate rule withdrawal; current configs include withdrawal in the episode")
    p.add_argument("--runs-root", default="outputs/positive_reward")
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
    if args.preview and args.rules:
        p.error("preview and rules are separate playback modes")
    if (args.preview or args.rules) and (args.checkpoint or args.best):
        p.error("preview/rules cannot be combined with checkpoint/best")
    if args.speed is not None and (not 0. < args.speed < float('inf')):
        p.error("speed must be finite and positive")
    if args.headless and (args.start_paused or args.hold_end):
        p.error("pause/hold requires a visible window")
    return args


def _wait_for_space(renderer):
    pygame = renderer.pygame
    while True:
        event = pygame.event.wait(100)
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            return True


def _run_episode(env, agent, *, seed: int, renderer: Renderer | None, max_steps=None,
                 speed=None, start_paused=False, hold_end=False, show_withdrawal=False):
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    done = False
    info = None
    if renderer is not None:
        renderer.reset_trails()
        renderer.render(env.render_frame())
        if start_paused and not _wait_for_space(renderer):
            return info, True
    started = time.perf_counter()
    playback_steps = 0
    while not done:
        if max_steps is not None and env.step_count>=max_steps: break
        if renderer is not None and not renderer.pump_events():
            return info, True
        obs, _, _, done, info = env.step(agent.act(obs, explore=False))
        playback_steps += 1
        if renderer is not None:
            renderer.render(env.render_frame())
            if speed is not None:
                remaining = started + playback_steps * env.dt / speed - time.perf_counter()
                while remaining > 0.:
                    if not renderer.pump_events():
                        return info, True
                    time.sleep(min(.02, remaining))
                    remaining = started + playback_steps * env.dt / speed - time.perf_counter()
    if (show_withdrawal and info is not None and getattr(info, 'success', False)
            and not getattr(env, 'post_capture_end_step', 0)):
        summary, closed = _run_withdrawal(env, renderer=renderer, speed=speed,
                                         controller=getattr(agent, 'controller', None))
        print(f"capture_success=True withdrawal={summary['status']} "
              f"verified={summary['withdrawal_verified']} elapsed={summary['elapsed_seconds']:.2f}s")
        if closed:
            return info, True
    if renderer is not None and hold_end and not _wait_for_space(renderer):
        return info, True
    return info, False


def _run_withdrawal(env, *, renderer=None, speed=None, record_frame=None, controller=None):
    from usv_env.withdrawal import WithdrawalReplay
    replay = WithdrawalReplay(env, controller=controller)
    if renderer is not None:
        renderer.render(replay.render_frame())
    if record_frame is not None:
        record_frame(replay)
    started = time.perf_counter()
    while not replay.done:
        if renderer is not None and not renderer.pump_events():
            return replay.trace_row(), True
        replay.step()
        if renderer is not None:
            renderer.render(replay.render_frame())
        if record_frame is not None:
            record_frame(replay)
        if renderer is not None and speed is not None:
            remaining = started + replay.elapsed_seconds / speed - time.perf_counter()
            while remaining > 0.:
                if not renderer.pump_events():
                    return replay.trace_row(), True
                time.sleep(min(.02, remaining))
                remaining = started + replay.elapsed_seconds / speed - time.perf_counter()
    return replay.trace_row(), False


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
    """Zero blue actuator commands solely for inspecting the current scene."""
    def __init__(self, env):
        import numpy as np
        self.actions = np.zeros_like(env.agent_action_limits)

    def act(self, obs, explore=False):
        return self.actions.copy()


class _LocalRules:
    """Rules use the same receiver estimates as the scene acceptance runner."""
    def __init__(self, env):
        from usv_env.red_policy import RuleController
        self.env = env
        self.controller = RuleController()

    def reset(self):
        self.controller.reset()

    def act(self, obs, explore=False):
        env = self.env
        return self.controller.actions(env.blue_states, env.submarine_states, None,
            env.protected_center, env.reefs, env.world_size, env.blue_dynamics,
            env.submarine_dynamics, env.step_count * env.dt,
            target_estimates=env.target_estimates, active_mask=env.agent_active_mask())


def main(argv=None):
    args = parse_args(argv)
    path = None
    if args.preview or args.rules:
        cfg = assemble(experiment=args.experiment, seed=args.seed)
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
    elif args.rules:
        agent = _LocalRules(env)
    speed = args.speed if args.speed is not None else (1. if args.rules else None)
    renderer = None if args.headless else Renderer(env.world_size,
        fps=0 if speed is not None else cfg['visual']['fps'],
        trail_length=cfg['visual']['trail_length'])
    if renderer is not None and (args.rules or args.start_paused or args.hold_end):
        renderer.pygame.display.set_caption("Reef interception | Local rules | Space: start/next | Esc: exit"
            if args.rules else "Reef interception | Space: start/next | Esc: exit")
    mode = "RULE_CONTROLLER_LOCAL" if args.rules else ("UNTRAINED_PREVIEW" if args.preview else "CHECKPOINT_POLICY")
    print(f"mode={mode}, experiment={cfg['revision']['experiment']}, mask={cfg['revision']['mask']}, checkpoint={path}, scene={cfg['env']['world_size']} metres, "
          f"variant={args.variant}, policy_step={getattr(agent, 'global_env_steps', 'none')}")
    try:
        for episode in range(args.episodes):
            info, stopped = _run_episode(env, agent, seed=args.seed+episode, renderer=renderer,
                max_steps=args.max_steps, speed=speed, start_paused=args.start_paused,
                hold_end=args.hold_end, show_withdrawal=args.show_withdrawal)
            if info is not None:
                print(f"episode={episode+1} success={info.success} reason={info.terminal_reason}")
            if stopped:
                break
    finally:
        env.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
