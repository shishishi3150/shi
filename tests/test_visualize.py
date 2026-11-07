from __future__ import annotations

from types import SimpleNamespace

from pathlib import Path

from scripts.visualize import ROOT, _checkpoint_path, _run_episode, parse_args
import pytest
from scripts.visualize import _check_scene


def test_default_historical_checkpoint_requires_explicit_selection():
    args = parse_args([])
    with pytest.raises(SystemExit, match="historical"):
        _check_scene({"env": {"world_size": [12., 12.]}}, args, Path("old.pt"))
    _check_scene({"env": {"world_size": [55., 55.]}}, args, Path("new.pt"))
    _check_scene({"env": {"world_size": [12., 12.]}}, parse_args(["--checkpoint", "old.pt"]), Path("old.pt"))


def test_preview_is_explicit_and_not_a_checkpoint_policy():
    assert parse_args(["--preview"]).preview
    with pytest.raises(SystemExit):
        parse_args(["--preview", "--checkpoint", "old.pt"])


def test_rule_playback_is_explicit_and_has_recording_controls():
    args = parse_args(["--rules", "--speed", "2", "--start-paused", "--hold-end", "--show-withdrawal"])
    assert args.rules and args.speed == 2. and args.start_paused and args.hold_end and args.show_withdrawal
    for arguments in (["--rules", "--preview"], ["--rules", "--best"],
                      ["--rules", "--checkpoint", "old.pt"], ["--speed", "0"],
                      ["--speed", "nan"], ["--headless", "--hold-end"]):
        with pytest.raises(SystemExit):
            parse_args(arguments)


def test_rule_playback_matches_acceptance_controller():
    import numpy as np
    from scripts.train import make_env
    from scripts.visualize import _LocalRules
    from usv_env.config import assemble
    from usv_env.red_policy import RuleController
    env = make_env(assemble(device="cpu"), 5, "fixed")
    try:
        obs, _ = env.reset(seed=5)
        agent = _LocalRules(env)
        expected = RuleController().actions(env.blue_states, env.submarine_states, None,
            env.protected_center, env.reefs, env.world_size, env.blue_dynamics,
            env.submarine_dynamics, env.step_count * env.dt,
            target_estimates=env.target_estimates, active_mask=env.agent_active_mask())
        np.testing.assert_array_equal(agent.act(obs), expected)
        agent.reset()
        assert not agent.controller._path_cache
    finally:
        env.close()


class _Environment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int):
        self.steps = 0
        return {"seed": seed}, {}

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        return {"step": self.steps}, 0.0, False, done, SimpleNamespace(
            strict_success=done,
            terminal_reason="success" if done else "",
        )

    def render_frame(self):
        return ("frame", self.steps)


class _Agent:
    def act(self, obs, *, explore: bool):
        assert not explore
        return obs


class _Renderer:
    def __init__(self) -> None:
        self.frames = []
        self.reset_count = 0

    def reset_trails(self) -> None:
        self.reset_count += 1

    def pump_events(self) -> bool:
        return True

    def render(self, frame) -> None:
        self.frames.append(frame)


def test_run_episode_uses_render_frame_and_standalone_renderer() -> None:
    env = _Environment()
    renderer = _Renderer()

    info, window_closed = _run_episode(env, _Agent(), seed=7, renderer=renderer)

    assert not window_closed
    assert info.strict_success
    assert env.steps == 2
    assert renderer.reset_count == 1
    assert renderer.frames == [("frame", 0), ("frame", 1), ("frame", 2)]


def test_run_episode_stops_cleanly_when_window_closes() -> None:
    env = _Environment()
    renderer = _Renderer()
    renderer.pump_events = lambda: False

    info, window_closed = _run_episode(env, _Agent(), seed=7, renderer=renderer)

    assert window_closed
    assert info is None
    assert env.steps == 0


def test_integrated_withdrawal_does_not_append_rule_replay(monkeypatch):
    env = _Environment()
    env.post_capture_end_step = 650
    original_step = env.step
    def advance(action):
        result = original_step(action)
        result[-1].success = result[-1].strict_success
        return result
    env.step = advance
    def forbidden(*args, **kwargs):
        raise AssertionError('integrated episode must keep its selected policy')
    monkeypatch.setattr('scripts.visualize._run_withdrawal', forbidden)
    info, closed = _run_episode(env, _Agent(), seed=7, renderer=None, show_withdrawal=True)
    assert info.success and not closed


def test_default_checkpoint_is_capture_aware_full_latest() -> None:
    args = parse_args([])
    path = _checkpoint_path({"outputs": {"root": "outputs"}}, args)
    assert path == ROOT / "outputs" / "positive_reward" / "Full" / "seed0" / "latest.pt"


def test_experiment_seed_best_and_explicit_checkpoint_precedence(tmp_path: Path) -> None:
    args = parse_args(["--runs-root", "outputs", "--experiment", "w_o_I2", "--seed", "3", "--best"])
    assert _checkpoint_path({"outputs": {"root": "outputs"}}, args) == (
        ROOT / "outputs" / "w_o_I2" / "seed3" / "best.pt"
    )
    explicit = tmp_path / "manual.pt"
    args = parse_args(["--experiment", "Base", "--checkpoint", str(explicit)])
    assert _checkpoint_path({"outputs": {"root": "outputs"}}, args) == explicit
