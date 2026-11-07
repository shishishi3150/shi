"""Configuration and terminal labels must describe the actual runtime behavior."""
import numpy as np
import pytest

from usv_env.config import assemble
from usv_env.environment import PursuitEvasionEnv


@pytest.mark.parametrize("patch", [
    {"loss_rate": .3},
    {"target_loss_range": [.2, .2]},
    {"target_loss_range": [.05, .2]},
])
def test_conflicting_loss_settings_are_rejected(patch):
    cfg = assemble(device="cpu")
    cfg["communication"].update(patch)
    with pytest.raises(ValueError, match="loss_rate.*target_loss_range"):
        PursuitEvasionEnv(cfg)


@pytest.mark.parametrize("rate", [0., .05, .1, .2, .3])
def test_explicit_loss_sweep_reaches_both_networks(rate):
    cfg = assemble(device="cpu")
    cfg["communication"].update(loss_rate=rate, target_loss_range=[rate, rate])
    env = PursuitEvasionEnv(cfg)
    assert env.communication.target_loss == rate
    assert env.target_communication.loss_rate == rate


@pytest.mark.parametrize("success,phase", [(False, "HOLD"), (True, "SUCCESS")])
def test_terminal_success_phase_is_not_overwritten_by_hold(monkeypatch, success, phase):
    env = PursuitEvasionEnv(assemble(device="cpu"))
    env.post_capture_end_step = 0  # Explicit historical immediate ending.

    def task_result(safe):
        assert safe
        env.hold_steps = env.required_hold_steps if success else 1
        return success

    monkeypatch.setattr(env, "_task_success", task_result)
    _, _, _, done, info = env.step(np.zeros((env.num_friendly, 2), dtype=np.float32))
    assert done is success
    assert info.success is success
    assert info.phase == phase
    assert env.phase == phase
