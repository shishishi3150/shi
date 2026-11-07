"""Every named ablation executes its intended paths during a real TD3 update."""
import numpy as np
import pytest
import torch
from torch import nn

from matd3.agent import Agent
from matd3.history_enhancement import JointHistory, conservative_value
from usv_env.config import PRESETS, assemble, refresh_hashes
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from usv_env.observation import build_actor_observation, build_critic_state
from usv_env.target_tracking import TargetEstimate, information_mode
from usv_env import target_tracking


# Independent expectations catch incorrect preset wiring rather than deriving
# the expected paths from the very mechanism flags under test.
EXPECTED = {
    "Base": (False, "hold", None),
    "Full": (True, "age_uncertainty", "conservative"),
    "w_o_I1": (False, "age_uncertainty", "conservative"),
    "w_o_I2": (True, "hold", "conservative"),
    "w_o_I3": (True, "age_uncertainty", None),
    "I1_only": (True, "hold", None),
    "I2_only": (False, "age_uncertainty", None),
    "I3_only": (False, "hold", "conservative"),
    "Traditional_CV": (False, "cv", None),
    "Random_History": (True, "age_uncertainty", "random_history"),
    "No_State_Matching": (True, "age_uncertainty", "no_state_matching"),
    "No_Safety_Screening": (True, "age_uncertainty", "no_safety_screening"),
    "Hard_Max": (True, "age_uncertainty", "hard_max"),
}


class ActionValue(nn.Module):
    """Deterministic target-Q oracle makes retrieved actions advantageous."""
    def __init__(self, heads, offset):
        super().__init__()
        self.heads, self.offset = heads, offset

    def forward(self, state, actions, obs):
        return (actions.flatten(1).mean(1, keepdim=True)+self.offset).expand(-1, self.heads)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_all_presets_execute_real_update_and_receiver_information_paths(monkeypatch, name):
    assert set(EXPECTED) == set(PRESETS)
    expected_i1, expected_mode, expected_variant = EXPECTED[name]
    torch.set_num_threads(1)
    cfg = assemble(experiment=name, device="cpu")
    cfg["training"].update(batch_size=2, n_step_return=1, policy_delay=1,
                           hidden_sizes=[8, 8], target_policy_noise=0.)
    cfg["replay_buffer"]["capacity"] = 8
    cfg["i3"].update(candidates=2, search_pool=8)
    cfg["communication"].update(delay_steps=[0, 0], loss_rate=0., target_loss_range=[0., 0.])
    refresh_hashes(cfg)
    env = HistoryEnv(PursuitEvasionEnv(cfg), history_length=1)
    env.reset()

    # Exercise the environment-created receiver, not an independently selected
    # estimator. All presets receive exactly the same two timestamped tracks.
    prediction_calls, fusion_calls = [], []
    original_prediction, original_fusion = target_tracking.predict_estimate, target_tracking.fuse_estimates

    def predict(*args, **kwargs):
        prediction_calls.append(args[0].source_agent_id)
        return original_prediction(*args, **kwargs)

    def fuse(*args, **kwargs):
        fusion_calls.append(True)
        return original_fusion(*args, **kwargs)

    monkeypatch.setattr(target_tracking, "predict_estimate", predict)
    monkeypatch.setattr(target_tracking, "fuse_estimates", fuse)
    a = TargetEstimate(True, np.array([11., 27., .4, 0.]), np.eye(4)*.2, 0., 0., .9, 2, "2:0")
    b = TargetEstimate(True, np.array([12., 27., .4, 0.]), np.eye(4)*.4, .15, .15, .8, 3, "3:0")
    bus = env.target_communication
    bus.broadcast(2, a, 0.); bus.broadcast(3, b, .15); bus.advance(.15)
    actual = bus.latest(0, 1.5)
    assert information_mode(cfg) == expected_mode and actual.processing_mode == expected_mode
    if expected_mode == "hold":
        np.testing.assert_array_equal(actual.state, b.state)
        np.testing.assert_array_equal(actual.covariance, b.covariance)
        assert not prediction_calls and not fusion_calls
    elif expected_mode == "cv":
        assert actual.position[0] == pytest.approx(12.54)
        assert actual.contribution_source_ids == (3,) and prediction_calls == [3]
        assert not fusion_calls
    else:
        assert actual.contribution_source_ids == (2, 3)
        assert 11.6 < actual.position[0] < 12.54
        assert len(prediction_calls) == 2 and len(fusion_calls) == 1

    env.env.step_count = 10
    env.env.estimators[2].estimate = a; env.env.estimators[3].estimate = b
    env.env.target_estimates = [bus.latest(i, 1.5, env.estimators[i].estimate) for i in range(5)]
    env.env.phase = "ENCIRCLE"
    raw = np.stack([build_actor_observation(env.env, i) for i in range(5)])
    obs = env._push(raw); state = build_critic_state(env.env)
    context = env.learning_context()
    monkeypatch.setattr("matd3.agent.nominal_actions", lambda _: np.zeros((5, 2), np.float32))
    agent = Agent(env, cfg)
    agent.critic_targets = [[ActionValue(2, .2), ActionValue(2, 0.)],
                            [ActionValue(3, .2), ActionValue(3, 0.)]]

    projection_calls, history_calls, value_calls = [], [], []

    def projection(config, states, histories, actions, reject_only=False, receivers=None, decision_only=False):
        projection_calls.append(bool(reject_only))
        # Distinct I1 output proves the update consumes the projection result;
        # I3's reject-only screening must preserve the complete historical row.
        output = actions.copy()
        if not reject_only:
            output[:, list(range(5) if receivers is None else receivers)] *= .5
        return output, dict(safe_mask=np.ones(len(actions), bool), intervention=np.ones(len(actions)),
                            projection_norm=np.ones(len(actions)), infeasible=np.zeros(len(actions)))

    original_retrieve = JointHistory.retrieve

    def retrieve(self, *args, **kwargs):
        assert expected_variant is not None, "disabled I3 must never retrieve history"
        found = original_retrieve(self, *args, **kwargs)
        history_calls.append((self.settings["variant"], found["valid"].copy()))
        return found

    def history_value(*args, **kwargs):
        assert expected_variant is not None, "disabled I3 must never augment target values"
        value_calls.append(args[4]["variant"])
        return conservative_value(*args, **kwargs)

    monkeypatch.setattr("usv_env.safety_projection.project_from_state", projection)
    monkeypatch.setattr(JointHistory, "retrieve", retrieve)
    monkeypatch.setattr("matd3.agent.conservative_value", history_value)
    historical = np.full((5, 2), .8, np.float32)
    for _ in range(4):
        agent.replay.add(obs, state, historical, np.ones(5, np.float32), obs, state,
            terminated=False, truncated=False, timeout=False, current_context=context, next_context=context)
    critic_before = [p.detach().clone() for p in agent.critics[0][0].parameters()]
    expected_policy = agent.joint_actions(agent.tensor(obs)[None], targets=True).detach().numpy()
    if expected_i1: expected_policy *= .5
    metrics = agent.update()
    assert agent.update_steps == agent.actor_updates == 1
    assert np.isfinite(metrics["usv_critic_loss"]) and np.isfinite(metrics["auv_critic_loss"])
    assert np.isfinite(metrics["usv_actor_loss"]) and np.isfinite(metrics["auv_actor_loss"])
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, agent.critics[0][0].parameters()))
    assert metrics["usv_target_q2"] == pytest.approx(float(expected_policy.mean()), abs=1e-6)
    # Target + shared Actor projection; identical current role actions reuse it.
    assert projection_calls.count(False) == (2 if expected_i1 else 0)
    expected_screen = expected_variant is not None and expected_variant != "no_safety_screening"
    assert projection_calls.count(True) == int(expected_screen)
    if expected_variant is None:
        assert not history_calls and not value_calls and agent.last_i3 is None
        assert metrics["i3_retrieved_candidates"] == metrics["i3_target_enhancement"] == 0.
    else:
        assert len(history_calls) == 1 and history_calls[0][0] == expected_variant
        assert history_calls[0][1].all() and value_calls == [expected_variant]
        assert metrics["i3_retrieved_candidates"] == metrics["i3_valid_candidates"] == 4.
        assert metrics["i3_target_enhancement"] > 0. and agent.last_i3["accepted"].all()
        advantage = np.maximum(agent.last_i3["q_history"]-agent.last_i3["q_policy"], 0.)
        if expected_variant == "hard_max":
            np.testing.assert_allclose(agent.last_i3["delta"], advantage)
        else:
            assert np.all(agent.last_i3["delta"] <= .5*advantage+1e-6)
