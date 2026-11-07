"""Evaluation must separate first capture from safe episode completion."""
import numpy as np
import pytest

from scripts import research_evaluation
from usv_env.config import assemble


def test_selection_uses_success_then_safety_before_return():
    def key(success=0., safety=0., capture=0., reward=0.):
        return research_evaluation.validation_key(dict(success_rate=success,
            safety_failure_rate=safety, capture_rate=capture, discounted_return_value=reward))
    assert key(safety=0., reward=1.) > key(safety=1., reward=100.)
    assert key(success=1., reward=1.) > key(success=0., reward=100.)
    assert key(capture=1., reward=1.) > key(capture=0., reward=100.)
    assert key(reward=2.) > key(reward=1.)


def evaluate_rows(monkeypatch, rows):
    class Environment:
        agent_action_limits = np.ones((5, 2), np.float32)
        index = -1

        def reset(self, seed):
            self.index += 1
            return np.zeros((5, 1), np.float32), None

        def step(self, action):
            return np.zeros((5, 1), np.float32), None, np.zeros(5), True, None

        def close(self):
            pass

    class Measurement:
        def __init__(self, env):
            pass

        def observe(self, *args):
            pass

        def summary(self, env, info):
            return rows[env.index].copy()

    class Policy:
        def act(self, obs, explore):
            return np.zeros((5, 2), np.float32)

    class Writer:
        def __init__(self):
            self.records = []

        def append(self, stream, row):
            self.records.append((stream, row))

    writer = Writer()
    monkeypatch.setattr(research_evaluation, 'make_env', lambda *args: Environment())
    monkeypatch.setattr(research_evaluation, 'EpisodeMetrics', Measurement)
    result = research_evaluation.evaluate_policy(
        Policy(), assemble(experiment='Base', device='cpu'), len(rows), 51,
        writer=writer, trace_episodes=0)
    written = [row for stream, row in writer.records if stream == 'evaluation_summary']
    assert written == result['summaries']
    return result, result['summaries'][0]


def episode(captured, success, withdrawn, first=None, post=0):
    return dict(capture_success=int(captured), success=int(success),
        withdrawal_cleared_entry=bool(withdrawn), first_capture_step=first,
        first_capture_time_s=None if first is None else first*.15,
        post_capture_steps=post, episode_steps=650 if success else 900,
        terminal_reason='success' if success else 'timeout',
        team_return=-10., discounted_team_return=-2.)


def test_first_capture_and_post_capture_outcomes_use_explicit_denominators(monkeypatch):
    result, summary = evaluate_rows(monkeypatch, [
        episode(True, True, True, 100, 550),
        episode(True, False, False, 200, 50),
        episode(True, True, False, 300, 350),
        episode(False, False, False),
        episode(True, False, True, 400, 20),
    ])
    assert summary['episodes'] == 5
    assert summary['capture_count'] == 4
    assert summary['capture_rate'] == pytest.approx(.8)
    assert summary['first_capture_step'] == pytest.approx(250.)
    assert summary['first_capture_step_n'] == 4
    assert summary['first_capture_time_s'] == pytest.approx(37.5)
    assert summary['first_capture_time_s_n'] == 4
    assert summary['post_capture_steps'] == pytest.approx(242.5)
    assert summary['post_capture_steps_n'] == 4
    assert summary['post_capture_success_count'] == 2
    assert summary['post_capture_success_rate'] == pytest.approx(2/5)
    assert summary['post_capture_success_rate_given_capture'] == pytest.approx(2/4)
    assert summary['withdrawal_cleared_entry_count'] == 2
    assert summary['withdrawal_cleared_entry_rate'] == pytest.approx(2/5)
    assert summary['withdrawal_cleared_entry_rate_given_capture'] == pytest.approx(2/4)
    # The checkpoint selection inputs keep their original episode-success meaning.
    assert result['success_rate'] == summary['success_rate'] == pytest.approx(2/5)
    assert result['return_value'] == -10.
    assert result['discounted_return_value'] == -2.


def test_no_capture_reports_zero_unconditional_rates_and_missing_conditional_rates(monkeypatch):
    _, summary = evaluate_rows(monkeypatch, [
        episode(False, False, False), episode(False, False, False),
    ])
    assert summary['capture_count'] == 0 and summary['capture_rate'] == 0.
    for field in ('first_capture_step', 'first_capture_time_s', 'post_capture_steps'):
        assert summary[field] is None and summary[field+'_n'] == 0
    for field in ('post_capture_success', 'withdrawal_cleared_entry'):
        assert summary[field+'_count'] == 0
        assert summary[field+'_rate'] == 0.
        assert summary[field+'_rate_given_capture'] is None
    assert summary['success_rate'] == 0.


def test_capture_without_safety_completion_is_not_reported_as_final_success(monkeypatch):
    result, summary = evaluate_rows(monkeypatch, [episode(True, False, True, 321, 40)])
    assert summary['capture_rate'] == 1.
    assert summary['post_capture_success_rate'] == 0.
    assert summary['post_capture_success_rate_given_capture'] == 0.
    # Crossing the entry is a latched event and does not certify a safe ending.
    assert summary['withdrawal_cleared_entry_rate'] == 1.
    assert summary['withdrawal_cleared_entry_rate_given_capture'] == 1.
    assert result['success_rate'] == 0.
