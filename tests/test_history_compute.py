"""Batched eligible distances retain stable ties, filters and random draws."""
import numpy as np
import pytest

from matd3.history_enhancement import JointHistory, VARIANTS, semantic_features
from matd3.replay_buffer import ReplayBuffer
from usv_env.config import assemble


@pytest.mark.parametrize('variant', VARIANTS)
def test_blocked_distances_match_rowwise_oracle_across_block_boundaries(variant):
    cfg = assemble(experiment='Full', device='cpu')
    cfg['i3'].update(variant=variant, candidates=4, search_pool=96)
    replay = ReplayBuffer(96, 5, 2, cfg['observation_schema']['critic_dim'], 2, n_step_return=1)
    rng = np.random.default_rng(147)
    observation = np.zeros((5, 2), np.float32)
    context = dict(phase_id=0, information_valid=np.ones(5), information_age=np.zeros(5),
        information_variance=np.ones(5), safety_margin=np.ones(5), quality=1.)
    states = rng.uniform(-.1, .1, (96, replay.state_dim)).astype(np.float32)
    states[:8] = states[0]  # Exact distance ties must retain resident pool order.
    for row, state in enumerate(states):
        current = dict(context)
        if row >= 90:
            current['quality'] = .01
        replay.add(observation, state, rng.uniform(-1., 1., (5, 2)), np.full(5, -.01),
            observation, state, terminated=False, truncated=False, timeout=False,
            current_context=current, next_context=context)
    queries = replay._sample_rows(np.arange(75), np.ones(75), np.ones(75))
    queries['next_context'][0, -1] = 0.
    queries['next_phase_ids'][1] = 2
    queries['agent_active_mask'][2, 0] = 0.
    args = (replay, queries['next_states'], queries['next_context'], queries['next_phase_ids'], queries['agent_active_mask'])
    new = JointHistory(cfg, 513)
    actual = new.retrieve(*args)
    assert actual['funnel']['i3_eligible_candidates'] > 2048
    assert not actual['valid'][:3].any()
    # An independent, row-at-a-time oracle replaces the deleted historical
    # validation directory. This fixture has three deliberately invalid query
    # rows and six low-quality resident rows; all others pass metadata filters.
    oracle_rng = np.random.default_rng(513)
    pool = oracle_rng.choice(96, 96, replace=False)
    features = semantic_features(replay.states[pool], replay.current_context[pool], cfg)
    query = semantic_features(queries['next_states'], queries['next_context'], cfg)
    returns = replay.agent_rewards[pool].mean(1)
    confidence = 1./(1.+np.exp(-returns/cfg['i3']['return_scale']))
    quality = replay.current_context[pool, -2]*confidence
    eligible_count = 0
    for row in range(3, 75):
        distances = np.sqrt(np.mean((features-query[row])**2, axis=1))
        candidates = np.flatnonzero(pool < 90)
        if variant not in ('random_history', 'no_state_matching'):
            candidates = candidates[distances[candidates] <= cfg['i3']['max_distance']]
        eligible_count += len(candidates)
        if variant == 'random_history': oracle_rng.shuffle(candidates)
        elif variant == 'no_state_matching':
            candidates = candidates[np.argsort(-quality[candidates], kind='stable')]
        else:
            candidates = candidates[np.argsort(distances[candidates], kind='stable')]
        selected = candidates[:4]
        resident = pool[selected]
        np.testing.assert_array_equal(actual['indices'][row], resident)
        np.testing.assert_array_equal(actual['actions'][row], replay.actions[resident])
        np.testing.assert_array_equal(actual['distance'][row], distances[selected])
        np.testing.assert_allclose(actual['quality'][row], quality[selected], rtol=1e-7)
        np.testing.assert_array_equal(actual['valid'][row], np.ones(4, bool))
    assert actual['funnel']['i3_eligible_candidates'] == eligible_count
    assert actual['funnel']['i3_metadata_rejected'] == 96
    assert actual['funnel']['i3_phase_rejected'] == 96
    assert actual['funnel']['i3_active_rejected'] == 96
    assert actual['funnel']['i3_quality_rejected'] == 72*6
    assert oracle_rng.bit_generator.state == new.rng.bit_generator.state
