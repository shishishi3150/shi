"""Training-only retrieval of previously executed complete joint actions.

No demonstration store, test trajectory, deployment retrieval or per-agent
action stitching is used. The resident Replay is the sole history source.
"""
import numpy as np
import torch

from .replay_buffer import TERMINAL_REASON_IDS


VARIANTS = ("conservative", "random_history", "no_state_matching", "no_safety_screening", "hard_max")
DEFAULTS = dict(variant="conservative", candidates=8, search_pool=512, max_distance=2.0,
                minimum_quality=.2, enhancement_weight=.5, advantage_clip=1.0,
                uncertainty_scale=1.0, similarity_temperature=1.0, return_scale=.05)

# Counts are sums per update (not rates). Retrieval rejection stages partition
# query x pool candidate pairs, in the order below. Row counts have suffix rows.
I3_COUNT_FIELDS = tuple('i3_'+name for name in (
    'queries', 'queries_without_history', 'pool_candidate_pairs', 'metadata_rejected',
    'phase_rejected', 'active_rejected', 'information_rejected', 'fatal_window_rejected',
    'quality_rejected', 'distance_rejected', 'eligible_candidates', 'topk_discarded',
    'retrieval_selected', 'terminal_rejected', 'safety_rejected',
    'qualified_rows', 'positive_advantage_missing_rows', 'selected_rows',
    'nonzero_enhancement_rows'))
I3_DISTRIBUTION_FIELDS = tuple('i3_'+name for name in (
    'history_return_p10', 'history_return_p50', 'history_return_p75', 'history_return_p90',
    'pool_quality_p50', 'pool_quality_p90', 'pool_instant_quality_p90'))


def validate_settings(settings):
    cfg = {**DEFAULTS, **settings}
    if set(cfg) != set(DEFAULTS) or cfg["variant"] not in VARIANTS:
        raise ValueError("unsupported I3 configuration or variant")
    for key in ("candidates", "search_pool"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"i3.{key} must be a positive integer")
    if cfg["candidates"] > cfg["search_pool"]:
        raise ValueError("I3 candidates cannot exceed bounded search_pool")
    for key in ("max_distance", "advantage_clip", "uncertainty_scale", "similarity_temperature", "return_scale"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"i3.{key} must be positive")
    for key in ("minimum_quality", "enhancement_weight"):
        if not np.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError(f"i3.{key} must be in [0,1]")
    return cfg


def semantic_features(states, contexts, cfg):
    """Dimensionless, weighted task geometry, kinematics and information.

    Scales are declared physical scales (10 m formation, .5 m/s motion,
    3 s information age, .3 m warning margin), never fitted on held-out data.
    Phase and validity are hard filters in retrieval rather than distances.
    """
    states, contexts = np.asarray(states), np.asarray(contexts)
    n = cfg["env"]["num_surface_blue"] + cfg["env"]["num_submarines"]
    world = float(max(cfg["env"]["world_size"]))
    friendly = states[:, :9*n].reshape(-1, n, 9)
    red = states[:, 9*n:9*n+7]
    positions = friendly[:, :, 2:4] * world
    velocity = friendly[:, :, 4:6] * 1.5
    red_position, red_velocity = red[:, :2]*world, red[:, 2:4]*1.5
    goal = np.asarray(cfg["protected_zone"]["center"])
    features = [(goal-red_position)/20., red_velocity/.5, red[:, 4:6],
                ((positions-red_position[:, None])/10.).reshape(-1, 2*n),
                ((velocity-red_velocity[:, None])/.5).reshape(-1, 2*n),
                .5*friendly[:, :, 6:8].reshape(-1, 2*n)]
    pairwise = [np.linalg.norm(positions[:, i]-positions[:, j], axis=1)/10.
                for i in range(n) for j in range(i+1, n)]
    features.append(.5*np.stack(pairwise, 1))
    for reef in cfg["reefs"]:
        center = np.asarray(reef["center"])
        radius = reef["radius"] + cfg["depth"]["reef_shallow_buffer_m"]
        features.extend([((np.linalg.norm(red_position-center, axis=1)-radius)/10.)[:, None],
                         .5*(np.linalg.norm(positions-center, axis=2)-radius)/5.])
    valid, age, variance, margins = np.split(contexts[:, :4*n], 4, axis=1)
    features.extend([.5*np.minimum(age, 12.)/3.,
                     .5*np.log1p(variance)/np.log(13.),
                     .5*np.clip(margins/.3, -2., 3.)])
    return np.concatenate(features, 1).astype(np.float32)


class JointHistory:
    def __init__(self, cfg, seed):
        self.cfg, self.settings = cfg, validate_settings(cfg.get("i3", {}))
        self.rng = np.random.default_rng(int(seed))

    def retrieve(self, replay, states, contexts, phases, masks):
        """Return whole action rows from a bounded resident-history pool."""
        settings = self.settings
        b, n, k = len(states), replay.num_agents, settings["candidates"]
        result = dict(actions=np.zeros((b, k, n, 2), np.float32),
                      indices=np.full((b, k), -1, np.int64), valid=np.zeros((b, k), bool),
                      distance=np.full((b, k), np.inf, np.float32),
                      quality=np.zeros((b, k), np.float32),
                      realized_return=np.zeros((b, k), np.float32),
                      outcome_confidence=np.zeros((b, k), np.float32), phase_match_fraction=0.,
                      funnel={name: 0 for name in I3_COUNT_FIELDS},
                      distributions={name: None for name in I3_DISTRIBUTION_FIELDS})
        funnel = result['funnel']; funnel['i3_queries'] = b
        if not len(replay):
            funnel['i3_queries_without_history'] = b
            return result
        pool = self.rng.choice(len(replay), min(len(replay), settings["search_pool"]), replace=False)
        pool_phase, pool_active = replay._unpack_phase_activity(replay.phase_ids[pool])
        pool_context = replay.current_context[pool]
        # Resident rows contain only committed, observed n-step windows. Their
        # discounted rewards are historical outcomes, never unknown episode
        # futures, target-network estimates, evaluation data or test labels.
        realized_return = ((replay.agent_rewards[pool] * pool_active).sum(axis=1)
                           / np.maximum(pool_active.sum(axis=1), 1))
        outcome_confidence = 1. / (1. + np.exp(-np.clip(
            realized_return / settings["return_scale"], -60., 60.)))
        quality = pool_context[:, -2] * outcome_confidence
        for percentile in (10, 50, 75, 90):
            result['distributions'][f'i3_history_return_p{percentile}'] = float(np.percentile(realized_return, percentile))
        for percentile in (50, 90):
            result['distributions'][f'i3_pool_quality_p{percentile}'] = float(np.percentile(quality, percentile))
        result['distributions']['i3_pool_instant_quality_p90'] = float(np.percentile(pool_context[:, -2], 90))
        fatal_ids = [TERMINAL_REASON_IDS[name] for name in (
            "collision", "reef_collision", "out_of_bounds", "scenario_anomaly", "numerical_error")]
        # A terminal event flushes every pending window at the event step, so
        # this endpoint reason excludes the entire observed unsafe window.
        fatal_window = np.isin(replay.terminal_reason_ids[pool, 0], fatal_ids)
        phase_matches = pool_phase[:, 0][None] == np.asarray(phases).reshape(-1, 1)
        stages = (
            ('metadata', (pool_context[None, :, -1] == 1) & (contexts[:, None, -1] == 1)),
            ('phase', phase_matches),
            ('active', np.all(pool_active[None] == masks[:, None], axis=2)),
            ('information', np.all(pool_context[None, :, :n] == contexts[:, None, :n], axis=2)),
            ('fatal_window', np.broadcast_to(~fatal_window, (b, len(pool)))),
            ('quality', np.broadcast_to(quality >= settings['minimum_quality'], (b, len(pool)))))
        eligible_rows = np.ones((b, len(pool)), bool)
        funnel['i3_pool_candidate_pairs'] = b*len(pool)
        for name, passes in stages:
            funnel[f'i3_{name}_rejected'] = int(np.count_nonzero(eligible_rows & ~passes))
            eligible_rows &= passes
        if not eligible_rows.any():
            # All per-row distance/ranking work would be empty. Preserve the
            # sampled pool, RNG state, rejection ledger and distributions.
            result["phase_match_fraction"] = float(np.mean(phase_matches))
            return result
        # Reduce the same float32 feature axis in bounded eligible-pair blocks. This
        # removes hundreds of Python reductions without changing the pool,
        # hard filters, distance, stable ranking or random-history RNG order.
        distance_rows = np.full((b, len(pool)), np.inf, np.float32)
        query_rows, pool_rows = np.nonzero(eligible_rows)
        if len(query_rows):
            features = semantic_features(replay.states[pool], pool_context, self.cfg)
            query = semantic_features(states, contexts, self.cfg)
            for start in range(0, len(query_rows), 2048):
                selected = slice(start, start+2048)
                rows, columns = query_rows[selected], pool_rows[selected]
                difference = features[columns]-query[rows]
                np.square(difference, out=difference)
                distance_rows[rows, columns] = np.sqrt(np.mean(difference, axis=1))
        if settings["variant"] != "random_history":
            # Batched stable sorting keeps the original pool-index tie order.
            # Random_History retains its per-row shuffle/RNG sequence below.
            if settings["variant"] != "no_state_matching":
                passes = distance_rows <= settings["max_distance"]
                funnel['i3_distance_rejected'] = int(np.count_nonzero(eligible_rows & ~passes))
                eligible_rows &= passes
                scores = np.where(eligible_rows, distance_rows, np.inf)
            else:
                scores = np.where(eligible_rows, -quality[None, :], np.inf)
            counts = eligible_rows.sum(1)
            order = np.argsort(scores, axis=1, kind="stable")[:, :k]
            rows, slots = np.nonzero(np.arange(order.shape[1])[None, :] < np.minimum(counts, k)[:, None])
            columns = order[rows, slots]
            resident = pool[columns]
            funnel['i3_eligible_candidates'] = int(counts.sum())
            funnel['i3_topk_discarded'] = int(np.maximum(counts-k, 0).sum())
            funnel['i3_retrieval_selected'] = len(rows)
            result["indices"][rows, slots] = resident
            result["actions"][rows, slots] = replay.actions[resident]
            result["valid"][rows, slots] = True
            result["distance"][rows, slots] = distance_rows[rows, columns]
            result["quality"][rows, slots] = quality[columns]
            result["realized_return"][rows, slots] = realized_return[columns]
            result["outcome_confidence"][rows, slots] = outcome_confidence[columns]
            result["phase_match_fraction"] = float(np.mean(phase_matches))
            return result
        for row in range(b):
            eligible = eligible_rows[row].copy()
            if not eligible.any():
                continue
            distances = distance_rows[row]
            variant = settings["variant"]
            if variant not in ("random_history", "no_state_matching"):
                passes = distances <= settings["max_distance"]
                funnel['i3_distance_rejected'] += int(np.count_nonzero(eligible & ~passes))
                eligible &= passes
            candidates = np.flatnonzero(eligible)
            funnel['i3_eligible_candidates'] += len(candidates)
            funnel['i3_topk_discarded'] += max(0, len(candidates)-k)
            if variant == "random_history":
                self.rng.shuffle(candidates)
            elif variant == "no_state_matching":
                candidates = candidates[np.argsort(-quality[candidates], kind="stable")]
            else:
                candidates = candidates[np.argsort(distances[candidates], kind="stable")]
            candidates = candidates[:k]
            count = len(candidates)
            funnel['i3_retrieval_selected'] += count
            resident = pool[candidates]
            result["indices"][row, :count] = resident
            result["actions"][row, :count] = replay.actions[resident]
            result["valid"][row, :count] = True
            result["distance"][row, :count] = distances[candidates]
            result["quality"][row, :count] = quality[candidates]
            result["realized_return"][row, :count] = realized_return[candidates]
            result["outcome_confidence"][row, :count] = outcome_confidence[candidates]
        result["phase_match_fraction"] = float(np.mean(phase_matches))
        return result


def conservative_value(q_policy, q_history, confidence, valid, settings, agent_mask=None):
    """Choose ONE complete candidate per row and augment each value head.

    q_history is [batch,candidates,agents]. No head may select a different
    action row. A missing or nonpositive candidate gives exact MATD3 fallback.
    """
    advantage = (q_history-q_policy[:, None]).clamp_min(0.)
    if settings["variant"] != "hard_max":
        advantage = advantage.clamp_max(settings["advantage_clip"])
        gain = settings["enhancement_weight"]*confidence[:, :, None]*advantage
    else:
        gain = advantage
    if agent_mask is not None:
        gain = gain * agent_mask[:, None]
    gain = torch.where(valid[:, :, None], gain, torch.zeros_like(gain))
    score = gain.mean(-1).masked_fill(~valid, -torch.inf)
    chosen = score.argmax(1)
    rows = torch.arange(len(q_policy), device=q_policy.device)
    delta = gain[rows, chosen]
    accepted = valid.any(1) & (delta.sum(-1) > 0)
    delta = torch.where(accepted[:, None], delta, torch.zeros_like(delta))
    return q_policy+delta, dict(chosen=chosen, accepted=accepted, delta=delta,
                                advantage=advantage[rows, chosen])
