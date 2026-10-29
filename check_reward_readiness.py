"""Check finite same-state reward ordering against the current reward contract.

Rescoring is valid only when all dense formulas/weights are unchanged. It does
not predict a trained policy or establish universal success dominance.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from usv_env.config import assemble


def dense_identity(reward):
    return {k: v for k, v in reward.items()
            if k not in ('learning_scale', 'terminal', 'implementation_revision')}


def rescore(row, old, new, gamma):
    if dense_identity(old) != dense_identity(new):
        raise ValueError('cannot rescore changed dense reward formulas/parameters')
    terminal_discount = gamma ** (row['steps'] - 1)
    mask = np.asarray(row['active_mask'])
    dense = (np.asarray(row['discounted']) / old['learning_scale'] -
             terminal_discount * old['terminal'][row['outcome']] * mask)
    return new['learning_scale'] * (dense + terminal_discount * new['terminal'][row['outcome']] * mask)


def check(folder, cfg=None):
    cfg = assemble() if cfg is None else cfg
    runtime = json.loads((folder / 'runtime.json').read_text(encoding='utf-8'))
    if runtime['gamma'] != cfg['training']['gamma']:
        raise ValueError('changing gamma requires rescoring complete step traces')
    rows = [r for p in sorted(folder.glob('seed*.json'))
            for r in json.loads(p.read_text(encoding='utf-8'))['rows']]
    if len({r['seed'] for r in rows}) != len(runtime['seeds']):
        raise ValueError('audit is incomplete')
    gamma = runtime['gamma']; reward = cfg['reward']
    scores = [rescore(r, runtime['reward'], reward, gamma) for r in rows]
    comparisons = []; tracking = []; cross_source = []
    for i, success in enumerate(rows):
        if success['outcome'] != 'success':
            continue
        for j, alternative in enumerate(rows):
            if success['initial_state_hash'] != alternative['initial_state_hash']:
                continue
            margin = scores[i] - scores[j]
            if alternative['outcome'] != 'success':
                item = dict(seed=success['seed'], start_step=success['start_step'],
                    success_mode=success['mode'], success_source=success['auv_source'],
                    failure_mode=alternative['mode'], failure_source=alternative['auv_source'],
                    failure_outcome=alternative['outcome'], margins=margin.tolist(),
                    passed=bool(np.all(margin > 1e-6)))
                if success['auv_source'] == alternative['auv_source']:
                    comparisons.append(item)
                else:
                    # Distinct AUV controllers alter the other players' actions.
                    # Retain individual conflicts explicitly; require team ordering
                    # and individual ordering for the intended reference controller.
                    item.update(team_margin=float(margin.mean()), individual_passed=item['passed'])
                    item['passed'] = bool(margin.mean() > 1e-6 and
                        (success['mode'] != 'reference' or np.all(margin > 1e-6)))
                    cross_source.append(item)
            if (success['mode'] == 'reference' and alternative['mode'] == 'station_keep'
                    and success['auv_source'] == alternative['auv_source'] == 'diagnostic'):
                tracking.append(dict(seed=success['seed'], surface_margin=margin[:2].tolist(),
                    passed=bool(np.all(margin[:2] > 1e-6))))
    result = dict(status='PASS' if comparisons and tracking and
                  all(r['passed'] for r in comparisons + tracking + cross_source) else 'FAIL',
        interpretation='Finite physical controller audit only, not convergence or universal outcome ordering. '
                       'Same-source comparisons require all individual returns. Cross-source pairs '
                       'require team ordering, plus individual ordering for reference-success branches; '
                       'other individual conflicts remain explicitly reported.',
        source=str(folder), gamma=gamma, target_reward=deepcopy(reward),
        target_reward_hash=cfg['hashes']['reward'], source_outcomes=dict(Counter(r['outcome'] for r in rows)),
        trajectories=len(rows), comparisons=len(comparisons),
        inversions=sum(not r['passed'] for r in comparisons),
        minimum_agent_margin=min((min(r['margins']) for r in comparisons), default=None),
        cross_source_comparisons=len(cross_source),
        cross_source_individual_conflicts=sum(not r['individual_passed'] for r in cross_source),
        cross_source_gate_failures=sum(not r['passed'] for r in cross_source),
        tracking=tracking, paired_results=comparisons, cross_source_results=cross_source)
    (folder / 'current_reward_readiness.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('target_reward', 'paired_results', 'cross_source_results')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if check(args.folder)['status'] == 'PASS' else 1)
