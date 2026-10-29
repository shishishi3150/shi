"""Rescore fixed physical audit trajectories; this is not policy retraining."""
import argparse
import json
from pathlib import Path
import numpy as np


def rescore(row, gamma, failure_penalty, original_terminal, learning_scale=1.):
    value = np.asarray(row['discounted'], dtype=float)
    if row['outcome'] in ('collision', 'landing', 'out_of_bounds'):
        value = value + gamma ** (row['steps'] - 1) * (
            -failure_penalty - original_terminal[row['outcome']]) * learning_scale * np.asarray(row['active_mask'])
    return value


def summarize(folder):
    runtime = json.loads((folder / 'runtime.json').read_text(encoding='utf-8'))
    gamma = runtime['gamma']
    terminal = runtime['reward']['terminal']
    rows = [r for p in sorted(folder.glob('seed*.json'))
            for r in json.loads(p.read_text(encoding='utf-8'))['rows']]
    comparisons = []
    for failure in rows:
        if failure['outcome'] not in ('collision', 'landing', 'out_of_bounds'):
            continue
        for success in rows:
            if (success['outcome'] != 'success' or
                    success['initial_state_hash'] != failure['initial_state_hash'] or
                    success['auv_source'] != failure['auv_source']):
                continue
            comparisons.append(dict(seed=failure['seed'], start_step=failure['start_step'],
                failure_mode=failure['mode'], success_mode=success['mode'],
                failure_steps=failure['steps'], success_steps=success['steps'],
                margins={str(p): (rescore(success, gamma, p, terminal, runtime['reward']['learning_scale']) -
                                 rescore(failure, gamma, p, terminal, runtime['reward']['learning_scale'])).tolist()
                         for p in (2, 4, 6, 8, 12, 14)}))
    result = dict(gamma=gamma, trajectories=len(rows), paired_comparisons=len(comparisons),
        note='Positive per-agent margin means successful continuation is preferred. '
             'Diagnostic controllers, finite sampled states; not learned-policy performance.',
        penalties={str(p): dict(
            reversed_comparisons=sum(min(c['margins'][str(p)]) <= 0 for c in comparisons),
            minimum_agent_margin=min((min(c['margins'][str(p)]) for c in comparisons), default=None))
            for p in (2, 4, 6, 8, 12, 14)}, comparisons=comparisons)
    (folder / 'reward_order.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder)
