"""Innovation-only compute budgets leave the learning algorithm unchanged."""
from copy import deepcopy
import json

import pytest

from matd3.agent import Agent
from scripts.train import main
from usv_env.config import assemble


@pytest.mark.parametrize('filename,pool,candidates', [
    ('fast.yaml', 256, 4), ('fast_economy.yaml', 128, 2),
])
def test_fast_presets_only_reduce_historical_search(filename, pool, candidates):
    original = assemble(experiment='Full', device='cpu')
    fast = assemble('configs/'+filename, experiment='Full', device='cpu')
    assert fast['hashes'] == original['hashes']
    assert fast['mechanisms'] == original['mechanisms']
    assert all(fast['mechanisms'].values())
    assert fast['i1'] == original['i1']
    assert fast['training']['random_warmup_env_steps'] == 10000
    assert fast['training']['batch_size'] == 256
    assert fast['training']['policy_delay'] == 2
    assert fast['training'] == original['training']
    assert fast['i3']['search_pool'] == pool and fast['i3']['candidates'] == candidates
    left, right = deepcopy(fast), deepcopy(original)
    left['i3'].update(search_pool=right['i3']['search_pool'], candidates=right['i3']['candidates'])
    left['purpose'] = right['purpose']
    for cfg in (left, right):
        cfg.pop('provenance')
    assert left == right


def test_real_collector_updates_every_step_after_warmup(tmp_path, monkeypatch):
    calls, successful = [], []
    original = Agent.update
    def observe_update(agent):
        calls.append(agent.global_env_steps)
        metrics = original(agent)
        if metrics is not None:
            successful.append(agent.global_env_steps)
        return metrics
    monkeypatch.setattr(Agent, 'update', observe_update)
    output = tmp_path/'fast_smoke'
    main(['--config', 'configs/fast.yaml', '--experiment', 'Full', '--smoke',
          '--output', str(output), '--steps', '30', '--smoke-episode-limit', '0',
          '--device', 'cpu', '--eval-every', '0', '--train-trace-every', '0'])
    assert calls == list(range(9, 31))
    assert successful and successful == list(range(successful[0], 31))
    manifest = json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['online_steps'] == 30
