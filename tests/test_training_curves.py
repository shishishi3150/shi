"""Observed return curves retain provenance and never fill absent evidence."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.csv_logging import CsvLog, LEGACY_TRAIN_FIELDS, TRAIN_FIELDS
from scripts.quxian import main, read_run
from usv_env.config import assemble, refresh_hashes


def make_run(root, experiment='Base', *, legacy=False, source='same-source', change=None):
    path=Path(root)/experiment/'seed0'/'train.csv'
    cfg=assemble(experiment=experiment,device='cpu')
    cfg['provenance']['code_hash']=source
    if change: change(cfg)
    refresh_hashes(cfg)
    path.parent.mkdir(parents=True,exist_ok=True)
    (path.parent/'resolved_config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    fields=LEGACY_TRAIN_FIELDS if legacy else TRAIN_FIELDS
    log=CsvLog(path,fields)
    for episode,total in ((1,-2.),(2,-1.)):
        row=dict.fromkeys(fields)
        row.update(episode=episode,env_steps=episode*10,episode_steps=10,
                   team_return=total,usv_return=total,auv_return=total,
                   process_return=total+.5,terminal_return=-.5,
                   process_abs_sum=abs(total+.5),success=0,terminal_reason='timeout',wall_time_s=episode)
        if not legacy:
            row.update(discounted_team_return=total*.9,discounted_usv_return=total*.9,
                       discounted_auv_return=total*.9,discounted_process_return=total*.9+.4,
                       discounted_terminal_return=-.4,reward_gamma=cfg['training']['gamma'])
        log.append(row)
    return path


def test_new_curve_preserves_raw_values_and_causal_trailing_average(tmp_path):
    path=make_run(tmp_path)
    steps,metrics,_=read_run(path,2)
    np.testing.assert_array_equal(steps,[10,20])
    np.testing.assert_allclose(metrics['raw_team_return'],[-2.,-1.])
    np.testing.assert_allclose(metrics['team_return'],[-2.,-1.5])
    np.testing.assert_allclose(metrics['discounted_team_return'],[-1.8,-1.35])
    np.testing.assert_allclose(metrics['raw_terminal_abs_fraction'],[.25,.5])
    np.testing.assert_array_equal(metrics['episode_steps'],[10,10])


def test_legacy_curve_reads_without_inventing_discounted_values(tmp_path):
    path=make_run(tmp_path,legacy=True)
    _,metrics,_=read_run(path,2)
    assert np.isnan(metrics['discounted_team_return']).all()
    np.testing.assert_allclose(metrics['team_return'],[-2.,-1.5])


def test_legacy_discounted_fallback_requires_exact_episode_and_online_step(tmp_path):
    path=make_run(tmp_path,legacy=True)
    directory=path.parent/'diagnostics'; directory.mkdir()
    records=[dict(episode_index=1,env_steps=10,discounted_team_return=-1.75),
             dict(episode_index=2,env_steps=999,discounted_team_return=123.)]
    (directory/'training_episodes.jsonl').write_text(
        ''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    _,metrics,_=read_run(path,1)
    assert metrics['discounted_team_return'][0]==-1.75
    assert np.isnan(metrics['discounted_team_return'][1])


@pytest.mark.parametrize('change,message',[
    (lambda cfg: cfg['provenance'].update(code_hash='different-source'), 'cannot mix'),
    (lambda cfg: cfg['task'].update(maximum_observer_angle_gap_degrees=159.), 'cannot mix'),
    (lambda cfg: cfg['training'].update(gamma=.98), 'cannot mix'),
    (lambda cfg: cfg.update(purpose='learning_pilot'), 'cannot mix'),
    # Recurrent inputs are now rejected before cohort comparison.
    (lambda cfg: cfg['revision'].update(history_length=4), 'single-frame MLP'),
    (lambda cfg: cfg['training'].update(batch_size=128), 'cannot mix'),
    (lambda cfg: cfg['i3'].update(search_pool=256, candidates=4), 'cannot mix'),
])
def test_curve_cohorts_reject_incompatible_provenance(tmp_path,change,message):
    make_run(tmp_path,'Base')
    make_run(tmp_path,'Full',change=change)
    with pytest.raises(ValueError,match=message):
        main(['--runs-root',str(tmp_path),'--experiments','Base','Full','--seeds','0'])
    assert not (tmp_path/'plots').exists()


def test_mixed_legacy_and_new_csv_layouts_can_share_verified_cohort(tmp_path):
    make_run(tmp_path,'Base',legacy=True)
    make_run(tmp_path,'Full')
    assert main(['--runs-root',str(tmp_path),'--experiments','Base','Full',
                 '--seeds','0','--window-episodes','2'])==0
    output=tmp_path/'plots'/'learning_seeds_0.png'
    assert output.is_file() and output.stat().st_size>10000
