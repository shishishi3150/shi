"""Real short-run output contracts for actor-only policy persistence."""
import csv
import json
import pytest

from matd3.checkpoint import read
from matd3.replay_buffer import ReplayBuffer
from scripts.research_evaluation import conditions
from scripts.train import main


def test_replay_has_no_full_state_persistence_api():
    assert not any(hasattr(ReplayBuffer, name) for name in (
        'state_dict', 'load_state_dict', '_snapshot_rows', '_restore_rows',
    ))


def test_communication_evaluation_grid_has_required_delays_and_losses():
    matrix=conditions('communication')
    cells=[override for _,override in matrix if override is not None]
    assert len(cells)==25
    assert {(cell['delay_steps'][0],cell['loss_rate']) for cell in cells} == {
        (delay,loss) for delay in (0,3,10,20,40) for loss in (0.,.05,.1,.2,.3)}
    assert all(cell['delay_steps'][0]==cell['delay_steps'][1] for cell in cells)
    assert all(override is None or override['target_loss_range']==[override['loss_rate']]*2 for _,override in matrix)


def test_exact_resume_is_explicitly_rejected_before_checkpoint_read(tmp_path):
    with pytest.raises(SystemExit): main(['--resume',str(tmp_path/'missing.pt')])
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('budget_args,warmup', [
    ([], 10000),
    (['--steps', '9999'], 10000),
    (['--config', 'configs/pilot.yaml', '--steps', '64'], 64),
])
def test_training_budget_must_extend_beyond_random_warmup(tmp_path, monkeypatch, capsys,
                                                        budget_args, warmup):
    def unexpected_environment(*args, **kwargs):
        pytest.fail('nonlearning run reached simulator allocation')
    monkeypatch.setattr('scripts.train.make_env', unexpected_environment)
    output = tmp_path / 'unstarted'
    with pytest.raises(SystemExit) as exc:
        main(['--output', str(output), '--device', 'cpu', *budget_args])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert 'no gradient updates' in error
    assert str(warmup) in error and '--steps' in error and '--smoke' in error
    assert not output.exists()


@pytest.mark.parametrize('config_args', [[], ['--config', 'configs/pilot.yaml']])
def test_prepare_only_allows_budget_within_warmup(tmp_path, capsys, config_args):
    output = tmp_path / 'prepared_only'
    main(['--prepare-only', '--steps', '1', '--output', str(output),
          '--device', 'cpu', *config_args])
    assert 'random_warmup_env_steps:' in capsys.readouterr().out
    assert not output.exists()


def run_smoke(output,steps=40,*extra):
    main(['--smoke','--output',str(output),'--steps',str(steps),'--device','cpu',
          '--eval-every','0','--train-trace-every','0',*extra])


def test_episode_budget_stops_at_exact_completed_count(tmp_path):
    output=tmp_path/'episodes'
    main(['--smoke','--output',str(output),'--episodes','3','--smoke-episode-limit','2',
          '--device','cpu','--eval-every','0','--train-trace-every','0'])
    manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    with (output/'train.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==3 and [int(r['episode']) for r in rows]==[1,2,3]
    assert manifest['online_steps']==6 and manifest['completed_episodes']==3
    assert manifest['partial_episode_steps']==0 and manifest['requested_episodes']==3
    assert manifest['config']['training_budget']['requested']==3
    assert read(output/'latest.pt')['metadata']['mid_episode'] is False


@pytest.mark.parametrize('arguments', [['--episodes','0'],['--episodes','-1'],
                                     ['--episodes','1000','--steps','900000']])
def test_invalid_or_ambiguous_episode_budget_rejected(arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code==2


def test_smoke_retains_lightweight_policies_and_expected_output_tree(tmp_path):
    output=tmp_path/'run'; run_smoke(output,40,'--train-trace-every','1')
    assert (output/'latest.pt').is_file()
    assert not (output/'checkpoint.pt').exists()
    assert not (output/'validation_checkpoints').exists()
    assert {path.name for path in output.glob('*.pt')}=={'latest.pt'}
    assert (output/'latest.pt').stat().st_size<5_000_000
    payload=read(output/'latest.pt')
    assert 'replay' not in payload and 'agent' not in payload and 'optimizers' not in payload
    manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['online_steps']==40 and manifest['storage_policy']['replay_saved'] is False
    assert manifest['storage_policy']['retained_policies']==['latest.pt','best.pt']
    assert (output/'resolved_config.yaml').is_file() and (output/'diagnostics').is_dir()
    assert not list((output/'diagnostics').glob('*.jsonl'))
    assert not list((output/'diagnostics').glob('*.csv'))
    assert list(csv.DictReader((output/'updates.csv').open(encoding='utf-8-sig')))
    # Recompute from recorded simulator rewards, independently of CSV counters.
    import numpy as np
    from scripts.experiment_data import iter_records
    rows=list(csv.DictReader((output/'train.csv').open(encoding='utf-8-sig')))
    assert len(rows)==1
    row=rows[0]; gamma=float(row['reward_gamma'])
    steps=[step for record in iter_records(output/'diagnostics'/'training_steps.jsonl.gz')
           if record['episode_index']==int(row['episode']) for step in record['steps']]
    rewards=np.asarray([step['rewards'] for step in steps],dtype=float)
    discounted=(gamma**np.arange(len(steps)))@rewards
    np.testing.assert_allclose(float(row['discounted_team_return']),discounted.mean(),rtol=1e-6)
    np.testing.assert_allclose(float(row['discounted_usv_return']),discounted[:2].mean(),rtol=1e-6)
    np.testing.assert_allclose(float(row['discounted_auv_return']),discounted[2:].mean(),rtol=1e-6)
    np.testing.assert_allclose(float(row['team_return']),rewards.sum(0).mean(),rtol=1e-6)
    for prefix in ('','discounted_'):
        assert float(row[prefix+'team_return']) == pytest.approx(
            float(row[prefix+'process_return'])+float(row[prefix+'terminal_return']),abs=1e-6)
    before=(output/'latest.pt').read_bytes()
    with pytest.raises(SystemExit): run_smoke(output)
    assert (output/'latest.pt').read_bytes()==before


def test_actor_weight_warm_start_is_a_new_training_run(tmp_path):
    source=tmp_path/'source'; target=tmp_path/'target'; run_smoke(source,16)
    run_smoke(target,8,'--warm-start',str(source/'latest.pt'))
    manifest=json.loads((target/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['online_steps']==8
    assert manifest['initialization']['kind']=='actor_only_warm_start_new_training'
    assert manifest['initialization']['source_online_steps']==16
    assert manifest['initialization']['restored_replay'] is False
    assert manifest['initialization']['restored_optimizers'] is False


def test_in_memory_validation_keeps_only_latest_and_best(tmp_path):
    output=tmp_path/'validation'
    run_smoke(output,12,'--smoke-episode-limit','8','--eval-every','1','--eval-episodes','1',
              '--eval-trace-episodes','0','--export-diagnostics-csv')
    assert {path.name for path in output.rglob('*.pt')}=={'latest.pt','best.pt'}
    assert not (output/'validation_checkpoints').exists()
    rows=list(csv.DictReader((output/'diagnostics'/'evaluation_summary.csv').open(encoding='utf-8-sig')))
    assert len(rows)==1 and rows[0]['checkpoint_env_steps']=='8'
    assert len(rows[0]['policy_sha256'])==64
    best=read(output/'best.pt')
    assert best['metadata']['policy_sha256']==rows[0]['policy_sha256']
    episodes=list(csv.DictReader((output/'diagnostics'/'evaluation_episodes.csv').open(encoding='utf-8-sig')))
    assert float(rows[0]['eval_discounted_team_return']) == pytest.approx(
        sum(float(row['discounted_team_return']) for row in episodes)/len(episodes))
    from scripts.evaluate import main as evaluate_main
    evaluation=tmp_path/'heldout.csv'
    evaluate_main(['--checkpoint',str(output/'best.pt'),'--episodes','1','--base-seed','13',
                   '--trace-episodes','0','--output',str(evaluation)])
    heldout=list(csv.DictReader(evaluation.open(encoding='utf-8-sig')))
    assert len(heldout)==1 and heldout[0]['split']=='test'
    assert heldout[0]['policy_sha256']==rows[0]['policy_sha256']


def test_best_policy_selection_uses_discounted_return_without_relabeling_eval_return(tmp_path,monkeypatch):
    # Equal success rates: the middle policy has the best Bellman objective,
    # while the last has the largest undiscounted episodic reward.
    results=iter([(-10.,-9.),(-20.,-2.),(-5.,-8.)])
    def evaluation(*args,**kwargs):
        raw,discounted=next(results)
        return dict(success_rate=0.,return_value=raw,discounted_return_value=discounted,
                    safety_failure_rate=0.,capture_rate=0.,summaries=[],rows=[])
    monkeypatch.setattr('scripts.train.evaluate_policy',evaluation)
    output=tmp_path/'selected'
    run_smoke(output,24,'--smoke-episode-limit','8','--eval-every','1','--eval-episodes','1')
    best=read(output/'best.pt')
    assert best['global_env_steps']==16
    assert best['metadata']['selection_metric']=='success_then_safety_then_capture_then_discounted_return'
    assert best['metadata']['selection_key']==[0.,0.,0.,-2.]
    manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['best_validation_key']==[0.,0.,0.,-2.]
    assert manifest['best_validation_metric']=='success_then_safety_then_capture_then_discounted_return'
    rows=list(csv.DictReader((output/'train.csv').open(encoding='utf-8-sig')))
    assert [float(row['eval_return']) for row in rows]==[-10.,-20.,-5.]


def test_headless_scene_preview_needs_no_policy_or_calibration(capsys):
    from scripts.visualize import main as preview_main
    preview_main(['--preview','--headless','--max-steps','2'])
    assert 'UNTRAINED_PREVIEW' in capsys.readouterr().out
