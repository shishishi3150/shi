"""Merge per-seed evaluation evidence into plot-ready CSVs; never pool test episodes as training seeds."""
import sys
from pathlib import Path
_PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(_PROJECT_ROOT))
import argparse
import csv
import json
from scripts.experiment_data import atomic_json,file_hash,iter_records,resolve_log
from scripts.experiment_identity import comparison_contract


def run_contract(root, row):
    """Use recorded conditions, never infer equal budgets from equal source."""
    recorded = row.get('comparison_contract')
    cfg = None
    configuration = root/'resolved_config.yaml'
    manifest = root/'run_manifest.json'
    if configuration.exists():
        import yaml
        cfg = yaml.safe_load(configuration.read_text(encoding='utf-8-sig'))
    elif manifest.exists():
        cfg = json.loads(manifest.read_text(encoding='utf-8-sig')).get('config')
    if cfg is not None:
        expected = comparison_contract(cfg)
        if recorded is not None and recorded != expected:
            raise ValueError(f'recorded comparison contract differs from configuration: {root}')
        recorded = expected
    if not isinstance(recorded, dict):
        raise ValueError(f'missing experiment comparison contract/configuration: {root}')
    names = dict(task='training_task_hash',reward='reward_hash',observation='observation_hash',
                 architecture='architecture_hash',i1='i1_config_hash')
    for key, name in names.items():
        if name in row and row[name] != recorded['hashes'][key]:
            raise ValueError(f'row {name} differs from comparison contract: {root}')
    if row.get('training_source_code_hash') != recorded.get('source'):
        raise ValueError(f'row source differs from comparison contract: {root}')
    # Unknown source is inspectable in one run, not proof that two runs match.
    if recorded.get('source') is None:
        recorded = {**recorded, 'source': 'unrecorded:'+str(root.resolve())}
    return recorded


def write_table(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='run directories, each containing diagnostics/')
    p.add_argument('--output',required=True)
    p.add_argument('--allow-smoke',action='store_true',help='implementation demonstration only')
    args=p.parse_args(argv); output=Path(args.output)
    if output.exists(): p.error('refusing to overwrite comparison output')
    rows=[]; provenance=[]; seen=set(); contract=None
    for root in map(Path,args.runs):
        source=resolve_log(root/'diagnostics/evaluation_summary.jsonl')
        current=list(iter_records(source))
        for row in current:
            if row['purpose']=='implementation_smoke' and not args.allow_smoke: p.error('smoke data is not research evidence')
            try: signature=run_contract(root,row)
            except ValueError as error: p.error(str(error))
            if contract is None: contract=signature
            if signature!=contract: p.error('incompatible task/reward/observation/architecture/I1/training/search/initialization contracts')
            key=tuple(row.get(k) for k in ('algorithm','experiment','train_seed','checkpoint_env_steps','condition_hash','split'))
            if key in seen: p.error('duplicate model/seed/checkpoint/condition; do not double-count runs')
            seen.add(key); rows.append(row)
        provenance.append(dict(source=str(source.resolve()),sha256=file_hash(source)))
    output.mkdir(parents=True)
    write_table(output/'comparison_by_seed.csv',rows)
    atomic_json(output/'sources.json',dict(sources=provenance,contract=contract,
        unit='one model-training-seed/checkpoint/condition per row',
        checkpoint_alignment='actual and planned steps retained; no interpolation',
        aggregation='none; preserve independent training seeds for downstream intervals'))
    print(f'Saved {len(rows)} per-seed evaluation rows to {output}')


if __name__=='__main__': main()
