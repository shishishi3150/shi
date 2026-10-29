"""Merge per-seed evaluation evidence into plot-ready CSVs; never pool test episodes as training seeds."""
import argparse
from pathlib import Path
import csv
import json
from collections import Counter,defaultdict
from scripts.experiment_data import atomic_json,file_hash,iter_records


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
    rows=[]; provenance=[]; seen=set(); contract=None; windows=[]
    for root in map(Path,args.runs):
        source=root/'diagnostics/evaluation_summary.jsonl'
        current=list(iter_records(source))
        for row in current:
            if row['purpose']=='implementation_smoke' and not args.allow_smoke: p.error('smoke data is not research evidence')
            signature=tuple(row.get(k) for k in ('training_task_hash','reward_hash','observation_hash','training_source_code_hash'))
            if contract is None: contract=signature
            if signature!=contract: p.error('incompatible task/reward/observation/source contracts')
            key=tuple(row.get(k) for k in ('algorithm','experiment','train_seed','checkpoint_env_steps','condition_hash','split'))
            if key in seen: p.error('duplicate model/seed/checkpoint/condition; do not double-count runs')
            seen.add(key); rows.append(row)
        provenance.append(dict(source=str(source.resolve()),sha256=file_hash(source)))
        teacher=root/'teacher.jsonl'
        if teacher.exists():
            groups=defaultdict(list)
            for record in iter_records(teacher): groups[int(record['env_steps'])//10000].append(record)
            for bucket,events in groups.items():
                started=sum(bool(e.get('search_started',False)) for e in events)
                accepted=sum(e['status']=='accepted' for e in events)
                windows.append(dict(run=str(root),experiment=events[0].get('experiment'),train_seed=events[0].get('train_seed'),
                    window_start=bucket*10000,window_end_exclusive=(bucket+1)*10000,
                    scheduled=len(events),search_started=started,accepted=accepted,
                    acceptance_rate=accepted/started if started else None,
                    simulator_steps=sum(e['simulator_steps'] for e in events),
                    reasons=dict(Counter(e['status'] for e in events))))
    output.mkdir(parents=True)
    write_table(output/'comparison_by_seed.csv',rows)
    write_table(output/'teacher_windows.csv',windows)
    atomic_json(output/'sources.json',dict(sources=provenance,contract=contract,
        unit='one model-training-seed/checkpoint/condition per row',
        checkpoint_alignment='actual and planned steps retained; no interpolation',
        aggregation='none; preserve independent training seeds for downstream intervals'))
    print(f'Saved {len(rows)} per-seed evaluation rows to {output}')


if __name__=='__main__': main()
