"""Causal CSV curves; shaded variation is across independent training seeds."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
import argparse
import csv
import gzip
import json
from copy import deepcopy
import numpy as np
import yaml
from usv_env.config import ROOT,PRESETS,refresh_hashes,validate
from scripts.csv_logging import CsvLog,TRAIN_FIELDS,LEGACY_TRAIN_FIELDS
from scripts.experiment_identity import comparison_contract, default_runs_root


def _episode_diagnostics(root):
    """Join observed episode summaries by BOTH episode index and online step."""
    paths=[root/'diagnostics'/'training_episodes.jsonl',root/'diagnostics'/'training_episodes.jsonl.gz']
    found={}
    for path in paths:
        if not path.exists(): continue
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                if not line.strip(): continue
                row=json.loads(line)
                key=(int(row['episode_index']),int(row['env_steps']))
                if key in found: raise ValueError(f'duplicate episode diagnostics: {path}: {key}')
                found[key]=row
    return found


def _trailing(values,window):
    # Missing observations remain missing, including in their trailing window.
    return np.array([values[max(0,i-window+1):i+1].mean() for i in range(len(values))])


def read_run(path,window):
    path=Path(path)
    if window<1: raise ValueError('window must be positive')
    cfg=yaml.safe_load((path.parent/"resolved_config.yaml").read_text(encoding="utf-8"))
    validate(cfg)
    expected=deepcopy(cfg["hashes"])
    if refresh_hashes(cfg)!=expected: raise ValueError(f"hash mismatch: {path}")
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    if fields not in (TRAIN_FIELDS,LEGACY_TRAIN_FIELDS): raise ValueError(f'CSV schema mismatch: {path}')
    CsvLog(path,fields,resume=True)
    if not rows: raise ValueError(f"no completed real episodes in {path}")
    steps=np.array([int(r["env_steps"]) for r in rows])
    if np.any(np.diff(steps)<=0): raise ValueError(f"duplicate/nonmonotonic steps in {path}")
    if 'reward_gamma' in fields and any(float(r['reward_gamma'])!=cfg['training']['gamma'] for r in rows if r['reward_gamma']):
        raise ValueError(f'reward gamma differs from resolved config: {path}')
    raw={}
    for key in ("team_return","usv_return","auv_return","process_return","terminal_return","episode_steps","success"):
        values=np.array([float(r[key]) for r in rows])
        raw[key]=values
    diagnostics=_episode_diagnostics(path.parent)
    discounted=[]
    for row in rows:
        value=row.get('discounted_team_return')
        if not value:
            observed=diagnostics.get((int(row['episode']),int(row['env_steps'])),{})
            value=observed.get('discounted_team_return')
        discounted.append(np.nan if value is None or value=='' else float(value))
    raw['discounted_team_return']=np.asarray(discounted,dtype=float)
    if np.isinf(raw['discounted_team_return']).any(): raise ValueError(f'nonfinite discounted return: {path}')
    terminal_abs=np.abs(raw['terminal_return'])
    process_abs=np.array([float(r['process_abs_sum']) for r in rows])
    denominator=process_abs+terminal_abs
    raw['terminal_abs_fraction']=np.divide(terminal_abs,denominator,out=np.full(len(rows),np.nan),where=denominator>0)
    failures=np.array([r["terminal_reason"] in ("collision","reef_collision","out_of_bounds") for r in rows])
    raw['safety_failure_rate']=failures
    for reason in ("breakthrough","collision","reef_collision","out_of_bounds","timeout","scenario_anomaly","numerical_error"):
        values=np.array([r["terminal_reason"]==reason for r in rows])
        raw[reason+"_rate"]=values
    metrics={key:_trailing(values,window) for key,values in raw.items()}
    metrics.update({'raw_'+key:values for key,values in raw.items()})
    source=cfg.get('provenance',{}).get('code_hash')
    manifest_path=path.parent/'run_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        recorded=manifest.get('source_code_hash') or manifest.get('training_source_code_hash')
        if source and recorded and source!=recorded: raise ValueError(f'source code hash mismatch: {path}')
        source=source or recorded
    # Unknown provenance is usable for one run but cannot silently form a cohort.
    source=source or 'unrecorded:'+str(path.parent.resolve())
    # Different training/search budgets are separate experimental cohorts even
    # when task/reward/observation and source code are identical.
    identity=comparison_contract(cfg,source=source)
    return steps,metrics,identity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root",default=None)
    p.add_argument("--seeds",nargs="+",type=int,default=[0])
    p.add_argument("--experiments",nargs="+",choices=PRESETS,default=["Base"])
    p.add_argument("--window-episodes",type=int,default=50)
    p.add_argument("--output-dir",default=None)
    args=p.parse_args(argv)
    if args.window_episodes<1: p.error("window must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(args.runs_root).resolve() if args.runs_root else default_runs_root()
    dest=Path(args.output_dir) if args.output_dir else root/"plots"
    groups={}; common_identity=None
    for experiment in args.experiments:
        paths=[root/experiment/f"seed{seed}"/"train.csv" for seed in args.seeds]
        if any(not path.exists() for path in paths):
            print(f"Skipped {experiment}: missing requested seed logs"); continue
        runs=[read_run(path,args.window_episodes) for path in paths]
        for _,_,identity in runs:
            if common_identity is not None and identity!=common_identity: raise ValueError("cannot mix source code, task/reward/schema hashes, gamma or purpose")
            common_identity=identity
        groups[experiment]=runs
    if not groups: p.error("no complete requested experiment/seed logs")
    keys=("team_return","discounted_team_return","success","process_return","terminal_abs_fraction","episode_steps","usv_return","auv_return","breakthrough_rate","collision_rate","reef_collision_rate","out_of_bounds_rate","timeout_rate","scenario_anomaly_rate","numerical_error_rate")
    fig,axes=plt.subplots(5,3,figsize=(15,17),constrained_layout=True)
    for experiment,runs in groups.items():
        start=max(x[0][0] for x in runs); end=min(x[0][-1] for x in runs)
        grid=np.unique(np.concatenate([x[0][(x[0]>=start)&(x[0]<=end)] for x in runs]))
        if not len(grid): raise ValueError("seed curves have no shared observed horizon")
        for axis,key in zip(axes.flat,keys):
            aligned=np.stack([metrics[key][np.searchsorted(steps,grid,side="right")-1] for steps,metrics,_ in runs])
            mean=aligned.mean(0); line,=axis.plot(grid,mean,label=experiment,marker=".",markersize=5)
            if len(runs)==1:
                steps,metrics,_=runs[0]
                axis.plot(steps,metrics['raw_'+key],color=line.get_color(),alpha=.25,linewidth=.8)
            if not np.isfinite(mean).any():
                axis.text(.02,.98-.06*list(groups).index(experiment),f'{experiment}: no recorded values',
                          transform=axis.transAxes,ha='left',va='top',fontsize=8)
            if len(runs)>1:
                sd=aligned.std(0,ddof=1); axis.fill_between(grid,mean-sd,mean+sd,alpha=.15)
            axis.set(xlabel="Actual online environment steps",ylabel=key.replace("_"," "))
            axis.grid(alpha=.25)
            if key in ('success','terminal_abs_fraction') or key.endswith("_rate"): axis.set_ylim(-.02,1.02)
    axes[0,0].legend()
    purpose="IMPLEMENTATION SMOKE — " if common_identity['purpose']=="implementation_smoke" else ""
    fig.suptitle(purpose+f"Trailing {args.window_episodes} completed episodes; faint raw line (single seed), ±1 SD (multiple seeds)")
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/("learning_seeds_"+"_".join(map(str,args.seeds))+".png")
    fig.savefig(path,dpi=160); plt.close(fig)
    print(f"Saved {path}")
    return 0


if __name__=="__main__": main()
