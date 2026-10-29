"""Causal CSV curves; shaded variation is across independent training seeds."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
import argparse
import csv
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
from usv_env.config import ROOT,PRESETS,refresh_hashes,validate
from scripts.csv_logging import CsvLog,TRAIN_FIELDS


def read_run(path,window):
    cfg=yaml.safe_load((path.parent/"resolved_config.yaml").read_text(encoding="utf-8"))
    validate(cfg)
    expected=deepcopy(cfg["hashes"])
    if refresh_hashes(cfg)!=expected: raise ValueError(f"hash mismatch: {path}")
    CsvLog(path,TRAIN_FIELDS,resume=True)
    with path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError(f"no completed real episodes in {path}")
    steps=np.array([int(r["env_steps"]) for r in rows])
    if np.any(np.diff(steps)<=0): raise ValueError(f"duplicate/nonmonotonic steps in {path}")
    metrics={}
    for key in ("team_return","usv_return","auv_return","process_return","success"):
        values=np.array([float(r[key]) for r in rows])
        metrics[key]=np.array([values[max(0,i-window+1):i+1].mean() for i in range(len(values))])
    failures=np.array([r["terminal_reason"] in ("landing","collision","out_of_bounds") for r in rows])
    metrics["safety_failure_rate"]=np.array([failures[max(0,i-window+1):i+1].mean() for i in range(len(failures))])
    for reason in ("landing","collision","out_of_bounds","timeout"):
        values=np.array([r["terminal_reason"]==reason for r in rows])
        metrics[reason+"_rate"]=np.array([values[max(0,i-window+1):i+1].mean() for i in range(len(values))])
    identity=tuple(expected[k] for k in ("task","reward","observation"))+(cfg["revision"]["history_length"],cfg.get("purpose","training"))
    return steps,metrics,identity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root",default=str(ROOT/"outputs"))
    p.add_argument("--seeds",nargs="+",type=int,default=[0])
    p.add_argument("--experiments",nargs="+",choices=PRESETS,default=["Base","w_o_I1","w_o_I2","w_o_I3","Full"])
    p.add_argument("--window-episodes",type=int,default=50)
    p.add_argument("--output-dir",default=None)
    args=p.parse_args(argv)
    if args.window_episodes<1: p.error("window must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(args.runs_root).resolve(); dest=Path(args.output_dir) if args.output_dir else root/"plots"
    groups={}; common_identity=None
    for experiment in args.experiments:
        paths=[root/experiment/f"seed{seed}"/"train.csv" for seed in args.seeds]
        if any(not path.exists() for path in paths):
            print(f"Skipped {experiment}: missing requested seed logs"); continue
        runs=[read_run(path,args.window_episodes) for path in paths]
        for _,_,identity in runs:
            if common_identity is not None and identity!=common_identity: raise ValueError("cannot mix task/reward/schema hashes")
            common_identity=identity
        groups[experiment]=runs
    if not groups: p.error("no complete requested experiment/seed logs")
    keys=("team_return","usv_return","auv_return","process_return","success","landing_rate","collision_rate","out_of_bounds_rate","timeout_rate")
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for experiment,runs in groups.items():
        start=max(x[0][0] for x in runs); end=min(x[0][-1] for x in runs)
        grid=np.unique(np.concatenate([x[0][(x[0]>=start)&(x[0]<=end)] for x in runs]))
        if not len(grid): raise ValueError("seed curves have no shared observed horizon")
        for axis,key in zip(axes.flat,keys):
            aligned=np.stack([metrics[key][np.searchsorted(steps,grid,side="right")-1] for steps,metrics,_ in runs])
            mean=aligned.mean(0); axis.plot(grid,mean,label=experiment,marker=".",markersize=5)
            if len(runs)>1:
                sd=aligned.std(0,ddof=1); axis.fill_between(grid,mean-sd,mean+sd,alpha=.15)
            axis.set(xlabel="Actual online environment steps",ylabel=key.replace("_"," "))
            axis.grid(alpha=.25)
            if key=="success" or key.endswith("_rate"): axis.set_ylim(-.02,1.02)
    axes[0,0].legend()
    purpose="IMPLEMENTATION SMOKE — " if common_identity[-1]=="implementation_smoke" else ""
    fig.suptitle(purpose+f"Trailing {args.window_episodes} completed episodes; seed mean ± 1 SD (multiple seeds only)")
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/("learning_seeds_"+"_".join(map(str,args.seeds))+".png")
    fig.savefig(path,dpi=160); plt.close(fig)
    print(f"Saved {path}")
    return 0


if __name__=="__main__": main()
