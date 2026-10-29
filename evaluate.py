import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT))
import argparse
import uuid
from copy import deepcopy
from pathlib import Path
import torch
from matd3.checkpoint import load_policy
from usv_env.config import ROOT,PRESETS,run_directory,code_hash
from scripts.research_evaluation import evaluate_policy
from scripts.experiment_data import ExperimentWriter,export_csv,file_hash


def main(argv=None):
    p=argparse.ArgumentParser(description="Frozen v4.3 evaluation with held-out seeds")
    p.add_argument("--checkpoint")
    p.add_argument("--runs-root",default=str(ROOT/"outputs"))
    p.add_argument("--experiment",choices=PRESETS,default="Base")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--base-seed",type=int,default=2000000)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="cpu")
    p.add_argument("--variant",choices=("fixed","randomized"),default="randomized")
    p.add_argument("--output")
    p.add_argument('--suite',choices=('nominal','screen','communication'),default='nominal')
    p.add_argument('--trace-episodes',type=int,default=1)
    args=p.parse_args(argv)
    if args.episodes<1 or args.trace_episodes<0: p.error("episodes must be positive; traces nonnegative")
    path=Path(args.checkpoint) if args.checkpoint else run_directory(args.runs_root,args.experiment,args.seed)/"latest.pt"
    destination=Path(args.output) if args.output else path.parent/"evaluation"/f"{path.stem}_{args.variant}_{args.suite}_{args.base_seed}.csv"
    if destination.suffix.lower()!='.csv': p.error('--output must be a .csv path')
    if destination.exists(): p.error("evaluation output already exists; choose another --output")
    data_root=destination.with_suffix('')
    if data_root.exists(): p.error('evaluation detail directory exists; choose another output')
    agent,_=load_policy(path,args.device); cfg=deepcopy(agent.runtime_config)
    steps=agent.global_env_steps
    identity=dict(run_id=uuid.uuid4().hex,algorithm='MATD3',experiment=cfg['revision']['experiment'],
                  train_seed=cfg['training']['seed'],purpose=cfg.get('purpose','held_out_evaluation'))
    torch.set_num_threads(cfg["training"]["torch_num_threads"])
    checkpoint_hash=file_hash(path)
    writer=ExperimentWriter(data_root,cfg,identity=identity)
    result=evaluate_policy(agent,cfg,args.episodes,args.base_seed,args.variant,suite=args.suite,
        writer=writer,checkpoint_id=checkpoint_hash,checkpoint_steps=steps,planned_steps=steps,
        split='test',trace_episodes=args.trace_episodes)
    writer.export()
    writer.manifest(steps,None,dict(evaluation_only=True,teacher_training_steps_known=False,
        evaluation_source_code_hash=code_hash(),checkpoint=str(path.resolve()),
        checkpoint_sha256=checkpoint_hash,evaluation_sim_steps=sum(s['episode_steps_total'] for s in result['summaries'])))
    export_csv(writer.logs['evaluation_episodes'].path,destination)
    print(f"Saved {destination}; nominal success_rate={result['success_rate']:.6f}, return={result['return_value']:.6f}; conditions={len(result['summaries'])}")


if __name__=="__main__": main()
