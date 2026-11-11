"""Current sustained-v5 acceptance; historical reward preflight is retired."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import subprocess
import sys
import yaml
from usv_env.config import ROOT,assemble,code_hash


def protected_integrity(backup):
    protected=['matd3/agent.py','matd3/history_enhancement.py','matd3/history_kernel.py',
        'matd3/networks.py','matd3/checkpoint.py','usv_env/vo_guidance.py','usv_env/safety_kernel.py',
        'usv_env/target_tracking.py','usv_env/communication.py','usv_env/dynamics.py',
        'usv_env/dynamics_kernel.py','usv_env/red_policy.py','usv_env/observation.py']
    with ZipFile(backup) as z:
        for n in protected:
            assert z.read(n)==(ROOT/n).read_bytes(),f'protected implementation changed: {n}'
        old=yaml.safe_load(z.read('configs/scene_template.yaml'))
        new=yaml.safe_load((ROOT/'configs/scene_template.yaml').read_text(encoding='utf-8-sig'))
        for key in old:
            if key not in ('task','reward','outputs'):assert old[key]==new[key],key
    return dict(passed=True,protected_byte_identical=protected,
        task_reward_output_only_config_changes=True,backup_sha256=hashlib.sha256(Path(backup).read_bytes()).hexdigest())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--backup',required=True);p.add_argument('--spec');args=p.parse_args(argv)
    root=Path(args.output)
    if root.exists():p.error('use a new output directory')
    root.mkdir(parents=True)
    result=dict(implementation_verified=False,learned_policy_success_verified=False,formal_training_launched=False,
        source_code_hash=code_hash(),protected_integrity=protected_integrity(args.backup),commands=[])
    if args.spec:result['spec_sha256']=hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()
    commands=[('tests',['-m','pytest','-q','-p','no:cacheprovider']),
        ('physical',['-m','scripts.validate_sustained_reward','--output',str(root/'physical')]),
        ('smoke',['-m','scripts.train','--config','configs/fast.yaml','--experiment','Full','--steps','96','--smoke','--device','cpu','--eval-every','0','--output',str(root/'smoke')])]
    for name,arguments in commands:
        with (root/(name+'.log')).open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B',*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        result['commands'].append(dict(name=name,exit_code=r.returncode))
        if r.returncode:raise RuntimeError(f'{name} failed; see {root/(name+".log")}')
    assert result['source_code_hash']==code_hash(),'source changed during acceptance'
    result['implementation_verified']=True
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    return 0

if __name__=='__main__':raise SystemExit(main())
