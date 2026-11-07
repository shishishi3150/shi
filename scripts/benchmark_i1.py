"""Small reproducible I1 batch stress benchmark; never starts training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import time
import numpy as np
from scripts.experiment_data import atomic_json
from usv_env.config import assemble, code_hash
from usv_env.environment import PursuitEvasionEnv
from usv_env.history import HistoryEnv
from usv_env.observation import build_actor_observation, build_critic_state
from usv_env.safety_projection import project_from_state, _PREDICTORS


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'validation/five_agent_acceptance/i1_optimization/batch_benchmark.json'))
    args=p.parse_args(argv); output=Path(args.output)
    if output.exists(): p.error('refusing to overwrite an existing benchmark')
    cfg=assemble(experiment='Full',device='cpu'); env=PursuitEvasionEnv(cfg)
    histories=HistoryEnv(env,cfg['revision']['history_length']); states=[]; observations=[]
    for i in range(256):
        histories.reset(seed=i)
        # Deliberately mix far-field and moving-toward-reef receivers. These
        # are synthetic local-input fixtures, not evaluation success samples.
        if i%2:
            own=env.blue_states[0]; own.pos=np.array([25.8+i*.001,28.],np.float32)
            own.heading=0.;own.speed=.4;own.sway_speed=.03;own.yaw_rate=.02
        raw=np.stack([build_actor_observation(env,j) for j in range(env.num_friendly)])
        observations.append(histories._push(raw));states.append(build_critic_state(env))
    env.close(); states=np.asarray(states);observations=np.asarray(observations)
    actions=np.broadcast_to(np.array([[2.2,.3]]*2+[[3.035,.3]]*3,np.float32),(256,5,2)).copy()
    # Load/compile once; explicitly exclude this from stable batch timings.
    started=time.perf_counter()
    project_from_state(cfg,states[:1],observations[:1],actions[:1])
    startup=time.perf_counter()-started; rows=[]
    for batch in (32,256):
        _PREDICTORS.clear(); started=time.perf_counter()
        expected,detail=project_from_state(cfg,states[:batch],observations[:batch],actions[:batch])
        cold=time.perf_counter()-started; warm=[]
        for _ in range(5):
            started=time.perf_counter()
            actual,again=project_from_state(cfg,states[:batch],observations[:batch],actions[:batch])
            warm.append(time.perf_counter()-started)
            np.testing.assert_array_equal(actual,expected)
            for key in detail:
                if key!='solver_time_s': np.testing.assert_array_equal(detail[key],again[key])
        rows.append(dict(batch_size=batch,cold_assessment_cache_s=cold,
            warm_assessment_cache_median_s=float(np.median(warm)),
            intervention_fraction=float(detail['intervention'].mean()),
            infeasible_fraction=float(detail['infeasible'].mean()),
            action_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
    atomic_json(output,dict(source_code_hash=code_hash(),settings=cfg['i1'],
        scope='Synthetic mixed far/near-reef batch projection, not training/evaluation episodes.',
        predictor_first_call_s=startup,rows=rows))
    print(output)
    for row in rows: print(row)


if __name__=='__main__': main()
