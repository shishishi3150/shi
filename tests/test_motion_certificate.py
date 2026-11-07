"""Free-space certificate cannot accept a trajectory rejected by full I1."""
import math
import numpy as np
import pytest
from usv_env.config import assemble
from usv_env.safety_projection import _Predictor
from usv_env.safety_kernel import assess_motion, assess_nominal_motion
from usv_env import safety_kernel


@pytest.mark.parametrize('submarine', [False, True])
def test_certificate_only_accepts_full_safe_recoverable_motion(submarine):
    certify = getattr(safety_kernel, 'certify_free_motion', None)
    assert callable(certify)
    cfg = assemble(experiment='Full', device='cpu')
    predictor = _Predictor(cfg)
    model = predictor.models[int(submarine)]
    arrays = predictor.model_data[int(submarine)]
    length, width = (cfg['ship']['submarine_collision_length_m'], cfg['ship']['submarine_collision_width_m']) if submarine else (cfg['ship']['surface_length_m'], cfg['ship']['surface_collision_width_m'])
    rng = np.random.default_rng(914)
    accepted = 0
    for row in range(240):
        pos = rng.uniform(1.,54.,2)
        q = np.array([*pos, rng.uniform(model.min_speed, model.max_speed),
            rng.uniform(-model.max_sway_speed, model.max_sway_speed),
            rng.uniform(-model.max_yaw_rate, model.max_yaw_rate), rng.uniform(-math.pi,math.pi)])
        obstacles = np.zeros((2,8))
        obstacles[:, :2] = pos+rng.uniform(-8.,8.,(2,2))
        obstacles[:, 2:4] = rng.uniform(-.4,.4,(2,2))
        obstacles[:, 4:6] = [.7,.1]
        action = rng.uniform(-1.,1.,2)*[model.max_surge_force,model.max_yaw_moment]
        args = (q, action, *arrays, cfg['env']['dt'],model.internal_substeps,
            length,width,predictor.world,predictor.reefs,obstacles,
            cfg['i1']['horizon_steps'],cfg['i1']['backup_steps'],
            cfg['i1']['stop_speed_mps'],cfg['i1']['stop_yaw_rate_rps'],cfg['i1']['margin_m'])
        full = assess_motion(*args)
        reused = assess_nominal_motion(*args)
        assert reused[1:] == full[1:]
        if full[0] < cfg['i1']['margin_m'] or not full[1]:
            assert reused[0] == full[0], (row, full, reused)
        else:
            assert cfg['i1']['margin_m'] <= reused[0] <= full[0]+1.e-9
        certificate = certify(*args)
        if certificate[0]:
            accepted += 1
            assert full[0] >= cfg['i1']['margin_m'] and full[1], (row,full,certificate)
            assert certificate[1] <= full[0]+1.e-9
            assert certificate[2] == full[2]
    assert accepted > 20
