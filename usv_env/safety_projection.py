"""Finite-candidate nonlinear force projection using receiver-local information.

This is a discrete predictive filter, not a proven CBF. Unknown vessels and
unobserved wave velocity cannot be certified. Infeasibility is never hidden.
"""
from collections import OrderedDict
import math
import time
import numpy as np

from .dynamics import USVDynamics, AUVDynamics3DOF
from .types import ShipState
from .history import schema_for_counts
from .safety_kernel import model_arrays, assess_motion, brake_action


def _model(cfg, submarine):
    ship=cfg['ship']; spec=ship['submarine_dynamics'] if submarine else ship['dynamics']
    model=AUVDynamics3DOF if submarine else USVDynamics
    force=spec['max_surge_force'] if submarine else ship['max_surge_force']
    moment=spec['max_yaw_moment'] if submarine else ship['max_yaw_moment']
    result=model(dt=cfg['env']['dt'],max_speed=ship['max_speed_submarine' if submarine else 'max_speed_blue'],
        allow_reverse=ship['allow_reverse'],reverse_speed_ratio=ship['submarine_reverse_speed_ratio' if submarine else 'blue_reverse_speed_ratio'],
        max_yaw_rate=ship['max_yaw_rate'],max_sway_speed=ship['max_sway_speed'],max_surge_force=force,max_yaw_moment=moment,
        parameters=spec['parameters'],internal_substeps=spec['internal_substeps'],
        reference_max_acceleration=spec['reference_max_acceleration'],reference_max_yaw_acceleration=spec['reference_max_yaw_acceleration'],
        reference_sway_damping=spec['reference_sway_damping'])
    result.record_trajectory=True
    return result


def _dimensions(cfg, surface):
    s=cfg['ship']
    return (s['surface_length_m'],s['surface_collision_width_m']) if surface else (s['submarine_collision_length_m'],s['submarine_collision_width_m'])


def _separated(cfg):
    d=cfg['depth']
    return d['auv_depth_m']-d['surface_keel_depth_m']>d['auv_vertical_radius_m']+d['vertical_safety_margin_m']


class _Predictor:
    def __init__(self,cfg,assess_cache_capacity=8192):
        self.cfg=cfg; self.models=[_model(cfg,False),_model(cfg,True)]
        self.model_data=[model_arrays(model) for model in self.models]
        self.world=np.asarray(cfg['env']['world_size'],np.float64)
        self.reefs=np.asarray([[*r['center'],r['radius']+cfg['depth']['reef_shallow_buffer_m']]
                               for r in cfg['reefs']],np.float64).reshape(-1,3)
        if type(assess_cache_capacity) is not int or assess_cache_capacity <= 0:
            raise ValueError('assessment cache capacity must be a positive integer')
        self._assess_cache_capacity=assess_cache_capacity
        self._assess_cache=OrderedDict()
        self._assess_cache_hits=self._assess_cache_misses=self._assess_cache_evictions=0

    @staticmethod
    def _assessment_key(own,obstacles,action):
        def exact_array(value):
            # Preserve original dtype, shape and bytes. In particular, obstacle
            # float32 arithmetic must never alias a float64 calculation.
            array=np.asarray(value)
            return array.dtype.str,array.shape,array.tobytes()
        return (own.unit_type,bool(own.surfaced),exact_array(own.pos),
            exact_array([own.speed,own.sway_speed,own.heading,own.yaw_rate]),
            tuple((exact_array(o['position']),exact_array(o['velocity']),o['surface'],
                   exact_array([o.get('uncertainty',0.)])) for o in obstacles),exact_array(action))

    def assess_cache_info(self):
        return dict(hits=self._assess_cache_hits,misses=self._assess_cache_misses,
            evictions=self._assess_cache_evictions,size=len(self._assess_cache),capacity=self._assess_cache_capacity)

    def _assessment(self,own,obstacles,action):
        # This predictor's configuration/models are fixed for its lifetime.
        # assess returns only a clearance scalar; model clamp counters are
        # diagnostic side effects and do not enter any predicted trajectory.
        key=self._assessment_key(own,obstacles,action)
        if key in self._assess_cache:
            self._assess_cache_hits+=1
            self._assess_cache.move_to_end(key)
            return self._assess_cache[key]
        self._assess_cache_misses+=1
        value=self._detail_uncached(own,obstacles,action)
        self._assess_cache[key]=value
        if len(self._assess_cache)>self._assess_cache_capacity:
            self._assess_cache.popitem(last=False)
            self._assess_cache_evictions+=1
        return value

    def assess(self,own,obstacles,action):
        return self._assessment(own,obstacles,action)[0]

    def _assess_uncached(self,own,obstacles,action):
        return self._detail_uncached(own,obstacles,action)[0]

    @staticmethod
    def _state_array(own):
        return np.array([*own.pos,own.speed,own.sway_speed,own.yaw_rate,own.heading],np.float64)

    def braking_action(self,own):
        index=int(own.unit_type!='surface')
        damping,mass,_,limits=self.model_data[index]
        return brake_action(self._state_array(own),damping,mass,limits,self.cfg['env']['dt'])

    def _detail_uncached(self,own,obstacles,action):
        cfg=self.cfg; surface=own.unit_type=='surface'; length,width=_dimensions(cfg,surface)
        packed=[]
        for obstacle in obstacles:
            if obstacle['surface']!=surface and _separated(cfg): continue
            other_length,_=_dimensions(cfg,obstacle['surface'])
            pos=np.asarray(obstacle['position']); velocity=np.asarray(obstacle['velocity'])
            packed.append([*pos,*velocity,.5*other_length,obstacle.get('uncertainty',0.),
                           velocity.dtype==np.float32,pos.dtype==np.float32])
        damping,mass,inverse,limits=self.model_data[int(not surface)]
        settings=cfg['i1']
        return assess_motion(self._state_array(own),np.asarray(action,np.float64),damping,mass,inverse,limits,
            cfg['env']['dt'],self.models[int(not surface)].internal_substeps,length,width,self.world,self.reefs,
            np.asarray(packed,np.float64).reshape(-1,8),int(settings['horizon_steps']),
            int(settings.get('backup_steps',0)),settings.get('stop_speed_mps',.02),
            settings.get('stop_yaw_rate_rps',.02),settings['margin_m'])

    def project_one(self,own,obstacles,nominal,limits,reject_only=False):
        nominal=np.clip(nominal,-limits,limits); margin=self.cfg['i1']['margin_m']
        initial,recovered,backup_seconds=self._assessment(own,obstacles,nominal)
        safe=initial>=margin and recovered
        if safe or reject_only:
            return nominal,dict(safe=safe,intervention=False,projection_norm=0.,infeasible=not safe,
                fallback=False,slack=max(0.,margin-initial),minimum_clearance=initial,candidates=1,
                terminal_recoverable=bool(recovered),backup_time_s=backup_seconds)
        # Enumerated actuator-space optimization, with exact same 3-DOF predictor.
        levels=int(self.cfg['i1'].get('candidate_levels',3))
        candidates=[np.array([f,y]) for f in np.linspace(-limits[0],limits[0],levels) for y in np.linspace(-limits[1],limits[1],levels)]
        candidates.extend([np.array([-limits[0],nominal[1]]),np.array([0.,nominal[1]])])
        if self.cfg['i1'].get('backup_steps',0): candidates.append(self.braking_action(own))
        candidates.sort(key=lambda a:float(np.sum(((a-nominal)/limits)**2)))
        best=nominal; best_clearance=initial; best_recovered=recovered; best_seconds=backup_seconds; evaluated=1
        for candidate in candidates:
            value,recovered,backup_seconds=self._assessment(own,obstacles,candidate); evaluated+=1
            if value>best_clearance:
                best,best_clearance,best_recovered,best_seconds=candidate,value,recovered,backup_seconds
            if value>=margin and recovered:
                return candidate,dict(safe=True,intervention=True,projection_norm=float(np.linalg.norm((candidate-nominal)/limits)),
                    infeasible=False,fallback=False,slack=0.,minimum_clearance=value,candidates=evaluated,
                    terminal_recoverable=bool(recovered),backup_time_s=backup_seconds)
        return best,dict(safe=False,intervention=bool(np.any(best!=nominal)),projection_norm=float(np.linalg.norm((best-nominal)/limits)),
            infeasible=True,fallback=True,slack=max(0.,margin-best_clearance),minimum_clearance=best_clearance,candidates=evaluated,
            terminal_recoverable=bool(best_recovered),backup_time_s=best_seconds)


def _pack(rows,elapsed):
    result={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    result['solver_time_s']=float(elapsed)
    return result


def _inactive_row(cfg):
    return dict(safe=True,intervention=False,projection_norm=0.,infeasible=False,fallback=False,
                slack=0.,minimum_clearance=float(max(cfg['env']['world_size'])),candidates=0,
                terminal_recoverable=True,backup_time_s=0.)


class SafetyProjector:
    def __init__(self,env):
        self.env=env; self.predictor=_Predictor(env.config)

    def project(self,actions):
        env=self.env; cfg=env.config; started=time.perf_counter(); rows=[]; result=[]
        for receiver,own in enumerate(env.friendly_states()):
            if not env.agent_active_mask()[receiver]:
                result.append(np.zeros(2)); rows.append(_inactive_row(cfg)); continue
            obstacles=[]
            for sender in range(env.num_friendly):
                packet=env.communication.cache.get((receiver,sender))
                if packet is None or sender==receiver or not env.agent_active_mask()[sender]: continue
                velocity=np.asarray(packet.velocity,float)
                obstacles.append(dict(position=np.asarray(packet.position,float),velocity=velocity,
                    heading=math.atan2(velocity[1],velocity[0]) if np.linalg.norm(velocity)>.02 else 0.,
                    surface=sender<env.num_surface_blue,uncertainty=cfg['i1']['uncertainty_margin_m']))
            estimate=env.target_estimates[receiver]
            if estimate.valid:
                velocity=np.asarray(estimate.velocity,float)
                obstacles.append(dict(position=np.asarray(estimate.position,float),velocity=velocity,
                    heading=math.atan2(velocity[1],velocity[0]) if np.linalg.norm(velocity)>.02 else 0.,surface=True,
                    uncertainty=cfg['i1']['uncertainty_margin_m']))
            action,row=self.predictor.project_one(own,obstacles,np.asarray(actions[receiver]),env.agent_action_limits[receiver])
            if not env.agent_active_mask()[receiver]: action=np.zeros(2)
            result.append(action); rows.append(row)
        return np.asarray(result,np.float32),_pack(rows,time.perf_counter()-started)


_PREDICTORS={}


def project_from_state(cfg,state,obs,actions,reject_only=False,receivers=None,decision_only=False):
    """Batch training projection: each receiver sees own state and its own last frame.

    Other rows of the centralized state are never read as obstacles. This is
    intentionally the same delayed/possibly missing information used at execution.
    """
    if decision_only and not reject_only: raise ValueError('decision_only requires reject_only')
    import json
    key=json.dumps({k:cfg[k] for k in ('env','ship','reefs','depth','i1')},sort_keys=True)
    if key not in _PREDICTORS: _PREDICTORS[key]=_Predictor(cfg)
    predictor=_PREDICTORS[key]; actions=np.asarray(actions,np.float32)
    state=np.asarray(state); obs=np.asarray(obs); single=actions.ndim==2
    if single: actions=actions[None]; state=state[None]; obs=obs[None]
    count=cfg['env']['num_surface_blue']+cfg['env']['num_submarines']; surface_count=cfg['env']['num_surface_blue']
    selected=set(range(count) if receivers is None else receivers)
    if not selected.issubset(range(count)): raise ValueError('invalid projection receivers')
    schema=schema_for_counts(count); frames=obs.reshape(len(obs),count,-1,schema.frame_dim)[:,:,-1]
    scale=max(cfg['env']['world_size']); results=[]; summaries=[]; started=time.perf_counter()
    if len(actions)>=32:
        from .safety_batch import project_batch
        result,diagnostics=project_batch(predictor,cfg,state,frames,actions,selected,reject_only,decision_only)
        diagnostics['solver_time_s']=time.perf_counter()-started
        return result,diagnostics
    def feature(frame,name): return float(frame[schema.index(name)])
    for batch in range(len(actions)):
        output=[]; rows=[]
        for i in range(count):
            if i not in selected:
                output.append(actions[batch,i]); rows.append(_inactive_row(cfg)); continue
            f=frames[batch,i]; row=state[batch,i*9:i*9+9]
            if not row[0]:
                output.append(np.zeros(2)); rows.append(_inactive_row(cfg)); continue
            heading=math.atan2(row[6],row[7]); c,s=math.cos(heading),math.sin(heading)
            rotation=np.array([[c,-s],[s,c]])
            vmax=cfg['ship']['max_speed_blue' if i<surface_count else 'max_speed_submarine']
            own=ShipState(np.asarray(row[2:4]*scale,np.float32),feature(f,'surge')*vmax,heading,
                feature(f,'yaw_rate')*cfg['ship']['max_yaw_rate'],'surface' if i<surface_count else 'submarine',i<surface_count,
                feature(f,'sway')*cfg['ship']['max_sway_speed'])
            obstacles=[]
            for j in range(count-1):
                if not feature(f,f'peer_{j}.active'): continue
                pos=own.pos+rotation@np.array([feature(f,f'peer_{j}.x'),feature(f,f'peer_{j}.y')])*scale
                vel=own.velocity+rotation@np.array([feature(f,f'peer_{j}.vx'),feature(f,f'peer_{j}.vy')])*1.5
                obstacles.append(dict(position=pos,velocity=vel,heading=math.atan2(vel[1],vel[0]) if np.linalg.norm(vel)>.02 else 0.,
                    surface=bool(feature(f,f'peer_{j}.surface')),uncertainty=cfg['i1']['uncertainty_margin_m']))
            if feature(f,'target_valid'):
                prefix='estimated_target_' if 'estimated_target_x' in schema.fields else 'red_'
                pos=own.pos+rotation@np.array([feature(f,prefix+'x'),feature(f,prefix+'y')])*scale
                vel=own.velocity+rotation@np.array([feature(f,prefix+'vx'),feature(f,prefix+'vy')])*1.5
                obstacles.append(dict(position=pos,velocity=vel,heading=math.atan2(vel[1],vel[0]) if np.linalg.norm(vel)>.02 else 0.,
                    surface=True,uncertainty=cfg['i1']['uncertainty_margin_m']))
            sub=cfg['ship']['submarine_dynamics']; limits=np.array([cfg['ship']['max_surge_force'],cfg['ship']['max_yaw_moment']]) if i<surface_count else np.array([sub['max_surge_force'],sub['max_yaw_moment']])
            projected,diagnostics=predictor.project_one(own,obstacles,actions[batch,i],limits,reject_only)
            if not row[0]: projected=np.zeros(2)
            output.append(projected); rows.append(diagnostics)
        results.append(output); summaries.append(dict(safe_mask=all(r['safe'] for r in rows),
            intervention=any(r['intervention'] for r in rows),projection_norm=sum(r['projection_norm'] for r in rows),
            infeasible=any(r['infeasible'] for r in rows),slack=max(r['slack'] for r in rows)))
    diagnostics={k:np.array([r[k] for r in summaries]) for k in summaries[0]}
    if decision_only: diagnostics={'safe_mask':diagnostics['safe_mask']}
    diagnostics['solver_time_s']=time.perf_counter()-started
    result=np.asarray(results,np.float32)
    return (result[0] if single else result),diagnostics
