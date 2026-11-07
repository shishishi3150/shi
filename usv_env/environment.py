"""Five-agent reef interception using the established heterogeneous 3-DOF physics."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
import math
import time
import numpy as np

from .config import validate, named_seed
from .dynamics import USVDynamics, AUVDynamics3DOF
from .geometry import capsule_segment, capsule_circle_clearance, capsule_inside_rectangle, capsules_overlap
from .history import schema_for_counts
from .communication import CommunicationNetwork, TargetMessageBus
from .target_tracking import LocalTargetSensor, TargetEstimator, TargetEstimate, information_mode
from .observation import build_actor_observation, build_critic_state, critic_state_dim
from .red_policy import RedNavigator
from .waves import RegularLinearWave
from .task import TaskState, encirclement_geometry
from .types import ShipState, EnvInfo
from .safety import center_path_enters_circle


class PursuitEvasionEnv:
    def __init__(self, config, seed=None):
        validate(config)
        from .reward import validate_settings
        validate_settings(config['reward'], config['training']['gamma'], config['env']['max_steps'])
        self.config=deepcopy(config)
        self.seed=int(config['scenario']['seed'] if seed is None else seed)
        e,s=config['env'],config['ship']
        self.world_size=np.asarray(e['world_size'],np.float32)
        self.dt=float(e['dt']); self.max_steps=int(e['max_steps'])
        # Missing in historical actor configs: preserve their immediate ending.
        self.post_capture_end_step=int(config['task'].get('post_capture_end_step',0))
        self.num_surface_blue=int(e['num_surface_blue']); self.num_submarines=int(e['num_submarines'])
        self.num_friendly=self.num_agents=self.num_surface_blue+self.num_submarines
        self.num_surface=self.num_surface_blue
        self.surface_slice=slice(0,self.num_surface_blue); self.submarine_slice=slice(self.num_surface_blue,self.num_friendly)
        self.surface_length=float(s['surface_length_m']); self.surface_collision_width=float(s['surface_collision_width_m'])
        self.submarine_length=float(s['submarine_collision_length_m']); self.submarine_collision_width=float(s['submarine_collision_width_m'])
        self.semi_major=float(s['semi_major']); self.semi_minor=float(s['semi_minor'])
        self.max_speed_by_type={'surface':float(s['max_speed_blue']),'submarine':float(s['max_speed_submarine'])}
        self.max_speed_red=float(s['max_speed_red']); self.max_yaw_rate=float(s['max_yaw_rate']); self.max_sway_speed=float(s['max_sway_speed'])
        self.surface_action_limit=np.array([s['max_surge_force'],s['max_yaw_moment']],np.float32)
        sub=s['submarine_dynamics']
        self.submarine_action_limit=np.array([sub['max_surge_force'],sub['max_yaw_moment']],np.float32)
        self.agent_action_limits=np.concatenate([np.tile(self.surface_action_limit,(self.num_surface_blue,1)),np.tile(self.submarine_action_limit,(self.num_submarines,1))])
        self.action_limit=self.surface_action_limit; self.action_dim=2
        self.blue_dynamics=self._dynamics(USVDynamics,s['dynamics'],s['max_speed_blue'],s['blue_reverse_speed_ratio'],self.surface_action_limit)
        self.red_dynamics=self._dynamics(USVDynamics,s['dynamics'],s['max_speed_red'],1.,self.surface_action_limit)
        self.submarine_dynamics=self._dynamics(AUVDynamics3DOF,sub,s['max_speed_submarine'],s['submarine_reverse_speed_ratio'],self.submarine_action_limit)
        for dynamics in (self.blue_dynamics,self.red_dynamics,self.submarine_dynamics): dynamics.record_trajectory=True
        self.reefs=deepcopy(config['reefs'])
        self.protected_center=np.asarray(config['protected_zone']['center'],np.float32)
        self.protected_radius=float(config['protected_zone']['radius'])
        self.route_origin=np.asarray(config['scenario']['red_start'],np.float32); self.route_endpoint=self.protected_center.copy()
        self.observation_schema=schema_for_counts(self.num_friendly)
        self.observation_dim=self.observation_schema.sensor_dim
        self.critic_state_dim=self.state_dim=critic_state_dim(self.num_friendly,len(self.reefs))
        self.required_hold_steps=math.ceil(config['task']['success_hold_seconds']/self.dt)
        self.variant='fixed'; self._renderer=None
        self._configured_active_mask=np.ones(self.num_friendly,np.float32)
        self.reset(seed=self.seed)

    def _dynamics(self, cls, cfg, speed, reverse_ratio, limits):
        return cls(dt=self.dt,max_speed=speed,allow_reverse=self.config['ship']['allow_reverse'],
            reverse_speed_ratio=reverse_ratio,max_yaw_rate=self.max_yaw_rate,max_sway_speed=self.max_sway_speed,
            max_surge_force=limits[0],max_yaw_moment=limits[1],parameters=cfg['parameters'],
            internal_substeps=cfg['internal_substeps'],reference_max_acceleration=cfg['reference_max_acceleration'],
            reference_max_yaw_acceleration=cfg['reference_max_yaw_acceleration'],reference_sway_damping=cfg['reference_sway_damping'])

    def set_evaluation_variant(self, variant):
        if variant not in ('fixed','randomized','training'): raise ValueError('invalid scenario variant')
        self.variant=variant

    def agent_active_mask(self): return self._agent_active_mask.copy()
    def set_active_agents(self, mask):
        value=np.asarray(mask,np.float32)
        if value.shape!=(self.num_friendly,) or not np.isin(value,[0,1]).all(): raise ValueError('invalid activity mask')
        self._configured_active_mask=value.copy()
    def friendly_states(self): return self.blue_states+self.submarine_states

    def reset(self, seed=None, options=None):
        if seed is not None: self.seed=int(seed)
        self.rng=np.random.default_rng(named_seed(self.seed,'scenario'))
        self._agent_active_mask=self._configured_active_mask.copy()
        cfg=self.config; scene=cfg['scenario']
        jitter=float(scene['position_jitter']) if self.variant!='fixed' else 0.
        heading_jitter=math.radians(scene['heading_jitter_degrees']) if self.variant!='fixed' else 0.
        def state(pos, heading, role, speed):
            return ShipState(np.asarray(pos,np.float32)+self.rng.uniform(-jitter,jitter,2).astype(np.float32),
                             speed,float(heading+self.rng.uniform(-heading_jitter,heading_jitter)),0.,role,role!='submarine')
        self.blue_states=[state(p,h,'surface',scene['initial_surface_speed']) for p,h in zip(scene['blue_starts'],scene['blue_headings'])]
        self.submarine_states=[state(p,h,'submarine',scene['initial_submarine_speed']) for p,h in zip(scene['submarine_starts'],scene['submarine_headings'])]
        self.red_state=state(scene['red_start'],scene['red_heading'],'surface',cfg['ship']['max_speed_red'])
        self.shadow_red_state=deepcopy(self.red_state)
        self.wave=RegularLinearWave.sample(cfg['wave'],named_seed(self.seed,'waves'))
        self.communication=CommunicationNetwork(self.num_friendly,cfg['communication'],named_seed(self.seed,'peer_communication'))
        self.communication.reset(self.friendly_states(),active_mask=self._agent_active_mask)
        self.target_communication=TargetMessageBus(self.num_friendly,cfg['communication'],named_seed(self.seed,'target_communication'),self.dt,
            mode=information_mode(cfg),information=cfg.get('information'),active_mask=self._agent_active_mask)
        self.sensors=[LocalTargetSensor(cfg['sensors'],named_seed(self.seed,'sensor',i),self.num_surface_blue+i) for i in range(self.num_submarines)]
        self.estimators=[TargetEstimator(i) for i in range(self.num_friendly)]
        self.target_estimates=[TargetEstimate.unknown() for _ in range(self.num_friendly)]
        self.red_navigator=RedNavigator(cfg['red_policy']); self.shadow_navigator=RedNavigator(cfg['red_policy'])
        self.step_count=0; self.hold_steps=0; self.task_state=TaskState(); self.phase='SEARCH'
        self.capture_success=False; self.first_capture_step=None
        self.withdrawal_cleared_entry=False
        self.last_nominal_actions=np.zeros((self.num_friendly,2),np.float32)
        self.last_safe_actions=self.last_nominal_actions.copy(); self.last_executed_actions=self.last_nominal_actions.copy()
        self.safety_diagnostics={}
        self._safety_projector=None
        self.sensor_detected=np.zeros(self.num_submarines,bool); self.step_detections=[None]*self.num_submarines
        self.tracking_errors=np.full(self.num_submarines,np.nan); self.observer_angle_gap=2*math.pi
        self.capture_geometry=encirclement_geometry(self.red_state.pos,
            [s.pos for i,s in enumerate(self.submarine_states) if self._agent_active_mask[self.num_surface_blue+i]])
        self.first_detection_steps=[None]*self.num_submarines
        self.team_first_detection_step=None
        self.local_target_estimates=[TargetEstimate.unknown() for _ in range(self.num_friendly)]
        self.received_information_gain=np.zeros(self.num_friendly)
        self.shared_information_gain=np.zeros(self.num_friendly)
        self.counterfactual_gap=self.progress_deficit=0.
        window=math.ceil(cfg['task']['attribution_window_seconds']/self.dt)+1
        self._progress_history=deque(maxlen=window)
        self._red_propulsion_history=deque(maxlen=window-1)
        self.last_red_reference=np.zeros(2,np.float32)
        self.last_red_action=np.zeros(2,np.float32)
        self.shadow_recent_progress=0.
        self.events=[]; self.terminal_reason=''; self._episode_done=False
        self._team_search_cells=set()
        self._new_cells=np.zeros(self.num_submarines)
        self.team_new_search_cells=0
        if cfg['reward'].get('search_coverage_diagnostics', False): self._update_search_coverage()
        self._new_cells.fill(0.); self.team_new_search_cells=0
        self.counterfactual_audit={}
        self.counterfactual_audit_new=False
        self.counterfactual_simulation_steps=0
        self.counterfactual_wall_time_s=0.
        self.red_incident={}
        self._record_progress()
        self.route_available=self._route_available()
        self.collision_pair=''
        return self._observations()

    def _route_available(self):
        from .red_policy import _route, _segment_clear
        margin=self.config['depth']['reef_shallow_buffer_m']+.5*self.surface_length
        circles=[(np.asarray(r['center'],float),r['radius']+margin) for r in self.reefs]
        path=_route(self.red_state.pos,self.protected_center,circles,self.world_size,.6,1)
        points=[self.red_state.pos,*path]
        return bool(path and all(_segment_clear(a,b,circles) for a,b in zip(points[:-1],points[1:])))

    def _observations(self):
        obs=np.stack([build_actor_observation(self,i) for i in range(self.num_friendly)])
        return obs,build_critic_state(self)

    def _update_search_coverage(self):
        """One shared cumulative sensor-footprint grid; no repeat credit by another AUV."""
        self._new_cells.fill(0.)
        grid=float(self.config['reward']['search_grid_m'])
        radius=float(self.config['sensors']['max_range_m'])
        candidates={}
        for j,ship in enumerate(self.submarine_states):
            if not self._agent_active_mask[self.num_surface_blue+j]: continue
            lo=np.maximum(0,np.floor((ship.pos-radius)/grid).astype(int))
            hi=np.minimum(np.ceil(self.world_size/grid).astype(int)-1,np.floor((ship.pos+radius)/grid).astype(int))
            for x in range(lo[0],hi[0]+1):
                for y in range(lo[1],hi[1]+1):
                    cell=(x,y); center=(np.array(cell)+.5)*grid
                    if cell in self._team_search_cells or np.linalg.norm(center-ship.pos)>radius: continue
                    if any(np.linalg.norm(center-np.asarray(r['center']))<=r['radius']+self.config['depth']['reef_shallow_buffer_m'] for r in self.reefs): continue
                    candidates.setdefault(cell,[]).append(j)
        self.team_new_search_cells=len(candidates)
        for cell,observers in candidates.items():
            self._team_search_cells.add(cell)
            for j in observers: self._new_cells[j]+=1./len(observers)

    def _information_quality(self, estimate, now):
        """Truth is used only for reward attribution, never an Actor or receiver update."""
        if not estimate.valid: return 0.
        error=float(np.linalg.norm(estimate.position-self.red_state.pos))
        age=max(0.,now-estimate.t_obs)
        variance=float(np.trace(estimate.covariance[:2,:2]))
        return float(math.exp(-error/3.)*math.exp(-age/self.config['communication']['max_age_seconds'])/(1.+variance/12.))

    def _sense_and_communicate(self):
        now=self.step_count*self.dt; self.sensor_detected.fill(False); self.step_detections=[None]*self.num_submarines
        self.received_information_gain.fill(0.)
        self.shared_information_gain.fill(0.)
        for j,(sensor,ship) in enumerate(zip(self.sensors,self.submarine_states)):
            i=self.num_surface_blue+j
            if not self._agent_active_mask[i]: continue
            detection=sensor.measure(ship,self.red_state,now)
            self.step_detections[j]=detection
            if detection is not None:
                self.sensor_detected[j]=True
                if self.first_detection_steps[j] is None:
                    self.first_detection_steps[j]=self.step_count
                    self.events.append({'event':f'AUV{j+1}_first_detection','step':self.step_count,'time_s':now})
                if self.team_first_detection_step is None:
                    self.team_first_detection_step=self.step_count
                    self.events.append({'event':'team_first_detection','step':self.step_count,'time_s':now})
                estimate=self.estimators[i].update_detection(detection)
                self.target_communication.broadcast(i,estimate,now)
        # Attribute only the receiver's change from newly DELIVERED information,
        # after applying this step's local detection and time propagation.
        self.local_target_estimates=[estimator.estimate for estimator in self.estimators]
        before_receive=[self.target_communication.latest(i,now,self.local_target_estimates[i]) for i in range(self.num_friendly)]
        previous=[self._information_quality(e,now) for e in before_receive]
        received=self.target_communication.advance(now)
        for i in range(self.num_friendly):
            local=self.local_target_estimates[i]
            estimate=self.target_communication.latest(i,now,local)
            self.target_estimates[i]=estimate if self._agent_active_mask[i] else TargetEstimate.unknown(now)
            if received[i] and self._agent_active_mask[i]:
                self.received_information_gain[i]=max(0.,self._information_quality(estimate,now)-previous[i])
                sources=set(packet.source_agent_id for packet in received[i]) & set(estimate.contribution_source_ids)
                for source in sources:
                    if self._agent_active_mask[source]: self.shared_information_gain[source]+=self.received_information_gain[i]/max(len(sources),1)
        self.tracking_errors=np.array([np.linalg.norm(self.target_estimates[i].position-self.red_state.pos) if self.target_estimates[i].valid else np.nan for i in range(self.num_surface_blue,self.num_friendly)])

    def _branch_contact(self, old_red, red, surfaces):
        from .safety import path_contact
        if path_contact(self,old_red,red,obstacle='boundary'): return 'boundary'
        for reef in self.reefs:
            if path_contact(self,old_red,red,obstacle=(np.asarray(reef['center']),reef['radius']+self.config['depth']['reef_shallow_buffer_m'])):
                return 'reef_'+reef['name']
        for index,old,new in surfaces:
            if path_contact(self,old_red,red,old,new): return 'USV_'+str(index)
        return ''

    def audit_counterfactual(self, actions, *, force=False):
        """Audit same-current-state Red under all, no, and leave-one-out USVs.

        All branches copy the actual navigation memory and actual red state.
        They use the same wave episode and absolute times. Blue trajectories
        integrate the current executed actuator input held open loop. This is
        a short-horizon intervention approximation, not a closed-loop causal
        identification theorem. Branch dynamics are private copies, so clamp
        counts, navigation memory, online sensors and RNG streams are untouched.
        """
        cfg=self.config['counterfactual']
        self.counterfactual_audit_new=False
        # Scripted mission withdrawal is not evidence of USV blocking.
        if self.capture_success: return self.counterfactual_audit
        if not cfg['enabled']: return self.counterfactual_audit
        if not force and self.step_count%cfg['cadence_steps']: return self.counterfactual_audit
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(self.num_friendly,2): raise ValueError('counterfactual actions must be a complete joint action')
        started=time.perf_counter(); horizon=int(cfg['horizon_steps']); now=self.step_count*self.dt
        active=[i for i in range(self.num_surface_blue) if self._agent_active_mask[i]]
        # The retained integrator clamps surge and sway every substep. If even
        # their maximum possible relative displacement cannot reach the Red
        # avoidance radius, all four interventions are exactly equivalent.
        # This analytic zero-effect shortcut avoids pretending unrun branches
        # have measured progress; those progress fields remain null.
        relative_ground_limit=math.hypot(self.max_speed_by_type['surface'],self.max_sway_speed)+math.hypot(self.max_speed_red,self.max_sway_speed)
        reach=self.config['red_policy']['surface_avoidance_range_m']+relative_ground_limit*horizon*self.dt
        distance=min((float(np.linalg.norm(self.blue_states[i].pos-self.red_state.pos)) for i in active),default=math.inf)
        if not force and distance>reach:
            branch={'progress_m':None,'final_position':None,'contact':None,'blue_contacts':[],
                    'valid':True,'steps':0,'entered_protection':None,'recovery_delta':0}
            self.counterfactual_audit={'step':self.step_count,'time_s':now,'horizon_steps':horizon,
                'horizon_seconds':horizon*self.dt,'cadence_steps':cfg['cadence_steps'],
                'branches':{name:deepcopy(branch) for name in ['all','none']+[f'without_{i}' for i in range(self.num_surface_blue)]},
                'valid':True,'total_effect_m':0.,'removal_effect_m':[0.]*self.num_surface_blue,
                'shapley_effect_m':[0.]*self.num_surface_blue,'cooperation_supported':False,
                'simulation_steps':0,'wall_time_s':time.perf_counter()-started,'proven_no_surface_interaction':True,
                'distance_m':distance if math.isfinite(distance) else None,'relative_reach_bound_m':reach,
                'approximation':'Analytic zero effect: bounded relative ground displacement cannot enter the configured USV avoidance range during the audit horizon.'}
            self.counterfactual_audit_new=True
            self.counterfactual_wall_time_s+=self.counterfactual_audit['wall_time_s']
            return self.counterfactual_audit
        surface_dynamics=deepcopy(self.blue_dynamics)
        tracks={i:[deepcopy(self.blue_states[i])] for i in active}
        blue_contacts={i:set() for i in active}
        from .safety import path_contact
        for k in range(horizon):
            for i in active:
                ship=tracks[i][-1]
                wave=self.wave.horizontal_velocity(ship.pos,now+k*self.dt)
                moved=surface_dynamics.step(ship,actions[i],wave)
                tracks[i].append(moved)
                if path_contact(self,ship,moved,obstacle='boundary'): blue_contacts[i].add(f'USV_{i}:boundary')
                for reef in self.reefs:
                    circle=(np.asarray(reef['center']),reef['radius']+self.config['depth']['reef_shallow_buffer_m'])
                    if path_contact(self,ship,moved,obstacle=circle): blue_contacts[i].add(f'USV_{i}:reef_{reef["name"]}')
        branches={}; initial_distance=float(np.linalg.norm(self.red_state.pos-self.protected_center))
        groups={'all':active,'none':[]}
        groups.update({f'without_{i}':[j for j in active if j!=i] for i in range(self.num_surface_blue)})
        sim_steps=horizon*len(active)
        for name,members in groups.items():
            state=deepcopy(self.red_state); navigator=deepcopy(self.red_navigator); dynamics=deepcopy(self.red_dynamics)
            contact=''; completed=0; reached=False
            for k in range(horizon):
                old=state
                visible=[tracks[i][k] for i in members]
                reference=navigator.reference(state,self.protected_center,self.reefs,visible,self.world_size,self.config['ship'])
                command=dynamics.reference_action(state,desired_yaw_rate=float(reference[0]),desired_surge_acceleration=float(reference[1]))
                state=dynamics.step(state,command,self.wave.horizontal_velocity(state.pos,now+k*self.dt))
                completed+=1; sim_steps+=1
                if not contact:
                    contact=self._branch_contact(old,state,[(i,tracks[i][k],tracks[i][k+1]) for i in members])
                reached |= center_path_enters_circle(old,state,self.protected_center,self.protected_radius)
            branches[name]={'progress_m':initial_distance-float(np.linalg.norm(state.pos-self.protected_center)),
                'final_position':state.pos.tolist(),'contact':contact,'steps':completed,
                'entered_protection':bool(reached),'recovery_delta':navigator.recovery_count-self.red_navigator.recovery_count}
            invalid_blue=set().union(*(blue_contacts[i] for i in members)) if members else set()
            for offset,i in enumerate(members):
                for j in members[offset+1:]:
                    if any(path_contact(self,tracks[i][k],tracks[i][k+1],tracks[j][k],tracks[j][k+1]) for k in range(horizon)):
                        invalid_blue.add(f'USV_{i}:USV_{j}')
            branches[name]['blue_contacts']=sorted(invalid_blue)
            branches[name]['valid']=not (contact or invalid_blue or branches[name]['recovery_delta']>0)
        total=branches['none']['progress_m']-branches['all']['progress_m']
        removal=[branches[f'without_{i}']['progress_m']-branches['all']['progress_m'] for i in range(self.num_surface_blue)]
        if self.num_surface_blue==2:
            shapley=[.5*((branches['none']['progress_m']-branches['without_1']['progress_m'])+removal[0]),
                     .5*((branches['none']['progress_m']-branches['without_0']['progress_m'])+removal[1])]
        else: shapley=removal.copy()
        valid=all(item['valid'] for item in branches.values())
        self.counterfactual_audit={'step':self.step_count,'time_s':now,'horizon_steps':horizon,
            'horizon_seconds':horizon*self.dt,'cadence_steps':cfg['cadence_steps'],'branches':branches,
            'valid':valid,'total_effect_m':total,'removal_effect_m':removal,'shapley_effect_m':shapley,
            'cooperation_supported':bool(valid and all(x>=cfg['minimum_effect_m'] for x in removal)),
            'simulation_steps':sim_steps,'wall_time_s':time.perf_counter()-started,
            'approximation':'Current executed blue force/moment held open loop; fixed current navigation memory; same wave episode and absolute time.'}
        self.counterfactual_audit_new=True
        self.counterfactual_simulation_steps+=sim_steps
        self.counterfactual_wall_time_s+=self.counterfactual_audit['wall_time_s']
        return self.counterfactual_audit

    def counterfactual_evidence_valid(self):
        audit=self.counterfactual_audit; cfg=self.config['counterfactual']
        return bool(not self.capture_success and audit and audit['valid'] and 0<=self.step_count-audit['step']<=cfg['max_age_steps']
                    and audit['total_effect_m']>=cfg['minimum_effect_m'])

    def _classify_red_incident(self, pair):
        audit=self.counterfactual_audit; now=self.step_count
        recent=bool(audit and now-audit['step']<=self.config['counterfactual']['max_age_steps'])
        none_contact=audit['branches']['none']['contact'] if recent else None
        all_contact=audit['branches']['all']['contact'] if recent else None
        if recent and audit.get('proven_no_surface_interaction'): category='natural_navigation_failure'
        elif recent and none_contact: category='natural_navigation_failure'
        elif recent and all_contact and not none_contact: category='blue_influenced_navigation_failure'
        else: category='attribution_unresolved'
        return {'category':category,'actual_contact':pair,'audit_step':audit.get('step'),
            'counterfactual_none_contact':none_contact,'counterfactual_all_contact':all_contact,
            'nearest_usv_center_distance_m':min((float(np.linalg.norm(s.pos-self.red_state.pos)) for i,s in enumerate(self.blue_states) if self._agent_active_mask[i]),default=None),
            'interpretation':'Short open-loop audit attribution, not a strict causal diagnosis.'}

    def learning_context(self):
        now=self.step_count*self.dt
        information_valid=np.array([e.valid for e in self.target_estimates],np.float32)
        ages=np.array([max(0.,now-e.t_obs) if e.valid else self.config['communication']['max_age_seconds'] for e in self.target_estimates],np.float32)
        variances=np.array([min(float(np.trace(e.covariance[:2,:2])),1e6) if e.valid else 1e6 for e in self.target_estimates],np.float32)
        margins=np.array([min(self._clearances(i)) for i in range(self.num_friendly)],np.float32)
        quality=float(np.clip(.35*np.mean(information_valid)+.35*max(0.,1.-self.observer_angle_gap/(2*math.pi))+.3*min(1.,self.hold_steps/self.required_hold_steps),0.,1.))
        return {'phase_id':{'SEARCH':0,'ENCIRCLE':1,'HOLD':2,'WITHDRAW':3,'SUCCESS':3}.get(self.phase,0),
                'information_valid':information_valid,'information_age':ages,'information_variance':variances,
                'safety_margin':margins,'quality':quality}

    def _capsule(self, state):
        length,width=(self.submarine_length,self.submarine_collision_width) if state.unit_type=='submarine' else (self.surface_length,self.surface_collision_width)
        return capsule_segment(state.pos,state.heading,length,width)

    def _can_collide(self, a, b):
        if a.unit_type==b.unit_type: return True
        d=self.config['depth']
        separation=d['auv_depth_m']-d['surface_keel_depth_m']
        return separation<=d['auv_vertical_radius_m']+d['vertical_safety_margin_m']

    def _clearances(self, i):
        ship=self.friendly_states()[i]; cap=self._capsule(ship)
        from .reward import boundary_clearance, capsule_clearance
        margins=[boundary_clearance(cap,self.world_size)]
        for reef in self.reefs:
            margins.append(capsule_circle_clearance(cap,np.asarray(reef['center']),reef['radius']+self.config['depth']['reef_shallow_buffer_m']))
        for j,other in enumerate(self.friendly_states()+[self.red_state]):
            if i!=j and (j==self.num_friendly or self._agent_active_mask[j]) and self._can_collide(ship,other): margins.append(capsule_clearance(cap,self._capsule(other)))
        return margins

    def _safety_cost(self, i):
        margin=self.config['depth']['horizontal_safety_margin_m']
        return float(np.clip((margin-min(self._clearances(i)))/max(margin,1e-6),0.,2.))

    def _contacts(self, old, new, *, shadow=False):
        from .safety import path_contact
        active=list(self._agent_active_mask)+[1]
        for i,ship in enumerate(new):
            if not active[i]: continue
            if path_contact(self,old[i],ship,obstacle='boundary'): return ('scenario_anomaly' if i==self.num_friendly else 'out_of_bounds',f'{i}-boundary')
            for reef in self.reefs:
                if path_contact(self,old[i],ship,obstacle=(np.asarray(reef['center']),reef['radius']+self.config['depth']['reef_shallow_buffer_m'])):
                    return ('scenario_anomaly' if i==self.num_friendly else 'reef_collision',f'{i}-reef-{reef["name"]}')
            for j in range(i+1,len(new)):
                if active[j] and self._can_collide(ship,new[j]) and path_contact(self,old[i],ship,old[j],new[j]): return 'collision',f'{i}-{j}'
        return '', ''

    def _record_progress(self):
        if self.capture_success:
            self.counterfactual_gap=self.progress_deficit=self.shadow_recent_progress=0.
            return
        actual=float(np.linalg.norm(self.protected_center-self.red_state.pos))
        shadow=float(np.linalg.norm(self.protected_center-self.shadow_red_state.pos))
        self.counterfactual_gap=actual-shadow
        self._progress_history.append((actual,shadow))
        start=self._progress_history[0]
        self.progress_deficit=(start[1]-shadow)-(start[0]-actual)
        self.shadow_recent_progress=start[1]-shadow

    def _task_success(self, safe):
        cfg=self.config['task']; now=self.step_count*self.dt
        self.capture_geometry=encirclement_geometry(self.red_state.pos,
            [s.pos for i,s in enumerate(self.submarine_states) if self._agent_active_mask[self.num_surface_blue+i]])
        observers=[]
        for j in range(self.num_submarines):
            i=self.num_surface_blue+j; estimate=self.estimators[i].predict(now)
            if (self._agent_active_mask[i] and estimate.valid and now-estimate.t_obs<=1.0
                and np.linalg.norm(estimate.position-self.red_state.pos)<=cfg['maximum_tracking_error_m']): observers.append(self.submarine_states[j].pos)
        if len(observers)>=2:
            angles=np.sort([math.atan2(*(p-self.red_state.pos)[::-1]) for p in observers])
            self.observer_angle_gap=float(np.max(np.diff(np.r_[angles,angles[0]+2*math.pi])))
        else: self.observer_angle_gap=2*math.pi
        informed=sum(e.valid and now-e.t_obs<=cfg['maximum_information_age_seconds'] and np.trace(e.covariance[:2,:2])<=cfg['maximum_position_variance_m2'] for e in self.target_estimates[:self.num_surface_blue])
        engaged=sum(bool(self._agent_active_mask[i]) and np.linalg.norm(s.pos-self.red_state.pos)<=cfg['minimum_usv_engagement_m'] for i,s in enumerate(self.blue_states))
        propulsion_fraction=float(np.mean(self._red_propulsion_history)) if self._red_propulsion_history else 0.
        intent=(len(self._red_propulsion_history)==self._red_propulsion_history.maxlen
                and self.red_navigator.config['target_speed']>.0 and propulsion_fraction>=cfg['minimum_propulsion_fraction']
                and self.shadow_recent_progress>=cfg['minimum_shadow_progress_m'])
        result=self.task_state.advance(config=cfg,dt=self.dt,red_x=float(self.red_state.pos[0]),engaged_usvs=engaged,
            shadow_gap=self.counterfactual_gap,progress_deficit=self.progress_deficit,observers=len(observers),angle_gap=self.capture_geometry['maximum_angle_gap_rad'],
            informed_usvs=informed,route_available=self.route_available,red_intent=intent,safe=safe,
            capture_distances=self.capture_geometry['radii_m'],capture_hull_margin=self.capture_geometry['hull_margin_m'],
            counterfactual_valid=self.counterfactual_evidence_valid())
        self.hold_steps=self.task_state.hold_steps
        return result

    def _confirm_capture(self):
        self.capture_success=True
        self.first_capture_step=self.step_count
        self.events.append({'event':'capture_confirmed','step':self.step_count,'time_s':self.step_count*self.dt})
        if self.post_capture_end_step:
            goal=np.array([max(3.,float(self.route_origin[0])-5.),float(self.route_origin[1])])
            self.red_navigator.request_withdrawal(goal,reason='capture_confirmed')
            self.events.append({'event':'red_withdraw','step':self.step_count,'time_s':self.step_count*self.dt})
            # Retire the intrusion-only shadow and all of its accumulated credit.
            self.counterfactual_audit={}; self.counterfactual_audit_new=False
            self._record_progress()

    def _update_withdrawal_exit(self):
        if not self.capture_success or not self.post_capture_end_step or self.withdrawal_cleared_entry: return
        left,right,radius=capsule_segment(self.red_state.pos,self.red_state.heading,
            self.surface_length,self.surface_collision_width)
        if max(left[0],right[0])+radius<=float(self.route_origin[0]) and self.red_state.velocity[0]<-.02:
            self.withdrawal_cleared_entry=True
            self.events.append({'event':'withdrawal_cleared_entry','step':self.step_count,'time_s':self.step_count*self.dt})

    def step(self, actions):
        if self._episode_done: raise RuntimeError('step called after terminal; reset required')
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(self.num_friendly,2) or not np.isfinite(actions).all(): raise ValueError(f'expected finite physical actions [{self.num_friendly},2]')
        from .reward import snapshot, evaluate
        previous=snapshot(self) if self.config['reward'].get('scheme')=='cooperative_pbrs_v1' else None
        old=deepcopy(self.friendly_states()+[self.red_state])
        self.last_nominal_actions=actions.copy()
        if self.config['mechanisms']['safety_projection']:
            from .safety_projection import SafetyProjector
            if self._safety_projector is None: self._safety_projector=SafetyProjector(self)
            safe,self.safety_diagnostics=self._safety_projector.project(actions)
        else:
            safe=actions.copy()
            self.safety_diagnostics={'enabled':False,'intervention':[False]*self.num_friendly,
                'projection_norm':[0.]*self.num_friendly,'infeasible':[False]*self.num_friendly,
                'fallback':[False]*self.num_friendly,'slack':[0.]*self.num_friendly,'solver_time_s':0.,
                'minimum_clearance':min(min(self._clearances(i)) for i in range(self.num_friendly))}
        self.last_safe_actions=np.clip(np.asarray(safe,np.float32),-self.agent_action_limits,self.agent_action_limits)
        self.last_executed_actions=self.last_safe_actions*self._agent_active_mask[:,None]
        self.audit_counterfactual(self.last_executed_actions)
        before=self.step_count*self.dt
        next_states=[]
        for i,ship in enumerate(self.friendly_states()):
            if not self._agent_active_mask[i]: next_states.append(deepcopy(ship)); continue
            dynamics=self.blue_dynamics if i<self.num_surface_blue else self.submarine_dynamics
            wave=self.wave.horizontal_velocity(ship.pos,before) if i<self.num_surface_blue else None
            next_states.append(dynamics.step(ship,self.last_executed_actions[i],wave))
        surfaces=[s for i,s in enumerate(self.blue_states) if self._agent_active_mask[i]]
        def red_step(state,navigator,surfaces):
            previous_mode=navigator.mode
            reference=navigator.reference(state,self.protected_center,self.reefs,surfaces,self.world_size,self.config['ship'])
            action=self.red_dynamics.reference_action(state,desired_yaw_rate=float(reference[0]),desired_surge_acceleration=float(reference[1]))
            if navigator is self.red_navigator:
                if previous_mode!=navigator.mode:
                    self.events.append({'event':'red_'+navigator.mode.lower(),'step':self.step_count,
                                        'time_s':before})
                self.last_red_reference=reference.copy(); self.last_red_action=action.copy()
                self._red_propulsion_history.append(float(action[0])>.05)
            return self.red_dynamics.step(state,action,self.wave.horizontal_velocity(state.pos,before))
        self.red_state=red_step(self.red_state,self.red_navigator,surfaces)
        old_shadow=deepcopy(self.shadow_red_state)
        if not self.capture_success:
            self.shadow_red_state=red_step(self.shadow_red_state,self.shadow_navigator,[])
        self.blue_states=next_states[:self.num_surface_blue]; self.submarine_states=next_states[self.num_surface_blue:]
        self.step_count+=1
        self.communication.step(self.friendly_states(),self.step_count,active_mask=self._agent_active_mask)
        self._sense_and_communicate(); self._record_progress()
        if self.config['reward'].get('search_coverage_diagnostics', False): self._update_search_coverage()
        reason,pair=self._contacts(old,self.friendly_states()+[self.red_state])
        if reason=='scenario_anomaly': self.red_incident=self._classify_red_incident(pair)
        # A stuck/invalid shadow trajectory is a scene anomaly, never a blue success.
        if not reason and not self.capture_success and self.shadow_navigator.recovery_count>8:
            reason='scenario_anomaly'; self.red_incident={'category':'shadow_stagnation','recovery_count':self.shadow_navigator.recovery_count}
        if not reason and not self.capture_success:
            from .safety import path_contact
            if path_contact(self,old_shadow,self.shadow_red_state,obstacle='boundary') or any(path_contact(self,old_shadow,self.shadow_red_state,obstacle=(np.asarray(r['center']),r['radius']+self.config['depth']['reef_shallow_buffer_m'])) for r in self.reefs):
                reason='scenario_anomaly'; self.red_incident={'category':'shadow_navigation_failure'}
        breakthrough=center_path_enters_circle(old[self.num_friendly],self.red_state,self.protected_center,self.protected_radius)
        if not reason and breakthrough: reason='breakthrough'
        success=self._task_success(safe=not bool(reason))
        capture_just_confirmed=bool(not reason and success and not self.capture_success)
        if capture_just_confirmed: self._confirm_capture()
        self._update_withdrawal_exit()
        if not reason and self.capture_success and self.step_count>=self.post_capture_end_step:
            reason='success'
        if not reason and self.step_count>=self.max_steps: reason='timeout'
        self.terminal_reason=reason; self.collision_pair=pair
        self.phase='SUCCESS' if reason=='success' else ('ENCIRCLE' if any(e.valid for e in self.target_estimates) else 'SEARCH')
        if not reason and self.hold_steps>0: self.phase='HOLD'
        if not reason and self.capture_success: self.phase='WITHDRAW'
        self._episode_done=bool(reason)
        rewards,terms=evaluate(previous,self,reason,geometry=self.capture_geometry)
        if not np.isfinite(rewards).all(): raise FloatingPointError('reward contains NaN/Inf')
        terms.update(terminated=bool(reason),truncated=False,timeout=reason=='timeout',bootstrap_mask=float(not bool(reason)),
                     canonical_team_dense_reward=float(rewards.mean()-terms['team_terminal_reward']),
                     process_abs_sum=float(np.mean(np.abs(rewards-terms['team_terminal_reward']))),hold_progress=self.hold_steps/self.required_hold_steps)
        info=EnvInfo(step=self.step_count,success=reason=='success',hold_steps=self.hold_steps,
            collision=reason in ('collision','reef_collision'),boundary_violation=reason=='out_of_bounds',landing_success=breakthrough,
            surface_intervention_success=all(self.task_state.support_gates.get(k,False) for k in ('engagement','causal_blocking','current_state_counterfactual')),
            cooperative_task_success=self.capture_success,task_name='reef_interception',stage_hold_seconds=self.hold_steps*self.dt,
            stage_success_gates=self.task_state.gates.copy(),unmet_success_gates=tuple(k for k,v in self.task_state.gates.items() if not v),
            support_gates=self.task_state.support_gates.copy(),capture_geometry=deepcopy(self.capture_geometry),
            phase=self.phase,terminated=bool(reason),truncated=False,timeout=reason=='timeout',terminal_reason=reason,
            collision_pair=pair,reward_terms=terms,safety_diagnostics=deepcopy(self.safety_diagnostics),
            counterfactual_audit=deepcopy(self.counterfactual_audit) if self.counterfactual_audit_new else {},red_incident=deepcopy(self.red_incident),
            capture_success=self.capture_success,capture_just_confirmed=capture_just_confirmed,
            first_capture_step=self.first_capture_step,
            first_capture_time_s=None if self.first_capture_step is None else self.first_capture_step*self.dt,
            post_capture_steps=0 if self.first_capture_step is None else self.step_count-self.first_capture_step,
            withdrawal_cleared_entry=self.withdrawal_cleared_entry)
        if reason: self.events.append({'event':reason,'step':self.step_count,'time_s':self.step_count*self.dt})
        obs,state=self._observations()
        return obs,state,rewards,self._episode_done,info

    @property
    def target_estimate(self):
        # Convenience for display only. Controllers must consume target_estimates per receiver.
        return self.target_estimates[0]

    def render_frame(self, *, show_shadow_red=None):
        from .rendering import RenderFrame
        return RenderFrame(world_size=self.world_size,island_center=np.array(self.reefs[0]['center']),island_radius=self.reefs[0]['radius'],
            red_state=self.red_state,blue_states=self.blue_states,submarine_states=self.submarine_states,
            route_start=self.route_origin,route_end=self.route_endpoint,phase=self.phase,physical_capture_success=False,
            cooperative_task_success=self.capture_success,semi_major=self.semi_major,semi_minor=self.semi_minor,
            surface_active_mask=self._agent_active_mask[self.surface_slice],submarine_active_mask=self._agent_active_mask[self.submarine_slice],
            shadow_red_state=self.shadow_red_state,show_shadow_red=not self.capture_success and (True if show_shadow_red is None else show_shadow_red),
            task_name='reef interception',simulation_time=self.step_count*self.dt,termination_reason=self.terminal_reason,
            red_behavior=self.red_navigator.mode,
            event_markers={e['event']:True for e in self.events},diagnostics={'hold_progress':self.hold_steps/self.required_hold_steps,'e_cf':self.counterfactual_gap,
                'capture_max_m':max(self.capture_geometry['radii_m'],default=0.),
                'enclosure_gap_deg':math.degrees(self.capture_geometry['maximum_angle_gap_rad'])},
            encirclement_valid=all(self.task_state.gates.get(k,False) for k in ('auv_participation','capture_distance','encirclement')),
            reefs=self.reefs,protected_center=self.protected_center,protected_radius=self.protected_radius)

    def render(self):
        from .rendering import Renderer
        if self._renderer is None: self._renderer=Renderer(self.world_size)
        self._renderer.render(self.render_frame())

    def close(self):
        if self._renderer is not None: self._renderer.close(); self._renderer=None
