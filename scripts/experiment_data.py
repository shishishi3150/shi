"""Observational metrics and transactional, bounded experiment logs (schema 1)."""
from copy import deepcopy
from pathlib import Path
import csv
import gzip
import hashlib
import json
import os
import platform
import uuid
import numpy as np
import torch
from usv_env.config import digest
from usv_env.reward import REWARD_COMPONENTS, REWARD_DIAGNOSTICS

SCHEMA = 'shi_experiment_data_v2'


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def policy_hash(policy):
    """Content identity for an in-memory policy, without writing a checkpoint."""
    h=hashlib.sha256()
    for index,actor in enumerate(policy.actors):
        for name,tensor in sorted(actor.state_dict().items()):
            array=tensor.detach().cpu().contiguous().numpy()
            h.update(f'{index}:{name}:{array.dtype}:{array.shape}'.encode('utf-8'))
            h.update(array.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(plain(value),ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


class JsonLog:
    """One JSON record per line; gzip files contain independently recoverable members."""
    def __init__(self,path,create=True):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        if create: self.path.touch(exist_ok=False)

    def append(self,record):
        blob=(json.dumps(plain(record),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
        if self.path.suffix=='.gz': blob=gzip.compress(blob,compresslevel=3,mtime=0)
        with self.path.open('ab') as f: f.write(blob)

    def commit(self):
        return dict(bytes=self.path.stat().st_size,sha256=file_hash(self.path))

    def verify(self,commit):
        remaining=commit['bytes']; h=hashlib.sha256()
        with self.path.open('rb') as f:
            while remaining:
                part=f.read(min(remaining,1024*1024))
                if not part: raise ValueError('diagnostic log committed prefix missing')
                h.update(part); remaining-=len(part)
        if h.hexdigest()!=commit['sha256']: raise ValueError('diagnostic log committed prefix corrupt')

    def recover(self,commit,archive):
        self.verify(commit)
        if self.path.stat().st_size>commit['bytes']:
            archive.mkdir(parents=True,exist_ok=True)
            with self.path.open('rb') as source, (archive/self.path.name).open('wb') as dest:
                source.seek(commit['bytes'])
                for block in iter(lambda:source.read(1024*1024),b''): dest.write(block)
            with self.path.open('r+b') as f: f.truncate(commit['bytes'])


def iter_records(path):
    path=resolve_log(path)
    if path.stat().st_size==0: return
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def resolve_log(path):
    """Read both historical plain JSONL and current compressed diagnostics."""
    path = Path(path)
    if not path.exists() and path.suffix == '.jsonl':
        compressed = path.with_suffix('.jsonl.gz')
        if compressed.exists(): return compressed
    return path


def export_csv(path, destination):
    """Keep arrays/structures as JSON cells. Blank is missing, never imputed zero."""
    fields=[]; seen=set()
    for row in iter_records(path):
        for k in row:
            if k not in seen: seen.add(k); fields.append(k)
    tmp=Path(destination).with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader()
        for row in iter_records(path):
            writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    os.replace(tmp,destination)


class EpisodeMetrics:
    """Bounded observed/search/communication metrics; truth is evaluation-only."""
    def __init__(self,env,state=None):
        if state is not None: self.__dict__.update(deepcopy(state)); return
        self.goal=np.asarray(env.config['protected_zone']['center'],dtype=np.float64)
        self.goal_radius=float(env.config['protected_zone']['radius'])
        distance=float(np.linalg.norm(env.red_state.pos-self.goal))
        self.steps=0; self.seconds=0.; self.follow_seconds=0.; self.surface_count=len(env.blue_states)
        self.initial_distance=distance; self.last_distance=distance; self.min_distance=distance
        n=len(env.submarine_states); self.returns=np.zeros(self.surface_count+n)
        self.error_integral=np.zeros(n); self.error_seconds=np.zeros(n)
        self.worst_integral=0.; self.worst_seconds=0.; self.follow_max=None
        self.lost_seconds=np.zeros(n); self.any_lost_seconds=0.; self.detected_seconds=np.zeros(n)
        self.first_discovery=[None]*n; self.message_age_sum=0.; self.message_age_samples=0
        self.threshold=float(env.config['task']['maximum_tracking_error_m'])
        self.angle_gap_sum=0.; self.angle_gap_seconds=0.; self.max_hold_seconds=0.
        self.component_sums={}; self.component_min={}; self.component_max={}; self.safety_interventions=0; self.safety_infeasible=0; self.safety_seconds=0.; self.minimum_clearance=None
        self.error_square_integral=np.zeros(n)
        self.discounted_returns=np.zeros(self.surface_count+n)
        self.gamma=float(env.config['training']['gamma'])
        self.reward_state_sums={}; self.reward_state_min={}; self.reward_state_max={}
        self.saturated_nominal=self.saturated_executed=self.active_actuator_samples=0

    def state_dict(self): return deepcopy(vars(self))

    def observe(self,env,reward,info,trace=False):
        dt=float(env.dt); self.steps+=1; self.seconds+=dt
        positions=np.asarray([s.pos for s in env.submarine_states]); errors=np.asarray(env.tracking_errors,dtype=float)
        detected=np.asarray(env.sensor_detected,dtype=bool); valid=np.isfinite(errors)
        self.error_square_integral+=np.where(valid,errors**2,0.)*dt
        for name,value in info.reward_terms.items():
            if name in (*REWARD_COMPONENTS, 'terminal_reward') and np.asarray(value).shape==self.returns.shape:
                array=np.asarray(value,float)
                # Historical zero columns are compatibility aliases, not data.
                # A component first appearing later still has earlier zeros.
                if not array.any() and name not in self.component_sums: continue
                initial = np.zeros_like(array) if self.steps > 1 else array
                self.component_sums[name]=self.component_sums.get(name,np.zeros_like(array))+array
                self.component_min[name]=np.minimum(self.component_min.get(name,initial),array)
                self.component_max[name]=np.maximum(self.component_max.get(name,initial),array)
            elif name in REWARD_DIAGNOSTICS:
                array=np.asarray(value,float)
                self.reward_state_sums[name]=self.reward_state_sums.get(name,np.zeros_like(array))+array
                self.reward_state_min[name]=np.minimum(self.reward_state_min.get(name,array),array)
                self.reward_state_max[name]=np.maximum(self.reward_state_max.get(name,array),array)
        self.discounted_returns+=self.gamma**(self.steps-1)*reward
        mask=env.agent_active_mask().astype(bool)
        self.active_actuator_samples+=2*int(mask.sum())
        self.saturated_nominal+=int(np.count_nonzero(np.abs(env.last_nominal_actions[mask]/env.agent_action_limits[mask])>=.98))
        self.saturated_executed+=int(np.count_nonzero(np.abs(env.last_executed_actions[mask]/env.agent_action_limits[mask])>=.98))
        safety=getattr(info,'safety_diagnostics',{})
        self.safety_interventions+=int(np.count_nonzero(safety.get('intervention',[])))
        self.safety_infeasible+=int(np.count_nonzero(safety.get('infeasible',[])))
        self.safety_seconds+=float(safety.get('solver_time_s',0.))
        if safety.get('minimum_clearance') is not None and np.asarray(safety['minimum_clearance']).size:
            value=float(np.min(safety['minimum_clearance']))
            self.minimum_clearance=value if self.minimum_clearance is None else min(value,self.minimum_clearance)
        self.returns+=reward; self.error_integral+=np.where(valid,errors,0.)*dt; self.error_seconds+=valid*dt
        self.detected_seconds+=detected*dt
        for index,is_detected in enumerate(detected):
            if is_detected and self.first_discovery[index] is None: self.first_discovery[index]=self.seconds
        if valid.any():
            worst=float(errors[valid].max()); self.worst_integral+=worst*dt; self.worst_seconds+=dt
            self.follow_seconds+=dt; self.follow_max=max(worst,self.follow_max or 0.)
        lost=(~valid)|(errors>self.threshold); self.lost_seconds+=lost*dt
        self.any_lost_seconds+=float(lost.any())*dt
        estimates=env.target_estimates
        ages=[max(0.,env.step_count*dt-e.t_obs) for e in estimates if e.valid]
        self.message_age_sum+=sum(ages); self.message_age_samples+=len(ages)
        gap=float(env.observer_angle_gap)
        if detected.sum()>=2 and np.isfinite(gap): self.angle_gap_sum+=gap*dt; self.angle_gap_seconds+=dt
        self.max_hold_seconds=max(self.max_hold_seconds,float(env.hold_steps*dt))
        distance=float(np.linalg.norm(env.red_state.pos-self.goal))
        self.last_distance=distance; self.min_distance=min(self.min_distance,distance)
        if not trace: return None
        communication=env.target_communication
        return dict(step=int(env.step_count),time_s=float(env.step_count*dt),dt_s=dt,phase=info.phase,
            capture_success=info.capture_success,capture_just_confirmed=info.capture_just_confirmed,
            first_capture_step=info.first_capture_step,post_capture_steps=info.post_capture_steps,
            withdrawal_cleared_entry=info.withdrawal_cleared_entry,
            follow_stage=bool(valid.any()),tracking_threshold_m=self.threshold,auv_positions=positions,
            auv_errors_m=[float(e) if good else None for e,good in zip(errors,valid)],sensor_detected=detected,
            red_position=env.red_state.pos,shadow_red_position=env.shadow_red_state.pos,
            red_goal_distance_m=distance,shadow_goal_distance_m=float(np.linalg.norm(env.shadow_red_state.pos-self.goal)),
            usv_positions=[s.pos for s in env.blue_states],actions=env.last_executed_actions.copy(),
            nominal_actions=env.last_nominal_actions.copy(),observer_angle_gap_rad=gap if np.isfinite(gap) else None,
            hold_seconds=env.hold_steps*dt,events=env.events,
            capture_geometry=env.capture_geometry.copy(),support_gates=info.support_gates.copy(),
            rewards=reward,reward_terms=info.reward_terms,terminal_reason=info.terminal_reason,
            vessel_states=[dict(position=s.pos,heading=s.heading,surge=s.speed,sway=s.sway_speed,yaw_rate=s.yaw_rate,ground_speed=s.ground_speed) for s in env.friendly_states()],
            red_state=dict(position=env.red_state.pos,heading=env.red_state.heading,surge=env.red_state.speed,yaw_rate=env.red_state.yaw_rate),
            safe_actions=getattr(env,'last_safe_actions',env.last_executed_actions),safety_diagnostics=safety,
            counterfactual_audit=getattr(info,'counterfactual_audit',{}),red_incident=getattr(info,'red_incident',{}),
            target_estimates=[dict(valid=e.valid,position=e.position,velocity=e.velocity,covariance=e.covariance if e.valid else None,t_obs=e.t_obs if e.valid else None,
                contribution_source_ids=getattr(e,'contribution_source_ids',()),contribution_measurement_ids=getattr(e,'contribution_measurement_ids',()),
                contribution_observation_times=getattr(e,'contribution_observation_times',()),contribution_weights=getattr(e,'contribution_weights',())) for e in estimates],
            target_estimates_valid=len(ages),target_information_age_mean_s=float(np.mean(ages)) if ages else None,
            communication_sent=communication.sent,communication_dropped=communication.dropped)

    def summary(self,env,info):
        follow=self.follow_seconds; comm=env.target_communication.statistics()
        delay=env.target_communication.delay_steps
        if isinstance(delay,int): delay=(delay,delay)
        result=dict(episode_steps=self.steps,episode_seconds=self.seconds,
            capture_success=int(info.capture_success),first_capture_step=info.first_capture_step,
            first_capture_time_s=info.first_capture_time_s,post_capture_steps=info.post_capture_steps,
            withdrawal_cleared_entry=info.withdrawal_cleared_entry,
            episode_complete=bool(info.terminated or info.truncated),success=int(info.success),
            terminal_reason=info.terminal_reason or 'collector_cut',
            team_return=float(self.returns.mean()),usv_return=float(self.returns[:self.surface_count].mean()),auv_return=float(self.returns[self.surface_count:].mean()),
            per_agent_return=self.returns,per_agent_discounted_return=self.discounted_returns,
            discounted_team_return=float(self.discounted_returns.mean()),reward_gamma=self.gamma,
            follow_stage_entered=bool(follow),follow_seconds=follow,
            follow_worst_mean_m=self.worst_integral/self.worst_seconds if self.worst_seconds else None,follow_worst_max_m=self.follow_max,
            follow_any_lost_fraction=self.any_lost_seconds/self.seconds if self.seconds else None,
            all_worst_mean_m=self.worst_integral/self.worst_seconds if self.worst_seconds else None,
            tracking_threshold_m=self.threshold,
            red_goal_initial_m=self.initial_distance,red_goal_final_m=self.last_distance,
            red_goal_progress_m=self.initial_distance-self.last_distance,red_goal_min_m=self.min_distance,
            shadow_progress_gap_m=float(np.linalg.norm(env.red_state.pos-self.goal)-np.linalg.norm(env.shadow_red_state.pos-self.goal)),
            protected_zone_clearance_min_m=self.min_distance-self.goal_radius,
            collision=int(info.collision),breakthrough=int(info.terminal_reason=='breakthrough'),
            reef_collision=int(info.terminal_reason=='reef_collision'),scenario_anomaly=int(info.terminal_reason=='scenario_anomaly'),
            numerical_error=int(info.terminal_reason=='numerical_error'),
            out_of_bounds=int(info.terminal_reason == 'out_of_bounds'),
            boundary_contact=int(info.boundary_violation),timeout=int(info.timeout),
            collision_pair=info.collision_pair,red_incident=getattr(info,'red_incident',{}),
            reward_component_sums=self.component_sums,reward_component_min=self.component_min,reward_component_max=self.component_max,
            reward_component_team_sums={key:float(values.mean()) for key,values in self.component_sums.items()},
            reward_state_sums=self.reward_state_sums,reward_state_min=self.reward_state_min,reward_state_max=self.reward_state_max,
            reward_state_means={key:values/max(self.steps,1) for key,values in self.reward_state_sums.items()},
            nominal_action_saturation_fraction=self.saturated_nominal/max(self.active_actuator_samples,1),
            executed_action_saturation_fraction=self.saturated_executed/max(self.active_actuator_samples,1),
            safety_intervention_rate=self.safety_interventions/max(self.steps*len(self.returns),1),
            safety_infeasible_rate=self.safety_infeasible/max(self.steps*len(self.returns),1),safety_solver_time_s=self.safety_seconds,
            predicted_minimum_clearance_m=self.minimum_clearance,
            joint_hold_max_seconds=self.max_hold_seconds,first_discovery_seconds=min((t for t in self.first_discovery if t is not None),default=None),
            capture_hold_max_seconds=self.max_hold_seconds,
            final_capture_geometry=env.capture_geometry.copy(),final_capture_gates=info.stage_success_gates.copy(),
            final_support_gates=info.support_gates.copy(),
            target_information_age_mean_seconds=self.message_age_sum/self.message_age_samples if self.message_age_samples else None,
            observer_angle_gap_mean_degrees=float(np.rad2deg(self.angle_gap_sum/self.angle_gap_seconds)) if self.angle_gap_seconds else None,
            communication_enabled=env.target_communication.enabled,
            configured_delay_min_steps=int(delay[0]),configured_delay_max_steps=int(delay[1]),
            dt_s=float(env.dt),**comm)
        for i in range(len(self.error_integral)):
            error=float(self.error_integral[i]/self.error_seconds[i]) if self.error_seconds[i] else None
            result[f'auv{i+1}_tracking_rmse_m']=float(np.sqrt(self.error_square_integral[i]/self.error_seconds[i])) if self.error_seconds[i] else None
            result[f'auv{i+1}_all_mean_m']=error; result[f'auv{i+1}_follow_mean_m']=error
            result[f'auv{i+1}_lost_fraction']=float(self.lost_seconds[i]/self.seconds) if self.seconds else None
            result[f'auv{i+1}_detection_fraction']=float(self.detected_seconds[i]/self.seconds) if self.seconds else None
            result[f'auv{i+1}_first_detection_seconds']=self.first_discovery[i]
        return result


LOGS=('training_episodes','evaluation_episodes','evaluation_summary','updates','reward_components','runtime')
TRACES=('training_steps','evaluation_steps')


class ExperimentWriter:
    def __init__(self,root,cfg,state=None,identity=None):
        if state is not None: raise ValueError('exact training-state resume is disabled; warm-start a new run from latest.pt')
        self.root=Path(root); self.path=self.root/'diagnostics'; self.path.mkdir(parents=True,exist_ok=True)
        self.cfg=deepcopy(cfg)
        self.identity=deepcopy(state['identity'] if state else identity or dict(
            run_id=uuid.uuid4().hex,algorithm='MATD3',experiment=cfg['revision']['experiment'],
            train_seed=cfg['training']['seed'],purpose=cfg.get('purpose','formal_training')))
        self.identity.update(schema=SCHEMA,training_task_hash=cfg['hashes']['task'],reward_hash=cfg['hashes']['reward'],
                             observation_hash=cfg['hashes']['observation'],architecture_hash=cfg['hashes']['architecture'],
                             training_source_code_hash=cfg['provenance'].get('code_hash'))
        self.logs={name:JsonLog(self.path/(name+'.jsonl.gz'),create=state is None)
                   for name in (*LOGS,*TRACES)}
        self.artifacts={}

    def append(self,name,record): self.logs[name].append({**self.identity,**record})

    def register_checkpoint(self,path):
        if Path(path).name not in ('latest.pt','best.pt'):
            raise ValueError('only latest.pt and best.pt are retained')
        self.artifacts[Path(path).relative_to(self.root).as_posix()]=file_hash(path)

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,artifacts=self.artifacts,
                    logs={name:log.commit() for name,log in self.logs.items()})

    def export(self):
        for name in LOGS: export_csv(self.logs[name].path,self.path/(name+'.csv'))

    def manifest(self,online_steps,teacher_steps,extra=None):
        device=self.cfg['training']['device']
        atomic_json(self.root/'run_manifest.json',dict(**self.identity,config=self.cfg,
            source_code_hash=self.cfg['provenance'].get('code_hash'),
            online_steps=online_steps,teacher_sim_steps=teacher_steps,
            training_interactions=online_steps+teacher_steps if teacher_steps is not None else None,
            hardware=dict(platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),
                torch_version=torch.__version__,torch_threads=torch.get_num_threads(),requested_device=device,
                cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            storage_policy=dict(actor_only=True,replay_saved=False,optimizer_saved=False,retained_policies=['latest.pt','best.pt'],
                diagnostics_compression='gzip',training_trace_every=self.cfg.get('diagnostics',{}).get('train_trace_every'),
                duplicate_diagnostic_csv=self.cfg.get('diagnostics',{}).get('export_csv',False)),
            metric_contract=dict(follow_stage='finite local/fused target estimate, unknown intervals count as lost',
                averaging='time within episode; then episodes within seed',
                success='three-AUV capture confirmed under unchanged gates, then safe continuation until task.post_capture_end_step (total episode step); legacy missing/zero ends immediately; USV blocking is auxiliary',
                missing='null in JSON; blank in CSV',truth='offline metrics only; never Actor input'),**(extra or {})))
