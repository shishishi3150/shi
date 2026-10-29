"""Observational metrics and transactional, bounded experiment logs (schema 1)."""
from copy import deepcopy
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import time
import uuid
import numpy as np
import torch
from usv_env.config import digest

SCHEMA = 'shi_experiment_data_v1'


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
    path=Path(path)
    if path.stat().st_size==0: return
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


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
    """O(1) storage. Truth used for measurement only, never appended to Actor inputs.

    Follow stage is post-step compaction_started (sticky after the entry gate).
    Also report whole-episode error so failure before entry is never hidden.
    """
    def __init__(self,env,state=None):
        if state is not None: self.__dict__.update(deepcopy(state)); return
        distance=float(np.linalg.norm(env.red_state.pos-env.island_center))
        self.steps=0; self.seconds=0.; self.follow_seconds=0.
        self.initial_distance=distance; self.last_distance=distance; self.min_distance=distance
        self.returns=np.zeros(5); self.error_integral=np.zeros(3); self.worst_integral=0.
        self.follow_errors=np.zeros(3); self.follow_worst=0.; self.follow_max=None
        self.lost_seconds=np.zeros(3); self.any_lost_seconds=0.
        self.threshold=float(env.encirclement_cfg['shrink_tracking']['max_slot_error_max'])

    def state_dict(self): return deepcopy(vars(self))

    def observe(self,env,reward,info,trace=False):
        dt=float(env.dt); self.steps+=1; self.seconds+=dt
        positions=np.asarray([s.pos for s in env.submarine_states]); slots=np.asarray(env.submarine_target_slots)
        errors=np.linalg.norm(positions-slots,axis=1)
        if not np.isfinite(errors).all(): raise ValueError('nonfinite measured follow errors')
        self.returns+=reward; self.error_integral+=errors*dt; self.worst_integral+=float(max(errors))*dt
        threshold=float(env.encirclement_cfg['shrink_tracking']['max_slot_error_max'])
        if threshold!=self.threshold: raise ValueError('metric schema 1 requires fixed tracking threshold')
        following=bool(env.compaction_started)
        if following:
            self.follow_seconds+=dt; self.follow_errors+=errors*dt
            self.follow_worst+=float(max(errors))*dt
            self.follow_max=max(float(max(errors)),self.follow_max or 0.)
            lost=errors>threshold; self.lost_seconds+=lost*dt
            self.any_lost_seconds+=float(any(lost))*dt
        distance=float(np.linalg.norm(env.red_state.pos-env.island_center))
        self.last_distance=distance; self.min_distance=min(self.min_distance,distance)
        if not trace: return None
        communication=env.communication
        ages=[max(0,env.step_count-p.send_step)*dt for p in communication.cache.values()]
        return dict(step=int(env.step_count),time_s=float(env.step_count*dt),dt_s=dt,phase=info.phase,
            follow_stage=following,tracking_threshold_m=threshold,auv_positions=positions,
            auv_slots=slots,auv_errors_m=errors,worst_error_m=float(max(errors)),
            red_position=env.red_state.pos,red_island_distance_m=distance,
            usv_positions=[s.pos for s in env.blue_states],actions=env.last_executed_actions.copy(),
            rewards=reward,terminal_reason=info.terminal_reason,
            messages_cached=len(ages),message_age_mean_s=float(np.mean(ages)) if ages else None,
            communication_sent=communication.sent,communication_dropped=communication.dropped)

    def summary(self,env,info):
        follow=self.follow_seconds; comm=env.communication.statistics()
        result=dict(episode_steps=self.steps,episode_seconds=self.seconds,
            episode_complete=bool(info.terminated),success=int(info.success),
            terminal_reason=info.terminal_reason or 'collector_cut',
            team_return=float(self.returns.mean()),usv_return=float(self.returns[:2].mean()),auv_return=float(self.returns[2:].mean()),
            follow_stage_entered=bool(follow),follow_seconds=follow,
            follow_worst_mean_m=self.follow_worst/follow if follow else None,follow_worst_max_m=self.follow_max,
            follow_any_lost_fraction=self.any_lost_seconds/follow if follow else None,
            all_worst_mean_m=self.worst_integral/self.seconds if self.seconds else None,
            tracking_threshold_m=self.threshold,
            red_island_initial_m=self.initial_distance,red_island_final_m=self.last_distance,
            red_island_progress_m=self.last_distance-self.initial_distance,red_island_min_m=self.min_distance,
            landing_clearance_min_m=self.min_distance-env.island_radius-env.landing_buffer if env.island_active else None,
            shrink_progress=float(info.reward_terms.get('shrink_progress',0.)),
            collision=int(info.collision),landing=int(info.landing_success),
            out_of_bounds=int(info.terminal_reason == 'out_of_bounds'),
            boundary_contact=int(info.boundary_violation),timeout=int(info.timeout),
            wall_contact=int(getattr(info,'wall_contact',False)),
            wall_contact_agents=list(getattr(info,'wall_contact_agents',())),
            full_protection_success=None,
            communication_enabled=env.communication.enabled,
            configured_delay_min_steps=int(env.communication.delay_range[0]),configured_delay_max_steps=int(env.communication.delay_range[1]),
            dt_s=float(env.dt),**comm)
        for i in range(3):
            result[f'auv{i+1}_all_mean_m']=float(self.error_integral[i]/self.seconds) if self.seconds else None
            result[f'auv{i+1}_follow_mean_m']=float(self.follow_errors[i]/follow) if follow else None
            result[f'auv{i+1}_lost_fraction']=float(self.lost_seconds[i]/follow) if follow else None
        return result


LOGS=('training_episodes','evaluation_episodes','evaluation_summary','i2_updates','guidance_events')
TRACES=('training_steps','evaluation_steps','teacher_branches')


class ExperimentWriter:
    def __init__(self,root,cfg,state=None,identity=None):
        self.root=Path(root); self.path=self.root/'diagnostics'; self.path.mkdir(parents=True,exist_ok=True)
        self.cfg=deepcopy(cfg)
        self.identity=deepcopy(state['identity'] if state else identity or dict(
            run_id=uuid.uuid4().hex,algorithm='MATD3',experiment=cfg['revision']['experiment'],
            train_seed=cfg['training']['seed'],purpose=cfg.get('purpose','formal_training')))
        self.identity.update(schema=SCHEMA,training_task_hash=cfg['hashes']['task'],reward_hash=cfg['hashes']['reward'],
                             observation_hash=cfg['hashes']['observation'],architecture_hash=cfg['hashes']['architecture'],
                             training_source_code_hash=cfg['provenance'].get('code_hash'))
        self.logs={name:JsonLog(self.path/(name+('.jsonl.gz' if name in TRACES else '.jsonl')),create=state is None)
                   for name in (*LOGS,*TRACES)}
        self.artifacts=deepcopy(state['artifacts']) if state else {}
        if state:
            if state['schema']!=SCHEMA: raise ValueError('unsupported experiment data checkpoint')
            # Check every committed dependency before recovering any tail.
            for name,log in self.logs.items(): log.verify(state['logs'][name])
            for relative,sha in self.artifacts.items():
                if file_hash(self.root/relative)!=sha: raise ValueError('committed validation checkpoint corrupt')
            archive=self.root/'orphaned_attempt'/uuid.uuid4().hex
            for name,log in self.logs.items(): log.recover(state['logs'][name],archive/'diagnostics')
            for file in (self.root/'validation_checkpoints').glob('step_*.pt'):
                if file.relative_to(self.root).as_posix() not in self.artifacts:
                    target=archive/'validation_checkpoints'; target.mkdir(parents=True,exist_ok=True)
                    file.replace(target/file.name)
            self.export()

    def append(self,name,record): self.logs[name].append({**self.identity,**record})

    def register_checkpoint(self,path):
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
            metric_contract=dict(follow_stage='post-step compaction_started',averaging='time within episode; then episodes within seed',
                success='existing geometry/compaction contract, not full protection certification',
                full_protection_success='unavailable',missing='null in JSON; blank in CSV',
                truth='offline metrics only; never Actor input'),**(extra or {})))


def strip_branch_traces(record):
    """Keep teacher decisions compact; move preselected trajectory arrays to gzip."""
    traces=[]
    def walk(value,path):
        if isinstance(value,dict):
            if 'trajectory' in value: traces.append(dict(branch_path=path,steps=value.pop('trajectory')))
            for k,v in value.items(): walk(v,path+'/'+k)
        elif isinstance(value,list):
            for i,v in enumerate(value): walk(v,path+'/'+str(i))
    walk(record,''); return traces
