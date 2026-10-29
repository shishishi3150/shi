"""Two fixed UTF-8-SIG CSVs, aggregate windows and lossless resume branching."""
import csv
import hashlib
import io
import math
from pathlib import Path
import uuid

TRAIN_FIELDS="episode,env_steps,episode_steps,team_return,usv_return,auv_return,process_return,terminal_return,process_abs_sum,success,terminal_reason,shrink_progress,wall_time_s,eval_success_rate,eval_return".split(",")
UPDATE_FIELDS="env_steps,update_step,usv_actor_loss,auv_actor_loss,usv_critic_loss,auv_critic_loss,usv_td_abs,auv_td_abs,i2_grad_ratio,i3_bc_loss,i3_grad_ratio,teacher_calls,teacher_accepted,teacher_sim_steps,wall_time_s".split(",")


class CsvLog:
    def __init__(self,path,fields,resume=False):
        self.path,self.fields=Path(path),list(fields)
        if self.path.exists():
            if not resume: raise FileExistsError(self.path)
            self.validate()
        else:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open("x",encoding="utf-8-sig",newline="") as f: csv.writer(f).writerow(fields)

    def validate(self):
        raw=self.path.read_bytes()
        if not raw.startswith(b"\xef\xbb\xbf") or raw.count(b"\xef\xbb\xbf")!=1: raise ValueError("invalid CSV BOM")
        rows=list(csv.reader(io.StringIO(raw.decode("utf-8-sig"),newline=""),strict=True))
        if not rows or rows[0]!=self.fields: raise ValueError("CSV schema mismatch")
        for index,row in enumerate(rows[1:],2):
            if len(row)!=len(self.fields): raise ValueError(f"corrupt CSV row {index}")
            for key,value in zip(self.fields,row):
                if value and key!="terminal_reason":
                    try: finite=math.isfinite(float(value))
                    except ValueError: finite=False
                    if not finite: raise ValueError(f"corrupt/nonfinite CSV {key} row {index}")

    def append(self,row):
        if set(row)!=set(self.fields): raise ValueError("row fields differ from fixed CSV schema")
        values=[]
        for name in self.fields:
            value=row[name]
            if value is None: value=""
            elif isinstance(value,bool): value=int(value)
            elif name!="terminal_reason" and not math.isfinite(float(value)): raise FloatingPointError(f"nonfinite CSV {name}")
            values.append(value)
        with self.path.open("a",encoding="utf-8",newline="") as f:
            csv.writer(f).writerow(values); f.flush()

    def commit(self):
        data=self.path.read_bytes()
        return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"rows":len(data.splitlines())-1}

    @staticmethod
    def recover(path,commit,fields):
        path=Path(path); data=path.read_bytes(); offset=commit["bytes"]
        if len(data)<offset or hashlib.sha256(data[:offset]).hexdigest()!=commit["sha256"]:
            raise ValueError("CSV committed prefix is missing/corrupt; refusing recovery")
        lines=data.splitlines()
        bad_last=False
        for index,line in enumerate(lines[1:],1):
            try:
                row=next(csv.reader([line.decode("utf-8")],strict=True))
                if len(row)!=len(fields): raise ValueError("column count")
                for name,value in zip(fields,row):
                    if name!="terminal_reason" and value and not math.isfinite(float(value)): raise ValueError("nonfinite")
            except (ValueError,csv.Error):
                if index!=len(lines)-1: raise ValueError(f"corrupt middle CSV row {index+1}; refusing recovery")
                bad_last=True
        # A damaged tail is preserved byte-for-byte, not interpreted as committed data.
        if len(data)>offset:
            archive=path.parent/"orphaned_attempt"/uuid.uuid4().hex
            archive.mkdir(parents=True,exist_ok=False)
            (archive/path.name).write_bytes(data[offset:])
            if bad_last:
                (archive/"corrupt_tail.txt").write_text("Final uncommitted CSV row was corrupt and preserved in the archived bytes.\n",encoding="utf-8")
            temporary=path.with_suffix(".recover.tmp"); temporary.write_bytes(data[:offset]); temporary.replace(path)
        return CsvLog(path,fields,resume=True)


class UpdateWindow:
    metrics=UPDATE_FIELDS[2:11]
    def __init__(self,state=None):
        self.count=0; self.sums={k:0. for k in self.metrics}; self.counts={k:0 for k in self.metrics}
        if state:
            self.count=state["count"]; self.sums=state["sums"].copy(); self.counts=state["counts"].copy()

    def add(self,values):
        self.count+=1
        for k in self.metrics:
            value=values[k]
            if value is not None:
                if not math.isfinite(value): raise FloatingPointError(f"nonfinite update {k}")
                self.sums[k]+=value; self.counts[k]+=1

    def flush(self,log,agent,teacher,wall_time):
        if not self.count: return
        row={k:(self.sums[k]/self.counts[k] if self.counts[k] else None) for k in self.metrics}
        rev=agent.runtime_config["revision"]
        if not rev["i2_enabled"]: row["i2_grad_ratio"]=0.
        if not rev["teacher_enabled"]: row.update(i3_bc_loss=0.,i3_grad_ratio=0.)
        elif row["i3_grad_ratio"] is None: row["i3_grad_ratio"]=0.
        row.update(env_steps=agent.global_env_steps,update_step=agent.update_steps,teacher_calls=teacher.calls,
                   teacher_accepted=teacher.accepted,teacher_sim_steps=teacher.sim_steps,wall_time_s=wall_time)
        log.append(row)
        self.__init__()

    def state_dict(self): return {"count":self.count,"sums":self.sums.copy(),"counts":self.counts.copy()}
