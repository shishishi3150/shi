"""Two fixed UTF-8-SIG CSV schemas with validated, lightweight append logging."""
import csv
import io
import math
from pathlib import Path
from matd3.history_enhancement import I3_COUNT_FIELDS, I3_DISTRIBUTION_FIELDS
from usv_env.reward import PAID_COMPONENTS

LEGACY_TRAIN_FIELDS="episode,env_steps,episode_steps,team_return,usv_return,auv_return,process_return,terminal_return,process_abs_sum,success,terminal_reason,shrink_progress,wall_time_s,eval_success_rate,eval_return".split(",")
TRAIN_FIELDS=LEGACY_TRAIN_FIELDS + 'discounted_team_return discounted_usv_return discounted_auv_return discounted_process_return discounted_terminal_return reward_gamma'.split()
TRAIN_FIELDS += 'capture_success first_capture_step first_capture_time_s post_capture_steps withdrawal_cleared_entry'.split()
MISSION_TRAIN_FIELDS = 'episode_return episode_return_team_sum episode_length capture_triggered full_mission_success post_capture_encirclement_ratio post_capture_mean_Q maximum_loss_duration final_5s_capture_ratio usv_red_collision_count usv_block_effect usv_follow_quality usv1_follow_quality usv2_follow_quality'.split()
TRAIN_FIELDS += MISSION_TRAIN_FIELDS
TRAIN_FIELDS += [prefix+key for prefix in ('component_','discounted_component_') for key in PAID_COMPONENTS]
UPDATE_FIELDS="env_steps,update_step,usv_actor_loss,auv_actor_loss,usv_critic_loss,auv_critic_loss,usv_td_abs,auv_td_abs,i2_grad_ratio,i3_bc_loss,i3_grad_ratio,teacher_calls,teacher_accepted,teacher_sim_steps,wall_time_s".split(",")
UPDATE_FIELDS += 'usv_q1 usv_q2 auv_q1 auv_q2 usv_target_q1 usv_target_q2 auv_target_q1 auv_target_q2 exploration_noise i1_target_intervention i1_target_projection_norm i1_target_infeasible i1_target_vo_time_s i3_queries i3_retrieved_candidates i3_valid_candidates i3_candidate_acceptance_rate i3_acceptance_rate i3_similarity_distance i3_phase_match_fraction i3_safety_rejection_fraction i3_history_quality i3_q_policy i3_q_history i3_advantage i3_confidence i3_enhancement_weight i3_target_enhancement i3_realized_nstep_return i3_outcome_confidence'.split()
UPDATE_FIELDS += ['i1_actor_vo_time_s']
UPDATE_FIELDS += 'i3_vo_screen_time_s i1_target_triggered i1_target_direction_correction_rad'.split()
UPDATE_FIELDS += [key for key in (*I3_COUNT_FIELDS, *I3_DISTRIBUTION_FIELDS) if key not in UPDATE_FIELDS]
UPDATE_FIELDS += 'usv_actor_grad_norm auv_actor_grad_norm usv_critic_grad_norm auv_critic_grad_norm'.split()


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
