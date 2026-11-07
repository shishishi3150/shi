"""Physical withdrawal replay after capture; no learning transitions or rewards."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np

from .geometry import capsule_segment
from .history import HistoryEnv
from .red_policy import RuleController
from .safety import center_path_enters_circle


class WithdrawalReplay:
    """Copy a successful endpoint without reopening or mutating its episode.

    The capture-confirmed signal ends Red's intrusion attempt. Blue continues
    local-rule pursuit, sensing, communication and configured safety screening.
    Departure is verified only when Red's whole hull crosses the entry west.
    """

    def __init__(self, env, *, max_seconds=120., controller=None):
        if env.terminal_reason != 'success' or not env._episode_done:
            raise ValueError('withdrawal replay requires a completed capture success')
        if not math.isfinite(max_seconds) or max_seconds <= 0.:
            raise ValueError('withdrawal replay duration must be positive and finite')
        base = env.env if isinstance(env, HistoryEnv) else env
        # Keep the terminated learning episode, sensor RNGs and queues intact.
        # A display and the predictor's disposable cache need not be copied.
        memo = {id(value): None for value in (base._renderer, base._safety_projector)
                if value is not None}
        self.env = deepcopy(base, memo)
        env = self.env
        self.dt = env.dt
        self.capture_step = env.step_count
        self.capture_time = env.step_count * self.dt
        self.steps = 0
        self.max_steps = math.ceil(max_seconds / self.dt)
        self.exit_x = float(env.route_origin[0])
        self.exit_goal = np.array([max(3., self.exit_x - 5.), float(env.route_origin[1])])
        self.red_state = env.red_state
        self.blue_states = env.blue_states
        self.submarine_states = env.submarine_states
        self.active = env.agent_active_mask()
        self.red_dynamics = env.red_dynamics
        self.blue_dynamics = env.blue_dynamics
        self.submarine_dynamics = env.submarine_dynamics
        self.wave = env.wave
        self.navigator = env.red_navigator
        self.controller = deepcopy(controller) if controller is not None else RuleController()
        self.navigator.request_withdrawal(self.exit_goal, reason='capture_confirmed')
        self.status = 'withdrawing'
        self.contact = ''
        self.done = self.verified = False
        self.last_blue_actions = np.zeros((env.num_friendly, 2), np.float32)
        self.last_red_action = np.zeros(2, np.float32)
        self.current_capture_valid = bool(env.task_state.gates) and all(env.task_state.gates.values())
        self.encircled_steps = self.encirclement_loss_steps = self.longest_encirclement_loss_steps = 0
        self.friendly_travel_m = np.zeros(env.num_friendly)

    @property
    def elapsed_seconds(self):
        return self.steps * self.dt

    def has_cleared_entry(self):
        left, right, radius = capsule_segment(self.red_state.pos, self.red_state.heading,
            self.env.surface_length, self.env.surface_collision_width)
        return bool(max(left[0], right[0]) + radius <= self.exit_x
                    and self.red_state.velocity[0] < -.02)

    def step(self):
        if self.done:
            raise RuntimeError('withdrawal replay already ended')
        old = deepcopy(self.blue_states + self.submarine_states + [self.red_state])
        now = self.capture_time + self.elapsed_seconds
        env = self.env
        nominal = self.controller.actions(self.blue_states, self.submarine_states, None,
            env.protected_center, env.reefs, env.world_size, self.blue_dynamics,
            self.submarine_dynamics, now, target_estimates=env.target_estimates,
            active_mask=self.active)
        if env.config['mechanisms']['safety_projection']:
            from .safety_projection import SafetyProjector
            if env._safety_projector is None:
                env._safety_projector = SafetyProjector(env)
            safe, env.safety_diagnostics = env._safety_projector.project(nominal)
        else:
            safe = nominal
        self.last_blue_actions = np.clip(safe, -env.agent_action_limits,
                                         env.agent_action_limits) * self.active[:, None]
        env.last_nominal_actions = nominal.copy()
        env.last_safe_actions = np.asarray(safe, np.float32).copy()
        env.last_executed_actions = self.last_blue_actions.copy()
        moved = []
        for i, state in enumerate(self.blue_states + self.submarine_states):
            if not self.active[i]:
                moved.append(deepcopy(state))
                continue
            dynamics = self.blue_dynamics if i < self.env.num_surface_blue else self.submarine_dynamics
            wave = self.wave.horizontal_velocity(state.pos, now) if i < self.env.num_surface_blue else None
            moved.append(dynamics.step(state, self.last_blue_actions[i], wave))
        surfaces = [s for i, s in enumerate(self.blue_states) if self.active[i]]
        reference = self.navigator.reference(self.red_state, self.env.protected_center,
            self.env.reefs, surfaces, self.env.world_size, self.env.config['ship'])
        self.last_red_action = self.red_dynamics.reference_action(self.red_state,
            desired_yaw_rate=float(reference[0]), desired_surge_acceleration=float(reference[1]))
        self.red_state = self.red_dynamics.step(self.red_state, self.last_red_action,
            self.wave.horizontal_velocity(self.red_state.pos, now))
        self.blue_states = moved[:self.env.num_surface_blue]
        self.submarine_states = moved[self.env.num_surface_blue:]
        self.steps += 1
        self.friendly_travel_m += [np.linalg.norm(after.pos - before.pos)
                                  for before, after in zip(old, moved)]
        env.blue_states, env.submarine_states, env.red_state = self.blue_states, self.submarine_states, self.red_state
        env.step_count = self.capture_step + self.steps
        env.communication.step(moved, env.step_count, active_mask=self.active)
        env._sense_and_communicate()
        reason, pair = self.env._contacts(old, moved + [self.red_state])
        if reason:
            self.status, self.contact, self.done = 'withdrawal_contact', pair, True
        elif center_path_enters_circle(old[-1], self.red_state,
                                       self.env.protected_center, self.env.protected_radius):
            self.status, self.done = 'withdrawal_breakthrough', True
        elif self.has_cleared_entry():
            self.status, self.done, self.verified = 'withdrawn', True, True
        elif self.steps >= self.max_steps:
            self.status, self.done = 'withdrawal_timeout', True
        # Reuse the original capture gates for diagnostics, without producing a
        # reward, reopening step(), or adding any learning/replay transition.
        env._task_success(safe=self.status not in ('withdrawal_contact', 'withdrawal_breakthrough'))
        self.current_capture_valid = all(env.task_state.gates.values())
        self.encircled_steps += self.current_capture_valid
        self.encirclement_loss_steps = 0 if self.current_capture_valid else self.encirclement_loss_steps + 1
        self.longest_encirclement_loss_steps = max(self.longest_encirclement_loss_steps,
                                                   self.encirclement_loss_steps)
        return self.trace_row()

    def trace_row(self):
        states = self.blue_states + self.submarine_states
        return dict(stage='post_capture_withdrawal', capture_step=self.capture_step,
            capture_time_s=self.capture_time, step=self.steps,
            time_s=self.capture_time + self.elapsed_seconds,
            elapsed_seconds=self.elapsed_seconds, status=self.status,
            withdrawal_verified=self.verified, contact=self.contact,
            exit_x_m=self.exit_x, red_x=float(self.red_state.pos[0]),
            red_y=float(self.red_state.pos[1]), red_heading_rad=self.red_state.heading,
            red_surge_mps=self.red_state.speed, red_sway_mps=self.red_state.sway_speed,
            red_yaw_rate_radps=self.red_state.yaw_rate,
            red_action=self.last_red_action.tolist(), blue_actions=self.last_blue_actions.tolist(),
            blue_behavior='CONTINUE_PURSUIT',
            capture_valid_now=self.current_capture_valid,
            capture_geometry=deepcopy(self.env.capture_geometry),
            capture_gates=self.env.task_state.gates.copy(),
            encircled_step_fraction=self.encircled_steps / self.steps if self.steps else None,
            longest_encirclement_loss_s=self.longest_encirclement_loss_steps * self.dt,
            friendly_travel_m=self.friendly_travel_m.tolist(),
            received_track_valid=[e.valid for e in self.env.target_estimates],
            received_track_age_s=[max(0., self.capture_time + self.elapsed_seconds - e.t_obs)
                                  if e.valid else None for e in self.env.target_estimates],
            friendly_positions=[s.pos.tolist() for s in states],
            friendly_headings=[s.heading for s in states],
            friendly_body_velocities=[s.body_velocity.tolist() for s in states])

    def render_frame(self):
        return replace(self.env.render_frame(show_shadow_red=False),
            red_state=self.red_state, blue_states=self.blue_states,
            submarine_states=self.submarine_states,
            phase='WITHDRAWN' if self.verified else 'WITHDRAWING' if not self.done else 'WITHDRAWAL_FAILED',
            simulation_time=self.capture_time + self.elapsed_seconds,
            termination_reason='capture success; ' + self.status,
            encirclement_valid=all(self.env.task_state.gates.get(k, False)
                for k in ('auv_participation', 'capture_distance', 'encirclement')),
            red_behavior='WITHDRAW',
            event_markers={'capture_confirmed': True, 'pursuit_continues': True,
                           'departure_verified': self.verified},
            diagnostics={'withdrawal_remaining_m':max(0., float(self.red_state.pos[0]) - self.exit_x),
                         'capture_max_m':max(self.env.capture_geometry['radii_m'], default=0.),
                         'enclosure_gap_deg':math.degrees(self.env.capture_geometry['maximum_angle_gap_rad'])})
