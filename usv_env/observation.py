"""Decentralized estimated observations and centralized training state."""
import math
import numpy as np
from .geometry import to_local
from .history import SCHEMA

ACTOR_OBSERVATION_DIM = SCHEMA.sensor_dim
CRITIC_STATE_DIM = 5*9 + 7 + 3*3 + 3 + 5*6 + 5


def critic_state_dim(num_agents=5, num_reefs=3):
    return num_agents*9 + 7 + num_reefs*3 + 3 + num_agents*6 + 5


def effective_end_step(env):
    """Known task deadline after the public capture-confirmed mission event.

    This exposes no target truth or controller reference. The same event
    triggers Red's withdrawal; every friendly agent can retain the event.
    """
    if not getattr(env, 'capture_success', False):
        return env.max_steps
    end = int(getattr(env, 'post_capture_end_step', 0))
    return max(1, end if end else env.step_count)


def build_actor_observation(env, receiver):
    schema = env.observation_schema
    ship = env.friendly_states()[receiver]
    active = env.agent_active_mask()
    scale = float(max(env.world_size))
    vmax = env.max_speed_by_type[ship.unit_type]
    estimate = env.target_estimates[receiver]
    # No controller reference, assigned location or search path crosses this boundary.
    values = [float(active[receiver]), float(ship.unit_type=="surface"), float(ship.unit_type=="submarine"),
              ship.speed/vmax, ship.sway_speed/env.max_sway_speed, ship.yaw_rate/env.max_yaw_rate,
              math.sin(ship.heading), math.cos(ship.heading)]
    if estimate.valid:
        relative = to_local(estimate.position-ship.pos, ship.heading)/scale
        velocity = to_local(estimate.velocity-ship.velocity, ship.heading)/1.5
        speed = float(np.linalg.norm(estimate.velocity))
        heading = math.atan2(estimate.velocity[1], estimate.velocity[0])-ship.heading if speed>.02 else 0.
        values.extend([*relative, *velocity, math.sin(heading) if speed>.02 else 0., math.cos(heading) if speed>.02 else 0.])
    else: values.extend([0.]*6)
    for sender in range(env.num_friendly):
        if sender == receiver: continue
        packet = env.communication.cache.get((receiver,sender))
        if packet is None:
            values.extend([0.]*9); continue
        age = (env.step_count-packet.send_step)*env.dt
        relative = to_local(packet.position-ship.pos,ship.heading)/scale
        velocity = to_local(packet.velocity-ship.velocity,ship.heading)/1.5
        values.extend([1., float(sender<env.num_surface_blue), float(packet.surfaced),
                       *relative, *velocity, float(env.communication.fresh[receiver,sender]), max(age,0.)])
    values.extend([ship.pos[0]/scale, (env.world_size[0]-ship.pos[0])/scale,
                   ship.pos[1]/scale, (env.world_size[1]-ship.pos[1])/scale])
    for reef in env.reefs:
        center = np.asarray(reef['center'])
        relative = to_local(center-ship.pos,ship.heading)/scale
        values.extend([*relative, (np.linalg.norm(center-ship.pos)-reef['radius'])/scale])
    values.extend([*list(to_local(env.protected_center-ship.pos,ship.heading)/scale), env.protected_radius/scale])
    values.extend(list(env.last_executed_actions[receiver]/env.agent_action_limits[receiver]))
    if estimate.valid:
        values.extend([1., max(0.,env.step_count*env.dt-estimate.t_obs),
                       min(1e3,float(np.trace(estimate.covariance[:2,:2])))/25.,
                       min(1e3,float(np.trace(estimate.covariance[2:,2:]))), estimate.confidence])
    else: values.extend([0.]*5)
    values.extend([float(env.sensor_detected[receiver-env.num_surface_blue]) if receiver>=env.num_surface_blue else 0.,
                   max(0.,1.-env.step_count/effective_end_step(env)),
                   float(getattr(env, 'capture_success', False))])
    result=np.asarray(values,np.float32)
    if result.shape != (schema.sensor_dim,) or not np.isfinite(result).all():
        raise ValueError(f"invalid actor observation shape {result.shape}, expected {schema.sensor_dim}")
    return result


def build_critic_state(env):
    """Truth is legal here under CTDE. It never crosses the Actor boundary."""
    scale=float(max(env.world_size)); values=[]
    for state,active in zip(env.friendly_states(),env.agent_active_mask()):
        values.extend([float(active),float(state.unit_type=='surface'),*list(state.pos/scale),
                       *list(state.velocity/1.5),math.sin(state.heading),math.cos(state.heading),state.yaw_rate/env.max_yaw_rate])
    red=env.red_state
    values.extend([*list(red.pos/scale),*list(red.velocity/1.5),math.sin(red.heading),math.cos(red.heading),red.yaw_rate/env.max_yaw_rate])
    for reef in env.reefs: values.extend([*list(np.asarray(reef['center'])/scale),reef['radius']/scale])
    values.extend([*list(env.protected_center/scale),env.protected_radius/scale])
    for estimate in env.target_estimates:
        values.extend([float(estimate.valid), *(list(estimate.position/scale) if estimate.valid else [0.,0.]),
                       *(list(estimate.velocity/1.5) if estimate.valid else [0.,0.]),
                       max(0.,env.step_count*env.dt-estimate.t_obs) if estimate.valid else 0.])
    values.extend([env.step_count/effective_end_step(env),env.hold_steps/max(env.required_hold_steps,1),
                   float(np.linalg.norm(env.protected_center-env.shadow_red_state.pos)/scale),env.counterfactual_gap/scale,
                   float(getattr(env, 'capture_success', False))])
    result=np.asarray(values,np.float32)
    if result.shape != (env.critic_state_dim,) or not np.isfinite(result).all(): raise ValueError("invalid critic state")
    return result
