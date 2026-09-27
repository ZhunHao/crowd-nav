"""Bounded migration controller replay with constant-velocity neighbours.

This is an acceptance harness, not the original CrowdSim simulator. One clock,
Cartesian metres/seconds, perfect observations, no intermediate goals or safety
interventions. Neighbours follow frozen initial velocities independently.
"""
from pathlib import Path
import math
import torch

MODEL = Path(__file__).resolve().parents[3]/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def replay(scenario, max_steps=64, policy=None, states=None):
    if max_steps <= 0:
        raise ValueError('max_steps must be positive')
    if policy is None:
        from shipnav.model import load_policy
        policy = load_policy(MODEL)
    if states is None:
        from shipnav.compat.crowd_sim.envs.utils import state as states
    dt = .25
    policy.query_env = False
    policy.time_step = dt
    own, others = list(scenario['own']), [list(h) for h in scenario['others']]
    trace = []
    status = 'timeout'
    with torch.inference_mode():
        for step in range(max_steps):
            if math.hypot(own[0]-own[5],own[1]-own[6]) < own[4]:
                status = 'goal'
                break
            action = policy.predict(states.JointState(states.FullState(*own),[states.ObservableState(*h) for h in others]))
            if not all(math.isfinite(v) for v in action):
                raise ValueError('Non-finite replay action')
            clearance = float('inf')
            for h in others:
                dx,dy = h[0]-own[0],h[1]-own[1]
                vx,vy = h[2]-action[0],h[3]-action[1]
                speed2 = vx*vx+vy*vy
                t = max(0,min(dt,-(dx*vx+dy*vy)/speed2)) if speed2 else 0
                clearance = min(clearance,math.hypot(dx+t*vx,dy+t*vy)-h[4]-own[4])
            own[0] += dt*action[0]; own[1] += dt*action[1]
            own[2:4] = action
            for h in others:
                h[0] += dt*h[2]; h[1] += dt*h[3]
            trace.append({'t':(step+1)*dt,'own':own.copy(),'others':[h.copy() for h in others],
                          'action':list(action),'clearance':clearance if others else None})
            if clearance < 0:
                status = 'collision'
                break
            if math.hypot(own[0]-own[5],own[1]-own[6]) < own[4]:
                status = 'goal'
                break
    return {'id':scenario['id'],'status':status,'trace':trace,'time_step':dt,
            'max_steps':max_steps,'safety_interventions':0}
