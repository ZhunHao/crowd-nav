from math import dist
from pathlib import Path

from shipnav.compat.crowd_sim.envs.utils.state import FullState, JointState, ObservableState


class Direct:
    def __call__(self, position, velocity, goal, neighbours, radius, speed, dt):
        distance = dist(position, goal)
        magnitude = min(speed, distance/dt)
        return tuple((b-a)/distance*magnitude if distance else 0.0
                     for a, b in zip(position, goal))


def _joint_state(position, velocity, goal, neighbours, radius, speed):
    state = FullState(*position, *velocity, radius, *goal, speed, 0.0)
    others = [ObservableState(*p, *v, r) for p, v, r in neighbours]
    return JointState(state, others)


class Learned:
    def __init__(self, model_dir: Path):
        from shipnav.model import load_policy
        self.policy = load_policy(model_dir)
        self.policy.query_env = False

    def __call__(self, position, velocity, goal, neighbours, radius, speed, dt):
        if not neighbours:
            return Direct()(position, velocity, goal, neighbours, radius, speed, dt)
        import torch
        self.policy.time_step = dt
        joint_state = _joint_state(position, velocity, goal, neighbours, radius, speed)
        with torch.inference_mode():
            action = self.policy.predict(joint_state)
        return float(action.vx), float(action.vy)


class Reciprocal:
    def __init__(self):
        from shipnav.compat.crowd_sim.envs.policy.orca import ORCA
        self.policy = ORCA()

    def __call__(self, position, velocity, goal, neighbours, radius, speed, dt):
        if not neighbours:
            return Direct()(position, velocity, goal, neighbours, radius, speed, dt)
        self.policy.time_step = dt
        joint_state = _joint_state(position, velocity, goal, neighbours, radius, speed)
        action = self.policy.predict(joint_state)
        return float(action.vx), float(action.vy)
