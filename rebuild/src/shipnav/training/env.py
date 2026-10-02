"""Gymnasium view of the shared marine step engine (observation schema 1)."""
from dataclasses import asdict, dataclass, field
from math import dist, isfinite

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from shipnav.episode import build_traffic, episode_limit, plan_route
from shipnav.maps import SeaMap
from shipnav.observations import Observer
from shipnav.scale import map_options, units
from shipnav.scenarios import scenario_hash, validate_observation, validate_scenario
from shipnav.simulation import episode_steps
from shipnav.training.observation import SIZE, action_to_velocity, encode


@dataclass(frozen=True)
class EnvConfig:
    observation: dict = field(default_factory=dict)
    filtered: bool = True
    penalty: float = 0.
    domain_m: float = 1.
    progress_weight: float = 1.
    time_penalty: float = .01
    success_reward: float = 10.
    collision_penalty: float = 10.
    limit: float | None = None

    def __post_init__(self):
        validate_observation(self.observation)
        numbers = (self.penalty, self.domain_m, self.progress_weight, self.time_penalty,
                   self.success_reward, self.collision_penalty)
        if not all(isfinite(x) and x >= 0 for x in numbers) or self.domain_m <= 0:
            raise ValueError('Reward/cost settings must be finite and nonnegative; domain_m positive')
        if self.limit is not None and not (isfinite(self.limit) and self.limit > 0):
            raise ValueError('Limit override must be positive and finite')

    def to_dict(self):
        return asdict(self)


def remaining_route(position, route, index):
    """Path length from `position` through the unvisited waypoints `route[index:]`."""
    index = min(index, len(route)-1)
    return dist(position, route[index]) + sum(dist(a, b) for a, b in zip(route[index:], route[index+1:]))


def domain_cost(last, length_m, time_s, domain_m):
    """Physical seconds of the last step spent inside the truth ship domain.

    Mirrors `metrics._exposure`: a step counts, weighted by its actual interval
    (`last['dt']`, so a partial final tick is shortened) in physical seconds, when
    its truth swept clearance (model units -> metres) is strictly below the domain.
    """
    if last is None or last['clearance'] is None:
        return 0.
    return last['dt']*time_s if last['clearance']*length_m < domain_m else 0.


class NavigationEnv(gym.Env):
    """Reward is route progress (remaining waypoint-path length), not straight-line goal distance.

    Known, accepted quirk: when the ego reaches a waypoint (within one ego radius) the
    target leg advances, so that single step can earn at most one ego-radius of extra
    progress. Cost is the truth ship-domain exposure and is never part of the observation.
    """
    metadata = {'render_modes': []}

    def __init__(self, scenarios, config=EnvConfig()):
        if not scenarios:
            raise ValueError('Empty training split')
        for scenario in scenarios:
            validate_scenario(scenario)
        self.scenarios, self.config, self.penalty = tuple(scenarios), config, config.penalty
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1., 1., (SIZE,), dtype=np.float32)
        self.engine, self.done = None, True

    def set_penalty(self, value):
        value = float(value)
        if not isfinite(value) or value < 0:
            raise ValueError('Penalty multiplier must be finite and nonnegative')
        self.penalty = value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        scenario = self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        sea = SeaMap.from_dict(scenario['map'])
        self.route = plan_route(sea, scenario['start'], scenario['goal'], **map_options(scenario['map']))
        self.length_m, self.time_s = units(scenario['map'])
        observer = Observer(seed=int(self.np_random.integers(2**31)), **self.config.observation)
        limit = self.config.limit or episode_limit(scenario['map'], self.route)
        self.engine = episode_steps(sea, self.route, build_traffic(scenario, sea), None, limit=limit,
                                    observer=observer, filtered=self.config.filtered, dynamics='marine',
                                    record=False, terminal_packet=True)
        try:
            self.packet = next(self.engine)
        except StopIteration as error:
            raise ValueError('Training split contains an initially terminal scenario') from error
        self.done, self.episode_cost = False, 0.
        self.remaining = remaining_route(self.packet['position'], self.route, self.packet['goal_index'])
        return encode(self.packet), {'scenario_hash': scenario_hash(scenario)}

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset() after an episode ends')
        command = action_to_velocity(action, self.packet['speed'])
        try:
            self.packet, status = self.engine.send((command, 0.)), 'running'
        except StopIteration as ended:
            result, self.packet = ended.value
            status = result['status']
        c = self.config
        cost = domain_cost(self.packet['last'], self.length_m, self.time_s, c.domain_m)
        remaining = 0. if status == 'success' else remaining_route(
            self.packet['position'], self.route, self.packet['goal_index'])
        progress = self.remaining-remaining
        reward = (c.progress_weight*progress - c.time_penalty + c.success_reward*(status == 'success')
                  - c.collision_penalty*(status == 'collision') - self.penalty*cost)
        self.remaining, self.episode_cost = remaining, self.episode_cost+cost
        terminated, truncated = status in ('success', 'collision'), status in ('timeout', 'cancelled')
        self.done = terminated or truncated
        info = {'cost': cost, 'progress': progress, 'status': status}
        if self.done:
            info['episode_cost'] = self.episode_cost
        return encode(self.packet), float(reward), terminated, truncated, info

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
