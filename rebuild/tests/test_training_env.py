from math import dist

from gymnasium.utils.env_checker import check_env as gym_check_env
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from shipnav.maps import SeaMap
from shipnav.scenarios import make_scenario
from shipnav.training.env import EnvConfig, NavigationEnv, remaining_route


def scene(land=(), start=(2., 2.), goal=(20., 2.), count=0, seed=1):
    return make_scenario(SeaMap((0, 0, 24, 24), land), start, goal, count, seed)


def run(env, action, steps=400):
    out = []
    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(action)
        out.append((observation, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return out


def test_checkers_pass_and_reset_is_seeded():
    gym_check_env(NavigationEnv([scene(count=2)], EnvConfig(filtered=False)), skip_render_check=True)
    sb3_check_env(NavigationEnv([scene(count=2)], EnvConfig(filtered=False)))
    env = NavigationEnv([scene(count=2, seed=s) for s in (1, 2, 3)], EnvConfig(filtered=False))
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b) and env.observation_space.contains(a)


def test_time_limit_truncates():
    env = NavigationEnv([scene()], EnvConfig(filtered=False, limit=.25))
    env.reset(seed=1)
    _, _, terminated, truncated, info = env.step([1., 0.])
    assert truncated and not terminated and info['status'] == 'timeout' and 'episode_cost' in info
    with pytest.raises(RuntimeError, match='reset'):
        env.step([1., 0.])


def test_land_collision_terminates_with_a_valid_final_observation():
    env = NavigationEnv([scene(land=((0, 5, 24, 6),))], EnvConfig(filtered=False))
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = run(env, [0., 1.])[-1]
    assert terminated and not truncated and info['status'] == 'collision'
    assert reward < -9 and env.observation_space.contains(observation)


def test_domain_cost_is_logged_penalised_and_never_observed():
    # A ship 1.3 m abeam: truth clearance 1.3-.5-.6 = .2 m < 1 m domain, but no collision.
    alongside = {**scene(), 'traffic': [{'start': [2., 3.3], 'goal': [22., 3.3], 'speed': .3, 'radius': .6}]}
    plain = NavigationEnv([alongside], EnvConfig(filtered=False))
    priced = NavigationEnv([alongside], EnvConfig(filtered=False, penalty=5.))
    plain.reset(seed=1), priced.reset(seed=1)
    (o1, r1, *_, i1), (o2, r2, *_, i2) = plain.step([1., 0.]), priced.step([1., 0.])
    assert i1['cost'] == pytest.approx(.25) and np.array_equal(o1, o2)
    assert r1-r2 == pytest.approx(5*.25)
    priced.set_penalty(0.)
    with pytest.raises(ValueError):
        priced.set_penalty(-1.)


def test_progress_follows_the_route_not_the_straight_line():
    env = NavigationEnv([scene(land=((6, 3.5, 8, 24),), start=(4., 20.), goal=(12., 20.))],
                        EnvConfig(filtered=False))
    env.reset(seed=1)
    total, before = 0., dist(env.packet['position'], (12., 20.))
    for _ in range(12):
        p, g = env.packet['position'], env.packet['goal']
        d = dist(p, g)
        *_, info = env.step([(g[0]-p[0])/d, (g[1]-p[1])/d])
        total += info['progress']
    assert total > 0 and dist(env.packet['position'], (12., 20.)) > before


def test_remaining_route_sums_the_unvisited_legs():
    assert remaining_route((0., 1.), [(0., 0.), (0., 4.), (3., 4.)], 1) == pytest.approx(6.)


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        NavigationEnv([])
    with pytest.raises(ValueError):
        EnvConfig(observation={'seed': 1})
    env = NavigationEnv([scene()], EnvConfig(filtered=False))
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step([np.inf, 0.])
