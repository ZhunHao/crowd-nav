from time import sleep

import pytest

from shipnav.maps import SeaMap
from shipnav.simulation import Traffic, episode_steps, run_episode


def toward(p, v, goal, neighbours, radius, speed, dt):
    from math import dist
    d = dist(p, goal)
    return tuple((b-a)/d*min(speed, d/dt) if d else 0 for a, b in zip(p, goal))


def drive(engine, pause=0.):
    try:
        packet = next(engine)
        while True:
            sleep(pause)
            action = toward(packet['position'], packet['velocity'], packet['goal'],
                            packet['neighbours'], packet['radius'], packet['speed'], packet['dt'])
            packet = engine.send((action, 0.))
    except StopIteration as ended:
        return ended.value


def test_learner_pause_changes_neither_latency_nor_physics():
    sea = SeaMap((0, 0, 20, 20))
    route, limit = [(2, 10), (18, 10)], 2.  # 8 steps keep the deliberate pauses short
    result, _ = drive(episode_steps(sea, route, [Traffic((10, 1), (10, 19))], filtered=True, limit=limit), pause=.2)
    reference = run_episode(sea, route, [Traffic((10, 1), (10, 19))], toward, filtered=True, limit=limit)
    assert result['frames'] == reference['frames'] and result['status'] == reference['status']
    assert max(d['decision_ms'] for d in result['diagnostics']) < 200
    assert result['inference_ms'] == [0.]*len(result['inference_ms'])


def test_initially_terminal_episode_returns_without_yielding():
    with pytest.raises(StopIteration) as ended:
        next(episode_steps(SeaMap((0, 0, 10, 10)), [(2, 2)], []))
    result, terminal = ended.value.value
    assert result['status'] == 'success' and terminal is None


def test_terminal_packet_reports_final_state_on_timeout():
    engine = episode_steps(SeaMap((0, 0, 10, 10)), [(2, 2), (8, 2)], [Traffic((5, 8), (5, 1))],
                           limit=.25, terminal_packet=True)
    result, terminal = drive(engine)
    assert result['status'] == 'timeout'
    assert terminal['t'] == .25 and terminal['goal_index'] == 1 and terminal['last']['dt'] == .25
    assert [o['id'] for o in terminal['observed']] == [0]


def test_land_collision_is_reported_in_last():
    sea = SeaMap((0, 0, 10, 10), ((4, 4, 6, 6),))
    result, terminal = drive(episode_steps(sea, [(2, 5), (8, 5)], [], terminal_packet=True))
    assert result['status'] == 'collision'
    assert terminal['last']['collision'] and terminal['last']['land_collision']


def test_close_releases_a_suspended_engine():
    engine = episode_steps(SeaMap((0, 0, 10, 10)), [(2, 2), (8, 2)], [])
    next(engine)
    engine.close()
    with pytest.raises(StopIteration):
        engine.send(((1., 0.), 0.))


def test_record_false_keeps_outcome_but_drops_traces():
    args = (SeaMap((0, 0, 20, 20)), [(2, 2), (8, 2), (8, 8)], [])
    full, _ = drive(episode_steps(*args))
    light, _ = drive(episode_steps(*args, record=False))
    assert (light['status'], light['elapsed']) == (full['status'], full['elapsed'])
    assert light['frames'] == [] and light['diagnostics'] == []


def test_run_episode_prefers_packet_policies():
    class PacketPolicy:
        seen = []

        def act(self, packet):
            self.seen.append(packet['goal_index'])
            return toward(packet['position'], packet['velocity'], packet['goal'], [], .5, 1., packet['dt'])

    policy = PacketPolicy()
    assert run_episode(SeaMap((0, 0, 20, 20)), [(2, 2), (8, 2)], [], policy)['status'] == 'success'
    assert policy.seen and set(policy.seen) == {1}
