from math import dist
from shipnav.maps import SeaMap
from shipnav.observations import Observer
from shipnav.reactive import ReactiveTraffic
from shipnav.simulation import Traffic, run_episode


def test_reaction_changes_path_but_preserves_history():
    s = ReactiveTraffic(Traffic((5, 5), (10, 5)), SeaMap((0, 0, 24, 24)))
    s.advance(0, .25, (6, 5))
    assert s.at(.25)[0][0] < 5
    assert s.at(0)[0] == (5, 5)


def _toward(p, v, goal, neighbours, radius, speed, dt):
    d = dist(p, goal)
    return tuple((b-a)/d*min(speed, d/dt) if d else 0 for a, b in zip(p, goal))


def _stay(p, v, goal, neighbours, radius, speed, dt):
    return (0.0, 0.0)


def test_advance_reacts_to_the_egos_realized_post_decision_position():
    # run_episode calls ship.advance(time, step, position) after the policy
    # has chosen this step's action and before truth checks -- so a reactive
    # target facing a policy that holds the ego still must never see it
    # "already" at some future, not-yet-realized position.
    sea = SeaMap((0, 0, 24, 24))
    target = ReactiveTraffic(Traffic((6, 12), (18, 12)), sea)
    run_episode(sea, [(3, 12), (21, 12)], [target], _stay, dt=.5, limit=1)
    assert len(target.history) == 3  # one appended entry per step (limit/dt)
    for (ta, pa, _), (tb, pb, _) in zip(target.history, target.history[1:]):
        assert tb > ta
    # Ego never moved, so every reaction was computed against the same,
    # already-realized ego position -- not a lookahead.
    assert target.at(0)[0] == (6, 12)


def test_paired_reactive_runs_share_initial_conditions_but_not_realized_target_paths():
    # Paired reactive experiments must compare the same initial
    # scenario/hash, target rule, and exogenous (observation) noise; they
    # must NOT assert identical realized target paths when the ego differs
    # -- the whole point of a reactive target is that it responds to the
    # ego it actually faces.
    sea = SeaMap((0, 0, 24, 24))
    route = [(3, 12), (21, 12)]
    seed_ship = Traffic((5, 12), (19, 12))

    target_toward = ReactiveTraffic(seed_ship, sea)
    target_stay = ReactiveTraffic(seed_ship, sea)
    run_episode(sea, route, [target_toward], _toward, dt=.5, limit=3, observer=Observer(seed=42))
    run_episode(sea, route, [target_stay], _stay, dt=.5, limit=3, observer=Observer(seed=42))

    # Shared initial conditions: same seed voyage, same start, same goal.
    assert target_toward.history[0] == target_stay.history[0] == (0., (5., 12.), (.3, 0.))
    assert target_toward.start == target_stay.start
    assert target_toward.goal == target_stay.goal
    assert target_toward.radius == target_stay.radius
    # Different ego policy -> free to realize a different target path; this
    # is never asserted to be identical.
    assert target_toward.history[-1][1] != target_stay.history[-1][1]


def test_initial_history_velocity_is_the_nominal_voyage_velocity():
    target = ReactiveTraffic(Traffic((5, 5), (10, 5), speed=.3), SeaMap((0, 0, 24, 24)))
    assert target.history[0] == (0., (5, 5), (.3, 0.))
    assert target.at(0.)[1] == (.3, 0.)
