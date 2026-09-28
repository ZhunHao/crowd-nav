from shipnav.observations import Observer
from shipnav.prediction import predict
from shipnav.simulation import CourseChangeTraffic, Traffic


def _visible(observer, traffic, ticks):
    return [tuple(s['id'] for s in observer.observe(traffic, tick*.25, tick) if s['age'] == 0)
            for tick in range(ticks)]


def test_dropout_is_keyed_per_seed_tick_and_target():
    traffic = [Traffic((4, 4), (20, 4)), Traffic((4, 8), (20, 8)), Traffic((4, 12), (20, 12))]
    a = _visible(Observer(seed=3, dropout=.5), traffic, 40)
    b = _visible(Observer(seed=3, dropout=.5), traffic, 40)
    assert a == b  # same (seed, tick, target) -> same draws
    # Keyed per target: at some tick one target is dropped while another is seen.
    assert any(0 < len(ids) < len(traffic) for ids in a)
    # Keyed per tick: the visible set changes over time for a fixed seed.
    assert len(set(a)) > 1
    # Keyed per seed.
    assert _visible(Observer(seed=4, dropout=.5), traffic, 40) != a


def test_delay_reads_a_true_past_state_not_the_current_one():
    # Turn at t=2: a 1.5 s delay at t=3 must measure the pre-turn state at
    # t=1.5 and extrapolate it, which differs from the post-turn truth at t=3.
    ship = CourseChangeTraffic(((0., (0., 0.)), (2., (2., 0.)), (4., (2., 2.))))
    [seen] = Observer(delay=1.5).observe([ship], 3., 12)
    assert seen['age'] == 1.5
    assert seen['velocity'] == ship.at(1.5)[1] == (1., 0.)
    assert seen['position'] == (3., 0.)
    assert seen['position'] != ship.at(3.)[0]


def test_stale_margin_grows_with_age_after_dropout():
    traffic = [Traffic((4, 4), (20, 4))]
    observer = Observer(noise=.1, stale_speed_bound=2.)
    fresh = observer.observe(traffic, 0., 0)[0]
    observer.dropout = 1.  # every later measurement is dropped
    later = [observer.observe(traffic, tick*.25, tick)[0] for tick in (1, 2, 3)]
    ages = [s['age'] for s in later]
    margins = [s['margin'] for s in later]
    assert fresh['age'] == 0 and fresh['margin'] == 3*.1
    assert ages == [.25, .5, .75]
    assert margins == sorted(margins) and len(set(margins)) == 3
    assert margins[-1] == 3*.1 + 2*2.*.75


def test_first_tick_dropout_leaves_target_unseen():
    assert Observer(dropout=1.).observe([Traffic((4, 4), (20, 4))], 0., 0) == []


def test_predict_without_uncertainty_has_zero_margins():
    observed = Observer(noise=.3).observe([Traffic((4, 4), (20, 4))], 1., 4)
    [prediction] = predict(observed, .25, uncertainty=False)
    assert prediction['margins'] == [0.]*len(prediction['points'])
    [uncertain] = predict(observed, .25)
    assert all(m > 0 for m in uncertain['margins'])
