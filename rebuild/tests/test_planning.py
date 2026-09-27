import pytest
from shipnav.maps import SeaMap
from shipnav.planning import plan, NoPath, PlanningLimit


@pytest.mark.parametrize('planner', ['astar', 'theta'])
def test_detour_continuous_clearance_determinism(planner):
    sea = SeaMap((0, 0, 24, 24), [(9, 5, 14, 18)])
    result = plan(sea, (2, 12), (22, 12), planner=planner, resolution=1, clearance=.7)
    assert result.points[0] == (2, 12) and result.points[-1] == (22, 12)
    assert len(result.points) > 2
    assert all(sea.clear(a, b, .7) for a, b in zip(result.points, result.points[1:]))
    assert result.points == plan(sea, (2, 12), (22, 12), planner=planner, resolution=1, clearance=.7).points
    assert result.stats['expanded_nodes'] > 0


@pytest.mark.parametrize('planner', ['astar', 'theta'])
def test_disconnected_and_invalid_endpoints(planner):
    sea = SeaMap((0, 0, 10, 10), [(4, 0, 6, 10)])
    with pytest.raises(NoPath):
        plan(sea, (2, 5), (8, 5), planner=planner, clearance=.2)
    with pytest.raises(NoPath):
        plan(sea, (5, 5), (8, 5), planner=planner, clearance=.2)
    assert plan(sea, (2, 2), (2, 2), planner=planner).points == [(2., 2.)]


def test_budget_failure_is_distinct_from_no_path():
    sea = SeaMap((0, 0, 24, 24), [(9, 5, 14, 18)])
    with pytest.raises(PlanningLimit) as error:
        plan(sea, (2, 12), (22, 12), max_nodes=1)
    assert error.value.stats['expanded_nodes'] == 1
    with pytest.raises(PlanningLimit):
        plan(sea, (2, 12), (22, 12), resolution=.001)


@pytest.mark.parametrize('kwargs', [{'resolution': 0}, {'timeout': -1}, {'clearance': float('nan')}, {'planner': 'invalid'}])
def test_bad_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        plan(SeaMap((0, 0, 10, 10)), (2, 2), (8, 8), **kwargs)


def test_diagonal_cannot_cut_touching_islands():
    sea = SeaMap((0, 0, 6, 6), [(0, 3, 3, 6), (3, 0, 6, 3)])
    with pytest.raises(NoPath):
        plan(sea, (1, 1), (5, 5), clearance=.1)


def test_validation_cannot_escape_deadline(monkeypatch):
    import shipnav.planning as planning
    clock=[0.]
    real=SeaMap((0,0,10,10))
    class SlowClearance:
        bounds=real.bounds
        clear=real.clear
        def minimum_clearance(self,a,b):
            clock[0]=2.
            return real.minimum_clearance(a,b)
    monkeypatch.setattr(planning,'perf_counter',lambda:clock[0])
    with pytest.raises(PlanningLimit):
        plan(SlowClearance(),(2,2),(8,8),timeout=1.)
