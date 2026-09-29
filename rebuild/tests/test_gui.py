from threading import Event
from time import sleep
import pytest
from shapely.geometry import box
from shipnav.maps import SeaMap
from shipnav.gui import Window
from shipnav.service import execute


def test_start_goal_and_land_editing(qtbot):
    window = Window(SeaMap((0, 0, 24, 24), metadata={'description': 'test'}))
    qtbot.addWidget(window)
    window.apply_click(3, 3)
    assert window.start == (3, 3)
    window.mode.setCurrentIndex(1)
    window.apply_click(21, 21)
    assert window.goal == (21, 21)
    window.mode.setCurrentIndex(2)
    window.apply_click(8, 8)
    window.apply_click(10, 10)
    assert len(window.sea.land) == 1 and window.sea.land[0].equals(box(8, 8, 10, 10))
    assert window.sea.to_dict()['metadata'] == {'description': 'test'}
    window.apply_click(2, 2)
    window.apply_click(4, 4)
    assert len(window.sea.land) == 1 and 'blocks start' in window.status.text()
    window.mode.setCurrentIndex(0)
    window.apply_click(9, 9)
    assert window.start == (3, 3) and 'clear of land' in window.status.text()
    window.close()


def test_real_map_loads_in_model_units_with_a_vessel_profile(qtbot):
    window = Window(SeaMap.load('maps/harbour.json'))
    qtbot.addWidget(window)
    window.set_map('maps/singapore-ubin.json')
    assert window.profile.name == 'harbour_craft' and window.vessel.currentText() == 'harbour_craft'
    assert window.placement.currentIndex() == 1
    assert window.sea.to_dict()['metadata']['model_scale']['length_m'] == 10
    assert window.sea.bounds == pytest.approx(tuple(v/10 for v in window.real.bounds))
    window.mode.setCurrentIndex(0)
    window.apply_click(-200, 50)                   # 2 km west, 270 m from shore
    assert window.start == (-200, 50)
    window.vessel.setCurrentText('coastal_ship')   # L = 50 m: same place in metres
    assert window.start == pytest.approx((-40, 10))
    window.vessel.setCurrentText('model')          # 1 m grid is far too large: refused
    assert window.profile.name == 'coastal_ship' and 'cells' in window.status.text()
    window.close()


def test_land_edit_on_a_scaled_map_is_stored_in_metres(qtbot):
    window = Window(SeaMap((0, 0, 1000, 1000)))
    qtbot.addWidget(window)
    window.vessel.setCurrentText('harbour_craft')
    window.mode.setCurrentIndex(2)
    window.apply_click(40, 40)
    window.apply_click(60, 50)
    assert window.real.land[0].equals(box(400, 400, 600, 500))
    assert window.sea.land[0].equals(box(40, 40, 60, 50))
    window.close()


def test_worker_leaves_gui_responsive_and_restores_controls(qtbot):
    entered, release = Event(), Event()
    def runner(*args, **kwargs):
        entered.set()
        release.wait(2)
        return execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    window = Window(SeaMap((0, 0, 24, 24)), runner=runner)
    qtbot.addWidget(window)
    window.start_run()
    qtbot.waitUntil(entered.is_set)
    assert not window.run_button.isEnabled()
    assert not window.planner.isEnabled()
    assert window.stop_button.isEnabled()
    window.apply_click(7, 7)
    assert window.start == (2, 2)
    release.set()
    qtbot.waitUntil(lambda: window.future is None)
    assert window.result['status'] == 'success'
    assert window.export_button.isEnabled()
    window.close()


def test_revised_controls_reach_the_runner_unchanged(qtbot):
    calls = []
    def capture(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('controller failed')
    window = Window(SeaMap((0, 0, 24, 24)), runner=capture)
    qtbot.addWidget(window)
    window.planner.setCurrentText('theta')
    window.dynamics.setCurrentText('marine')
    window.filter_mode.setCurrentIndex(1)
    window.perception.setCurrentIndex(1)
    window.start_run()
    qtbot.waitUntil(lambda: window.future is None)
    assert calls == [{'planner': 'theta', 'dynamics': 'marine', 'filtered': True,
                      'observation': {'noise': .1, 'delay': .5, 'dropout': .1},
                      'placement': 'uniform', 'resolution': 1., 'clearance': .7}]
    assert 'controller failed' in window.status.text()
    assert window.run_button.isEnabled() and window.planner.isEnabled()
    window.close()


def test_stop_cancels_the_active_run_not_a_stale_event(qtbot):
    entered = Event()
    def runner(*args, **kwargs):
        cancel = args[8]
        entered.set()
        while not cancel():
            sleep(.01)
        return execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (20, 2), policy_name='direct',
                       count=0, cancel=cancel)
    window = Window(SeaMap((0, 0, 24, 24)), runner=runner)
    qtbot.addWidget(window)
    window.start_run()  # replaces the Event created in __init__
    qtbot.waitUntil(entered.is_set)
    window.stop_button.click()
    try:
        qtbot.waitUntil(lambda: window.future is None, timeout=2000)
        assert window.result['status'] == 'cancelled'
        assert window.run_button.isEnabled()
    finally:
        window.cancel_event.set()
        qtbot.waitUntil(lambda: window.future is None, timeout=2000)
        window.close()


def test_worker_error_is_visible(qtbot):
    def broken(*args, **kwargs):
        raise ValueError('Bad checkpoint')
    window = Window(SeaMap((0, 0, 24, 24)), runner=broken)
    qtbot.addWidget(window)
    window.start_run()
    qtbot.waitUntil(lambda: window.future is None)
    assert 'Bad checkpoint' in window.status.text()
    assert window.run_button.isEnabled()
    assert window.result is None
    window.close()


def test_default_model_points_to_supplied_assets():
    from pathlib import Path
    from shipnav.gui import MODEL
    root = Path(__file__).resolve().parents[2]
    assert MODEL == root / 'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'
    assert (MODEL / 'policy.config').is_file()
    assert (MODEL / 'rl_model.pth').is_file()


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('navigation', ['zoom', 'pan'])
def test_toolbar_navigation_does_not_edit_scene(qtbot, mode, navigation):
    from types import SimpleNamespace
    window = Window(SeaMap((0, 0, 24, 24)))
    qtbot.addWidget(window)
    window.mode.setCurrentIndex(mode)
    before = (window.start, window.goal, window.sea.to_dict(), window.corner)
    getattr(window.navigation, navigation)()
    window.clicked(SimpleNamespace(inaxes=window.ax, xdata=8., ydata=8.))
    window.clicked(SimpleNamespace(inaxes=window.ax, xdata=10., ydata=10.))
    assert (window.start, window.goal, window.sea.to_dict(), window.corner) == before
    getattr(window.navigation, navigation)()
    window.clicked(SimpleNamespace(inaxes=window.ax, xdata=8., ydata=8.))
    if mode == 0:
        assert window.start == (8., 8.)
    elif mode == 1:
        assert window.goal == (8., 8.)
    else:
        assert window.corner == (8., 8.)
    window.close()


@pytest.mark.parametrize('speed', [0, 1, 2, 3])
def test_replay_uses_recorded_trace_and_draws_final_frame(qtbot, monkeypatch, speed):
    import shipnav.gui as gui
    window = Window(SeaMap((0, 0, 24, 24)),
                    runner=lambda *args, **kwargs: pytest.fail('Replay must not run simulation'))
    qtbot.addWidget(window)
    result = execute(window.sea.to_dict(), (2, 2), (4, 2),
                     policy_name='direct', count=0)
    window.result = result
    drawn = []
    monkeypatch.setattr(gui, 'draw', lambda ax, value, frame: drawn.append((value, frame)))
    window.replay_speed.setCurrentIndex(speed)
    window.replay()
    assert window.playback.interval() >= 40
    assert window.replay_step >= 1
    for _ in range(len(result['frames']) + 1):
        window.next_frame()
    assert drawn[0] == (result, 0)
    assert drawn[-1] == (result, len(result['frames']) - 1)
    assert not window.playback.isActive()
    window.close()


def test_export_runs_in_worker_and_restores_controls(qtbot, monkeypatch, tmp_path):
    import json
    import shipnav.gui as gui
    window = Window(SeaMap((0, 0, 24, 24)))
    qtbot.addWidget(window)
    window.result = execute(window.sea.to_dict(), (2, 2), (4, 2),
                            policy_name='direct', count=0)
    filename = tmp_path / 'run.json'
    monkeypatch.setattr(gui.QFileDialog, 'getSaveFileName',
                        lambda *args: (str(filename), 'JSON (*.json)'))
    window.export()
    assert not window.run_button.isEnabled()
    assert not window.stop_button.isEnabled()
    qtbot.waitUntil(lambda: window.future is None)
    assert json.loads(filename.read_text())['status'] == 'success'
    assert window.status.text() == 'Export saved'
    assert window.run_button.isEnabled()
    assert window.export_button.isEnabled()
    window.close()
