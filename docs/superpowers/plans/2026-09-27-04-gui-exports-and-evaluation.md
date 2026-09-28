# 04 — GUI, exports and evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the interactive maritime-navigation demo and reproducible experiment outputs, and record the pretrained/classical baselines that plan 05 retraining is compared against.

**Architecture:** Use the same headless service for CLI, GUI and evaluation. Execute simulations in a worker thread with cooperative cancellation; display and replay results on the Qt main thread, then export the same recorded trace.

**Tech Stack:** PySide6, Matplotlib QtAgg, Shapely, FFmpeg, Python concurrent.futures, pytest-qt, standard-library CSV/JSON

**Spec:** [../specs/2026-09-27-solo-rebuild-design.md](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` (now tracked in Git).
- Use `.venv-legacy` only for baseline verification and migration comparison; remove the legacy environment, build dependencies and runtime support after modern acceptance passes.
- Retain baseline results, hashes and frozen comparison fixtures as verification evidence; do not retain a legacy fallback or maintain a second runtime.
- After that gate, attempt the newest stable Python and dependency releases available at execution time; prereleases require an explicit separate experiment.
- All subsequent phases use `.venv-modern` and its committed lockfile; Linux EC2 training uses a separately verified CUDA environment with the same application version and Python minor.
- Do not silently downgrade the modern stack. Record compatibility blockers, isolate optional native dependencies, and re-run parity checks after any port.
- Run migration parity on CPU first; validate MPS and CUDA separately before using their results.
- Use metres, seconds, radians, and Cartesian East–North `(x, y)` coordinates in the rebuilt simulation.
- Keep one episode clock and continuous vessel state across intermediate goals.
- Freeze scenarios independently of planners; pair initial conditions and exogenous randomness across comparisons.
- Keep perceived observations separate from scoring truth; report collisions, failures, safety interventions and missed deadlines honestly.
- Use AWS CLI for EC2 retraining operations; choose account, region, resource sizes and an explicit spending limit when executing that phase.
- Do not implement literature work, academic report writing, or competition presentation preparation in these plans.

---

All file paths below are relative to the workspace root. Run commands from `rebuild/` unless explicitly stated otherwise. A code step can be worked through function by function in 2–5 minute increments; commit only after the task's tests pass. Commit only explicitly named implementation files.

## Revision 2026-09-28 — aligned with the implemented 02/03/03b code

The first version of this plan was written before plans 02–03b landed. Its snippets no longer matched the code, and several were wrong in ways the tests exposed. This revision rewrites the code blocks against the current `execute()` schema-2 result and polygon `SeaMap`.

**Verification of this revision.** Every code block below was copied into a scratch copy of `rebuild/` and run: **17 tests passed**, including the GUI tests under `QT_QPA_PLATFORM=offscreen`, an FFmpeg MP4 frame-count check, and benchmark runs on the frozen `head_on`, `unreachable`, `crossing` and `detour_harbour` scenarios. That run used Python 3.11 with current Shapely/Matplotlib/PySide6 wheels, **not** the locked 3.14 `.venv-modern`, and did not load the SARL checkpoint (it is not in the cloud checkout). Rerun every step in `.venv-modern`; these results are not a substitute for it.

What changed and why:

| Area | Old plan | Now |
|---|---|---|
| Land drawing (Task 7) | Read `result['map']['land']` as rectangles; KeyError on schema-1 maps | Draws polygon `features`, including holes and MultiPolygons |
| Diagnostics overlay and per-step CSV | Deferred to Task 10 | In Task 7: `execute()` already records `diagnostics` |
| GUI planner/dynamics/filter/perception controls | Deferred to Task 10 | In Task 8: `execute()` already accepts these keywords |
| Stop button (Task 8) | `clicked.connect(self.cancel_event.set)` bound the *first* Event; `start_run` replaces it, so **Stop never cancelled a run** | Connects through a lambda; a test fails on the old wiring |
| Land editing (Task 8) | Test compared `sea.land` with a tuple; the rebuilt map dropped metadata | Compares Shapely geometry; keeps `metadata` |
| Map scale (Task 8) | Assumed the 24 m harbour everywhere | GUI rejects maps over 250,000 m² with a visible message; real maps use `python -m shipnav.map_demo`. **Decision:** the interactive simulator stays on synthetic maps for this phase. `execute()` plans on a 1 m grid (planner limit 250,000 cells) with a 1 m/s, 100 s episode, which cannot cross a 12 × 8 km map. |
| Default endpoints (Task 8) | Fixed `(2,2)`/`(22,22)` | 2 m inside the map corners if clear, else unset and Run asks for them |
| Default filter (Task 8) | Unfiltered | Predictive filter. The supplied SARL never observed land or map edges: unfiltered on `harbour.json` with 5 ships it hit "land" at 4.5 s. From `(2,2)` the island is ≥7 m away, so that was most likely the map edge (which `SeaMap.clear` treats as land). |
| `metrics()` (Task 11) | `sum(d['no_feasible_action'])` crashed on every unfiltered run (`None` = not evaluated) and aborted the benchmark | Reports `None` for unfiltered runs |
| `land_clearance` (Task 11) | Unpacked polygons as rectangles | Signed polygon distance (negative inside land) |
| `geometry_metrics` (Task 11) | Assumed marine runs start at heading 0; a run whose first leg is not due east counted a false heading violation | Starts on the first route leg's bearing, matching `run_episode`; returns `{}` for runs with no frames |
| Benchmark rows | Only `metrics()` | Merges `geometry_metrics()`; adds `orca_filtered` so filter claims have a matched pair |

## File structure and prerequisites

Create `rebuild/src/shipnav/export.py`, `gui.py`, `evaluate.py`, `metrics.py`, `benchmark.py`; create `rebuild/tests/test_export.py`, `test_gui.py`, `test_evaluate.py`, `test_metrics.py`, `test_geometric_metrics.py`; extend `rebuild/README.md`.

Plans 01b, 02, 03 and 03b are complete (see `git log`). Run all commands in `.venv-modern` with `uv sync --locked --all-extras` (the `gui` extra supplies PySide6; the `dev` group supplies pytest-qt). FFmpeg is a system binary (`brew install ffmpeg`). Run Qt tests with `QT_QPA_PLATFORM=offscreen`, then perform the native-window acceptance checklist. Headless Linux needs the system EGL/GL/xkbcommon/fontconfig libraries for PySide6 to import.

The GUI replays a completed simulation. It shows a running status during inference, remains responsive, and allows cancellation. Live streaming during inference is a possible extension, not needed to meet the briefing's visualization requirement.

### Task 7: Export and draw recorded navigation results

**Files:** `rebuild/src/shipnav/export.py`, `rebuild/tests/test_export.py`.

**Interfaces:** Consumes the schema-2 `execute()` result. Produces `draw(ax, result: dict, frame_index: int = -1) -> None`, `land_polygons(map_data)`, `csv_rows(result)` and `export_run(result: dict, path: Path) -> None`.

- JSON is the complete reproducibility artifact, including nested observations and predictions.
- CSV has one row per recorded frame. Action columns (nominal/executed/actual velocity, intervention, no-feasible-action, decision latency, deadline miss, oldest observed target age, collision type, clearance) describe the step leaving that frame. The terminal frame has no action, so those columns are empty; nothing is invented.
- PNG is a final overview. MP4 replays every recorded frame at `1/dt` FPS.
- The overlay draws diagnostic `k` only on frame `k`: nominal and executed velocity arrows, predicted target tracks, and the filter's chosen rollout. `no_feasible_action` shows `n/a` when the filter is off.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_export.py` with:

```python
import copy
import csv
import json
import shutil
import subprocess
import pytest
from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.export import export_run


def test_exports_use_the_recorded_run(tmp_path):
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    export_run(result, tmp_path/'run.json')
    export_run(result, tmp_path/'run.csv')
    export_run(result, tmp_path/'run.png')
    loaded = json.loads((tmp_path/'run.json').read_text())
    assert loaded['frames'] == result['frames']
    with (tmp_path/'run.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result['frames'])
    assert float(rows[-1]['time_s']) == result['elapsed']
    assert (tmp_path/'run.png').read_bytes().startswith(b'\x89PNG')
    with pytest.raises(ValueError):
        export_run(result, tmp_path/'run.exe')


def test_polygon_land_and_diagnostics_come_from_the_stored_trace(tmp_path):
    sea = SeaMap.load('maps/harbour.json')
    result = execute(sea.to_dict(), (2, 2), (22, 22), policy_name='direct', count=3, seed=1, filtered=True)
    stored = json.loads(json.dumps(result))  # no policy or map object survives this
    export_run(stored, tmp_path/'run.png')
    export_run(stored, tmp_path/'run.csv')
    with (tmp_path/'run.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(stored['frames']) == len(stored['diagnostics'])+1
    first, d = rows[0], stored['diagnostics'][0]
    assert float(first['executed_vx']) == d['executed'][0]
    assert first['override'] == str(d['override'])
    assert rows[-1]['executed_vx'] == ''  # terminal frame: no invented action
    assert stored == json.loads(json.dumps(result))  # export did not mutate the trace


@pytest.mark.video
@pytest.mark.skipif(shutil.which('ffprobe') is None, reason='FFmpeg not installed')
def test_mp4_keeps_every_frame_at_one_over_dt(tmp_path):
    result = execute(SeaMap((0, 0, 10, 10)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    export_run(result, tmp_path/'run.mp4')
    probe = json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
         'stream=nb_read_frames,r_frame_rate', '-of', 'json', str(tmp_path/'run.mp4')],
        check=True, capture_output=True, text=True).stdout)['streams'][0]
    assert probe['r_frame_rate'] == '4/1'
    assert int(probe['nb_read_frames']) == len(result['frames'])
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_export.py -v
```

Expected: failure because the module does not exist. Resolve missing environment dependencies (including FFmpeg for the `video` test) before treating a failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following file.**

Create `rebuild/src/shipnav/export.py` with:

```python
"""Draw and export a recorded `execute()` result. Never re-runs a policy."""
from pathlib import Path
import csv
import json
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.animation import FFMpegWriter, writers
from shapely.geometry import shape

CSV_FIELDS = ['time_s', 'x_m', 'y_m', 'vx_mps', 'vy_mps', 'goal_index',
              'nominal_vx', 'nominal_vy', 'executed_vx', 'executed_vy', 'actual_vx', 'actual_vy',
              'override', 'no_feasible_action', 'decision_ms', 'deadline_miss',
              'max_target_age_s', 'ship_collision', 'land_collision', 'clearance_m']


def land_polygons(map_data: dict):
    """Yield every land Polygon of a schema-1 map, splitting MultiPolygons."""
    for feature in map_data['features']:
        geometry = shape(feature['geometry'])
        yield from getattr(geometry, 'geoms', [geometry])


def _patch(polygon, **style):
    # One compound path per polygon so interior rings (lakes/holes) stay open water.
    vertices, codes = [], []
    for ring in (polygon.exterior, *polygon.interiors):
        points = list(ring.coords)
        vertices += points
        codes += [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), **style)


def draw(ax, result: dict, frame_index: int = -1) -> None:
    ax.clear()
    x0, y0, x1, y1 = result['map']['bounds']
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel='x (m)', ylabel='y (m)', aspect='equal')
    ax.set_facecolor('#dceef5')
    for polygon in land_polygons(result['map']):
        ax.add_patch(_patch(polygon, facecolor='#9b987f', edgecolor='none'))
    route = result['route']
    if route:
        ax.plot(*zip(*route), '--', color='#555555', label='Global route')
    goals = result['goals']
    if goals:
        ax.scatter(*zip(*goals), marker='*', color='#cf4b35', label='Goals', zorder=3)
    frames = result['frames']
    if frames:
        i = len(frames)-1 if frame_index < 0 else min(frame_index, len(frames)-1)
        frame = frames[i]
        ax.plot(*zip(*(f['position'] for f in frames[:i+1])), color='#17659c', label='Travelled')
        ax.add_patch(Circle(frame['position'], result['settings']['radius'], color='#17659c'))
        for position, ship in zip(frame['traffic'], result['traffic_definitions']):
            ax.add_patch(Circle(position, ship['radius'], color='#df9a36'))
        title = f"{result['status']} — {frame['t']:.2f} s"
        diagnostics = result.get('diagnostics', [])
        # Diagnostic k is the decision from frame k to k+1: the terminal frame has none,
        # and an earlier frame never shows a later tick's prediction.
        if i < len(diagnostics):
            d = diagnostics[i]
            for key, color in (('nominal', '#b05090'), ('executed', '#248060')):
                ax.arrow(*frame['position'], *d[key], color=color, width=.025, label=key.capitalize())
            for target in d['predictions']:
                ax.plot(*zip(*target['points']), ':', color='#e5a02e')
            if d['path']:
                ax.plot(*zip(*d['path']), '-', color='#248060', alpha=.5)
            infeasible = 'n/a' if d['no_feasible_action'] is None else d['no_feasible_action']
            title += f" | override={d['override']} | no feasible action={infeasible}"
        ax.set_title(title)
    ax.legend(loc='upper left')


def csv_rows(result: dict):
    """One row per recorded frame. Action columns describe the step leaving that
    frame; the terminal frame has no action, so those columns stay empty."""
    diagnostics = result.get('diagnostics', [])
    for i, f in enumerate(result['frames']):
        row = {'time_s': f['t'], 'x_m': f['position'][0], 'y_m': f['position'][1],
               'vx_mps': f['velocity'][0], 'vy_mps': f['velocity'][1], 'goal_index': f['goal_index']}
        if i < len(diagnostics):
            d = diagnostics[i]
            ages = [s['age'] for s in d['observed']]
            row.update(nominal_vx=d['nominal'][0], nominal_vy=d['nominal'][1],
                       executed_vx=d['executed'][0], executed_vy=d['executed'][1],
                       actual_vx=d['actual_velocity'][0], actual_vy=d['actual_velocity'][1],
                       override=d['override'], no_feasible_action=d['no_feasible_action'],
                       decision_ms=d['decision_ms'], deadline_miss=d['deadline_miss'],
                       max_target_age_s=max(ages) if ages else None,
                       ship_collision=d['ship_collision'], land_collision=d['land_collision'],
                       clearance_m=d['clearance'])
        yield row


def export_run(result: dict, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in ('.json', '.csv', '.png', '.mp4'):
        raise ValueError('Choose JSON, CSV, PNG or MP4')
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.json':
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        return
    if suffix == '.csv':
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(result))
        return
    figure = Figure(figsize=(7, 7))
    ax = figure.subplots()
    if suffix == '.png':
        draw(ax, result)
        figure.savefig(path, dpi=150)
        return
    if not writers.is_available('ffmpeg'):
        raise RuntimeError('FFmpeg is required to export MP4')
    frames = result['frames']
    if not frames:
        raise ValueError('No frames to export')
    writer = FFMpegWriter(fps=1/result['settings']['dt'], codec='libx264')
    with writer.saving(figure, str(path), dpi=100):
        for i in range(len(frames)):
            draw(ax, result, i)
            writer.grab_frame()
```

- [ ] **Step 4: Run the same test command and inspect the result.** Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/export.py rebuild/tests/test_export.py
git commit -m "feat: export and draw recorded navigation results"
```

### Task 8: Build the navigation GUI

**Files:** `rebuild/src/shipnav/gui.py`, `rebuild/tests/test_gui.py`.

**Interfaces:** Consumes `SeaMap`, `execute`, `draw`, and `export_run`. Produces `Window(sea: SeaMap, model_dir: str = "", runner=execute)`, `.apply_click(x, y)`, `.set_map(path)`, `.start_run()`, `.check_worker()` and module entry `python -m shipnav.gui`.

- Worker functions never touch Qt widgets. `start_run` snapshots every control value on the GUI thread and passes them positionally plus `planner=`, `dynamics=`, `filtered=`, `observation=` keywords. Runners must accept `**kwargs`. A Qt timer polls the future and moves results to the main thread.
- Two control rows: map/model/click mode/policy/ships/seed/run/stop/replay/export, then planner/dynamics/filter/perception. All edit controls are disabled while a job runs.
- Maps larger than 250,000 m² are rejected with a message pointing to `shipnav.map_demo`.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_gui.py` with:

```python
from threading import Event
from time import sleep
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


def test_real_scale_map_is_rejected_visibly(qtbot):
    window = Window(SeaMap.load('maps/harbour.json'))
    qtbot.addWidget(window)
    window.set_map('maps/singapore-ubin.json')
    assert 'map_demo' in window.status.text()
    assert window.sea.bounds == (0, 0, 24, 24)
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
                      'observation': {'noise': .1, 'delay': .5, 'dropout': .1}}]
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
    qtbot.waitUntil(lambda: window.future is None, timeout=2000)
    assert window.result['status'] == 'cancelled'
    assert window.run_button.isEnabled()
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
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui.py -v
```

Expected: failure because the module does not exist.

- [ ] **Step 3: Implement the following file.**

Create `rebuild/src/shipnav/gui.py` with:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.export import draw, export_run

# The service plans on a 1 m grid (planning.plan max_cells=250_000) with a 1 m/s,
# 100 s episode. Larger maps are route-preview only: use `python -m shipnav.map_demo`.
MAX_INTERACTIVE_AREA_M2 = 250_000
CLEARANCE = .7
DEGRADED = {'noise': .1, 'delay': .5, 'dropout': .1}
MODEL = Path(__file__).resolve().parents[3]/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def check_interactive(sea: SeaMap) -> SeaMap:
    x0, y0, x1, y1 = sea.bounds
    if (x1-x0)*(y1-y0) > MAX_INTERACTIVE_AREA_M2:
        raise ValueError(f'Map is {x1-x0:.0f} x {y1-y0:.0f} m; the interactive simulator supports '
                         f'up to {MAX_INTERACTIVE_AREA_M2} m². Use python -m shipnav.map_demo for real-map routes')
    return sea


def default_endpoints(sea: SeaMap):
    x0, y0, x1, y1 = sea.bounds
    start, goal = (x0+2, y0+2), (x1-2, y1-2)
    return (start if sea.clear(start, start, CLEARANCE) else None,
            goal if sea.clear(goal, goal, CLEARANCE) else None)


class Window(QWidget):
    def __init__(self, sea: SeaMap, model_dir: str = '', runner=execute):
        super().__init__()
        self.setWindowTitle('Ship navigation rebuild')
        self.resize(1100, 850)
        self.sea, self.model_dir, self.runner = check_interactive(sea), model_dir, runner
        self.start, self.goal = default_endpoints(sea)
        self.result, self.future, self.corner = None, None, None
        self.job_kind, self.replay_index = '', 0
        self.cancel_event = Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        layout, toolbar, options = QVBoxLayout(self), QHBoxLayout(), QHBoxLayout()
        layout.addLayout(toolbar)
        layout.addLayout(options)
        self.map_button = QPushButton('Load map')
        self.model_button = QPushButton('Load model')
        self.mode = QComboBox()
        self.mode.addItems(['Set start', 'Set goal', 'Add land (two corners)'])
        self.mode.currentIndexChanged.connect(self.clear_corner)
        self.policy = QComboBox()
        self.policy.addItems(['sarl', 'orca', 'mpc', 'direct'])
        self.count = QSpinBox()
        self.count.setRange(0, 15)
        self.count.setValue(5)
        self.count.setPrefix('Ships: ')
        self.seed = QSpinBox()
        self.seed.setRange(0, 9999)
        self.seed.setPrefix('Seed: ')
        self.run_button, self.stop_button = QPushButton('Run'), QPushButton('Stop')
        self.replay_button, self.export_button = QPushButton('Replay'), QPushButton('Export')
        self.planner = QComboBox()
        self.planner.addItems(['astar_smooth', 'theta'])
        self.dynamics = QComboBox()
        self.dynamics.addItems(['holonomic', 'marine'])
        self.filter_mode = QComboBox()
        self.filter_mode.addItems(['Unfiltered', 'Predictive filter'])
        # The supplied SARL never observed land or map edges; the demo defaults to the
        # filter. Choose Unfiltered explicitly for the ablation.
        self.filter_mode.setCurrentIndex(1)
        self.perception = QComboBox()
        self.perception.addItems(['Exact observations', 'Noisy delayed observations'])
        self.edit_controls = [self.map_button, self.model_button, self.mode,
                              self.policy, self.count, self.seed, self.run_button,
                              self.planner, self.dynamics, self.filter_mode, self.perception]
        for widget in self.edit_controls[:7] + [self.stop_button, self.replay_button, self.export_button]:
            toolbar.addWidget(widget)
        for widget in self.edit_controls[7:]:
            options.addWidget(widget)
        self.status = QLabel('Select start and goal, then run')
        layout.addWidget(self.status)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.subplots()
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.clicked)
        self.map_button.clicked.connect(self.load_map)
        self.model_button.clicked.connect(self.load_model)
        self.run_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(lambda: self.cancel_event.set())
        self.replay_button.clicked.connect(self.replay)
        self.export_button.clicked.connect(self.export)
        self.poll = QTimer(self)
        self.poll.setInterval(50)
        self.poll.timeout.connect(self.check_worker)
        self.poll.start()
        self.playback = QTimer(self)
        self.playback.timeout.connect(self.next_frame)
        self.set_busy(False)
        self.preview()

    def clear_corner(self):
        self.corner = None

    def set_busy(self, busy):
        for widget in self.edit_controls:
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy and self.job_kind == 'run')
        self.replay_button.setEnabled(not busy and self.result is not None)
        self.export_button.setEnabled(not busy and self.result is not None)

    def invalidate(self):
        self.playback.stop()
        self.result = None
        self.set_busy(False)
        self.preview()

    def preview(self):
        preview = {'map': self.sea.to_dict(), 'route': [],
                   'goals': [p for p in (self.start, self.goal) if p is not None],
                   'frames': [], 'settings': {'radius': .5}, 'traffic_definitions': []}
        draw(self.ax, preview)
        self.ax.set_title('Start and destination')
        self.canvas.draw_idle()

    def clicked(self, event):
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            self.apply_click(event.xdata, event.ydata)

    def apply_click(self, x, y):
        if self.future is not None:
            return
        point = (float(x), float(y))
        try:
            if self.mode.currentIndex() < 2:
                if not self.sea.clear(point, point, CLEARANCE):
                    raise ValueError('Choose a point clear of land and map edges')
                if self.mode.currentIndex() == 0:
                    self.start = point
                else:
                    self.goal = point
            elif self.corner is None:
                self.corner = point
                self.status.setText('Choose the opposite land corner')
                return
            else:
                a, b = self.corner, point
                self.corner = None
                rect = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
                sea = SeaMap(self.sea.bounds, self.sea.land + (rect,), self.sea.to_dict()['metadata'])
                if not all(p is None or sea.clear(p, p, CLEARANCE) for p in (self.start, self.goal)):
                    raise ValueError('New land blocks start or goal')
                self.sea = sea
            self.invalidate()
            self.status.setText('Scene updated')
        except ValueError as error:
            self.status.setText(str(error))

    def load_map(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Load map', '', 'Map (*.json)')
        if filename:
            self.set_map(Path(filename))

    def set_map(self, path: Path):
        try:
            self.sea = check_interactive(SeaMap.load(path))
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.status.setText(f'Map not loaded: {error}')
            return
        self.corner = None
        self.start, self.goal = default_endpoints(self.sea)
        self.invalidate()
        self.status.setText('Map loaded; select valid start and goal')

    def load_model(self):
        directory = QFileDialog.getExistingDirectory(self, 'Folder with policy.config and rl_model.pth')
        if directory:
            self.model_dir = directory
            self.invalidate()
            self.status.setText('Model folder selected; weights are validated when Run starts')

    def start_run(self):
        if self.future is not None:
            return
        if self.start is None or self.goal is None:
            self.status.setText('Select a start and a goal clear of land first')
            return
        self.playback.stop()
        self.result = None
        self.cancel_event = Event()
        self.job_kind = 'run'
        self.status.setText('Running navigation…')
        self.set_busy(True)
        # Snapshot every widget value here; the worker never reads Qt objects.
        self.future = self.pool.submit(self.runner, self.sea.to_dict(), self.start, self.goal,
            self.model_dir, self.policy.currentText(), True, self.seed.value(), self.count.value(),
            self.cancel_event.is_set,
            planner=self.planner.currentText(),
            dynamics=self.dynamics.currentText(),
            filtered=self.filter_mode.currentIndex() == 1,
            observation={} if self.perception.currentIndex() == 0 else dict(DEGRADED))

    def check_worker(self):
        if self.future is None or not self.future.done():
            return
        try:
            value = self.future.result()
            if self.job_kind == 'run':
                self.result = value
                draw(self.ax, value)
                self.canvas.draw_idle()
                self.status.setText(f"{value['status']}: {value['elapsed']:.2f} simulation seconds")
            else:
                self.status.setText('Export saved')
        except Exception as error:
            self.status.setText(f'Failed: {error}')
        finally:
            self.future = None
            self.set_busy(False)

    def replay(self):
        if self.result is not None:
            self.replay_index = 0
            self.playback.start(round(1000*self.result['settings']['dt']))
            self.next_frame()

    def next_frame(self):
        if self.result is None or self.replay_index >= len(self.result['frames']):
            self.playback.stop()
            return
        draw(self.ax, self.result, self.replay_index)
        self.canvas.draw_idle()
        self.replay_index += 1

    def export(self):
        if self.result is None or self.future is not None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Export', 'run.json',
            'JSON (*.json);;CSV (*.csv);;PNG (*.png);;MP4 (*.mp4)')
        if filename:
            self.playback.stop()
            self.job_kind = 'export'
            self.set_busy(True)
            self.status.setText('Exporting…')
            self.future = self.pool.submit(export_run, self.result, Path(filename))

    def closeEvent(self, event):
        if self.future is not None:
            self.cancel_event.set()
            self.status.setText('Waiting for the active job to stop; close again when it finishes')
            event.ignore()
            return
        self.poll.stop()
        self.playback.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        event.accept()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window(SeaMap.load(Path('maps/harbour.json')), str(MODEL))
    window.show()
    sys.exit(app.exec())
```

- [ ] **Step 4: Run the same test command and inspect the result.** Expected: PASS. `test_stop_cancels_the_active_run_not_a_stale_event` must fail if Stop is connected directly to `self.cancel_event.set`; check that once.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/gui.py rebuild/tests/test_gui.py
git commit -m "feat: build the navigation gui"
```

### Task 9: Run paired smoke experiments

**Files:** `rebuild/src/shipnav/evaluate.py`, `rebuild/tests/test_evaluate.py`.

**Interfaces:** Consumes `execute` and its trace schema. Produces `summarize(runs: list[dict]) -> dict` and `evaluate(map_data, model_dir, output: Path, seeds=range(10), counts=(5,8,12,15), runner=execute) -> list[dict]`. No-global-goals ablates waypoint guidance; direct ablates local collision avoidance; ORCA supplies a rule-based comparator.

This is a smoke utility for `maps/harbour.json` only: endpoints are fixed at `(2,2)` → `(22,22)` and all variants are unfiltered. Comparative results come from Task 11's frozen-scenario benchmark. The code below is unchanged from the first revision and passes against the current service.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_evaluate.py` with:

```python
from shipnav.evaluate import summarize, evaluate
from shipnav.maps import SeaMap


def test_failures_do_not_improve_average_completion_time():
    rows = [{'status': 'success', 'elapsed': 10, 'distance': 9, 'inference_ms': [2]},
            {'status': 'collision', 'elapsed': 1, 'distance': 1, 'inference_ms': [4]},
            {'status': 'error', 'error': 'placement failed'}]
    summary = summarize(rows)
    assert summary['runs'] == 3 and summary['successes'] == 1
    assert summary['mean_success_time_s'] == 10
    assert summary['collisions'] == 1 and summary['errors'] == 1
    assert summary['mean_step_inference_ms'] == 3


def test_evaluation_pairs_seed_and_scenario_and_saves_every_run(tmp_path):
    calls = []
    def runner(map_data, start, goal, model, policy, global_goals, seed, count):
        calls.append((policy, global_goals, seed, count))
        return {'status': 'success', 'elapsed': 10, 'distance': 9,
                'inference_ms': [1], 'traffic_definitions': [{'seed': seed, 'count': count}]}
    rows = evaluate(SeaMap((0, 0, 24, 24)).to_dict(), '', tmp_path,
                    seeds=range(2), counts=(5,), runner=runner)
    assert len(calls) == 8 and len(rows) == 4
    assert len(list(tmp_path.glob('*.json'))) == 8
    assert (tmp_path/'summary.csv').exists()
    assert all(r['runs'] == 2 for r in rows)
```

- [ ] **Step 2: Run `python -m pytest tests/test_evaluate.py -v` before implementation.** Expect a missing module.

- [ ] **Step 3: Implement the following file.**

Create `rebuild/src/shipnav/evaluate.py` with:

```python
from pathlib import Path
import csv
import json
from statistics import mean
from shipnav.maps import SeaMap
from shipnav.service import execute

VARIANTS = [('global_sarl', 'sarl', True), ('final_goal_sarl', 'sarl', False),
            ('global_direct', 'direct', True), ('global_orca', 'orca', True)]


def summarize(runs: list[dict]) -> dict:
    successful = [r for r in runs if r['status'] == 'success']
    all_latency = [t for r in runs for t in r.get('inference_ms', [])]
    return {'runs': len(runs),
            'successes': len(successful),
            'collisions': sum(r['status'] == 'collision' for r in runs),
            'timeouts': sum(r['status'] == 'timeout' for r in runs),
            'errors': sum(r['status'] == 'error' for r in runs),
            'mean_success_time_s': mean(r['elapsed'] for r in successful) if successful else None,
            'mean_success_distance_m': mean(r['distance'] for r in successful) if successful else None,
            'mean_step_inference_ms': mean(all_latency) if all_latency else None}


def evaluate(map_data: dict, model_dir: str, output: Path, seeds=range(10),
             counts=(5, 8, 12, 15), runner=execute) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for count in counts:
        grouped = {name: [] for name, _, _ in VARIANTS}
        for seed in seeds:
            scenario = None
            for name, policy, global_goals in VARIANTS:
                try:
                    result = runner(map_data, (2, 2), (22, 22), model_dir,
                                    policy, global_goals, seed, count)
                    if scenario is None:
                        scenario = result['traffic_definitions']
                    if result['traffic_definitions'] != scenario:
                        raise AssertionError('Paired scenarios differ')
                except (ValueError, FileNotFoundError, RuntimeError) as error:
                    result = {'status': 'error', 'error': str(error), 'seed': seed, 'count': count}
                (output/f'{name}-n{count}-seed{seed}.json').write_text(
                    json.dumps(result, indent=2, allow_nan=False))
                grouped[name].append(result)
        for name, runs in grouped.items():
            summaries.append({'variant': name, 'traffic_count': count, **summarize(runs)})
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=Path, default=Path('maps/harbour.json'))
    parser.add_argument('--model', default='../CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    parser.add_argument('--output', type=Path, default=Path('results/evaluation'))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--counts', type=int, nargs='+', default=[5, 8, 12, 15])
    args = parser.parse_args()
    if args.seeds <= 0 or any(n < 0 for n in args.counts):
        parser.error('Use positive seed count and non-negative traffic counts')
    print(evaluate(SeaMap.load(args.map).to_dict(), args.model, args.output,
                   range(args.seeds), tuple(args.counts)))
```

- [ ] **Step 4: Run the same test command.** Expected: PASS.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/evaluate.py rebuild/tests/test_evaluate.py
git commit -m "feat: run paired navigation smoke experiments"
```

### Task 10: Native-window acceptance

The revised controls and diagnostics from the first revision's Task 10 are now in Tasks 7 and 8. What remains is manual and needs the real checkpoint and a display.

- [ ] Run the complete suite once: `QT_QPA_PLATFORM=offscreen python -m pytest -v`.
- [ ] Run `python -m shipnav.gui` in a native window. It opens `maps/harbour.json` with the supplied model folder, start `(2,2)` and destination `(22,22)`. Run SARL with 5 ships and the default predictive filter; record the status.
- [ ] Switch to *Unfiltered* and rerun the same seed. The expected outcome is a collision (observed: land collision at 4.5 s). Replay it and check the last diagnostic: confirm whether `land_collision` was the map edge or the island. Record the result; it is the motivating case for plan 05's constraint penalty, not a GUI defect.
- [ ] While running, move/resize the window. Confirm scene edits, the second control row and duplicate runs are disabled. Press Stop; wait for the current inference step and confirm `cancelled`, then rerun.
- [ ] Change the destination and add a land rectangle using two clicks. Run again and verify the global route changes. Try covering the start with land and selecting a point on land; both must be rejected with an understandable status message.
- [ ] Load `maps/singapore-ubin.json`. It must be rejected with a message pointing to `shipnav.map_demo`, and the harbour must stay loaded.
- [ ] Select an empty model folder and run SARL. Verify an error appears and controls are restored. Then select the correct model folder and verify recovery.
- [ ] Select Theta*, marine dynamics and noisy delayed observations; run and replay. Check the overlay arrows, predicted tracks and the title's override / no-feasible-action fields frame by frame.
- [ ] Replay a completed trace; export JSON, CSV, PNG and MP4 to different filenames. Verify exported coordinates/times match the display and that a 10-second simulated interval takes about 10 seconds in the MP4.
- [ ] Check the video with `ffprobe -v error -count_frames -show_entries stream=codec_name,nb_read_frames,r_frame_rate -of json results/demo.mp4`. Expected: 4 FPS at `dt=.25`, and `nb_read_frames` equal to the number of recorded frames, including the initial and terminal frame.
- [ ] Try an impossible map with a wall from one boundary to the opposite boundary; the GUI must show the planning failure (`No route at this grid resolution`), not an empty successful route.
- [ ] Run `python -m shipnav.evaluate --seeds 2 --counts 5`. Inspect all eight traces and `summary.csv`. Then run `python -m shipnav.evaluate` (160 episodes). This is a smoke batch only.
- [ ] If placement fails for dense traffic, retain the error rows. Reduce the count or enlarge the scenario in a documented subsequent experiment; do not drop difficult seeds from only one algorithm's results. Latencies include cold first inference and are descriptive, not real-time guarantees.
- [ ] Do not claim the application works solely because the offscreen GUI tests pass.

### Task 11: Metrics and paired evidence

**Files:** `rebuild/src/shipnav/metrics.py`, `rebuild/src/shipnav/benchmark.py`, `rebuild/tests/test_metrics.py`, `rebuild/tests/test_geometric_metrics.py`.

**Interfaces:** `metrics(run) -> dict`; `geometry_metrics(run) -> dict`; `quantile(values, q)`; `paired_interval(a: dict[id, float], b: dict[id, float]) -> dict`; `hierarchical_interval(a, b)` over training seed → scenario ID → value; `benchmark(scenarios, model_dir, output, variants=VARIANTS, runner=execute) -> list[dict]`.

- Failed runs keep their scenario hash and stay in the denominator. Planning failures are `planning_failure`; any other exception is `error`.
- `no_feasible_actions` is `None` when the filter was off (not evaluated), never 0.
- Cross-track error is measured against the run's own route. For planner comparisons, save one fixed reference route per scenario in the protocol and score against that; a planner's own route is only appropriate for its controller's tracking error.
- `domain_time_s` is dt-weighted exposure below 1 m ship clearance (an experimental domain definition). `sampled_land_clearance_m` is sampled at frames; collision status itself uses swept checks.
- Heading/speed violations use `abs(wrapped Δheading) > .35·dt` or `abs(Δspeed) > .2·dt` between successive marine diagnostics, starting from the first route leg's bearing at rest.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_metrics.py` with:

```python
import json
from pathlib import Path
import pytest
from shipnav.metrics import paired_interval,quantile
from shipnav.benchmark import benchmark

def test_tail_and_paired_direction():
    assert quantile([1,2,3,100],.99)>90
    r=paired_interval({'a':1,'b':1},{'a':0,'b':0})
    assert r['difference']==r['low']==r['high']==1
    with pytest.raises(ValueError): paired_interval({'a':1},{'b':1})

def test_failed_runs_remain_in_denominator(tmp_path):
    scene={'map':{},'start':[0,0],'goal':[1,1],'seed':0}
    def broken(*args,**kwargs): raise RuntimeError('controller failed')
    rows=benchmark([scene],'',tmp_path,{'a':{},'b':{}},runner=broken)
    assert len(rows)==2 and all(r['status']=='error' for r in rows)
    assert len(list(tmp_path.glob('*-scenario.json')))==1

def test_frozen_scenarios_score_filtered_unfiltered_and_unreachable(tmp_path):
    scenes=[json.loads(Path('scenarios',n).read_text()) for n in ('head_on.json','unreachable.json')]
    rows=benchmark(scenes,'',tmp_path,{'direct':dict(policy_name='direct'),
                                       'direct_filtered':dict(policy_name='direct',filtered=True)})
    by={(r['scenario_hash'][:8],r['variant']):r for r in rows}
    assert len(rows)==4
    unfiltered=[r for r in rows if r['variant']=='direct' and r['status']!='planning_failure']
    assert unfiltered and all(r['no_feasible_actions'] is None for r in unfiltered)
    assert sum(r['status']=='planning_failure' for r in rows)==2
    assert all('cross_track_max_m' in r for r in rows if r['status']!='planning_failure')
    json.loads((tmp_path/'metrics.json').read_text())
```

Create `rebuild/tests/test_geometric_metrics.py` with:

```python
import pytest
from shipnav.maps import SeaMap
from shipnav.metrics import point_segment_distance,land_clearance,hierarchical_interval,geometry_metrics
from shipnav.service import execute

def test_geometric_metrics_have_known_values():
    assert point_segment_distance((3,4),(0,0),(10,0))==4
    assert land_clearance(SeaMap((0,0,10,10)),(2,5),.5)==1.5
    assert land_clearance(SeaMap((0,0,10,10),((4,4,6,6),)),(5,5),.5)==-1.5

def test_hierarchical_interval_uses_both_seed_levels():
    a={0:{'x':1,'y':1},1:{'x':1,'y':1}}; b={0:{'x':0,'y':0},1:{'x':0,'y':0}}
    assert hierarchical_interval(a,b)['low']==1
    with pytest.raises(ValueError): hierarchical_interval(a,{0:b[0]})

def test_marine_run_starting_off_east_has_no_false_dynamics_violation():
    run=execute(SeaMap((0,0,24,24)).to_dict(),(2,2),(2,20),policy_name='direct',count=0,dynamics='marine')
    m=geometry_metrics(run)
    assert run['status']=='success' and m['dynamics_violations']==0
    assert m['cross_track_max_m']<1e-6
    assert geometry_metrics({'status':'error','error':'x'})=={}
```

- [ ] **Step 2: Run `python -m pytest tests/test_metrics.py tests/test_geometric_metrics.py -v`.** Expect a missing module, not a missing third-party dependency.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/metrics.py` with:

```python
from math import dist
from statistics import mean
from random import Random

def quantile(values,q):
    if not values: return None
    x=sorted(values); p=(len(x)-1)*q; a=int(p); b=min(a+1,len(x)-1)
    return x[a]+(p-a)*(x[b]-x[a])

def metrics(run):
    ds=run.get('diagnostics',[]); ms=[d['decision_ms'] for d in ds]
    dt=run.get('settings',{}).get('dt',.25)
    route=run.get('route',[])
    length=sum(dist(a,b) for a,b in zip(route,route[1:]))
    cs=[d['clearance'] for d in ds if d.get('clearance') is not None]
    return {'status':run['status'], 'ship_collision':any(d['ship_collision'] for d in ds),
            'land_collision':any(d['land_collision'] for d in ds),
            'min_ship_clearance':min(cs) if cs else None,
            'domain_time_s':sum(c<1. for c in cs)*dt,
            'detour_ratio':run.get('distance',0)/length if length else None,
            'overrides':sum(d['override'] for d in ds),
            'override_rate':mean(d['override'] for d in ds) if ds else None,
            # None = filter off, so feasibility was never evaluated (not "zero infeasible").
            'no_feasible_actions':None if any(d['no_feasible_action'] is None for d in ds) else sum(d['no_feasible_action'] for d in ds),
            'solver_failures':sum(d.get('solver_failed',False) for d in ds),
            'decision_mean_ms':mean(ms) if ms else None,
            'decision_p95_ms':quantile(ms,.95),'decision_p99_ms':quantile(ms,.99),
            'deadline_misses':sum(v>1000*dt for v in ms)}

def paired_interval(a,b,seed=0,samples=2000):
    if not a or set(a)!=set(b):
        raise ValueError('Require complete matching scenario IDs')
    diffs=[a[k]-b[k] for k in sorted(a)]; rng=Random(seed)
    draws=[mean(rng.choices(diffs,k=len(diffs))) for _ in range(samples)]
    return {'difference':mean(diffs),'low':quantile(draws,.025),'high':quantile(draws,.975),'pairs':len(diffs)}
def point_segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    f=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0.
    return dist(p,(a[0]+f*dx,a[1]+f*dy))

def land_clearance(sea,p,radius):
    # Signed: negative inside land. SeaMap.minimum_clearance saturates at 0, so it cannot be used here.
    from shapely.geometry import Point
    x,y=p; x0,y0,x1,y1=sea.bounds
    values=[x-x0,y-y0,x1-x,y1-y]; point=Point(p)
    for polygon in sea.land:
        values.append(-polygon.boundary.distance(point) if polygon.contains(point) else polygon.distance(point))
    return min(values)-radius

def geometry_metrics(run):
    from shipnav.maps import SeaMap
    from math import pi, atan2
    frames=run.get('frames',[]); route=run.get('route',[])
    if not frames:
        return {}
    sea=SeaMap.from_dict(run['map'])
    errors=[min((point_segment_distance(f['position'],a,b) for a,b in zip(route,route[1:])),default=0.) for f in frames]
    land=[land_clearance(sea,f['position'],run['settings']['radius']) for f in frames]
    ds=run.get('diagnostics',[]); violations=0; exposure=0.
    # Marine runs start at rest on the first route leg's bearing (east if already arrived),
    # matching simulation.run_episode.
    heading=atan2(route[1][1]-route[0][1],route[1][0]-route[0][0]) if len(route)>1 else 0.; speed=0.
    for a,b,d in zip(frames,frames[1:],ds):
        dt=b['t']-a['t']
        if d.get('clearance') is not None and d['clearance']<1.: exposure+=dt
        if 'heading' in d:
            turn=(d['heading']-heading+pi)%(2*pi)-pi
            violations+=int(abs(turn)> .35*dt+1e-6 or abs(d['speed']-speed)>.2*dt+1e-6)
            heading,speed=d['heading'],d['speed']
    return {'cross_track_mean_m':mean(errors),
            'cross_track_max_m':max(errors),
            'sampled_land_clearance_m':min(land),
            'domain_time_s':exposure,'dynamics_violations':violations}

def hierarchical_interval(a,b,seed=0,samples=2000):
    # a and b: training seed -> scenario ID -> scalar. Pair both levels.
    if not a or set(a)!=set(b): raise ValueError('Training seeds differ')
    for key in a:
        if not a[key] or set(a[key])!=set(b[key]): raise ValueError('Scenario IDs differ')
    rng=Random(seed); keys=sorted(a); draws=[]
    for _ in range(samples):
        means=[]
        for key in rng.choices(keys,k=len(keys)):
            ids=sorted(a[key]); chosen=rng.choices(ids,k=len(ids))
            means.append(mean(a[key][i]-b[key][i] for i in chosen))
        draws.append(mean(means))
    return {'difference':mean(mean(a[k][i]-b[k][i] for i in a[k]) for k in keys),
            'low':quantile(draws,.025),'high':quantile(draws,.975),
            'training_seeds':len(keys)}
```

Create `rebuild/src/shipnav/benchmark.py` with:

```python
from pathlib import Path
import json
from shipnav.service import execute
from shipnav.scenarios import scenario_hash
from shipnav.planning import NoPath
from shipnav.metrics import metrics, geometry_metrics

VARIANTS={
 'sarl_reference':dict(policy_name='sarl'),
 'sarl_theta':dict(policy_name='sarl',planner='theta'),
 'sarl_no_goals':dict(policy_name='sarl',global_goals=False),
 'sarl_filtered':dict(policy_name='sarl',filtered=True),
 'sarl_cv_filter':dict(policy_name='sarl',filtered=True,uncertainty=False),
 'sarl_degraded':dict(policy_name='sarl',filtered=True,observation={'noise':.1,'delay':.5,'dropout':.1}),
 'orca_reference':dict(policy_name='orca'),
 'orca_filtered':dict(policy_name='orca',filtered=True),
 'direct_reference':dict(policy_name='direct'),
 'marine_sarl':dict(policy_name='sarl',dynamics='marine',filtered=True),
 'marine_mpc':dict(policy_name='mpc',dynamics='marine',filtered=True),
 'marine_mpc_unfiltered':dict(policy_name='mpc',dynamics='marine',filtered=False)}

def benchmark(scenarios,model_dir,output,variants=VARIANTS,runner=execute):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); rows=[]
    for scenario in scenarios:
        identity=scenario_hash(scenario)
        (output/f'{identity}-scenario.json').write_text(json.dumps(scenario,allow_nan=False))
        for name,config in variants.items():
            try:
                run=runner(scenario['map'],scenario['start'],scenario['goal'],model_dir,
                           scenario=scenario,**config)
            except NoPath as error:
                run={'status':'planning_failure','error':str(error)}
            except Exception as error:
                run={'status':'error','error':f'{type(error).__name__}: {error}'}
            run.update(scenario_hash=identity,variant=name,variant_config=config)
            (output/f'{identity}-{name}.json').write_text(json.dumps(run,allow_nan=False))
            rows.append({'scenario_hash':identity,'variant':name,**metrics(run),**geometry_metrics(run)})
    (output/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows
```

- [ ] **Step 4: Run the same test command.** Expected: PASS. Inspect failures before proceeding.

- [ ] **Step 5: Review the diff and commit.**

```bash
git add rebuild/src/shipnav/metrics.py rebuild/src/shipnav/benchmark.py rebuild/tests/test_metrics.py rebuild/tests/test_geometric_metrics.py
git commit -m "feat: score paired benchmark runs with geometry metrics"
```

### Task 12: Freeze the protocol and record pre-retraining baselines

Plan 05 compares new policies against these numbers, so complete this task before any training run.

- [ ] Stratify inference-only and full decision timing by device/platform, cold versus warmed runs, traffic count and dynamics. For asynchronous CUDA timing, synchronize before/after measured work. Report p95/p99/deadline misses with sample counts.
- [ ] Freeze a `benchmark_protocol.json` before the held-out run: scenario IDs/hashes from `scenarios/splits.json`, split boundaries, seeds, primary metrics, variant pairs, the fixed reference route per scenario, runtime hardware and stopping rules. Set the success/collision targets that `2026-09-27-rebuild-decisions.md` §7 leaves open now, before seeing results. Ten seeds remain a smoke test. Begin with at least 100 distinct held-out scenarios across the named families (the current frozen test split has 11 canonical encounters; generate the rest with `tools/freeze_scenarios.py`), then choose additional sample size from pilot uncertainty/precision, not a desired significance result.
- [ ] Run `benchmark()` on the held-out split with every `VARIANTS` entry using the supplied SARL checkpoint. At minimum record `sarl_reference` (unfiltered) against `sarl_filtered`, `orca_reference` against `orca_filtered`, and `marine_mpc` against `marine_mpc_unfiltered`. Keep every trace and `metrics.json` under `results/` (ignored) and copy the protocol and summary into a tracked evidence file.
- [ ] Use scenario-paired bootstrap (`paired_interval`) for fixed policies. For learned policies (plan 05), keep independent training seeds (at least 3; preferably 5) and use `hierarchical_interval`; do not treat thousands of rollouts from one checkpoint as independent models. Publish seed-level results and intervals alongside pooled values. A failed/missing run is never deleted from only one arm.
- [ ] Isolate one main factor per comparison: A* vs Theta*, goals on/off, filter on/off, constant-velocity vs uncertainty-aware prediction, exact vs degraded sensing, SARL vs MPC **under the same dynamics/filter/observations**. The variant table provides defaults, not permission to compare mismatched controls.
- [ ] Update `rebuild/README.md`: CLI and GUI commands; GUI scope (synthetic maps up to 250,000 m²; real maps through `map_demo`); click workflow and both control rows; export formats and the CSV column meanings; experiment and benchmark commands and metric definitions; the no-neighbour direct fallback; constant-velocity traffic assumption; limits of encounter-rule coverage and the selected marine dynamics; model hash interpretation; results location; where the baseline benchmark evidence lives. This is the lightweight user manual, not an academic report.
- [ ] Commit the README and the tracked baseline evidence once the native-window checklist in Task 10 passes.
