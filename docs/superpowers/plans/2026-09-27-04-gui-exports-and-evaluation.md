# 04 — GUI, exports and evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the interactive maritime-navigation demo and reproducible experiment outputs.

**Architecture:** Use the same headless service for CLI, GUI and evaluation. Execute simulations in a worker thread with cooperative cancellation; display and replay results on the Qt main thread, then export the same recorded trace.

**Tech Stack:** PySide6, Matplotlib QtAgg, FFmpeg, Python concurrent.futures, pytest-qt, standard-library CSV/JSON

**Spec:** [../specs/2026-09-27-solo-rebuild-design.md](../specs/2026-09-27-solo-rebuild-design.md)

## Global Constraints

- Never use `codex` in branch names or worktree directory names.
- Preserve `CrowdNav-20250813-DIP/` and the supplied briefing unchanged.
- Implement only inside `rebuild/`; keep planning documents inside `docs/superpowers/` and Git-ignored.
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

All file paths below are relative to the workspace root. Run commands from `rebuild/` unless explicitly stated otherwise. Code blocks are proposed implementation content, not evidence of passing tests. A code step can be worked through function by function in 2–5 minute increments; commit only after the task's tests pass. Git is initialized on `main`; commit only explicitly named implementation files. Documentation remains ignored.

## File structure and prerequisites

Create `rebuild/src/shipnav/export.py`, `gui.py`, `evaluate.py`; create `rebuild/tests/test_export.py`, `test_gui.py`, `test_evaluate.py`; extend `rebuild/README.md`. Complete plans 01b, 03 and 03b first. Run all commands in `.venv-modern`; never activate `.venv-legacy` for this phase. Qt integration tests require a display; use `QT_QPA_PLATFORM=offscreen` for tests where supported, then perform the native-window acceptance checklist.

The GUI uses replay of a completed simulation. It shows a running status during inference, remains responsive, and allows cancellation. Live streaming during inference is a possible extension, not needed to meet the briefing's visualization requirement.

### Task 7: Export and draw recorded navigation results

**Files:** `rebuild/src/shipnav/export.py`, `rebuild/tests/test_export.py`.

**Interfaces:** Consumes the `execute()` result schema. Produces `draw(ax, result: dict, frame_index: int = -1) -> None` and `export_run(result: dict, path: Path) -> None`. JSON is the complete reproducibility artifact; CSV exports ego trajectory only; PNG is a final overview; MP4 replays at `1/dt` FPS.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_export.py` with:

```python
import csv
import json
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
```

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_export.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

Create `rebuild/src/shipnav/export.py` with:

```python
from pathlib import Path
import csv
import json
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FFMpegWriter, writers


def draw(ax, result: dict, frame_index: int = -1) -> None:
    ax.clear()
    x0, y0, x1, y1 = result['map']['bounds']
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel='x (m)', ylabel='y (m)', aspect='equal')
    ax.set_facecolor('#dceef5')
    for a, b, c, d in result['map']['land']:
        ax.add_patch(Rectangle((a, b), c-a, d-b, facecolor='#9b987f'))
    route = result['route']
    if route:
        ax.plot(*zip(*route), '--', color='#555555', label='Global route')
    goals = result['goals']
    if goals:
        ax.scatter(*zip(*goals), marker='*', color='#cf4b35', label='Goals')
    frames = result['frames']
    if frames:
        i = len(frames)-1 if frame_index < 0 else min(frame_index, len(frames)-1)
        frame = frames[i]
        ax.plot(*zip(*(f['position'] for f in frames[:i+1])), color='#17659c', label='Travelled')
        ax.add_patch(Circle(frame['position'], result['settings']['radius'], color='#17659c'))
        for position, ship in zip(frame['traffic'], result['traffic_definitions']):
            ax.add_patch(Circle(position, ship['radius'], color='#df9a36'))
        ax.set_title(f"{result['status']} — {frame['t']:.2f} s")
    ax.legend(loc='upper left')


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
            writer = csv.writer(handle)
            writer.writerow(['time_s', 'x_m', 'y_m', 'vx_mps', 'vy_mps', 'goal_index'])
            for f in result['frames']:
                writer.writerow([f['t'], *f['position'], *f['velocity'], f['goal_index']])
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

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_export.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/export.py rebuild/tests/test_export.py
git commit -m "feat: export and draw recorded navigation results"
```

### Task 8: Build the navigation GUI

**Files:** `rebuild/src/shipnav/gui.py`, `rebuild/tests/test_gui.py`.

**Interfaces:** Consumes `SeaMap`, `execute`, `draw`, and `export_run`. Produces `Window(sea: SeaMap, model_dir: str = "", runner=execute)`, `.apply_click(x,y)`, `.start_run()`, `.check_worker()` and module entry `python -m shipnav.gui`. Worker functions never touch Qt widgets; Qt timer polling transfers results to the main thread.

- [ ] **Step 1: Write the behaviour tests.**

Create `rebuild/tests/test_gui.py` with:

```python
from threading import Event
from shipnav.maps import SeaMap
from shipnav.gui import Window
from shipnav.service import execute


def test_start_goal_and_land_editing(qtbot):
    window = Window(SeaMap((0, 0, 24, 24)))
    qtbot.addWidget(window)
    window.apply_click(3, 3)
    assert window.start == (3, 3)
    window.mode.setCurrentIndex(1)
    window.apply_click(21, 21)
    assert window.goal == (21, 21)
    window.mode.setCurrentIndex(2)
    window.apply_click(8, 8)
    window.apply_click(10, 10)
    assert window.sea.land == ((8, 8, 10, 10),)
    window.close()


def test_worker_leaves_gui_responsive_and_restores_controls(qtbot):
    entered, release = Event(), Event()
    def runner(*args):
        entered.set()
        release.wait(2)
        return execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (4, 2), policy_name='direct', count=0)
    window = Window(SeaMap((0, 0, 24, 24)), runner=runner)
    qtbot.addWidget(window)
    window.start_run()
    qtbot.waitUntil(entered.is_set)
    assert not window.run_button.isEnabled()
    assert window.stop_button.isEnabled()
    window.apply_click(7, 7)
    assert window.start == (2, 2)
    release.set()
    qtbot.waitUntil(lambda: window.future is None)
    assert window.result['status'] == 'success'
    assert window.export_button.isEnabled()
    window.close()


def test_worker_error_is_visible(qtbot):
    def broken(*args):
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
python -m pytest tests/test_gui.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

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


class Window(QWidget):
    def __init__(self, sea: SeaMap, model_dir: str = '', runner=execute):
        super().__init__()
        self.setWindowTitle('Ship navigation rebuild')
        self.resize(1000, 800)
        self.sea, self.model_dir, self.runner = sea, model_dir, runner
        self.start, self.goal = (2, 2), (22, 22)
        self.result, self.future, self.corner = None, None, None
        self.job_kind, self.replay_index = '', 0
        self.cancel_event = Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        layout, toolbar = QVBoxLayout(self), QHBoxLayout()
        layout.addLayout(toolbar)
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
        self.edit_controls = [self.map_button, self.model_button, self.mode,
                              self.policy, self.count, self.seed, self.run_button]
        for widget in self.edit_controls + [self.stop_button, self.replay_button, self.export_button]:
            toolbar.addWidget(widget)
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
        self.stop_button.clicked.connect(self.cancel_event.set)
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
        preview = {'map': self.sea.to_dict(), 'route': [], 'goals': [self.start, self.goal],
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
                if not self.sea.clear(point, point, .7):
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
                sea = SeaMap(self.sea.bounds, self.sea.land + (rect,))
                if not all(sea.clear(p, p, .7) for p in (self.start, self.goal)):
                    raise ValueError('New land blocks start or goal')
                self.sea = sea
            self.invalidate()
            self.status.setText('Scene updated')
        except ValueError as error:
            self.status.setText(str(error))

    def load_map(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Load map', '', 'Map (*.json)')
        if not filename:
            return
        try:
            self.sea = SeaMap.load(Path(filename))
            self.corner = None
            self.invalidate()
            self.status.setText('Map loaded; select valid start and goal')
        except (ValueError, TypeError, OSError) as error:
            self.status.setText(str(error))

    def load_model(self):
        directory = QFileDialog.getExistingDirectory(self, 'Folder with policy.config and rl_model.pth')
        if directory:
            self.model_dir = directory
            self.invalidate()
            self.status.setText('Model folder selected; weights are validated when Run starts')

    def start_run(self):
        if self.future is not None:
            return
        self.playback.stop()
        self.result = None
        self.cancel_event = Event()
        self.job_kind = 'run'
        self.status.setText('Running navigation…')
        self.set_busy(True)
        self.future = self.pool.submit(self.runner, self.sea.to_dict(), self.start, self.goal,
            self.model_dir, self.policy.currentText(), True, self.seed.value(), self.count.value(),
            self.cancel_event.is_set)

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
    window = Window(SeaMap.load(Path('maps/harbour.json')),
                    str(Path('../CrowdNav-20250813-DIP/crowd_nav/data/output_trained').resolve()))
    window.show()
    sys.exit(app.exec())
```

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_gui.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/gui.py rebuild/tests/test_gui.py
git commit -m "feat: build the navigation gui"
```

### Task 9: Run paired navigation experiments

**Files:** `rebuild/src/shipnav/evaluate.py`, `rebuild/tests/test_evaluate.py`.

**Interfaces:** Consumes `execute` and its trace schema. Produces `summarize(runs: list[dict]) -> dict` and `evaluate(map_data, model_dir, output: Path, seeds=range(10), counts=(5,8,12,15), runner=execute) -> list[dict]`. No-global-goals ablates waypoint guidance; direct ablates local collision avoidance; ORCA supplies a rule-based comparator.

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

- [ ] **Step 2: Run the tests before implementation.**

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: failure because the new module or behaviour does not exist. Missing environment dependencies must be resolved before treating a test failure as evidence about application behaviour.

- [ ] **Step 3: Implement the following files.**

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

- [ ] **Step 4: Run the same test command and inspect the result.**

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: PASS. Do not weaken assertions to conceal an algorithm or interface failure.

- [ ] **Step 5: Review the task's diff and save a checkpoint.**

From the workspace root, if it is a Git repository:

```bash
git add rebuild/src/shipnav/evaluate.py rebuild/tests/test_evaluate.py
git commit -m "feat: run paired navigation experiments"
```

## End-to-end acceptance and operating notes

- [ ] Run the complete suite once after all stages: `python -m pytest -v`.
- [ ] Run `python -m shipnav.gui` in a native window. Load `maps/harbour.json`, select the supplied model folder, set start `(2,2)` and destination `(22,22)`, and run SARL with 5 moving ships.
- [ ] While running, move/resize the window. Confirm scene edits and duplicate runs are disabled. Press Stop; wait for the current inference step and confirm `cancelled`, then rerun.
- [ ] Change the destination and add a land rectangle using two clicks. Run again and verify the global route changes. Try covering the start with land and selecting a point on land; both must be rejected with an understandable status message.
- [ ] Select an empty model folder and run SARL. Verify an error appears and controls are restored. Then select the correct model folder and verify recovery.
- [ ] Replay a completed trace; export JSON, CSV, PNG and MP4 using different filenames. Verify exported coordinates/times match the display; verify a 10-second simulated interval takes approximately 10 seconds in the MP4. One extra terminal display frame is acceptable.
- [ ] Check an exported video with `ffprobe -v error -show_entries stream=codec_name,nb_frames,r_frame_rate -of json results/demo.mp4` after selecting `results/demo.mp4` in the GUI. Expected frame rate is 4 FPS at `dt=.25`; all recorded frames including the initial and terminal frame must be present.
- [ ] Try an impossible map with a wall from one boundary to the opposite boundary; show planning failure instead of an empty successful route.
- [ ] Run a small evaluation first: `python -m shipnav.evaluate --seeds 2 --counts 5`. Inspect all eight traces and `summary.csv`. Then run `python -m shipnav.evaluate` for 160 episodes across four variants, four traffic counts and ten seeds. This is a smoke batch only. Complete Task 10 below before drawing comparative conclusions.
- [ ] If placement fails for dense traffic, retain the error rows. Reduce the count or enlarge the scenario in a documented subsequent experiment; do not drop difficult seeds from only one algorithm's results. Latencies include cold first inference and are descriptive, not rigorous real-time guarantees.
- [ ] Extend `rebuild/README.md` with: baseline evidence location and modern-only setup/locks from plan 01b; CLI and GUI commands above; the map JSON schema from plan 02; click workflow and export formats; experiment command and metric definitions; the no-neighbour direct fallback; constant-velocity traffic assumption; limits of encounter-rule coverage and the selected marine dynamics; model hash interpretation; results location. This is the lightweight user manual, not an academic report.
- [ ] Save an application checkpoint after the manual workflow passes. Do not claim the full application works solely because mocked GUI tests pass.

## Task 10 — Expose revised controls and diagnostics

**Files:** modify `src/shipnav/gui.py`, `export.py`, `tests/test_gui.py`, `tests/test_export.py`; create `metrics.py`, `benchmark.py` and their tests.

The preceding `evaluate.py` remains a small smoke utility. Final comparisons use the frozen-scenario benchmark below. Its four original variants remain useful controls, but no longer define the full evaluation protocol.

- [ ] Add a second row of controls in `Window.__init__` before creating `self.status`:

```python
options = QHBoxLayout()
layout.addLayout(options)
self.planner = QComboBox(); self.planner.addItems(['astar_smooth', 'theta'])
self.dynamics = QComboBox(); self.dynamics.addItems(['holonomic', 'marine'])
self.filter_mode = QComboBox(); self.filter_mode.addItems(['Unfiltered', 'Predictive filter'])
self.perception = QComboBox(); self.perception.addItems(['Exact observations', 'Noisy delayed observations'])
for control in (self.planner, self.dynamics, self.filter_mode, self.perception):
    options.addWidget(control)
    self.edit_controls.append(control)
```

- [ ] Add keyword arguments to the existing `self.pool.submit(self.runner, ...)` call, after the cancellation callback. Snapshot these values on the GUI thread, never read widgets in a worker:

```python
planner=self.planner.currentText(),
dynamics=self.dynamics.currentText(),
filtered=self.filter_mode.currentIndex() == 1,
observation={} if self.perception.currentIndex() == 0 else {'noise': .1, 'delay': .5, 'dropout': .1}
```

- [ ] Existing mocked runners must accept `**kwargs`. Add a Qt test that selects Theta*, marine and predictive filter, runs a capturing fake runner, and asserts those three keywords arrive unchanged and controls restore after failure/cancellation. Retain native-window responsiveness and export tests; ensure controls fit at normal window size.
- [ ] Append the following overlay inside `draw`, after drawing ego/traffic but before `ax.legend`. Each diagnostic at tick k describes the action from frame k to k+1. Do not draw next-tick predictions on earlier frames.

```python
diagnostics = result.get('diagnostics', [])
if frames and diagnostics:
    d = diagnostics[min(i, len(diagnostics)-1)]
    origin = frames[min(i, len(diagnostics)-1)]['position']
    for key, color in [('nominal', '#b05090'), ('executed', '#248060')]:
        ax.arrow(*origin, *d[key], color=color, width=.025, label=key)
    for target in d['predictions']:
        ax.plot(*zip(*target['points']), ':', color='#e5a02e')
    if d['path']:
        ax.plot(*zip(*d['path']), '-', color='#248060', alpha=.5)
    ax.set_title(f"{result['status']} | override={d['override']} | no feasible action={d['no_feasible_action']}")
```

- [ ] Extend CSV with one row per executed step containing nominal/executed/actual velocity, intervention, no-feasible-action, controller/decision latency, observed target age and collision type; keep full nested observations/predictions in JSON. Initial/terminal frames have no invented action. Test that export uses stored data after the policy is removed, and that frame timing stays 1/dt.

### Task 11 — Metrics and paired evidence

### Task 11a — Persist every paired attempt

**Files:** `metrics.py`, `benchmark.py`, `tests/test_metrics.py`

**Interfaces:** `metrics(run)->dict`; `paired_interval(a:dict[id,float],b:dict[id,float])->dict`; `benchmark(scenarios,model_dir,output,variants=VARIANTS,runner=execute)->list[dict]`. Failures retain the scenario hash and experiment denominator.

- [ ] Write the behavior test:

File: `rebuild/tests/test_metrics.py`

```python
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
```

- [ ] Run `python -m pytest tests/test_metrics.py -v`. Expect a missing module or unmet behavior, not a missing third-party dependency.
- [ ] Implement the following content:

File: `rebuild/src/shipnav/metrics.py`

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
            'no_feasible_actions':sum(d['no_feasible_action'] for d in ds),
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
```

File: `rebuild/src/shipnav/benchmark.py`

```python
from pathlib import Path
import json
from shipnav.service import execute
from shipnav.scenarios import scenario_hash
from shipnav.planning import NoPath
from shipnav.metrics import metrics

VARIANTS={
 'sarl_reference':dict(policy_name='sarl'),
 'sarl_theta':dict(policy_name='sarl',planner='theta'),
 'sarl_no_goals':dict(policy_name='sarl',global_goals=False),
 'sarl_filtered':dict(policy_name='sarl',filtered=True),
 'sarl_cv_filter':dict(policy_name='sarl',filtered=True,uncertainty=False),
 'sarl_degraded':dict(policy_name='sarl',filtered=True,observation={'noise':.1,'delay':.5,'dropout':.1}),
 'orca_reference':dict(policy_name='orca'),
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
            rows.append({'scenario_hash':identity,'variant':name,**metrics(run)})
    (output/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows
```

- [ ] Run `python -m pytest tests/test_metrics.py -v` again. Inspect failures before proceeding.
- [ ] Review and commit only the implementation/test files named above after the check passes; keep `docs/` ignored.

- [ ] Extend metric tests with known geometry: point `(3,4)` against horizontal segment `(0,0)→(10,0)` gives cross-track error 4; radius .5 at x=2 in a `[0,10]` empty map gives minimum land/boundary clearance 1.5. Compute exact point-to-rectangle/boundary distances, subtract radius, and take the minimum across frames. For swept clearance retain the existing continuous geometry test; label sampled diagnostic minima separately.
- [ ] Compute cross-track mean/max against the **fixed reference route**, and heading/acceleration violations from successive recorded marine state (`abs(wrapped Δheading)/dt > .35+1e-6`, `abs(Δspeed)/dt > .2+1e-6`). Include dt-weighted domain exposure (the 1 m extra circular domain above is an experimental definition), not just count of frames. Use actual interval lengths for a fractional final step.
- [ ] Stratify inference-only and full decision timing by device/platform, cold versus warmed runs, traffic count and dynamics. For asynchronous CUDA timing, synchronize before/after measured work. Report p95/p99/deadline misses with sample counts; use repeated hardware trials only after behavior is deterministic.
- [ ] Freeze a `benchmark_protocol.json` before the held-out run: scenario IDs/hashes, split boundaries, seeds, primary metrics, variant pairs, runtime hardware and stopping rules. Ten seeds remain a smoke test. Begin with at least 100 distinct held-out scenarios across the named families, then choose additional sample size from pilot uncertainty/precision, not a desired significance result. Rare collision probabilities require more evidence than this starting sample provides.
- [ ] Use scenario-paired bootstrap for fixed policies. For learned policies, preserve independent training seeds (at least 3; preferably 5 if budget allows) and resample training seeds then scenarios hierarchically; do not treat thousands of rollouts from one checkpoint as independent learned models. Publish seed-level results and confidence intervals alongside pooled values. A failed/missing run is never deleted from only one arm.
- [ ] Isolate one main factor per comparison: A* vs Theta*, goals on/off, filter on/off, constant-velocity vs uncertainty-aware prediction, exact vs degraded sensing, SARL vs MPC **under the same dynamics/filter/observations**. Add both filtered and unfiltered ORCA/MPC comparison rows when making claims about the filter. The variant table provides defaults, not permission to compare mismatched controls.
- [ ] Complete original GUI acceptance again using the modern environment. Update README with retained baseline evidence, modern-only lock recreation, CLI/GUI controls, scenario import/limitations, metric definitions, training/EC2 commands and model manifest interpretation. No academic literature deliverable is added.

### Task 11b — Geometry and seed hierarchy reference code

Append these functions to `metrics.py`; merge `geometry_metrics(run)` into each successful/failed episode's metric row when frames exist. Its dt-weighted exposure replaces the earlier fixed-step approximation. Compare cross-track against the same reference route across variants (save that route in the protocol); a planner's own route is appropriate only for reporting its controller's tracking error, not a fair planner-independent score. Record the initial heading/speed if they become configurable; this implementation matches the fixed zero-heading/zero-speed initial state.

File: `rebuild/src/shipnav/metrics.py (append)`

```python
def point_segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    f=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0.
    return dist(p,(a[0]+f*dx,a[1]+f*dy))

def land_clearance(sea,p,radius):
    x,y=p; x0,y0,x1,y1=sea.bounds
    values=[x-x0,y-y0,x1-x,y1-y]
    for a,b,c,d in sea.land:
        if a<=x<=c and b<=y<=d:
            values.append(-min(x-a,c-x,y-b,d-y))
        else:
            values.append(dist(p,(min(c,max(a,x)),min(d,max(b,y)))))
    return min(values)-radius

def geometry_metrics(run):
    from shipnav.maps import SeaMap
    from math import pi
    sea=SeaMap.from_dict(run['map']); frames=run['frames']; route=run['route']
    errors=[min((point_segment_distance(f['position'],a,b) for a,b in zip(route,route[1:])),default=0.) for f in frames]
    land=[land_clearance(sea,f['position'],run['settings']['radius']) for f in frames]
    ds=run.get('diagnostics',[]); heading=0.; speed=0.; violations=0; exposure=0.
    for a,b,d in zip(frames,frames[1:],ds):
        dt=b['t']-a['t']
        if d.get('clearance') is not None and d['clearance']<1.: exposure+=dt
        if 'heading' in d:
            turn=(d['heading']-heading+pi)%(2*pi)-pi
            violations+=int(abs(turn)> .35*dt+1e-6 or abs(d['speed']-speed)>.2*dt+1e-6)
            heading,speed=d['heading'],d['speed']
    return {'cross_track_mean_m':mean(errors) if errors else None,
            'cross_track_max_m':max(errors) if errors else None,
            'sampled_land_clearance_m':min(land) if land else None,
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
File: `rebuild/tests/test_geometric_metrics.py`

```python
import pytest
from shipnav.maps import SeaMap
from shipnav.metrics import point_segment_distance,land_clearance,hierarchical_interval

def test_geometric_metrics_have_known_values():
    assert point_segment_distance((3,4),(0,0),(10,0))==4
    assert land_clearance(SeaMap((0,0,10,10)),(2,5),.5)==1.5
    assert land_clearance(SeaMap((0,0,10,10),((4,4,6,6),)),(5,5),.5)==-1.5

def test_hierarchical_interval_uses_both_seed_levels():
    a={0:{'x':1,'y':1},1:{'x':1,'y':1}}; b={0:{'x':0,'y':0},1:{'x':0,'y':0}}
    assert hierarchical_interval(a,b)['low']==1
    with pytest.raises(ValueError): hierarchical_interval(a,{0:b[0]})
```

- [ ] Run `python -m pytest tests/test_geometric_metrics.py -v` before/after implementation. Collision status uses swept checks; the land-clearance diagnostic above is explicitly sampled.
