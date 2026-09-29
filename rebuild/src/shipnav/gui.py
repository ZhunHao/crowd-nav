from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from bisect import bisect_right
from time import monotonic
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.export import draw, export_run
from shipnav.scale import PROFILES, check_grid, default_profile, scaled, units

DEGRADED = {'noise': .1, 'delay': .5, 'dropout': .1}
REPLAY_SPEEDS = (1, 10, 50, 200)
MODEL = Path(__file__).resolve().parents[3]/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def default_endpoints(sea: SeaMap, clearance: float):
    """Two model units inside opposite corners when clear; otherwise unset."""
    x0, y0, x1, y1 = sea.bounds
    start, goal = (x0+2, y0+2), (x1-2, y1-2)
    return (start if sea.clear(start, start, clearance) else None,
            goal if sea.clear(goal, goal, clearance) else None)


class Window(QWidget):
    def __init__(self, sea: SeaMap, model_dir: str = '', runner=execute):
        super().__init__()
        self.setWindowTitle('Ship navigation rebuild')
        self.resize(1100, 850)
        # `real` is the metric map (with any land edits); `sea` is what the service runs:
        # the same map in model units for the selected vessel profile.
        self.model_dir, self.runner = model_dir, runner
        self.real, self.profile = sea, default_profile(sea)
        check_grid(sea, self.profile)
        self.sea = scaled(sea, self.profile)
        self.start, self.goal = default_endpoints(self.sea, self.clearance)
        self.result, self.future, self.corner = None, None, None
        self.job_kind, self.replay_index = '', -1
        self.replay_started, self.replay_rate, self.replay_times = 0., 1, []
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
        self.vessel = QComboBox()
        self.vessel.addItems(list(PROFILES))
        self.vessel.setCurrentText(self.profile.name)
        self.vessel.currentTextChanged.connect(self.set_profile)
        self.placement = QComboBox()
        self.placement.addItems(['Traffic anywhere', 'Traffic along route'])
        self.placement.setCurrentIndex(0 if self.profile.name == 'model' else 1)
        self.replay_speed = QComboBox()
        self.replay_speed.addItems([f'Replay {s}x' for s in REPLAY_SPEEDS])
        self.edit_controls = [self.map_button, self.model_button, self.mode,
                              self.policy, self.count, self.seed, self.run_button,
                              self.planner, self.dynamics, self.filter_mode, self.perception,
                              self.vessel, self.placement]
        for widget in self.edit_controls[:7] + [self.stop_button, self.replay_button, self.export_button]:
            toolbar.addWidget(widget)
        for widget in self.edit_controls[7:] + [self.replay_speed]:
            options.addWidget(widget)
        self.status = QLabel('Select start and goal, then run')
        layout.addWidget(self.status)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.subplots()
        self.navigation = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.navigation)  # zoom/pan for kilometre maps
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

    @property
    def clearance(self):
        return self.profile.service_options()['clearance']

    def clear_corner(self):
        self.corner = None

    def set_profile(self, name):
        old, profile = self.profile, PROFILES[name]
        try:
            check_grid(self.real, profile)
        except ValueError as error:
            self.vessel.blockSignals(True)
            self.vessel.setCurrentText(old.name)
            self.vessel.blockSignals(False)
            self.status.setText(str(error))
            return
        ratio = old.length/profile.length
        self.profile, self.sea, self.corner = profile, scaled(self.real, profile), None
        # Keep the chosen endpoints at the same place in metres when still clear.
        self.start, self.goal = (None if p is None or not self.sea.clear(q := (p[0]*ratio, p[1]*ratio), q, self.clearance)
                                 else q for p in (self.start, self.goal))
        self.invalidate()
        self.status.setText(f'Vessel profile {name}: radius {profile.radius_m:g} m, '
                            f'{profile.speed_mps:g} m/s, grid {profile.resolution_m:g} m')

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
        if self.navigation.mode:
            return
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            self.apply_click(event.xdata, event.ydata)

    def apply_click(self, x, y):
        if self.future is not None:
            return
        point = (float(x), float(y))
        try:
            if self.mode.currentIndex() < 2:
                if not self.sea.clear(point, point, self.clearance):
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
                L, _ = units(self.sea.to_dict())
                rect = (min(a[0], b[0])*L, min(a[1], b[1])*L, max(a[0], b[0])*L, max(a[1], b[1])*L)
                real = SeaMap(self.real.bounds, self.real.land + (rect,), self.real.to_dict()['metadata'])
                sea = scaled(real, self.profile)
                if not all(p is None or sea.clear(p, p, self.clearance) for p in (self.start, self.goal)):
                    raise ValueError('New land blocks start or goal')
                self.real, self.sea = real, sea
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
            real = SeaMap.load(path)
            profile = default_profile(real)
            check_grid(real, profile)
            sea = scaled(real, profile)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            self.status.setText(f'Map not loaded: {error}')
            return
        self.real, self.profile, self.sea, self.corner = real, profile, sea, None
        self.vessel.blockSignals(True)
        self.vessel.setCurrentText(profile.name)
        self.vessel.blockSignals(False)
        self.placement.setCurrentIndex(0 if profile.name == 'model' else 1)
        self.start, self.goal = default_endpoints(self.sea, self.clearance)
        self.invalidate()
        self.status.setText(f'Map loaded with vessel profile {profile.name}; select valid start and goal')

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
            observation={} if self.perception.currentIndex() == 0 else dict(DEGRADED),
            placement='uniform' if self.placement.currentIndex() == 0 else 'corridor',
            **self.profile.service_options())

    def check_worker(self):
        if self.future is None or not self.future.done():
            return
        try:
            value = self.future.result()
            if self.job_kind == 'run':
                self.result = value
                draw(self.ax, value)
                self.canvas.draw_idle()
                _, T = units(value['map'])
                self.status.setText(f"{value['status']}: {value['elapsed']*T:.1f} simulated seconds")
            else:
                self.status.setText('Export saved')
        except Exception as error:
            self.status.setText(f'Failed: {error}')
        finally:
            self.future = None
            self.set_busy(False)

    def replay(self):
        if self.result is not None and self.result['frames']:
            _, T = units(self.result['map'])
            # Snapshot the physical clock and selected rate when Replay starts.
            self.replay_rate = REPLAY_SPEEDS[self.replay_speed.currentIndex()]
            self.replay_times = [f['t']*T for f in self.result['frames']]
            interval = 1000*self.result['settings']['dt']*T/self.replay_rate
            self.replay_index = -1
            self.replay_started = monotonic()
            self.playback.start(max(40, round(interval)))
            self.next_frame()

    def next_frame(self):
        if self.result is None or not self.replay_times:
            self.playback.stop()
            return
        # A slow render delays callbacks. Select by elapsed time, dropping stale
        # frames instead of stretching the requested replay speed.
        elapsed = max(0., monotonic()-self.replay_started)*self.replay_rate
        index = max(0, bisect_right(self.replay_times, elapsed)-1)
        if index != self.replay_index:
            draw(self.ax, self.result, index)
            self.canvas.draw_idle()
            self.replay_index = index
        if index == len(self.result['frames'])-1:
            self.playback.stop()

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
