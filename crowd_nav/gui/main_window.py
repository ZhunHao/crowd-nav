"""PyQt5 main window for the CrowdNav GUI demo (WP-5 / R5)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout

from crowd_nav.gui.canvas import SimCanvas
from crowd_nav.gui.dialogs import show_error
from crowd_nav.gui.obstacle_editor import ObstacleEditor
from crowd_nav.gui.toolbar import build_toolbar
from crowd_nav.gui.workers.sim_worker import SimWorker


class MainWindow(QMainWindow):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CrowdNav DIP - Maritime Navigation Demo")
        self.controller = controller

        self.canvas = SimCanvas(world_bounds=(-15.0, 15.0, -15.0, 15.0), parent=self)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        self.toolbar = build_toolbar(
            self,
            handlers={
                "Load static map": self.on_load_map,
                "Load DRL model": self.on_load_model,
                "Set start": self.on_set_start,
                "Set goal": self.on_set_goal,
                "Add static obstacles": self.on_add_obstacles,
                "Visualize": self.on_visualize,
                "Export": self.on_export,
            },
        )
        self.addToolBar(self.toolbar)

        self._canvas_click_handler: Optional[Callable[[float, float], None]] = None
        self.canvas.clicked.connect(self._route_canvas_click)

        self._editor: Optional[ObstacleEditor] = None
        self._worker: Optional[SimWorker] = None
        self._last_states: list[dict] = []

    # ---------- slot router ----------

    def _route_canvas_click(self, x: float, y: float) -> None:
        if self._canvas_click_handler is not None:
            handler = self._canvas_click_handler
            self._canvas_click_handler = None
            handler(x, y)
        elif self._editor is not None:
            if self._editor._press_xy is None:
                self._editor.on_press((x, y))
            else:
                self._editor.on_release((x, y))

    # ---------- handlers ----------

    def on_load_map(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select map", "", "Maps (*.png *.npy)"
        )
        if not path_str:
            return
        try:
            self.controller.load_map(Path(path_str))
        except Exception as exc:
            show_error(self, f"Cannot load map: {exc}")
            return
        if self.controller.static_map is not None:
            self.canvas.draw_static_map(self.controller.static_map)

    def on_load_model(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select DRL model", "", "Models (*.pth)"
        )
        if not path_str:
            return
        try:
            self.controller.load_model(Path(path_str), policy_name="sarl")
        except Exception as exc:
            show_error(self, f"Cannot load model: {exc}")

    def on_set_start(self) -> None:
        self._canvas_click_handler = lambda x, y: self._safe(
            self.controller.set_start, (x, y), label="start"
        )

    def on_set_goal(self) -> None:
        self._canvas_click_handler = lambda x, y: self._safe(
            self.controller.set_goal, (x, y), label="goal"
        )

    def on_add_obstacles(self) -> None:
        if self._editor is not None:
            try:
                self._editor.obstacle_created.disconnect(self.controller.add_obstacle)
            except TypeError:
                pass  # no prior connection
            self._editor.deleteLater()
        self._editor = ObstacleEditor(mode="rect", parent=self)
        self._editor.obstacle_created.connect(self.controller.add_obstacle)

    def on_visualize(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            show_error(self, "A rollout is already in progress.")
            return
        try:
            self._worker = SimWorker(self.controller)
            self._worker.frame_ready.connect(self._on_frame)
            self._worker.episode_done.connect(self._on_done)
            self._worker.failed.connect(lambda msg: show_error(self, msg))
            self._worker.start()
        except Exception as exc:
            show_error(self, f"Cannot start rollout: {exc}")

    def on_export(self) -> None:
        dir_str = QFileDialog.getExistingDirectory(self, "Export to directory")
        if not dir_str:
            return
        try:
            from crowd_sim.envs.utils.export_writer import write_exports

            write_exports(
                Path(dir_str),
                states=self._last_states,
                waypoints=self.controller.last_waypoints,
            )
        except Exception as exc:
            show_error(self, f"Cannot export: {exc}")

    # ---------- worker slots ----------

    def _on_frame(self, i: int, step: dict) -> None:
        self.canvas.push_frame(
            robot_xy=(step["robot"][0], step["robot"][1]),
            humans_xy=[(h[0], h[1]) for h in step["humans"]],
            waypoints=[],
        )

    def _on_done(self, summary: dict) -> None:
        self._last_states = summary.get("states", [])
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ---------- helpers ----------

    def _safe(self, fn, arg, *, label: str) -> None:
        try:
            fn(arg)
        except Exception as exc:
            show_error(self, f"{label}: {exc}")
