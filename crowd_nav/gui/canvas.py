"""matplotlib FigureCanvas wrapper with click->world coord signal (WP-5)."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SimCanvas(FigureCanvasQTAgg):
    clicked = pyqtSignal(float, float)  # world x, world y

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
        parent=None,
    ):
        self._fig = Figure(figsize=(6, 6))
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlim(world_bounds[0], world_bounds[1])
        self._ax.set_ylim(world_bounds[2], world_bounds[3])
        self._ax.set_aspect("equal")
        self._ax.set_xlabel("x (m)")
        self._ax.set_ylabel("y (m)")
        super().__init__(self._fig)
        if parent is not None:
            self.setParent(parent)
        self.mpl_connect("button_press_event", self._on_button_press)
        self._overlay: list = []

    @property
    def ax(self):
        return self._ax

    def _on_button_press(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:  # left-click only
            return
        self.clicked.emit(float(event.xdata), float(event.ydata))

    def draw_static_map(self, static_map) -> None:
        """Render occupancy as grey patches. Called from MainWindow after load_map."""
        from matplotlib.patches import Rectangle, Circle

        # Clear the frame overlay first so later push_frame doesn't reference dead artists.
        for artist in self._overlay:
            artist.remove()
        self._overlay = []
        for patch in list(self._ax.patches):
            patch.remove()
        for obs in static_map.obstacles:
            if obs.kind == "rect":
                self._ax.add_patch(
                    Rectangle(
                        (obs.cx - obs.w / 2.0, obs.cy - obs.h / 2.0),
                        obs.w, obs.h, facecolor="#888", edgecolor="none",
                    )
                )
            elif obs.kind == "circle":
                self._ax.add_patch(
                    Circle((obs.cx, obs.cy), obs.r, facecolor="#888", edgecolor="none")
                )
        self.draw_idle()

    def push_frame(self, robot_xy, humans_xy, waypoints) -> None:
        """Update the overlay for a single rollout step."""
        from matplotlib.patches import Circle

        for artist in self._overlay:
            artist.remove()
        overlay = []
        overlay.append(
            self._ax.add_patch(Circle(robot_xy, 0.5, facecolor="gold", edgecolor="k"))
        )
        for hxy in humans_xy:
            overlay.append(
                self._ax.add_patch(Circle(hxy, 0.3, facecolor="white", edgecolor="k"))
            )
        for wx, wy in waypoints:
            (pt,) = self._ax.plot([wx], [wy], "r*", markersize=10)
            overlay.append(pt)
        self._overlay = overlay
        self.draw_idle()
