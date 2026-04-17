"""Click-drag static obstacle drawing helper (WP-5).

MainWindow activates this when the user clicks 'Add static obstacles'. Stays
Qt-object so MainWindow can ``connect`` the ``obstacle_created`` signal directly
into ``SimController.add_obstacle``. The canvas proxies mouse events via
``on_press`` / ``on_release``.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from PyQt5.QtCore import QObject, pyqtSignal


class ObstacleEditor(QObject):
    obstacle_created = pyqtSignal(dict)

    def __init__(self, mode: Literal["rect", "circle"] = "rect", parent=None):
        super().__init__(parent)
        self.mode: Literal["rect", "circle"] = mode
        self._press_xy: Optional[tuple[float, float]] = None

    def on_press(self, xy: tuple[float, float]) -> None:
        self._press_xy = xy

    def on_release(self, xy: tuple[float, float]) -> None:
        if self._press_xy is None:
            return
        a = self._press_xy
        self._press_xy = None
        if self.mode == "rect":
            cx = (a[0] + xy[0]) / 2.0
            cy = (a[1] + xy[1]) / 2.0
            w = abs(xy[0] - a[0])
            h = abs(xy[1] - a[1])
            if w <= 0 or h <= 0:
                return
            self.obstacle_created.emit(
                {"type": "rect", "cx": cx, "cy": cy, "w": w, "h": h}
            )
        else:
            r = math.hypot(xy[0] - a[0], xy[1] - a[1])
            if r <= 0:
                return
            self.obstacle_created.emit(
                {"type": "circle", "cx": a[0], "cy": a[1], "r": r}
            )
