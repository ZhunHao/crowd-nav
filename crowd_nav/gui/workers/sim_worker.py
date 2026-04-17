"""Background QThread driving the rollout (WP-5).

Keeps the Qt UI thread responsive (NF1). Signals:

* ``frame_ready(int, dict)`` - step index + state dict per rollout step.
* ``episode_done(dict)``     - summary at end: ``num_frames``, ``states``.
* ``failed(str)``            - uncaught exception message; UI shows dialog.
"""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import QThread, pyqtSignal


class SimWorker(QThread):
    frame_ready = pyqtSignal(int, dict)
    episode_done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, controller: Any, parent=None):
        super().__init__(parent)
        self.controller = controller

    def run(self) -> None:  # called on the worker thread
        try:
            states = self.controller.run_episode(frame_callback=self._emit_frame)
        except Exception as exc:  # surface to UI thread
            self.failed.emit(str(exc))
            return
        self.episode_done.emit({"num_frames": len(states), "states": states})

    def _emit_frame(self, i: int, step: dict) -> None:
        self.frame_ready.emit(i, step)
