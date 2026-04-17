"""CLI entry: ``python -m crowd_nav.gui`` launches the demo GUI.

Env var ``CROWDNAV_ENV_CONFIG`` overrides the default env.config path so
tests and packaged demos can point at bundled configs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from crowd_nav.gui.controllers.sim_controller import SimController
from crowd_nav.gui.main_window import MainWindow


def build_window() -> MainWindow:
    cfg_str = os.environ.get("CROWDNAV_ENV_CONFIG")
    if cfg_str:
        cfg_path = Path(cfg_str)
    else:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "env.config"
    controller = SimController(env_config_path=cfg_path)
    return MainWindow(controller=controller)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    window = build_window()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
