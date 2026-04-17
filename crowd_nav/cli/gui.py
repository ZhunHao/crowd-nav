"""`crowdnav-gui` — launch the PyQt5 MainWindow without typing `python -m ...`."""

from __future__ import annotations

import crowd_nav.gui.__main__ as _gui_main_mod


def main() -> None:
    _gui_main_mod.main()


if __name__ == "__main__":
    main()
