"""Toolbar factory - 7 actions matching slide 14 verbatim."""

from __future__ import annotations

from PyQt5.QtWidgets import QToolBar


BUTTON_LABELS = (
    "Load static map",
    "Load DRL model",
    "Set start",
    "Set goal",
    "Add static obstacles",
    "Visualize",
    "Export",
)


def build_toolbar(parent, handlers: dict) -> QToolBar:
    tb = QToolBar("Main", parent)
    for label in BUTTON_LABELS:
        action = tb.addAction(label)
        if label not in handlers:
            raise ValueError(f"toolbar handler missing for label {label!r}")
        handler = handlers[label]
        action.triggered.connect(handler)
    return tb
