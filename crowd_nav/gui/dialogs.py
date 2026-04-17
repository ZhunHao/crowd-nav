"""QMessageBox wrappers - isolate dialog calls so tests can monkeypatch them."""

from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox, QWidget


def show_error(parent: QWidget, message: str) -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle("Error")
    box.setText(message)
    box.exec_()
