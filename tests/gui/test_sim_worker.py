"""SimWorker: emits frame_ready per step, episode_done at end."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.gui


def test_worker_emits_frames_and_done(qtbot) -> None:
    from crowd_nav.gui.workers.sim_worker import SimWorker

    fake_controller = MagicMock()
    fake_controller.run_episode.side_effect = lambda frame_callback=None: [
        _and_push(frame_callback, i) for i in range(3)
    ]

    worker = SimWorker(controller=fake_controller)
    frames: list[tuple[int, dict]] = []
    finished: list[dict] = []
    worker.frame_ready.connect(lambda i, s: frames.append((i, s)))
    worker.episode_done.connect(finished.append)

    with qtbot.waitSignal(worker.episode_done, timeout=5000):
        worker.start()

    assert [i for i, _ in frames] == [0, 1, 2]
    assert finished and "num_frames" in finished[0]
    assert finished[0]["num_frames"] == 3


def _and_push(cb, i: int) -> dict:
    step = {
        "t": 0.25 * i,
        "robot": (0.0, 0.0, 0.0, 0.0),
        "humans": [],
        "waypoint_idx": 0,
        "reward": 0.0,
    }
    if cb is not None:
        cb(i, step)
    return step
