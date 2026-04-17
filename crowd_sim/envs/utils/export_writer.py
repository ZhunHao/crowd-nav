"""Persist episode telemetry + video (WP-5).

Called once per Visualize run. Produces three artifacts in ``out_dir``:

* ``run.csv``     - per-step telemetry (t, robot+humans state, waypoint, reward)
* ``trajectory.png`` - static overlay of the robot's path + waypoints
* ``run.mp4``     - matplotlib animation (requires ffmpeg on PATH)

No dependency on Qt; callable from SimController or tests.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Sequence, TypedDict

import matplotlib

from matplotlib import animation, pyplot as plt

_FFMPEG = shutil.which("ffmpeg")
if _FFMPEG is not None:
    matplotlib.rcParams["animation.ffmpeg_path"] = _FFMPEG


class _Step(TypedDict):
    t: float
    robot: tuple[float, float, float, float]
    humans: Sequence[tuple[float, float, float, float]]
    waypoint_idx: int
    reward: float


def write_exports(
    out_dir: Path,
    states: Sequence[_Step],
    waypoints: Sequence[tuple[float, float]],
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "run.csv"
    png_path = out_dir / "trajectory.png"
    mp4_path = out_dir / "run.mp4" if _FFMPEG is not None else None

    _write_csv(csv_path, states)
    _write_trajectory_png(png_path, states, waypoints)
    if mp4_path is not None:
        _write_mp4(mp4_path, states, waypoints)

    result: dict[str, Path] = {"csv": csv_path, "png": png_path}
    if mp4_path is not None:
        result["mp4"] = mp4_path
    return result


def _write_csv(path: Path, states: Sequence[_Step]) -> None:
    if not states:
        path.write_text("")
        return
    n_humans = len(states[0]["humans"])
    header = ["t", "robot_x", "robot_y", "robot_vx", "robot_vy"]
    for i in range(n_humans):
        header += [f"h{i}_x", f"h{i}_y", f"h{i}_vx", f"h{i}_vy"]
    header += ["waypoint_idx", "reward"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in states:
            row: list[float | int] = [s["t"], *s["robot"]]
            for h in s["humans"]:
                row += list(h)
            row += [s["waypoint_idx"], s["reward"]]
            w.writerow(row)


def _write_trajectory_png(
    path: Path, states: Sequence[_Step], waypoints: Sequence[tuple[float, float]]
) -> None:
    if not states:
        raise ValueError("write_exports: states is empty — nothing to plot")
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        xs = [s["robot"][0] for s in states]
        ys = [s["robot"][1] for s in states]
        ax.plot(xs, ys, "-", linewidth=1.5, label="robot")
        if waypoints:
            wx = [w[0] for w in waypoints]
            wy = [w[1] for w in waypoints]
            ax.plot(wx, wy, "r*", markersize=12, label="waypoints")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def _write_mp4(
    path: Path, states: Sequence[_Step], waypoints: Sequence[tuple[float, float]]
) -> None:
    if not states:
        raise ValueError("write_exports: states is empty — nothing to animate")
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        ax.set_aspect("equal")
        xs_all = [s["robot"][0] for s in states]
        ys_all = [s["robot"][1] for s in states]
        x_lo = min(xs_all) - 1.0
        x_hi = max(xs_all) + 1.0
        y_lo = min(ys_all) - 1.0
        y_hi = max(ys_all) + 1.0
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        (robot_dot,) = ax.plot([], [], "yo", markersize=14)

        def _init():
            robot_dot.set_data([], [])
            return (robot_dot,)

        def _step(i: int):
            robot_dot.set_data([states[i]["robot"][0]], [states[i]["robot"][1]])
            return (robot_dot,)

        anim = animation.FuncAnimation(
            fig, _step, frames=len(states), init_func=_init, interval=50, blit=True
        )
        writer = animation.writers["ffmpeg"](fps=8, bitrate=1200)
        anim.save(str(path), writer=writer)
    finally:
        plt.close(fig)
