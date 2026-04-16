import shutil
import pytest


@pytest.mark.unit
def test_render_uses_autodetected_ffmpeg(monkeypatch):
    """render() should honour shutil.which('ffmpeg'), not a hardcoded /usr/bin path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: F401 — import triggers module-level side effects

    fake_path = "/tmp/fake-ffmpeg-for-test"
    monkeypatch.setattr(shutil, "which", lambda name: fake_path if name == "ffmpeg" else None)

    # Reload the module so the path re-evaluates under the monkeypatch.
    import importlib
    import crowd_sim.envs.crowd_sim as mod
    importlib.reload(mod)

    assert plt.rcParams["animation.ffmpeg_path"] == fake_path, (
        "render() must read ffmpeg path from shutil.which, not a hardcoded /usr/bin/ffmpeg"
    )
