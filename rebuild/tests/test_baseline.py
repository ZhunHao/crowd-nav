from pathlib import Path
import math
import pytest
import torch
from shipnav.baseline import build_baseline, load_policy

MODEL = Path(__file__).resolve().parents[2] / 'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def test_missing_model_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path)


@pytest.mark.integration
def test_checkpoint_runs_one_original_step():
    env, policy, observation = build_baseline(MODEL)
    before = env.global_time
    with torch.inference_mode():
        action = env.robot.act(observation)
    assert math.isfinite(action.vx) and math.isfinite(action.vy)
    observation, reward, done, info = env.step(action)
    assert env.global_time == pytest.approx(before + env.time_step)
    assert len(observation) == env.human_num
    assert len(env.states) == 1
    assert next(policy.get_model().parameters()).device.type == 'cpu'


def test_renderer_resolves_ffmpeg_and_restores_on_failure(monkeypatch, tmp_path):
    from matplotlib import animation, rcParams
    from shipnav.baseline import render_video
    monkeypatch.setattr('shipnav.baseline.shutil.which', lambda name: '/selected/ffmpeg')
    original = animation.FFMpegWriter.bin_path()
    original_setting = rcParams['animation.ffmpeg_path']
    class BrokenRenderer:
        def render(self, *args):
            rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            assert animation.FFMpegWriter.bin_path() == '/selected/ffmpeg'
            raise RuntimeError('renderer failed')
    with pytest.raises(RuntimeError, match='renderer failed'):
        render_video(BrokenRenderer(), tmp_path / 'video.mp4')
    assert animation.FFMpegWriter.bin_path() == original
    assert rcParams['animation.ffmpeg_path'] == original_setting


def test_renderer_missing_ffmpeg_is_explicit(monkeypatch, tmp_path):
    from shipnav.baseline import render_video
    monkeypatch.setattr('shipnav.baseline.shutil.which', lambda name: None)
    with pytest.raises(FileNotFoundError, match='FFmpeg'):
        render_video(None, tmp_path / 'video.mp4')
