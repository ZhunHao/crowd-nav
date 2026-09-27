from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

MODEL = Path(__file__).resolve().parents[2] / 'CrowdNav-20250813-DIP/crowd_nav/data/output_trained'


def test_import_without_original_gym_or_native():
    result = subprocess.run([sys.executable, '-c', "import sys; import shipnav.model; from shipnav.compat.crowd_sim.envs.policy.orca import ORCA; assert not any(n == 'gym' or n == 'rvo2' or n.startswith('crowd_sim') or n.startswith('crowd_nav') for n in sys.modules)"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_checkpoint_load_cpu_strict_eval():
    from shipnav.model import load_policy
    policy = load_policy(MODEL)
    assert policy.phase == 'test'
    assert policy.device == torch.device('cpu')
    assert not policy.get_model().training
    weights = torch.load(MODEL / 'rl_model.pth', weights_only=True, map_location='cpu')
    for name, value in policy.get_model().state_dict().items():
        assert torch.equal(value, weights[name])


@pytest.mark.parametrize('filename', ['policy.config', 'rl_model.pth'])
def test_missing_files(tmp_path, filename):
    from shipnav.model import load_policy
    for name in ('policy.config', 'rl_model.pth'):
        if name != filename:
            shutil.copyfile(MODEL / name, tmp_path / name)
    with pytest.raises(FileNotFoundError, match=filename):
        load_policy(tmp_path)


@pytest.mark.parametrize('failure', ['nan', 'inf', 'missing_key', 'shape', 'unicycle'])
def test_reject_invalid_checkpoint(tmp_path, failure):
    from shipnav.model import load_policy
    config = (MODEL / 'policy.config').read_text()
    if failure == 'unicycle':
        config = config.replace('holonomic', 'unicycle')
    (tmp_path / 'policy.config').write_text(config)
    weights = torch.load(MODEL / 'rl_model.pth', weights_only=True, map_location='cpu')
    first = next(iter(weights))
    if failure in ('nan', 'inf'):
        weights[first].flatten()[0] = float(failure)
    elif failure == 'missing_key':
        del weights[first]
    elif failure == 'shape':
        weights[first] = torch.zeros(1)
    torch.save(weights, tmp_path / 'rl_model.pth')
    with pytest.raises((ValueError, RuntimeError)):
        load_policy(tmp_path)

@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_reject_nonfinite_network_output(bad):
    from shipnav.model import load_policy
    policy = load_policy(MODEL)
    with torch.no_grad():
        policy.model.mlp3[-1].bias.fill_(bad)
    with pytest.raises(ValueError, match='Non-finite policy output'):
        policy.model(torch.ones(1, 1, 13))


def test_empty_neighbour_adapter():
    from shipnav.model import load_policy
    from shipnav.compat.crowd_sim.envs.utils.state import FullState, JointState
    policy = load_policy(MODEL)
    policy.time_step = .25
    own = FullState(0, 0, 0, 0, .1, 3, 4, 1, 0)
    assert policy.predict(JointState(own, [])) == pytest.approx((.6, .8))
    own.gx, own.gy = 0, 0
    assert policy.predict(JointState(own, [])) == (0, 0)
