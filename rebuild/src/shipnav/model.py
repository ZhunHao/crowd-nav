from pathlib import Path
import configparser
import torch
from shipnav.compat.crowd_nav.policy.sarl import SARL


def load_policy(model_dir: Path) -> SARL:
    config = configparser.RawConfigParser()
    config_path = model_dir / 'policy.config'
    weights = model_dir / 'rl_model.pth'
    for path in (config_path, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    config.read(config_path)
    policy = SARL()
    policy.configure(config)
    if policy.kinematics != 'holonomic':
        raise ValueError('This rebuild requires a holonomic checkpoint')
    policy.set_device(torch.device('cpu'))
    state_dict = torch.load(weights, map_location='cpu', weights_only=True)
    for name, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite checkpoint tensor: {name}')
    policy.get_model().load_state_dict(state_dict, strict=True)
    policy.get_model().eval()
    policy.set_phase('test')
    return policy
