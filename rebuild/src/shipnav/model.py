from pathlib import Path
import configparser
import math
import torch
from shipnav.compat.crowd_nav.policy.sarl import SARL


class InferenceSARL(SARL):
    """Keep nonempty SARL semantics; steer directly when no neighbours exist."""

    def predict(self, state):
        if not state.human_states:
            from shipnav.compat.crowd_sim.envs.utils.action import ActionXY
            own = state.self_state
            dx, dy = own.gx - own.px, own.gy - own.py
            distance = math.hypot(dx, dy)
            if distance < own.radius or distance == 0:
                return ActionXY(0, 0)
            if self.time_step is None or self.time_step <= 0:
                raise ValueError('Positive time_step required')
            speed = min(own.v_pref, distance / self.time_step)
            return ActionXY(speed * dx / distance, speed * dy / distance)
        return super().predict(state)


def _check_output(module, args, output):
    if not torch.isfinite(output).all():
        raise ValueError('Non-finite policy output')


def load_policy(model_dir: Path) -> SARL:
    config = configparser.RawConfigParser()
    config_path = model_dir / 'policy.config'
    weights = model_dir / 'rl_model.pth'
    for path in (config_path, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    config.read(config_path)
    policy = InferenceSARL()
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
    policy.get_model().register_forward_hook(_check_output)
    policy.set_phase('test')
    return policy
