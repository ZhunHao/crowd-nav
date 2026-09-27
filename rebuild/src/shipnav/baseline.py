from pathlib import Path
import configparser
import json
import math
import shutil
from unittest.mock import patch
import torch
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot


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
    policy.get_model().load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    policy.get_model().eval()
    policy.set_phase('test')
    return policy


def build_baseline(model_dir: Path, seed: int = 0):
    config = configparser.RawConfigParser()
    if not config.read(model_dir / 'env.config'):
        raise FileNotFoundError(model_dir / 'env.config')
    env = CrowdSim()
    env.configure(config)
    policy = load_policy(model_dir)
    robot = Robot(config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    policy.set_env(env)
    env.curr_post = [env.robot_initx, env.robot_inity]
    env.local_goal = [env.robot_goalx, env.robot_goaly]
    observation = env.reset('test', seed)
    return env, policy, observation


def run_baseline(model_dir: Path, output: Path, seed: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    env, policy, observation = build_baseline(model_dir, seed)
    info = None
    trace = []
    with torch.inference_mode():
        for _ in range(math.ceil(env.time_limit / env.time_step) + 1):
            action = env.robot.act(observation)
            trace.append({'t': float(env.global_time),
                          'robot': vars(env.robot.get_full_state()),
                          'humans': [vars(h.get_observable_state()) for h in env.humans],
                          'action': list(action)})
            observation, reward, done, info = env.step(action)
            trace[-1].update(reward=float(reward), done=bool(done), info=type(info).__name__)
            if done:
                break
        else:
            raise RuntimeError('Original simulator failed to terminate')
    result = {'seed': seed, 'status': type(info).__name__,
              'simulation_time': env.global_time, 'frames': len(env.states),
              'time_step': env.time_step, 'policy': policy.name,
              'query_env': policy.query_env,
              'terminal_state': vars(env.robot.get_full_state())}
    (output / 'baseline.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (output / 'trace.json').write_text(json.dumps(trace, indent=2, allow_nan=False))
    render_video(env, output / 'baseline.mp4')
    return result


def render_video(env, output: Path):
    """Use the host FFmpeg without altering the original renderer or its timing."""
    from matplotlib import animation, rc_context
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise FileNotFoundError('FFmpeg executable not found on PATH')
    with rc_context(), patch.object(animation.FFMpegWriter, 'bin_path', return_value=executable):
        env.render('video', None, str(output))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/baseline'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(run_baseline(args.model_dir, args.output, args.seed))
