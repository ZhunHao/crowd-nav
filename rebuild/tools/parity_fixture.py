from pathlib import Path
import argparse, json
import torch


def capture(model_dir):
    from shipnav.model import load_policy
    from shipnav.compat.crowd_sim.envs.utils import state as states
    policy = load_policy(model_dir)
    policy.query_env = False
    policy.time_step = .25
    rows = []
    for n in (1, 4, 12):
        for vx in (0., .7):
            calls = []
            def hook(module, args, result):
                calls.append({'input': args[0].detach().cpu().tolist(),
                              'value': result.detach().cpu().tolist()})
            handle = policy.model.register_forward_hook(hook)
            own = states.FullState(2., 2., vx, 0., .5, 12., 9., 1., 0.)
            others = [states.ObservableState(4.+i, 4.+i%3, -.2, .1, .6) for i in range(n)]
            with torch.inference_mode():
                action = policy.predict(states.JointState(own, others))
            handle.remove()
            rows.append({'n': n, 'vx': vx, 'action': list(action), 'calls': calls})
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = capture(Path('../CrowdNav-20250813-DIP/crowd_nav/data/output_trained'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, allow_nan=False))
