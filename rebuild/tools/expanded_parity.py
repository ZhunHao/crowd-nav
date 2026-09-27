"""Frozen recorded/synthetic inputs and full candidate inference evidence."""
import argparse
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def setup():
    from shipnav.model import load_policy
    from shipnav.compat.crowd_sim.envs.utils import state as states
    policy = load_policy(ROOT.parent/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
    policy.query_env = False
    policy.time_step = .25
    return policy, states


def evaluate(policy, states, row, hooks=True):
    calls = []
    def hook(module, args, result):
        calls.append({'input':args[0].detach().cpu().tolist(), 'value':result.detach().cpu().tolist()})
    handle = policy.model.register_forward_hook(hook) if hooks else None
    try:
        own = states.FullState(*row['own'])
        humans = [states.ObservableState(*h) for h in row['others']]
        with torch.inference_mode():
            action = policy.predict(states.JointState(own, humans))
        values = list(policy.action_values)
        ordered = sorted(values, reverse=True)
        return {'id':row['id'], 'action':list(action), 'calls':calls,
                'candidate_values':values, 'margin':ordered[0]-ordered[1]}
    finally:
        if handle: handle.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixtures = ROOT/'migration/expanded-inputs.json'
    policy, states = setup()
    rows = json.loads(fixtures.read_text())['rows']
    args.output.write_text(json.dumps([evaluate(policy, states, r) for r in rows], allow_nan=False))
