"""Frozen recorded/synthetic inputs and full candidate inference evidence."""
import argparse
import importlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OWN = ('px','py','vx','vy','radius','gx','gy','v_pref','theta')
OTHER = OWN[:5]


def setup(modern):
    prefix = 'shipnav.compat.' if modern else ''
    module = importlib.import_module('shipnav.model' if modern else 'shipnav.baseline')
    states = importlib.import_module(prefix+'crowd_sim.envs.utils.state')
    policy = module.load_policy(ROOT.parent/'CrowdNav-20250813-DIP/crowd_nav/data/output_trained')
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


def freeze():
    policy, states = setup(False)
    recorded = []
    for seed in range(3):
        trace = json.loads((ROOT/f'reference/episodes/seed-{seed}-trace.json').read_text())
        for i, frame in enumerate(trace):
            row = {'id':f'episode-{seed}-frame-{i}', 'own':[frame['robot'][k] for k in OWN],
                   'others':[[h[k] for k in OTHER] for h in frame['humans']]}
            # Terminal-at-goal inputs have no candidate search.
            if ((row['own'][0]-row['own'][5])**2+(row['own'][1]-row['own'][6])**2)**.5 < row['own'][4]:
                continue
            row['selection_margin'] = evaluate(policy, states, row, False)['margin']
            recorded.append(row)
    picked = [next(r for r in recorded if r['id']==f'episode-{s}-frame-0') for s in range(3)]
    picked += sorted(recorded, key=lambda r:r['selection_margin'])[:3]
    for n in (1, 2, 8, 24):
        picked.append({'id':f'neighbours-{n}', 'own':[2,2,.7,0,.5,12,9,1,0],
                       'others':[[4+i,4+i%3,-.2,.1,.6] for i in range(n)]})
    return {'selection':'Three episode starts, three minimum legacy candidate margins across all recorded frames, and 1/2/8/24 neighbours; query_env=False.',
            'rows':picked}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modern', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixtures = ROOT/'migration/expanded-inputs.json'
    if args.freeze:
        if args.modern or fixtures.exists(): raise ValueError('Freeze once using legacy before retirement')
        fixtures.write_text(json.dumps(freeze(), indent=2, allow_nan=False))
    policy, states = setup(args.modern)
    rows = json.loads(fixtures.read_text())['rows']
    args.output.write_text(json.dumps([evaluate(policy, states, r) for r in rows], allow_nan=False))
