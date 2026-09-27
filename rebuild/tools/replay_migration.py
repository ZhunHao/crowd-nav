"""Capture complete bounded controller traces on shared frozen scenarios."""
import argparse
import json
from pathlib import Path
from expanded_parity import ROOT, setup
from shipnav.replay import replay

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modern', action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    scenarios = json.loads((ROOT/'migration/expanded-inputs.json').read_text())['rows'][:3]
    results = []
    for scenario in scenarios:
        policy,states = setup(args.modern)
        results.append(replay(scenario,policy=policy,states=states))
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False))
