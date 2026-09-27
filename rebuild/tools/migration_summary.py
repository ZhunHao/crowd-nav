"""Summarize action margins and complete controller differences without waiver."""
from pathlib import Path
import json
import numpy as np
from reference_manifest import manifest

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    path = ROOT/'migration'
    old = json.loads((path/'expanded-legacy.json').read_text())
    new = json.loads((path/'expanded-modern.json').read_text())
    margins = [{'id':a['id'],'legacy_margin':a['margin'],'modern_margin':b['margin'],
                'action_equal_1e_7':bool(np.allclose(a['action'],b['action'],rtol=0,atol=1e-7)),
                'max_candidate_value_delta':float(np.max(np.abs(np.array(a['candidate_values'])-b['candidate_values'])))} for a,b in zip(old,new)]
    old_traces = json.loads((path/'replay-legacy.json').read_text())
    new_traces = json.loads((path/'replay-modern.json').read_text())
    traces = [{'id':a['id'],'legacy_status':a['status'],'modern_status':b['status'],
               'legacy_steps':len(a['trace']),'modern_steps':len(b['trace']),
               'exact_trace_equal':a==b} for a,b in zip(old_traces,new_traces)]
    before = json.loads((path/'source-before-m3.json').read_text())
    after = manifest(ROOT.parent/'CrowdNav-20250813-DIP')
    (path/'source-after-m3.json').write_text(json.dumps(after,indent=2))
    if before != after: raise ValueError('Original source changed')
    result = {'expanded':margins,'controller':traces,'source_unchanged':True,
              'scope':'CPU. Controller uses fixed constant-velocity neighbours and a 64-step bound; it is not original simulator trajectory reproduction. Timeout is a bounded failure, not success.'}
    (path/'m3-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__ == '__main__': summarize()
