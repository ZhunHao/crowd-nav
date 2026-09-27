import json
from pathlib import Path
import subprocess
import sys
from shipnav.maps import SeaMap


def test_offline_cli_creates_valid_route_and_preview(tmp_path):
    sea = SeaMap((0, 0, 24, 24), [(9, 5, 14, 18)])
    source = tmp_path/'map.json'
    sea.save(source)
    run = subprocess.run([sys.executable, '-m', 'shipnav.map_demo', 'route', str(source),
        '--start', '2', '12', '--goal', '22', '12', '--radius', '.5', '--margin', '.2',
        '--resolution', '1', '--output', str(tmp_path/'route')], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads((tmp_path/'route.json').read_text())
    assert result['status'] == 'success'
    assert all(sea.clear(a,b,.7) for a,b in zip(result['points'],result['points'][1:]))
    assert (tmp_path/'route.png').read_bytes().startswith(b'\x89PNG')


def test_cli_blocked_endpoint_is_failure_with_artifact(tmp_path):
    source = tmp_path/'map.json'
    SeaMap((0,0,10,10), [(4,4,6,6)]).save(source)
    run = subprocess.run([sys.executable, '-m', 'shipnav.map_demo', 'route', str(source),
        '--start','5','5','--goal','8','8','--output',str(tmp_path/'failed')], capture_output=True, text=True)
    assert run.returncode != 0
    assert json.loads((tmp_path/'failed.json').read_text())['status'] == 'no_path'


def test_benchmark_retains_failures_and_reports_paired_outcomes(tmp_path):
    source = tmp_path/'map.json'
    SeaMap((0,0,10,10), [(4,0,6,10)]).save(source)
    cases = {'schema':1, 'cases':[
        {'id':'blocked','split':'test','map':'map.json','start':[2,5],'goal':[8,5], 'resolution':1, 'radius':.5, 'margin':.2},
        {'id':'clear','split':'dev','map':'map.json','start':[2,2],'goal':[2,8], 'resolution':1, 'radius':.5, 'margin':.2}]}
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(cases))
    run = subprocess.run([sys.executable,'-m','shipnav.map_demo','benchmark',str(manifest),
        '--repeats','2','--output',str(tmp_path/'benchmark.json')],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    data = json.loads((tmp_path/'benchmark.json').read_text())
    assert len(data['runs']) == 8
    assert sum(r['status']=='no_path' for r in data['runs']) == 4
    assert all('latency_p99_s' in x for x in data['summary'].values())


def test_benchmark_accepts_explicit_local_metre_coordinates(tmp_path):
    source = tmp_path/'map.json'
    SeaMap((0,0,10,10), [(4,0,6,10)]).save(source)
    cases = {'schema':1, 'cases':[
        {'id':'local','split':'dev','map':'map.json','coordinates':'local',
         'start':[2,2],'goal':[2,8], 'resolution':1, 'radius':.5, 'margin':.2}]}
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(cases))
    run = subprocess.run([sys.executable,'-m','shipnav.map_demo','benchmark',str(manifest),
        '--repeats','1','--output',str(tmp_path/'benchmark.json')],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    data = json.loads((tmp_path/'benchmark.json').read_text())
    assert all(r['status'] == 'success' for r in data['runs'])


def test_benchmark_rejects_unknown_coordinates_value(tmp_path):
    source = tmp_path/'map.json'
    SeaMap((0,0,10,10), [(4,0,6,10)]).save(source)
    cases = {'schema':1, 'cases':[
        {'id':'bad','split':'dev','map':'map.json','coordinates':'utm',
         'start':[2,2],'goal':[2,8], 'resolution':1, 'radius':.5, 'margin':.2}]}
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps(cases))
    run = subprocess.run([sys.executable,'-m','shipnav.map_demo','benchmark',str(manifest),
        '--repeats','1','--output',str(tmp_path/'benchmark.json')],capture_output=True,text=True)
    assert run.returncode != 0
    assert 'coordinates' in run.stderr


def test_rejected_final_route_is_preserved_as_failure():
    from shipnav.map_demo import execute
    class RejectFinalValidation:
        calls=0
        def clear(self,a,b,clearance):
            self.calls+=1
            return self.calls <= 3
    result=execute(RejectFinalValidation(),(2,2),(8,8),planner='astar',resolution=1,radius=.5,margin=.2)
    assert result['status']=='invalid_route'
    assert result['points']==[]
