"""Inspect every retained evaluation trace and reconcile summaries (Task10)."""
import argparse
from collections import Counter
import csv
import json
import subprocess
from pathlib import Path
from shipnav.evaluate import summarize, VARIANTS
from shipnav.export import default_speedup, video_frames


def audit(directory, expected):
    files=sorted(directory.glob('*.json'))
    assert len(files)==expected,(directory,len(files),expected)
    runs={p.name:json.loads(p.read_text()) for p in files}
    rows=list(csv.DictReader((directory/'summary.csv').open()))
    outcomes=Counter(r['status'] for r in runs.values())
    for row in rows:
        selected=[r for n,r in runs.items() if n.startswith(f"{row['variant']}-n{row['traffic_count']}-")]
        computed=summarize(selected)
        for key,value in computed.items():
            actual=row[key]
            assert (actual=='' if value is None else abs(float(actual)-value)<1e-8),(key,actual,value)
    paired=0
    for seed in range(2 if expected==8 else 10):
        for count in ([5] if expected==8 else [5,8,12,15]):
            group=[runs[f'{name}-n{count}-seed{seed}.json'] for name,_,_ in VARIANTS]
            traces=[r for r in group if r['status']!='error']
            if traces:
                assert all(r['traffic_definitions']==traces[0]['traffic_definitions'] for r in traces)
                paired+=1
    details=[]
    for name,run in runs.items():
        if run['status']=='error':
            details.append(dict(file=name,status='error',error=run['error']))
            continue
        frames=run['frames'];ds=run['diagnostics']
        assert len(frames)==len(ds)+1
        assert frames[0]['t']==0
        assert abs(frames[-1]['t']-run['elapsed'])<1e-8
        assert all(b['t']>a['t'] for a,b in zip(frames,frames[1:]))
        assert len(run['traffic_definitions'])==run['settings']['actual_traffic']
        details.append(dict(file=name,status=run['status'],frames=len(frames),elapsed_s=run['elapsed'],scenario_hash=run['scenario_hash'],first_inference_ms=run['inference_ms'][0] if run['inference_ms'] else None))
    return dict(directory=str(directory),expected=expected,traces=len(files),summary_rows=len(rows),outcomes=dict(outcomes),paired_scenarios=paired,all_traces=details,notes=['Every failure retained; no seeds removed or counts reduced.','Latencies include cold first inference and concurrent acceptance work; descriptive only.'])

def audit_exports(directory):
    identity=json.loads((directory/'identity-timing.json').read_text())
    rows=list(csv.DictReader((directory/'identity-timing.csv').open()))
    assert len(rows)==len(identity['frames'])==80
    for row,frame in zip(rows,identity['frames']):
        assert abs(float(row['time_s'])-frame['t'])<1e-8
        assert all(abs(float(row[key])-frame['position'][i])<1e-8
                   for i,key in enumerate(('x_m','y_m')))
    assert rows[-1]['nominal_vx']==''
    videos=[]
    for name in ('identity-timing','ubin-sarl','ubin-mpc'):
        run=json.loads((directory/f'{name}.json').read_text())
        indices,fps=video_frames(run,default_speedup(run))
        assert indices[0]==0 and indices[-1]==len(run['frames'])-1
        result=subprocess.run(['ffprobe','-v','error','-count_frames',
            '-show_entries','stream=codec_name,nb_read_frames,r_frame_rate,duration',
            '-of','json',str(directory/f'{name}.mp4')],capture_output=True,text=True,check=True)
        stream=json.loads(result.stdout)['streams'][0]
        rate=stream['r_frame_rate'].split('/')
        assert stream['codec_name']=='h264'
        assert int(stream['nb_read_frames'])==len(indices)
        assert abs(int(rate[0])/int(rate[1])-fps)<1e-8
        videos.append(dict(name=name,recorded_frames=len(run['frames']),
                           exported_frames=len(indices),speedup=default_speedup(run),**stream))
    timestamps=json.loads(subprocess.run(['ffprobe','-v','error','-show_frames',
        '-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time',
        '-of','json',str(directory/'identity-timing.mp4')],
        capture_output=True,text=True,check=True).stdout)['frames']
    assert len(timestamps)==80
    assert all(abs(float(p['best_effort_timestamp_time'])-f['t'])<1e-8
               for p,f in zip(timestamps,identity['frames']))
    assert float(timestamps[40]['best_effort_timestamp_time'])==10
    return dict(videos=videos,csv_rows=len(rows),identity_initial_pts_s=0,
                identity_ten_second_frame=40,identity_ten_second_pts_s=10,
                identity_terminal_pts_s=float(timestamps[-1]['best_effort_timestamp_time']),
                all_identity_frame_pts_match_trace=True,terminal_action_empty=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--expected',type=int,choices=(8,160))
    parser.add_argument('--exports',action='store_true')
    args=parser.parse_args()
    if args.exports:
        result=audit_exports(args.directory)
        path=args.directory/'exports-audit.json'
    else:
        if args.expected is None:parser.error('--expected is required for smoke traces')
        result=audit(args.directory,args.expected)
        path=args.directory.parent/f'{args.directory.name}-audit.json'
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='all_traces'},indent=2))
