"""Offline route previews and paired planner measurements. No navigation certification."""
import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import platform
import sys
from time import perf_counter
import tracemalloc

import numpy as np

from shipnav.map_import import sha256
from shipnav.maps import LocalFrame, SeaMap, canonical_json
from shipnav.planning import NoPath, PlanningLimit, InvalidRoute, plan


def execute(sea, start, goal, *, planner, resolution, radius, margin, timeout=30., max_nodes=250_000):
    if any(not isfinite(v) or v < 0 for v in (radius, margin)):
        raise ValueError('Radius and margin must be finite and nonnegative')
    begin = perf_counter()
    result = {'planner': planner, 'start': list(start), 'goal': list(goal), 'radius_m': radius,
              'margin_m': margin, 'resolution_m': resolution, 'mode': 'coastline-only',
              'timeout_s': timeout, 'max_nodes': max_nodes}
    try:
        route = plan(sea, start, goal, planner=planner, resolution=resolution,
                     clearance=radius+margin, timeout=timeout, max_nodes=max_nodes)
        result.update(status='success', points=route.points, stats=route.stats)
        result['stats']['minimum_body_clearance_m'] = route.stats['minimum_centre_clearance_m']-radius
    except NoPath as error:
        result.update(status='no_path', error=str(error), points=[], stats=getattr(error,'stats',{}))
    except PlanningLimit as error:
        result.update(status='budget_exceeded', error=str(error), points=[], stats=getattr(error,'stats',{}))
    except InvalidRoute as error:
        result.update(status='invalid_route', error=str(error), points=[], stats=getattr(error,'stats',{}))
    result['elapsed_s'] = perf_counter()-begin
    return result


def render(sea, result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from shapely.geometry.polygon import orient
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.88)
    ax.set_facecolor('#e5f2f7')
    for polygon in sea.land:
        polygon = orient(polygon, sign=1.)
        vertices, codes = [], []
        for ring in [polygon.exterior, *polygon.interiors]:
            points = list(ring.coords)
            vertices.extend(points)
            codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(points)-2)+[PlotPath.CLOSEPOLY])
        ax.add_patch(PathPatch(PlotPath(vertices,codes), facecolor='#cbd9c1', edgecolor='#5c7360', linewidth=.6))
    if result['points']:
        xs,ys = zip(*result['points'])
        ax.plot(xs,ys,'o-',color='#d95f02',lw=2,ms=3,label=f"{result['planner']} route")
    for key,marker,color in [('start','o','#157f3b'),('goal','*','#8636ab')]:
        x,y=result[key]
        ax.scatter([x],[y],marker=marker,s=110,color=color,zorder=5,label=key.capitalize())
    x0,y0,x1,y1 = sea.bounds
    ax.set(xlim=(x0,x1),ylim=(y0,y1),aspect='equal',xlabel='East from map origin (m)',ylabel='North from map origin (m)')
    metadata=sea.to_dict()['metadata']
    title='Real coastline • offline route' if metadata.get('geographic_bbox') else 'Polygon route validation'
    if 'title' in result:
        title=result['title']
    ax.set_title(title+'\nCoastline-only simulation — depth and restrictions unknown',loc='left',fontsize=13)
    ax.grid(alpha=.16)
    ax.legend(loc='upper right')
    length=10**np.floor(np.log10((x1-x0)/5))
    sx,sy=x0+.06*(x1-x0),y0+.06*(y1-y0)
    ax.plot([sx,sx+length],[sy,sy],color='#233747',lw=3)
    ax.text(sx,sy+.015*(y1-y0),f'{length:g} m',fontsize=9)
    stats=result.get('stats',{})
    label=f"{result['status']} | radius {result['radius_m']:g} m + margin {result['margin_m']:g} m | grid {result['resolution_m']:g} m"
    if 'length_m' in stats:
        label+=f" | route {stats['length_m']/1000:.2f} km"
    fig.text(.04,.035,label+'\n© OpenStreetMap contributors • ODbL • source metadata accompanies map',fontsize=8,color='#34495e')
    fig.savefig(output,dpi=160,bbox_inches='tight')
    plt.close(fig)


def benchmark(manifest_path, output, repeats):
    if repeats <= 0:
        raise ValueError('repeats must be positive')
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema') != 1 or not manifest.get('cases'):
        raise ValueError('Invalid benchmark manifest')
    runs=[]
    for case in manifest['cases']:
        path=manifest_path.parent/case['map']
        map_hash=sha256(path)
        if case.get('map_sha256') and case['map_sha256'] != map_hash:
            raise ValueError(f"Frozen map changed for {case['id']}")
        sea=SeaMap.load(path)
        start,goal=case['start'],case['goal']
        coordinates=case.get('coordinates','local')
        if coordinates not in ('lonlat','local'):
            raise ValueError(f"Unknown coordinates '{coordinates}'; expected lonlat or local")
        if coordinates == 'lonlat':
            frame=LocalFrame(*sea.to_dict()['metadata']['frame']['origin_lonlat'])
            start,goal=frame.project(*start),frame.project(*goal)
        settings={k:case[k] for k in ('resolution','radius','margin')}
        settings['timeout']=case.get('timeout',30.)
        settings['max_nodes']=case.get('max_nodes',250_000)
        for repeat in range(repeats):
            # Alternate ordering to reduce systematic cache/order effects.
            for planner in (('astar','theta') if repeat%2 == 0 else ('theta','astar')):
                run=execute(sea,start,goal,planner=planner,**settings)
                run.update(case=case['id'],split=case['split'],repeat=repeat,map_sha256=map_hash)
                runs.append(run)
        # Memory measured separately; tracing is not allowed to distort latency runs.
        for planner in ('astar','theta'):
            tracemalloc.start()
            measured=execute(sea,start,goal,planner=planner,**settings)
            _,peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            for run in runs:
                if run['case']==case['id'] and run['planner']==planner:
                    run['python_peak_allocated_bytes']=peak
                    run['memory_probe_status']=measured['status']
    summary={}
    for split in sorted({r['split'] for r in runs}):
        for planner in ('astar','theta'):
            selected=[r for r in runs if r['planner']==planner and r['split']==split]
            times=[r['elapsed_s'] for r in selected]
            successes=[r for r in selected if r['status']=='success']
            summary[f'{split}/{planner}']={'runs':len(selected),'outcomes':dict(Counter(r['status'] for r in selected)),
                'success_rate':len(successes)/len(selected),
                **{f'latency_p{q}_s':float(np.percentile(times,q)) for q in (50,95,99)},
                'mean_success_length_m':float(np.mean([r['stats']['length_m'] for r in successes])) if successes else None}
    data={'schema':1,'manifest_sha256':sha256(manifest_path),'repeats':repeats,
          'hardware':{'platform':platform.platform(),'machine':platform.machine(),'python':sys.version},
          'measurement_notes':['wall latency includes search, smoothing and route validation; map loading excluded',
             'memory is a separate tracemalloc pass: Python allocations only, not native GEOS/RSS',
             'small frozen smoke benchmark; latency quantiles are descriptive, not a SOTA claim'],
          'runs':runs,'summary':summary}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(canonical_json(data))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    route=sub.add_parser('route')
    route.add_argument('map',type=Path)
    route.add_argument('--start',nargs=2,type=float,required=True)
    route.add_argument('--goal',nargs=2,type=float,required=True)
    route.add_argument('--lonlat',action='store_true')
    route.add_argument('--planner',choices=['astar','theta'],default='astar')
    route.add_argument('--radius',type=float,default=5.)
    route.add_argument('--margin',type=float,default=10.)
    route.add_argument('--resolution',type=float,default=50.)
    route.add_argument('--timeout',type=float,default=30.)
    route.add_argument('--output',type=Path,required=True,help='Output prefix for JSON and PNG')
    bench=sub.add_parser('benchmark')
    bench.add_argument('manifest',type=Path)
    bench.add_argument('--repeats',type=int,default=3)
    bench.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='benchmark':
            data=benchmark(args.manifest,args.output,args.repeats)
            print(json.dumps(data['summary'],indent=2))
            return
        sea=SeaMap.load(args.map)
        start,goal=args.start,args.goal
        if args.lonlat:
            frame=LocalFrame(*sea.to_dict()['metadata']['frame']['origin_lonlat'])
            start,goal=frame.project(*start),frame.project(*goal)
        result=execute(sea,start,goal,planner=args.planner,resolution=args.resolution,
                       radius=args.radius,margin=args.margin,timeout=args.timeout)
        result['map_sha256']=sha256(args.map)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.json').write_text(canonical_json(result))
        render(sea,result,args.output.with_suffix('.png'))
        print(json.dumps(result.get('stats',result),indent=2))
        if result['status']!='success':
            raise SystemExit(2)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,f'{error}\n')


if __name__=='__main__':
    main()
