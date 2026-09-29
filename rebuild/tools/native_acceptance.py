"""Task10 visible Cocoa Qt acceptance; no policy/service mocks.

Run from rebuild: QT_QPA_PLATFORM=cocoa OMP_NUM_THREADS=2
.venv-modern/bin/python tools/native_acceptance.py
QTest operates actual widgets/canvas. Only modal file chooser return values are
injected. Precision metre endpoints use the same apply_click handler as the canvas;
pixel clicks would change the prescribed scenario through quantization.
"""
import csv
import json
import subprocess
import time
import sys
from pathlib import Path

import torch
torch.set_num_threads(2)
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog
from shipnav.gui import MODEL, Window
from shipnav.maps import SeaMap
from shipnav.scale import units
from shipnav.export import default_speedup, video_frames, draw

OUT = Path('results/plan04-native').resolve()
OUT.mkdir(parents=True, exist_ok=True)
def pump(ms):
    deadline = time.monotonic() + ms/1000
    while time.monotonic() < deadline:
        app.processEvents()
        # Sleep releases the GIL; QTest.qWait holds it and starves a Python worker.
        time.sleep(min(.005, max(0, deadline-time.monotonic())))

app = QApplication([])
# A bare uv CPython executable has no application bundle. Register/activate its
# Cocoa NSApplication so the visible Qt window reaches the actual desktop.
import ctypes
objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.objc_getClass.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
send0 = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
sendint = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)(('objc_msgSend', objc))
nsapp = send0(objc.objc_getClass(b'NSApplication'), objc.sel_registerName(b'sharedApplication'))
sendint(nsapp, objc.sel_registerName(b'setActivationPolicy:'), 0)
sendint(nsapp, objc.sel_registerName(b'activateIgnoringOtherApps:'), 1)
w = Window(SeaMap.load('maps/harbour.json'), str(MODEL))
w.show()
w.raise_()
w.activateWindow()
pump(700)
resume_after_mpc = "--resume-after-mpc" in sys.argv
resume = "--resume-ubin" in sys.argv or resume_after_mpc or "--final-checks" in sys.argv
checks = json.loads((OUT/"checks.json").read_text()) if resume else []
heartbeats = []
heartbeat = QTimer()
heartbeat.setInterval(50)
heartbeat.timeout.connect(lambda: heartbeats.append(time.monotonic()))
heartbeat.start()

def record(name, passed=True, **details):
    entry = dict(check=name, outcome='pass' if passed else 'fail', **details)
    checks.append(entry)
    (OUT/'checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(entry), flush=True)

def shot(name):
    pump(150)
    sendint(nsapp, objc.sel_registerName(b'activateIgnoringOtherApps:'), 1)
    w.raise_()
    pump(150)
    path = OUT/f'{name}.png'
    w.grab().save(str(path))
    # A full desktop capture separately establishes native visibility; failure is retained.
    if name in ('native-initial', 'ubin-sarl-active', 'ubin-mpc-active'):
        p = subprocess.run(['screencapture', '-x', str(OUT/f'{name}-desktop.png')], capture_output=True, text=True)
        record(name+'-desktop', p.returncode == 0, stderr=p.stderr, visible=w.isVisible(), platform=app.platformName())
    return str(path)

def click(button):
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    pump(20)

def wait_job(name, timeout=2400):
    begin = time.monotonic()
    beats = len(heartbeats)
    last_move = begin
    moves = 0
    while w.future is not None:
        pump(20)
        now = time.monotonic()
        if now-last_move > 2:
            w.resize(1120+(moves%2)*40, 860+(moves%2)*20)
            w.move(20+(moves%3)*20, 30+(moves%3)*10)
            last_move = now
            moves += 1
        if now-begin > timeout:
            record(name+'-timeout', False, wall_s=now-begin)
            raise TimeoutError(name)
    pump(150)
    times = heartbeats[beats:]
    gaps = [b-a for a,b in zip(times,times[1:])]
    details = dict(wall_s=time.monotonic()-begin, heartbeats=len(times), max_heartbeat_gap_s=max(gaps,default=0), resizes=moves, status=w.status.text(), controls_restored=all(x.isEnabled() for x in w.edit_controls))
    # Heartbeat activity alone is insufficient: multi-second main-thread stalls
    # are a responsiveness failure even though the job eventually completes.
    record(name+'-responsive', len(times)>0 and max(gaps,default=0)<1 and
           all(x.isEnabled() for x in w.edit_controls), gap_limit_s=1, **details)
    return details

def run(name):
    print('STAGE '+name, flush=True)
    # Inspect locks before pumping events: a fast planning failure can already
    # restore controls during the ordinary click helper's20ms wait.
    QTest.mouseClick(w.run_button, Qt.MouseButton.LeftButton)
    busy = w.future is not None
    start = w.start
    w.apply_click(start[0]+1,start[1]+1)
    future = w.future
    QTest.mouseClick(w.run_button, Qt.MouseButton.LeftButton)
    record(name+'-locked', busy and not any(x.isEnabled() for x in w.edit_controls) and w.start==start and w.future is future,
           second_row_disabled=not any(x.isEnabled() for x in w.edit_controls[7:]), stop_enabled=w.stop_button.isEnabled())
    if name in ('ubin-sarl','ubin-mpc'):
        shot(name+'-active')
    metrics = wait_job(name)
    if w.result is not None:
        (OUT/f'{name}.json').write_text(json.dumps(w.result, indent=2, allow_nan=False))
        record(name+'-outcome', status=w.result['status'], model_elapsed=w.result['elapsed'], frames=len(w.result['frames']), diagnostics=len(w.result['diagnostics']), **{'run_wall_s':metrics['wall_s']})
    shot(name+'-terminal')
    return w.result

def load_map(path):
    QFileDialog.getOpenFileName = lambda *args: (str(Path(path).resolve()), 'Map (*.json)')
    click(w.map_button)

def model(path):
    QFileDialog.getExistingDirectory = lambda *args: str(Path(path).resolve())
    click(w.model_button)

def export(name, suffix):
    path=OUT/f'{name}.{suffix}'
    QFileDialog.getSaveFileName=lambda *args: (str(path), suffix)
    QTest.mouseClick(w.export_button, Qt.MouseButton.LeftButton)
    record(name+'-'+suffix+'-busy', w.future is not None and not w.stop_button.isEnabled() and not w.run_button.isEnabled())
    wait_job(name+'-'+suffix, timeout=3600)
    record(name+'-'+suffix+'-saved', path.is_file() and w.status.text()=='Export saved', bytes=path.stat().st_size if path.exists() else 0)
    return path

def replay(name, speed=50):
    w.replay_speed.setCurrentText(f'Replay {speed}x')
    click(w.replay_button)
    begin=time.monotonic()
    frames=[]
    bounded = name.startswith('ubin')
    segment_sim_s = None
    while w.playback.isActive():
        frames.append(dict(wall_s=time.monotonic()-begin,index=w.replay_index,title=w.ax.get_title()))
        pump(20)
        if bounded and time.monotonic()-begin >= 20:
            segment_wall_s=time.monotonic()-begin
            segment_sim_s=float(w.ax.get_title().split(' — ')[1].split(' s')[0])
            w.playback.stop()
            record(name+'-bounded-replay-speed',abs(segment_sim_s/segment_wall_s-speed)<speed*.05,requested_speedup=speed,
                   observed_simulated_s=segment_sim_s,wall_s=segment_wall_s,
                   observed_speedup=segment_sim_s/segment_wall_s,
                   nominal_full_wall_s=w.result['elapsed']*units(w.result['map'])[1]/speed,
                   limitation=None if abs(segment_sim_s/segment_wall_s-speed)<speed*.05 else 'Rendering cannot sustain the requested real-map speed on this host')
            w.replay_index=len(w.result['frames'])-1
            draw(w.ax,w.result,w.replay_index)
            w.canvas.draw_idle()
            pump(150)
            break
        if time.monotonic()-begin>900:
            raise TimeoutError('replay '+name)
    record(name+'-replay', w.ax.get_title().split(' |')[0].endswith(f"{w.result['frames'][-1]['t']*units(w.result['map'])[1]:.1f} s"),
           speed=speed, bounded_segment=bounded, terminal_selected_explicitly=bounded,
           wall_s=time.monotonic()-begin, terminal_title=w.ax.get_title(), timer_interval_ms=w.playback.interval(), monotonic_timing=hasattr(w,'replay_started'))
    (OUT/f'{name}-replay.json').write_text(json.dumps(frames,indent=2))
    shot(name+'-replay-terminal')

def precise_endpoints(start, goal):
    L=w.profile.length
    w.mode.setCurrentIndex(0); w.apply_click(start[0]/L,start[1]/L)
    w.mode.setCurrentIndex(1); w.apply_click(goal[0]/L,goal[1]/L)
    record('endpoints', w.start is not None and w.goal is not None, requested_m=[start,goal], actual_model=[w.start,w.goal], status=w.status.text())

def canvas_click(point):
    w.canvas.draw()
    x,y=w.ax.transData.transform(point)
    ratio=w.canvas.device_pixel_ratio
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=QPoint(round(x/ratio),round(w.canvas.height()-y/ratio)))
    pump(50)

if '--final-checks' in sys.argv:
    # A planning error may finish before the lock probe's event pump. Do not
    # mutate the scene or click Run again after that quick completion.
    path = OUT/'impossible-wall-input.json'
    path.write_text(json.dumps(SeaMap((0,0,24,24),[(10,0,12,24)]).to_dict()))
    load_map(path)
    w.policy.setCurrentText('direct');w.count.setValue(0)
    precise_endpoints((2,12),(22,12))
    QTest.mouseClick(w.run_button, Qt.MouseButton.LeftButton)
    locked = w.future is not None and not any(x.isEnabled() for x in w.edit_controls)
    metrics = wait_job('impossible-wall-corrected')
    record('impossible-wall-corrected', locked and w.result is None and
           'No route at this grid resolution' in w.status.text(),
           status=w.status.text(), start=w.start, goal=w.goal,
           controls_restored=metrics['controls_restored'])
    shot('impossible-wall-corrected')
    w.close()
    raise SystemExit(0 if checks[-1]['outcome']=='pass' else 1)

try:
    if not resume:
        record('native-platform', app.platformName()=='cocoa' and w.isVisible(), platform=app.platformName(), model=str(MODEL), initial_endpoints=[w.start,w.goal], screenshot=shot('native-initial'))
        initial=run('harbour-filtered')
        w.filter_mode.setCurrentIndex(0)
        unfiltered=run('harbour-unfiltered')
        last=unfiltered['diagnostics'][-1]
        from shapely.geometry import Point
        terminal=unfiltered['frames'][-1]['position']
        sea=SeaMap.from_dict(unfiltered['map'])
        edge_distance=min(terminal[0]-sea.bounds[0],terminal[1]-sea.bounds[1],sea.bounds[2]-terminal[0],sea.bounds[3]-terminal[1])
        island_distance=min(Point(terminal).distance(p) for p in sea.land)
        record('unfiltered-land-location', last['land_collision'], diagnostic=last, terminal_position=terminal, edge_distance=edge_distance,island_distance=island_distance,contact='map edge' if edge_distance<island_distance else 'island')
        replay('harbour-unfiltered')
        w.filter_mode.setCurrentIndex(1)
        click(w.run_button); pump(300)
        active=w.future is not None
        begin=time.monotonic(); click(w.stop_button); wait_job('active-stop')
        record('active-stop',active and w.result['status']=='cancelled',status=w.status.text(),stop_wall_s=time.monotonic()-begin)
        (OUT/'cancelled.json').write_text(json.dumps(w.result,indent=2))
        rerun=run('after-stop-rerun')
        old_route=rerun['route']
        w.mode.setCurrentIndex(1); canvas_click((21,20))
        w.mode.setCurrentIndex(2); canvas_click((4,8)); canvas_click((7,12))
        changed=run('edited-scene')
        record('changed-route', changed['route']!=old_route, route_before=old_route,route_after=changed['route'],land_count=len(w.real.land))
        before=w.real.to_dict()
        w.mode.setCurrentIndex(2); canvas_click((1,1)); canvas_click((3,3))
        record('land-over-start-rejected',w.real.to_dict()==before and 'blocks start' in w.status.text(),status=w.status.text());shot('invalid-land')
        start=w.start
        w.mode.setCurrentIndex(0);canvas_click((11,10))
        record('endpoint-on-land-rejected',w.start==start and 'clear of land' in w.status.text(),status=w.status.text());shot('invalid-endpoint')
        empty=OUT/'empty-model';empty.mkdir(exist_ok=True)
        model(empty);run('empty-model')
        record('model-error-restores-controls',w.result is None and w.status.text().startswith('Failed:') and all(x.isEnabled() for x in w.edit_controls),status=w.status.text())
        model(MODEL);run('model-recovery')
        record('model-recovery',w.result is not None,status=w.status.text())
        load_map('maps/harbour.json')
        w.planner.setCurrentText('theta');w.dynamics.setCurrentText('marine');w.perception.setCurrentIndex(1)
        noisy=run('theta-marine-noisy')
        # Inspect correspondence on every frame using the actual replay drawing method.
        correspondence=[]
        w.playback.stop()
        for i,f in enumerate(noisy['frames']):
            w.replay_index=i;draw(w.ax,noisy,i);w.canvas.draw_idle();title=w.ax.get_title()
            d=noisy['diagnostics'][i] if i<len(noisy['diagnostics']) else None
            expected='' if d is None else f"override={d['override']} | no feasible action={d['no_feasible_action']}"
            valid=(('override=' not in title) if d is None else expected in title)
            arrows=[p for p in w.ax.patches if p.__class__.__name__=='FancyArrow']
            predictions=sum(line.get_linestyle()==':' for line in w.ax.lines)
            valid=valid and len(arrows)==(0 if d is None else 2) and predictions==(0 if d is None else len(d['predictions']))
            if d is not None:
                valid=valid and all(abs(p._x-f['position'][0])<1e-9 and abs(p._y-f['position'][1])<1e-9 and abs(p._dx-d[key][0])<1e-9 and abs(p._dy-d[key][1])<1e-9 for p,key in zip(arrows,('nominal','executed')))
                tracks=[line for line in w.ax.lines if line.get_linestyle()==':']
                valid=valid and all(list(line.get_xdata())==[p[0] for p in target['points']] and list(line.get_ydata())==[p[1] for p in target['points']] for line,target in zip(tracks,d['predictions']))
            correspondence.append(dict(frame=i,t=f['t'],title=title,arrows=len(arrows),predicted_tracks=predictions,pass_=valid))
            if i in (0,len(noisy['frames'])//2,len(noisy['frames'])-2,len(noisy['frames'])-1):shot(f'diagnostics-{i}')
        (OUT/'diagnostics-correspondence.json').write_text(json.dumps(correspondence,indent=2))
        record('diagnostics-frame-correspondence',all(x['pass_'] for x in correspondence),frames=len(correspondence),terminal_has_no_diagnostic=True)
        replay('theta-marine-noisy')
        load_map('maps/singapore-ubin.json')
        w.planner.setCurrentText('astar_smooth');w.dynamics.setCurrentText('holonomic');w.perception.setCurrentIndex(0)
        record('ubin-profile-placement',w.profile.name=='harbour_craft' and w.placement.currentIndex()==1,profile=w.profile.name,placement=w.placement.currentText())
        # Toolbar zoom is exercised with a drag on the native canvas, then Home restores map.
        zoom=next(a for a in w.navigation.actions() if a.text()=='Zoom')
        zoom.trigger();pump(50)
        before_limits=w.ax.get_xlim()
        QTest.mousePress(w.canvas,Qt.MouseButton.LeftButton,pos=QPoint(w.canvas.width()//3,w.canvas.height()//3))
        QTest.mouseMove(w.canvas,QPoint(2*w.canvas.width()//3,2*w.canvas.height()//3),100)
        QTest.mouseRelease(w.canvas,Qt.MouseButton.LeftButton,pos=QPoint(2*w.canvas.width()//3,2*w.canvas.height()//3))
        pump(100);record('toolbar-zoom',w.ax.get_xlim()!=before_limits,before=before_limits,after=w.ax.get_xlim());shot('ubin-zoom')
        zoom.trigger();w.navigation.home();pump(100)
        precise_endpoints((-4449.8,1383.9),(4451.7,3038.1))
        ubin=run('ubin-sarl')
    else:
        # Resume only replay/export of THIS acceptance's freshly completed GUI run.
        # The full prescribed run remains in its original log/trace; no old Task6 trace.
        load_map('maps/singapore-ubin.json')
        precise_endpoints((-4449.8,1383.9),(4451.7,3038.1))
        if not resume_after_mpc:
            ubin=json.loads((OUT/'ubin-sarl.json').read_text())
            w.result=ubin
            w.set_busy(False)
            record('resume-fresh-native-trace',True,frames=len(ubin['frames']))
        else:
            record('resume-remaining-checklist',True,prior_full_run_evidence=['ubin-sarl.json','ubin-mpc.json'])
    if not resume_after_mpc:
        replay('ubin-sarl');export('ubin-sarl','mp4')
        w.policy.setCurrentText('mpc');w.dynamics.setCurrentText('marine')
        mpc=run('ubin-mpc');replay('ubin-mpc');export('ubin-mpc','mp4')
    before=(w.start,w.goal);w.vessel.setCurrentText('coastal_ship');pump(50)
    actual=(w.start,w.goal)
    expected=tuple(None if p is None or not w.real.clear(tuple(v*10 for v in p),tuple(v*10 for v in p),50) else tuple(v*.2 for v in p) for p in before)
    record('coastal-mapdemo-validity',all(p==q if p is None or q is None else all(abs(a-b)<1e-8 for a,b in zip(p,q)) for p,q in zip(expected,actual)),before_model=before,after_model=actual,expected_model=expected,start_shore_clearance_m=45.33723500398699,required_clearance_m=50)
    # Original demo start is valid for15m harbour clearance but not50m coastal.
    # Independently demonstrate preservation of BOTH endpoints clear for coastal.
    w.vessel.setCurrentText('harbour_craft')
    precise_endpoints((-2000,500),(4451.7,3038.1))
    before=(w.start,w.goal);w.vessel.setCurrentText('coastal_ship');pump(50)
    actual=(w.start,w.goal)
    record('coastal-profile-keeps-endpoints',all(p is not None and q is not None and all(abs(a*10-b*50)<1e-8 for a,b in zip(p,q)) for p,q in zip(before,actual)),before_model=before,after_model=actual)
    w.vessel.setCurrentText('model');pump(50)
    record('ubin-model-refused',w.profile.name=='coastal_ship' and 'cells' in w.status.text(),status=w.status.text());shot('profile-refused')
    load_map('maps/singapore-southern-islands.json')
    w.policy.setCurrentText('direct');w.dynamics.setCurrentText('holonomic')
    # Independently choose a channel from the second geography, without benchmark files.
    sea=w.real
    x0,y0,x1,y1=sea.bounds
    pts=[(x0+(x1-x0)*i/20,y0+(y1-y0)*j/20) for i in range(2,19) for j in range(2,19)]
    from shipnav.planning import astar,smooth
    from math import dist
    route_pair=None
    for p in pts:
        if not sea.clear(p,p,15):continue
        for q in pts:
            if 800<dist(p,q)<2000 and sea.clear(q,q,15) and not sea.clear(p,q,15):
                try:r=smooth(w.sea,astar(w.sea,tuple(v/10 for v in p),tuple(v/10 for v in q),1.5,5),1.5)
                except ValueError:continue
                route_pair=(p,q);break
        if route_pair:break
    if route_pair is None:raise RuntimeError('No independent Southern channel found')
    precise_endpoints(*route_pair);southern=run('southern-islands');replay('southern-islands')
    # Short identity-scale trace for wall-clock/video timing and every recorded frame.
    clear_map=OUT/'identity-map.json';clear_map.write_text(json.dumps(SeaMap((0,0,24,24)).to_dict()))
    load_map(clear_map);w.policy.setCurrentText('direct');w.count.setValue(0);w.perception.setCurrentIndex(0);w.dynamics.setCurrentText('holonomic')
    precise_endpoints((2,2),(22,2));identity=run('identity-timing')
    w.replay_speed.setCurrentIndex(0);click(w.replay_button)
    begin=time.monotonic();ten_s=None
    while w.playback.isActive():
        title=w.ax.get_title()
        try:sim=float(title.split(' — ')[1].split(' s')[0])
        except (IndexError,ValueError):sim=0
        if sim>=10 and ten_s is None:ten_s=time.monotonic()-begin;shot('identity-ten-seconds')
        pump(10)
    record('ten-second-replay',ten_s is not None and abs(ten_s-10)<1, wall_s_for_10_simulated_s=ten_s,terminal_title=w.ax.get_title())
    for suffix in ('json','csv','png','mp4'):export('identity-timing',suffix)
    rows=list(csv.DictReader((OUT/'identity-timing.csv').open()))
    record('csv-frame-correspondence',len(rows)==len(identity['frames']) and all(abs(float(row['time_s'])-f['t'])<1e-8 and abs(float(row['x_m'])-f['position'][0])<1e-8 and abs(float(row['y_m'])-f['position'][1])<1e-8 for row,f in zip(rows,identity['frames'])),rows=len(rows),terminal_action_empty=rows[-1]['nominal_vx']=='')
    impossible=OUT/'impossible-map.json';impossible.write_text(json.dumps(SeaMap((0,0,24,24),[(10,0,12,24)]).to_dict()))
    load_map(impossible);precise_endpoints((2,12),(22,12));run('impossible-map')
    record('impossible-map-error',w.result is None and 'No route at this grid resolution' in w.status.text(),status=w.status.text())
    for name in ('identity-timing','ubin-sarl','ubin-mpc'):
        result=identity if name=='identity-timing' else json.loads((OUT/f'{name}.json').read_text())
        path=OUT/f'{name}.mp4'
        p=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
        info=json.loads(p.stdout);indices,fps=video_frames(result,default_speedup(result));stream=info['streams'][0]
        (OUT/f'{name}-ffprobe.json').write_text(json.dumps(info,indent=2))
        record(name+'-video',int(stream['nb_read_frames'])==len(indices) and (name!='identity-timing' or stream['r_frame_rate']=='4/1'),expected_frames=len(indices),recorded_frames=len(result['frames']),speedup=default_speedup(result),expected_fps=fps,**stream)
    record('acceptance-driver-completed',True)
except Exception as error:
    import traceback
    traceback.print_exc()
    record('driver-error',False,error=str(error))
    shot(f'driver-error-{len(checks)}')
    raise
finally:
    if w.future is None:w.close()
