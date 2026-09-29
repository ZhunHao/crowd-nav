"""Localize heartbeat pauses with fresh versus harness-retained prior trace.

Two actual native MPC+marine episodes, same seed/count/endpoints, bounded to200
model seconds only for diagnosis. Main acceptance's full default-limit MPC run
remains separate. No policy/controller/draw mocks and no GC settings changed.
"""
import ctypes
import gc
import json
import threading
import time
import sys
from pathlib import Path
import torch
torch.set_num_threads(2)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import shipnav.gui as gui
from shipnav.maps import SeaMap
from shipnav.service import execute
from shipnav.controllers.mpc import MPC

ROOT=Path('results/plan04-native').resolve()
corrected='--cached-preview' in sys.argv
OUT=ROOT/'cached-preview-round1' if corrected else ROOT
OUT.mkdir(parents=True,exist_ok=True)
app=QApplication([])
objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.argtypes=[ctypes.c_char_p];objc.objc_getClass.restype=ctypes.c_void_p
objc.sel_registerName.argtypes=[ctypes.c_char_p];objc.sel_registerName.restype=ctypes.c_void_p
send0=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(('objc_msgSend',objc))
sendi=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long)(('objc_msgSend',objc))
nsapp=send0(objc.objc_getClass(b'NSApplication'),objc.sel_registerName(b'sharedApplication'))
sendi(nsapp,objc.sel_registerName(b'setActivationPolicy:'),0)
sendi(nsapp,objc.sel_registerName(b'activateIgnoringOtherApps:'),1)
events=[]
origin=time.monotonic()
main_thread=threading.get_ident()
def event(kind, **values):
    events.append(dict(kind=kind,t=time.monotonic()-origin,thread=threading.get_ident(),**values))

def collector(phase,info):event('gc-'+phase,generation=info['generation'])
gc.callbacks.append(collector)
original_draw=gui.draw
def draw(*args,**kwargs):
    event('draw-start');begin=time.monotonic();original_draw(*args,**kwargs)
    event('draw-stop',duration_s=time.monotonic()-begin)
gui.draw=draw
original_call=MPC.__call__
def decision(*args,**kwargs):
    begin=time.monotonic();value=original_call(*args,**kwargs)
    event('mpc-decision',begin=begin-origin,duration_s=time.monotonic()-begin)
    return value
MPC.__call__=decision

def runner(*args,**kwargs):
    event('worker-entry');value=execute(*args,**kwargs,limit=200.)
    event('worker-return',status=value['status']);return value

w=gui.Window(SeaMap.load('maps/singapore-ubin.json'),str(gui.MODEL),runner=runner)
w.show();w.raise_();w.activateWindow()
w.mode.setCurrentIndex(0);w.apply_click(-444.98,138.39)
w.mode.setCurrentIndex(1);w.apply_click(445.17,303.81)
w.policy.setCurrentText('mpc');w.dynamics.setCurrentText('marine')
original_canvas=w.canvas.draw
def render(*args,**kwargs):
    event('canvas-start');begin=time.monotonic();original_canvas(*args,**kwargs)
    event('canvas-stop',duration_s=time.monotonic()-begin)
w.canvas.draw=render
heartbeats=[]
timer=QTimer();timer.setInterval(50)
timer.timeout.connect(lambda: heartbeats.append(time.monotonic()-origin));timer.start()

def pump(ms):
    end=time.monotonic()+ms/1000
    while time.monotonic()<end:
        begin=time.monotonic();app.processEvents();duration=time.monotonic()-begin
        if duration>.1:event('slow-pump',begin=begin-origin,duration_s=duration)
        time.sleep(.005)
pump(700)
summaries=[]
retained=None
for label in (('fresh','retained-sarl','stop') if corrected else ('fresh','retained-sarl')):
    if label=='retained-sarl':
        retained=json.loads((ROOT/'ubin-sarl.json').read_text())
        pump(200)
    events=[];heartbeats=[];origin=time.monotonic()
    event('run-button')
    QTest.mouseClick(w.run_button,Qt.MouseButton.LeftButton)
    last_resize=time.monotonic();moves=0;stop_requested=None
    while w.future is not None:
        pump(20)
        if time.monotonic()-last_resize>2:
            event('resize-start');w.resize(1120+40*(moves%2),860+20*(moves%2));w.move(30,30)
            event('resize-stop');last_resize=time.monotonic();moves+=1
            if corrected and moves==1:
                # Capture the actual static native preview; its small cost is
                # included in the measured interval and explicitly timestamped.
                event('screenshot-start')
                w.grab().save(str(OUT/f'{label}-busy.png'))
                event('screenshot-stop')
        if label=='stop' and stop_requested is None and time.monotonic()-origin>3:
            stop_requested=time.monotonic()
            event('stop-click')
            QTest.mouseClick(w.stop_button,Qt.MouseButton.LeftButton)
            event('stop-handler-return',cancel_requested=w.cancel_event.is_set())
        if time.monotonic()-origin>180:raise TimeoutError(label)
    stop_latency=None if stop_requested is None else time.monotonic()-stop_requested
    pump(150)
    elapsed=time.monotonic()-origin
    gaps=[dict(start=a,end=b,duration_s=b-a) for a,b in zip(heartbeats,heartbeats[1:])]
    gaps.sort(key=lambda g:g['duration_s'],reverse=True)
    gc_starts={};collections=[]
    for e in events:
        if e['kind']=='gc-start':gc_starts[e['thread']]=e
        elif e['kind']=='gc-stop' and e['thread'] in gc_starts:
            start=gc_starts.pop(e['thread']);collections.append(dict(start=start['t'],end=e['t'],duration_s=e['t']-start['t'],generation=e['generation'],thread=e['thread']))
    collections.sort(key=lambda c:c['duration_s'],reverse=True)
    summary=dict(condition=label,wall_s=elapsed,heartbeat_ticks=len(heartbeats),
                 max_gap_s=max((g['duration_s'] for g in gaps),default=0),
                 top_heartbeat_gaps=gaps[:10],top_gc_collections=collections[:10],
                 longest_draw_s=max((e['duration_s'] for e in events if e['kind']=='draw-stop'),default=0),
                 longest_canvas_s=max((e['duration_s'] for e in events if e['kind']=='canvas-stop'),default=0),
                 main_thread=main_thread,resizes=moves,status=w.result['status'],frames=len(w.result['frames']),
                 retained_frames=0 if retained is None else len(retained['frames']))
    if corrected:
        entry=next(e['t'] for e in events if e['kind']=='worker-entry')
        returned=next(e['t'] for e in events if e['kind']=='worker-return')
        active_renders=[e for e in events if e['kind']=='canvas-start' and entry<=e['t']<=returned]
        summary.update(active_canvas_renders=len(active_renders),stop_latency_s=stop_latency,
                       canvas_restored=w.canvas.isVisible() and not w.canvas.render_suspended,
                       cache_released=w.cached_plot.image.isNull(),
                       toolbar_restored=w.navigation.isEnabled())
        passed=summary['max_gap_s']<1 and not active_renders and summary['canvas_restored'] and summary['cache_released'] and summary['toolbar_restored']
        if label=='stop':passed=passed and stop_requested is not None and stop_latency<1 and w.result['status']=='cancelled'
        summary['outcome']='pass' if passed else 'fail'
        w.grab().save(str(OUT/f'{label}-restored.png'))
    summaries.append(summary)
    (OUT/f'mpc-responsiveness-{label}-events.json').write_text(json.dumps(dict(summary=summary,events=events,heartbeats=heartbeats),indent=2))
    (OUT/f'mpc-responsiveness-{label}-trace.json').write_text(json.dumps(w.result,indent=2))
    print(json.dumps(summary),flush=True)
(OUT/'mpc-responsiveness-comparison.json').write_text(json.dumps(summaries,indent=2))
gc.callbacks.remove(collector);timer.stop();w.close()
if corrected:assert all(s['outcome']=='pass' for s in summaries)
