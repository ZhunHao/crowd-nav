"""Measure corrected replay in a visible Cocoa window using fresh Task10 traces.

From rebuild: QT_QPA_PLATFORM=cocoa .venv-modern/bin/python
 tools/native_replay_timing.py results/plan04-native/ubin-sarl.json [...ubin-mpc.json]
No service rerun or draw mock. 50x: bounded20-second measured segment.
200x: complete real playback, including the terminal frame.
"""
import ctypes
import json
import sys
import time
from pathlib import Path
import torch
torch.set_num_threads(2)
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shipnav.gui import MODEL, Window
from shipnav.maps import SeaMap
from shipnav.scale import units

OUT=Path('results/plan04-native').resolve()
app=QApplication([])
objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.argtypes=[ctypes.c_char_p];objc.objc_getClass.restype=ctypes.c_void_p
objc.sel_registerName.argtypes=[ctypes.c_char_p];objc.sel_registerName.restype=ctypes.c_void_p
send0=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(('objc_msgSend',objc))
sendi=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long)(('objc_msgSend',objc))
nsapp=send0(objc.objc_getClass(b'NSApplication'),objc.sel_registerName(b'sharedApplication'))
sendi(nsapp,objc.sel_registerName(b'setActivationPolicy:'),0)
sendi(nsapp,objc.sel_registerName(b'activateIgnoringOtherApps:'),1)
w=Window(SeaMap.load('maps/singapore-ubin.json'),str(MODEL))
w.show();w.raise_();w.activateWindow()

def pump(ms=10):
    end=time.monotonic()+ms/1000
    while time.monotonic()<end:app.processEvents();time.sleep(.005)

def shot(name):
    sendi(nsapp,objc.sel_registerName(b'activateIgnoringOtherApps:'),1)
    w.raise_();pump(100);w.grab().save(str(OUT/f'{name}.png'))

pump(700)
for path in map(Path,sys.argv[1:]):
    trace=json.loads(path.read_text())
    settings=trace['settings']
    w.policy.setCurrentText(settings['policy'])
    w.planner.setCurrentText(settings['planner'])
    w.dynamics.setCurrentText(settings['dynamics'])
    w.filter_mode.setCurrentIndex(int(settings['filtered']))
    w.perception.setCurrentIndex(int(bool(settings['observation'])))
    w.count.setValue(settings['requested_traffic']);w.seed.setValue(settings['seed'])
    w.start=tuple(trace['scenario']['start']);w.goal=tuple(trace['scenario']['goal'])
    w.result=trace;w.set_busy(False)
    name=path.stem
    _,T=units(trace['map'])
    checks=[]
    samples=[]
    w.replay_speed.setCurrentText('Replay 50x')
    QTest.mouseClick(w.replay_button,Qt.MouseButton.LeftButton)
    while w.playback.isActive() and time.monotonic()-w.replay_started<20:
        pump(10)
        samples.append(dict(wall_s=time.monotonic()-w.replay_started,frame=w.replay_index,
                            simulated_s=trace['frames'][w.replay_index]['t']*T,
                            title=w.ax.get_title()))
    wall=time.monotonic()-w.replay_started
    sim=trace['frames'][w.replay_index]['t']*T
    w.playback.stop()
    passed=abs(sim-wall*50)<wall*50*.05
    checks.append(dict(check='native50x-clock',outcome='pass' if passed else 'fail',
                       requested_speedup=50,wall_s=wall,displayed_simulated_s=sim,
                       observed_speedup=sim/wall,frame=w.replay_index,
                       expected_simulated_s=wall*50,timer_ms=w.playback.interval(),
                       platform=app.platformName(),visible=w.isVisible()))
    shot(name+'-corrected-50x')
    w.replay_speed.setCurrentText('Replay 200x')
    QTest.mouseClick(w.replay_button,Qt.MouseButton.LeftButton)
    while w.playback.isActive():
        pump(10)
        if time.monotonic()-w.replay_started>120:
            raise TimeoutError(name+' corrected200x')
    wall=time.monotonic()-w.replay_started
    nominal=trace['elapsed']*T/200
    terminal=w.replay_index==len(trace['frames'])-1
    checks.append(dict(check='native200x-terminal-clock',outcome='pass' if terminal and abs(wall-nominal)<1 else 'fail',
                       requested_speedup=200,nominal_wall_s=nominal,actual_wall_s=wall,
                       final_frame=w.replay_index,recorded_frames=len(trace['frames']),
                       title=w.ax.get_title(),terminal_diagnostics_absent='override=' not in w.ax.get_title()))
    shot(name+'-corrected-terminal')
    (OUT/f'{name}-corrected-replay-samples.json').write_text(json.dumps(samples,indent=2))
    (OUT/f'{name}-corrected-replay-checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks),flush=True)
    assert all(c['outcome']=='pass' for c in checks)
w.close()
