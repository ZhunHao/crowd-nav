"""Visible native scene-edit supplement after correcting Retina pixel conversion."""
import ctypes
import json
import time
from pathlib import Path
import torch
torch.set_num_threads(2)
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shipnav.gui import MODEL, Window
from shipnav.maps import SeaMap
from shipnav.export import draw

out=Path('results/plan04-native').resolve()
app=QApplication([])
objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.argtypes=[ctypes.c_char_p];objc.objc_getClass.restype=ctypes.c_void_p
objc.sel_registerName.argtypes=[ctypes.c_char_p];objc.sel_registerName.restype=ctypes.c_void_p
send0=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(('objc_msgSend',objc))
sendi=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long)(('objc_msgSend',objc))
nsapp=send0(objc.objc_getClass(b'NSApplication'),objc.sel_registerName(b'sharedApplication'))
sendi(nsapp,objc.sel_registerName(b'setActivationPolicy:'),0)
sendi(nsapp,objc.sel_registerName(b'activateIgnoringOtherApps:'),1)
w=Window(SeaMap.load('maps/harbour.json'),str(MODEL));w.show();w.raise_();w.activateWindow()
checks=[]
def pump(ms):
    end=time.monotonic()+ms/1000
    while time.monotonic()<end:app.processEvents();time.sleep(.005)
def record(name,passed,**details):
    entry=dict(check=name,outcome='pass' if passed else 'fail',**details)
    checks.append(entry);(out/'scene-checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(entry),flush=True)
def shot(name):pump(100);w.grab().save(str(out/f'{name}.png'))
def click(p):
    w.canvas.draw()
    x,y=w.ax.transData.transform(p);ratio=w.canvas.device_pixel_ratio
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=QPoint(round(x/ratio),round(w.canvas.height()-y/ratio)));pump(100)
def run(name):
    QTest.mouseClick(w.run_button,Qt.MouseButton.LeftButton)
    while w.future is not None:pump(20)
    (out/f'{name}.json').write_text(json.dumps(w.result,indent=2));shot(name)
    return w.result
pump(700)
initial=run('scene-baseline')
w.mode.setCurrentIndex(1);click((21,20))
record('destination-canvas-click',abs(w.goal[0]-21)<.1 and abs(w.goal[1]-20)<.1,goal=w.goal,device_pixel_ratio=w.canvas.device_pixel_ratio)
w.mode.setCurrentIndex(2);click((4,8));click((7,12))
record('land-two-canvas-clicks',len(w.real.land)==2,status=w.status.text(),land_count=len(w.real.land));shot('scene-edited-preview')
changed=run('scene-changed-route')
record('changed-route',changed['route']!=initial['route'],before=initial['route'],after=changed['route'],status=changed['status'])
before=w.real.to_dict();w.mode.setCurrentIndex(2);click((1,1));click((3,3))
record('land-over-start-rejected',w.real.to_dict()==before and 'blocks start' in w.status.text(),status=w.status.text());shot('scene-invalid-land')
before=w.start;w.mode.setCurrentIndex(0);click((11,10))
record('endpoint-on-land-rejected',w.start==before and 'clear of land' in w.status.text(),status=w.status.text());shot('scene-invalid-endpoint')
# Repeat stored-trace geometry correspondence in this actual native window.
w.result=json.loads((out/'theta-marine-noisy.json').read_text())
geometry=[]
for i,frame in enumerate(w.result['frames']):
    w.replay_index=i;draw(w.ax,w.result,i);w.canvas.draw_idle()
    d=w.result['diagnostics'][i] if i<len(w.result['diagnostics']) else None
    arrows=[p for p in w.ax.patches if p.__class__.__name__=='FancyArrow']
    tracks=[line for line in w.ax.lines if line.get_linestyle()==':']
    valid=len(arrows)==(0 if d is None else 2) and len(tracks)==(0 if d is None else len(d['predictions']))
    if d is not None:
        valid=valid and all(abs(p._x-frame['position'][0])<1e-9 and abs(p._y-frame['position'][1])<1e-9 and abs(p._dx-d[key][0])<1e-9 and abs(p._dy-d[key][1])<1e-9 for p,key in zip(arrows,('nominal','executed')))
        valid=valid and all(list(line.get_xdata())==[p[0] for p in target['points']] and list(line.get_ydata())==[p[1] for p in target['points']] for line,target in zip(tracks,d['predictions']))
    geometry.append(dict(frame=i,pass_=valid))
    if i in (0,200,399,400):shot(f'scene-diagnostic-{i}')
(out/'diagnostic-geometry.json').write_text(json.dumps(geometry,indent=2))
record('diagnostics-arrow-track-geometry',all(x['pass_'] for x in geometry),frames=len(geometry))
w.close()
assert all(x['outcome']=='pass' for x in checks)
