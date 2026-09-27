"""Exercise the local modern runtime and write platform evidence."""
import json
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

def require_local_platform():
    """These artifacts record only the accepted macOS 26 arm64 CPU host."""
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
            or platform.mac_ver()[0].split('.')[0] != '26'):
        raise RuntimeError('This recorder supports only macOS 26 arm64; other platforms require separate acceptance tooling')


require_local_platform()

import gymnasium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy
import PySide6
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel
import torch

out = Path(__file__).resolve().parents[1] / 'migration/macos-arm64'
out.mkdir(parents=True, exist_ok=True)
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = (x.square()).sum()
y.backward()
assert torch.equal(x.grad, torch.tensor([2., 4., 6.]))
app = QApplication([])
window = QLabel('ShipNav modern runtime smoke: native Qt window')
window.resize(560, 160)
window.show()
app.processEvents()
assert window.isVisible()
window.grab().save(str(out / 'qt-window.png'))
QTimer.singleShot(2000, app.quit)
app.exec()
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])
fig.savefig(out / 'plot.png')
writer = FFMpegWriter(fps=2)
with writer.saving(fig, str(out / 'animation.mp4'), dpi=100):
    for frame in range(4):
        ax.set_title(f'Modern export frame {frame}')
        writer.grab_frame()
plt.close(fig)
evidence = {'python': sys.version, 'executable': sys.executable,
            'base_executable': sys._base_executable, 'gil_disabled': sysconfig.get_config_var('Py_GIL_DISABLED'),
            'platform': platform.platform(), 'versions': {m.__name__: m.__version__ for m in [torch,numpy,gymnasium,PySide6,matplotlib]},
            'tensor_backward': x.grad.tolist(), 'qt_platform': app.platformName(), 'qt_visible': True,
            'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(), 'mps_validated': False,
            'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
            'uv': subprocess.check_output(['uv', '--version'], text=True).strip()}
(out / 'smoke.json').write_text(json.dumps(evidence, indent=2)+'\n')
print(json.dumps(evidence, indent=2))
