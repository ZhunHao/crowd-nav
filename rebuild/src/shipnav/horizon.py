"""The single shared lookahead-horizon definition.

Prediction (`prediction.predict`), the safety filter (`safety.choose`) and the
MPC comparator (`controllers.mpc.MPC`) must all roll out the same number of
steps; `safety.assess` refuses mismatched lengths rather than silently
truncating. Holonomic runs keep a fixed 12-step horizon. Marine runs extend it
so the horizon always covers the zero-command stopping time from full speed
under the bounded deceleration used by `dynamics.advance`:
``max(12, ceil(max_speed/acceleration/dt) + 1)`` steps.
"""
from math import ceil, isfinite

from shipnav.dynamics import ACCELERATION

HOLONOMIC_STEPS = 12


def horizon_steps(dt, dynamics='holonomic', max_speed=1., acceleration=ACCELERATION):
    if not all(isfinite(v) and v > 0 for v in (dt, max_speed, acceleration)):
        raise ValueError('Horizon requires positive finite dt, speed and acceleration')
    if dynamics == 'holonomic':
        return HOLONOMIC_STEPS
    if dynamics == 'marine':
        return max(HOLONOMIC_STEPS, ceil(max_speed/acceleration/dt)+1)
    raise ValueError('Unknown dynamics')
