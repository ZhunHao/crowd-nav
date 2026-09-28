"""Reachable marine dynamics shared by motion (simulation.py) and filtering (safety.py).

Integration scheme (explicit, discrete-time):
    1. Heading and speed are advanced toward the desired velocity reference, each clamped
       to its own reachability limit for the step: heading turns at most `yaw_rate*dt`
       radians, speed changes at most `acceleration*dt` metres/second.
    2. Position is then updated by forward-Euler integration using the *new* (already
       clamped) heading and speed: `x += speed*cos(heading)*dt`, `y += speed*sin(heading)*dt`.

This is a first-order, single-step scheme — not a curved/RK integrator — so a heading
change within a step is applied as if it happened instantaneously at the step's end
before the position update. Callers that need tighter curved-path fidelity should take
smaller `dt` (substeps) rather than relying on chord accuracy at coarse steps; see
`tests/test_dynamics.py::test_timestep_refinement_unfiltered_integration_agrees_closely`
and `tests/test_dynamics.py::test_timestep_refinement_filtered_status_agrees_but_horizon_shifts_trajectory`
for the substep/tolerance comparisons used to validate this.

Demo parameter caveat: `acceleration=0.2` m/s^2 and `yaw_rate=0.35` rad/s are placeholder
demo limits sized for a 24 m synthetic test scene. They are NOT calibrated ship data and
must not be presented as hydrodynamically realistic. A Fossen/NTNU model is expected to
replace this tier once units, step size, hull form and control limits have been validated
against real vessel data.
"""
from dataclasses import dataclass
from math import atan2, hypot, sin, cos, pi

ACCELERATION = .2
YAW_RATE = .35


@dataclass(frozen=True)
class Vessel:
    x: float
    y: float
    heading: float
    speed: float


def advance(state, desired, dt, max_speed=1., acceleration=ACCELERATION, yaw_rate=YAW_RATE):
    """Advance `state` one step toward the velocity reference `desired`.

    Heading and speed are each clamped to what is reachable within `dt` given
    `yaw_rate` and `acceleration`; position is then forward-Euler integrated using the
    resulting (already-clamped) heading and speed. See the module docstring for the
    full integration scheme and the demo-parameter caveat on the default limits.
    """
    if dt <= 0 or min(max_speed, acceleration, yaw_rate) <= 0:
        raise ValueError('Positive dynamics limits required')
    target = atan2(desired[1], desired[0]) if hypot(*desired) > 1e-9 else state.heading
    error = (target-state.heading+pi) % (2*pi)-pi
    heading = state.heading+max(-yaw_rate*dt, min(yaw_rate*dt, error))
    preferred = min(max_speed, hypot(*desired))*max(0., cos(error))
    speed = state.speed+max(-acceleration*dt, min(acceleration*dt, preferred-state.speed))
    return Vessel(state.x+speed*cos(heading)*dt, state.y+speed*sin(heading)*dt, heading, speed)


def motion_from(state, max_speed=1.):
    """Return a `motion(desired, dt, steps) -> list[Point]` rollout callable.

    The returned callable repeatedly applies `advance` starting from `state`, so the
    predicted path it returns is itself bounded by the same acceleration/yaw-rate
    reachability limits used to actually move the vessel — the safety filter's
    candidate rollouts (`safety.rollout`/`safety.choose`) can use this instead of an
    instantaneous-velocity approximation that would let a "stop" command teleport to a
    standstill.
    """
    def motion(desired, dt, steps):
        current = state
        path = [(current.x, current.y)]
        for _ in range(steps):
            current = advance(current, desired, dt, max_speed)
            path.append((current.x, current.y))
        return path
    return motion


def runout(state, dt, max_speed=1.):
    """Zero-command run-out after arrival: the positions the vessel would still
    sweep while decelerating to a stop under the same bounded dynamics."""
    current, path = state, [(state.x, state.y)]
    for _ in range(int(max_speed/ACCELERATION/dt)+2):
        if current.speed <= 1e-9:
            break
        current = advance(current, (0., 0.), dt, max_speed)
        path.append((current.x, current.y))
    return path
