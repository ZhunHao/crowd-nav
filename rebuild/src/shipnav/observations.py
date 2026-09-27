from hashlib import sha256
from random import Random
from math import isfinite


class Observer:
    def __init__(self, seed=0, noise=0., delay=0., dropout=0., stale_speed_bound=2.):
        if not all(isfinite(x) for x in (noise, delay, dropout, stale_speed_bound)) or min(noise, delay, stale_speed_bound) < 0 or not 0 <= dropout <= 1:
            raise ValueError('Invalid observation configuration')
        self.seed, self.noise, self.delay = seed, noise, delay
        self.dropout, self.bound, self.last = dropout, stale_speed_bound, {}

    def observe(self, traffic, t, tick):
        result = []
        for target, ship in enumerate(traffic):
            key = sha256(f'{self.seed}:{tick}:{target}'.encode()).digest()
            rng = Random(int.from_bytes(key, 'big'))
            stamp = max(0., t-self.delay)
            if rng.random() >= self.dropout:
                p, v, r = ship.at(stamp)
                measured = tuple(x+rng.gauss(0, self.noise) for x in p)
                self.last[target] = (measured, v, r, stamp)
            if target in self.last:
                p, v, r, stamp = self.last[target]
                age = t-stamp
                estimate = tuple(x+age*y for x, y in zip(p, v))
                # Bound missed motion conservatively; Gaussian margin is not a hard bound.
                result.append({'id': target, 'position': estimate, 'velocity': v,
                               'radius': r, 'age': age,
                               'margin': 3*self.noise+2*self.bound*age})
        return result
