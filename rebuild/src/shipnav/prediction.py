def predict(observed, dt, steps=12, uncertainty=True, growth=.15):
    if dt <= 0 or steps < 1 or growth < 0:
        raise ValueError('Invalid prediction horizon')
    return [{'id': s['id'], 'radius': s['radius'],
             'points': [[x+k*dt*v for x, v in zip(s['position'], s['velocity'])] for k in range(steps+1)],
             'margins': [(s['margin']+growth*k*dt) if uncertainty else 0. for k in range(steps+1)]}
            for s in observed]
