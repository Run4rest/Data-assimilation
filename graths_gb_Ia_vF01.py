
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict as OD
from scipy.integrate import odeint

SEED = 42; 
RNG = np.random.default_rng (SEED)
EPSILON = 1.0e-10; 

#point = [-0.396448, -0.799938, -0.475885, -0.930292, -0.784327]
point = [-0.471446, -0.799961, -0.477072, -0.930323, -0.784455]
#point = [-0.318911, -0.800012, -0.471538, -0.901017, -0.784729]
point = [-0.471051, -0.799961, -0.477111, -0.931345, -0.784482]
point = [-0.411034, -0.799983, -0.475076, -0.929001, -0.785014]
point = [-0.452266, -0.800025, -0.475508, -0.929806, -0.785177]
point = [-0.471144, -0.799969, -0.476927, -0.93,     -0.784626]
point = [-0.516935, -0.799983, -0.477915, -0.931054, -0.784588]
point = [-0.537595, -0.799999, -0.478251, -0.930851, -0.784683]
point = [-0.558845, -0.799996, -0.47886,  -0.93083,  -0.78462 ]
#point = [-0.471253, -0.799963, -0.477061, -0.930532, -0.784463]


class BoundError (Exception): pass
class SmoothError (Exception): pass

def ode_system (t, state, dt, D, si, gamma, ga, gb, va, mu0, mua, mub, Ks, Ka, Ia, y0, a0, b0):

#    global step_num

    if D.__class__.__name__ == 'ndarray': D = D[int ((t + EPSILON)//dt)]

    s, x, y, a, b = state

    y = y + y0; a = a + a0; b = b + b0

    vb = 0.003*gb; Ib = Ia/100.; Kb = Ka/10.

    mux = mu0*s/(s + Ks)*Ia/(Ia + a)*Ib/(Ib + b)
    muy = (mua*a/Ka + mub*b/Kb)/(1 + a/Ka + b/Kb)
    qa = ga/(1 + a/Ka + b/Kb)*mua*a/Ka
    qb = gb/(1 + a/Ka + b/Kb)*mub*b/Kb

#    ds = D*(si - s) - gamma*mux*x*s/(s + Ks)
#    dx = (mux*s/(s + Ks) - D)*x
    ds = D*(si - s) - gamma*mux*x
    dx = (mux - D)*x
    dy = (muy - D)*y
    da = va*x - D*a - qa*y
    db = vb*x - D*b - qb*y

    if s < 20.0 or x > 60.0 or y < 0.0 or a < 0.0 or b < 0.0: raise BoundError
    if np.abs (db) > 0.05 or np.abs (dy) > 0.05 or np.abs (da) > 0.05: raise SmoothError(db, dy, da, t)

    return [ds, dx, dy, da, db]

def get_noise (size, freq, rng):

    nawhite = rng.standard_normal (size)
    naspwht = np.fft.rfft (nawhite)

    S = np.fft.rfftfreq (size)
    S[0] = EPSILON
    Sn = np.where (S < 0.000001*freq, S, 0) 

    naspclr = naspwht*Sn
    nacolor = np.fft.irfft (naspclr)
    nacolor = nacolor/np.sqrt (np.mean (nacolor**2))

    return nacolor

###################################################################
START = 4320; PERIOD = 40.0; INTERVAL = 0.25

GLOBAL_PERIOD = 10000
PARAMS = OD ((
  ('dt', 0.01), ('D', 0.25), ('si', 100), ('gamma', 1.0), 
  ('ga', 0.3), ('gb', 3.0), ('va', 0.0001), 
  ('mu0', 0.3), ('mua', 0.1), ('mub', 0.7),
  ('Ks', 10.0), ('Ka', 1.0),  ('Ia', 100.0),
  ('y0', 0.0), ('a0', 0.0), ('b0', 0.0)
  ))

DT = PARAMS['dt']
gtime = np.arange (0.0, GLOBAL_PERIOD, DT)

PARAMS['D'] = PARAMS['D'] + 0.03*PARAMS['D']*get_noise (gtime.size + 50, 1/DT, RNG)
#ginistate = OD ((('s0', 65.0), ('x0', 30.0), ('y0', 0.34), ('a0', 0.01), ('b0', 0.055)))
ginistate = OD ((('s0', 67.2180), ('x0', 30.9196), ('y0', 0.355404), ('a0', 0.011528), ('b0', 0.0557069)))
#66.48058849977323, 30.4252135335915, 0.3477038551235918, 0.010888340720282174, 0.056116570188560866
#67.21803507610971, 30.91957965104299, 0.3554042568481911, 0.011528069990014257, 0.05570693780641679, 2.1188255656201003

global_graths = odeint (ode_system, list (ginistate.values()), gtime, tuple (PARAMS.values()), tfirst = True)

i0 = int ((START + EPSILON)//DT); i1 = i0 + int ((PERIOD + EPSILON)//DT)    
PARAMS['D'] = PARAMS['D'][i0:i1 + 100]
TIME = np.arange (0.0, PERIOD + EPSILON, DT)
try:
    GRATHS = odeint (ode_system, global_graths[i0], TIME, tuple (PARAMS.values()), tfirst = True)
except SmoothError: 
    print ('SmoothError in 2')


LABELS = GRATHS[0 : int ((PERIOD + EPSILON)//DT) + 1 : int ((INTERVAL + EPSILON)//DT), :]

params_space = {
                'gb':(1.0, 10.0), 'Ia':(0.0001, 1000.0), 
                'y0':(0.0, 2.0), 'a0':(0.0, 0.5), 'b0':(0.0, 0.5)
                }         

span_of_params = OD ((k, (params_space[k][1] - params_space[k][0])) for k in PARAMS if k in params_space)

number_of_steps = int ((PERIOD + EPSILON)//INTERVAL) 

time = np.arange (0.0, INTERVAL + EPSILON, PARAMS['dt'])


calculations = []; 
#calculations = []; a_min = 100.; a_max = -100.#; d2b_min = 100.; d2b_max = -100. 
d1b = np.empty ((0,), dtype = np.float64)
d1a = np.empty ((0,), dtype = np.float64)

bounds = LABELS[0, 0:2].tolist() + [0.0]*3

try:
    params = PARAMS.copy()

    updated = {(k, params_space[k][0] + 0.5*span_of_params[k]*(1 + v)) for k, v in zip (span_of_params, point)}

    params.update (updated)
    
    for step_num in range (number_of_steps):

        params['D'] = PARAMS['D'][(time.size - 1)*step_num : (time.size - 1)*(step_num + 1) + 50]

#    graths = odeint (ode_system, list (bounds.values()), self.time, tuple (params.values()), tfirst = True)
        graths = odeint (ode_system, bounds, time, tuple (params.values()), tfirst = True)

#        bounds = self.labels[step_num + 1, 0:2].tolist() + graths[-1, -3:].tolist()
        bounds = graths[-1, :].tolist()

        if graths[-1, 0] < 10 or graths[-1, 1] > 50 or graths[-1, 2] > 10.0: raise BoundError 

        calculations.append (graths [-1, 0:2])

#        reward.append (np.exp (-10000.0*np.sum (np.abs (LABELS[step_num + 1, 0:2] - graths [-1, 0:2])/(LABELS[step_num + 1, 0:2] + 1.0e-8))))
        # a = np.max (graths[:, 3]) + params['a0']
        # a_max = a if a > a_max else a_max
        # a = np.min (graths[:, 3]) + params['a0']
        # a_min = a if a < a_min else a_min

        d1b = np.concatenate ((d1b, np.diff (graths[:, 4], n = 1, axis = 0)/params['dt']), axis = 0, dtype = np.float64)
        d1a = np.concatenate ((d1a, np.diff (graths[:, 3], n = 1, axis = 0)/params['dt']), axis = 0, dtype = np.float64)

except BoundError: pass
except SmoothError: 
    print (f'SmoothError in 3, step:{step_num}')

if len (calculations) > 0:
    
    values = np.zeros ((number_of_steps, 2))
    values[0 : len (calculations), 0:2] = np.array (calculations)
    
    reward = 10.*np.exp (-100.*np.sqrt (((LABELS[1:, 0:2] - values)**2).mean (axis = 0))/LABELS[1:, 0:2].mean (axis = 0)).mean()
else:
    reward = 0.

reward0 = reward

d2b = np.diff (d1b, n = 1, axis = 0)/params['dt']
d2a = np.diff (d1a, n = 1, axis = 0)/params['dt']

#reward *= (1.0 - 0.2*(a_max - a_min)/a_max)
kd2b = np.exp (-100.*(np.abs (d2b).max()))
reward = reward*kd2b
kd2a = np.exp (-100.*(np.abs (d2a).max()))
reward = reward*kd2a

# print (f'(a_max - a_min)/a_max > 0.2: {(a_max - a_min)/a_max:.4f}   x1: {x1:.4f}  5*x2: {5*x2:.4f} x1 > 5.*x2: {x1 > 5.*x2}')
print (f'RWD0 {reward0:.2f}  RWD {reward:.2f}  SN: {step_num}')

b = LABELS[0, 0:2].tolist() + [0.0]*3
params['D'] = PARAMS['D']
g = odeint (ode_system, b, TIME, tuple (params.values()), tfirst = True)

GRATHS = GRATHS[:TIME.size - 1, :]; g = g[:TIME.size - 1, :]

t = TIME[:TIME.size - 1]
fig = plt.figure (figsize = [12.0, 10.0])#638.4])
gs = GridSpec (3, 2, figure = fig)

ax = fig.add_subplot (gs[0, 0])
ax.plot (t, GRATHS[:, 0], label = "S(1)")
ax.plot (t, g[:, 0],      label = 'S')
ax.legend (loc = "upper right")

ax = fig.add_subplot (gs[0, 1])
#ax.plot (t, GRATHS[:, 1], t, g[:, 1])
ax.plot (t, GRATHS[:, 1], label = "X(1)")
ax.plot (t, g[:, 1],      label = 'X')
ax.legend (loc = "upper left")

ax = fig.add_subplot (gs[1, 0])
#ax.plot (t, GRATHS[:, 2], t, g[:, 2])
ax.plot (t, GRATHS[:, 2], label = "Y(1)")
ax.plot (t, g[:, 2] + params['y0'],      label = 'Y')
ax.legend (loc = "upper left")

ax = fig.add_subplot (gs[1, 1])
#ax.plot (t, GRATHS[:, 3], t, g[:, 3])
ax.plot (t, GRATHS[:, 3], label = "A(1)")
ax.plot (t, g[:, 3] + params['a0'],      label = 'A')
ax.legend (loc = "upper right")

ax = fig.add_subplot (gs[2, 0])
#ax.plot (t, GRATHS[:, 4], t, g[:, 4])
ax.plot (t, GRATHS[:, 4], label = "B(1)")
ax.plot (t, g[:, 4] + params['b0'],      label = 'B')
ax.legend (loc = "upper right")

d1b = np.diff (g[:, 4], n = 1)*100.
d2b = np.diff (d1b, n = 1)*100.

# ax = fig.add_subplot (gs[2, 1])
# #ax.plot (t, GRATHS[:, 4], t, g[:, 4])
# ax.plot (t[:d1b.size], d1b, label = "d1B")
# ax.plot (t[:d2b.size], d2b, label = "d2B")
# ax.legend (loc = "upper right")

d1a = np.diff (g[:, 3], n = 1)*100.
d2a = np.diff (d1a, n = 1)*100.

ax = fig.add_subplot (gs[2, 1])
#ax.plot (t, GRATHS[:, 4], t, g[:, 4])
ax.plot (t[:d1a.size], d1a, label = "d1a")
ax.plot (t[:d2a.size], d2a, label = "d2a")
ax.legend (loc = "upper right")

#plt.show()

fres = '.\\graths\\graths_gb_Ia.png'

try: 
    os.remove (fres)
except FileNotFoundError: pass   

fig.savefig (fres)

print ('params: ')
for k in ('gb', 'Ka', 'Ia', 'y0', 'a0', 'b0'):
    print (f'{k}: {params[k]:.3f}')
