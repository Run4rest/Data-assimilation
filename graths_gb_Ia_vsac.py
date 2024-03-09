
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict as OD
from scipy.integrate import odeint
from replay_buffer_v01 import ReplayBuffer as rbclass

SEED = 42; 
RNG = np.random.default_rng (SEED)
EPSILON = 1.0e-10; 

OBS_DIM = 13
ACT_DIM = 5
DBFILE  = '.\\log\\db.npy'

LOGFILE = '.\\log\\graths v1.0.log'

rb = rbclass (OBS_DIM, ACT_DIM, 0, DBFILE)

obs = np.concatenate ((rb.data.steps[0][-200:], rb.data.rewards[0][-200:]), axis = 1)
ix = np.where (obs[:, 12] == 1)
obs = list (obs [ix])

params_space = OD ((('gb', (1.0, 10.0)), ('Ia', (0.0001, 1000.0)), ('y0', (0.0, 2.0)), ('a0', (0.0, 0.5)), ('b0', (0.0, 0.5))))         

span_of_params = dict ((k, (params_space[k][1] - params_space[k][0])) for k in params_space)

with open (LOGFILE, 'w') as flog:
    with np.printoptions (precision = 3, suppress = True, linewidth = 150):          
        for x in obs:
            ps = OD ((k, params_space[k][0] + 0.5*span_of_params[k]*(1 + v)) for k, v in zip (params_space, x[:5]))
            sout = ' '.join ([f'{k}: {ps[k]:.4f}' for k in ps]) + f' step: {x[12]}' + f' reward: {x[-1]:.4f}' 
            print (sout, file = flog)

n = int ((rb.data.steps[0].shape[0]//20)*20)
print ('n = ', n)

rws = np.reshape (rb.data.rewards[0][:n, 0], (-1, 20)).sum (axis = 1)

ix = np.argpartition (np.array (obs)[:, -1], -1)[-1]

point = obs[ix][:5].tolist()


def ode_system (t, state, dt, D, si, gamma, ga, gb, va, mu0, mua, mub, Ks, Ka, Ia, y0, a0, b0):

    if D.__class__.__name__ == 'ndarray': D = D[int ((t + EPSILON)//dt)]

    s, x, y, a, b = state

    y = y + y0; a = a + a0; b = b + b0

    vb = 0.003*gb; Ib = Ia/100.; Kb = Ka/10.

    mux = mu0*s/(s + Ks)*Ia/(Ia + a)*Ib/(Ib + b)
    muy = (mua*a/Ka + mub*b/Kb)/(1 + a/Ka + b/Kb)
    qa = ga/(1 + a/Ka + b/Kb)*mua*a/Ka
    qb = gb/(1 + a/Ka + b/Kb)*mub*b/Kb

    ds = D*(si - s) - gamma*mux*x
    dx = (mux - D)*x
    dy = (muy - D)*y
    da = va*x - D*a - qa*y
    db = vb*x - D*b - qb*y

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


START = 4320; PERIOD = 5.0; INTERVAL = 0.25

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

ginistate = OD ((('s0', 67.2180), ('x0', 30.9196), ('y0', 0.355404), ('a0', 0.011528), ('b0', 0.0557069)))

global_graths = odeint (ode_system, list (ginistate.values()), gtime, tuple (PARAMS.values()), tfirst = True)

i0 = int ((START + EPSILON)//DT); i1 = i0 + int ((PERIOD + EPSILON)//DT)    
PARAMS['D'] = PARAMS['D'][i0:i1 + 100]
TIME = np.arange (0.0, PERIOD + EPSILON, DT)

GRATHS = odeint (ode_system, global_graths[i0], TIME, tuple (PARAMS.values()), tfirst = True)

LABELS = GRATHS[0 : int ((PERIOD + EPSILON)//DT) + 1 : int ((INTERVAL + EPSILON)//DT), :]

params_space = {
                'gb':(1.0, 10.0), 'Ia':(0.0001, 1000.0), 
                'y0':(0.0, 2.0), 'a0':(0.0, 0.5), 'b0':(0.0, 0.5)
                }         

span_of_params = OD ((k, (params_space[k][1] - params_space[k][0])) for k in PARAMS if k in params_space)

number_of_steps = int ((PERIOD + EPSILON)//INTERVAL) 

time = np.arange (0.0, INTERVAL + EPSILON, PARAMS['dt'])


calculations = []; 
d1b = np.empty ((0,), dtype = np.float64)
d1a = np.empty ((0,), dtype = np.float64)

bounds = LABELS[0, 0:2].tolist() + [0.0]*3

params = PARAMS.copy()

updated = {(k, params_space[k][0] + 0.5*span_of_params[k]*(1 + v)) for k, v in zip (span_of_params, point)}

params.update (updated)

for step_num in range (number_of_steps):

    params['D'] = PARAMS['D'][(time.size - 1)*step_num : (time.size - 1)*(step_num + 1) + 50]

    graths = odeint (ode_system, bounds, time, tuple (params.values()), tfirst = True)

    bounds = graths[-1, :].tolist()

    calculations.append (graths [-1, 0:2])

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
ax.plot (t, GRATHS[:, 2], label = "Y(1)")
ax.plot (t, g[:, 2] + params['y0'],      label = 'Y')
ax.legend (loc = "upper left")

ax = fig.add_subplot (gs[1, 1])
#ax.plot (t, GRATHS[:, 3], t, g[:, 3])
ax.plot (t, GRATHS[:, 3], label = "A(1)")
ax.plot (t, g[:, 3] + params['a0'],      label = 'A')
ax.legend (loc = "upper right")

ax = fig.add_subplot (gs[2, 0])
ax.plot (t, GRATHS[:, 4], label = "B(1)")
ax.plot (t, g[:, 4] + params['b0'],      label = 'B')
ax.legend (loc = "upper right")

ax = fig.add_subplot (gs[2, 1])
ax.plot (rws, label = "reward")
ax.legend (loc = "upper left")

fres = '.\\log\\graths_gb_Ia.png'

try: 
    os.remove (fres)
except FileNotFoundError: pass   

fig.savefig (fres)

print ('params: ')
for k in ('gb', 'Ka', 'Ia', 'y0', 'a0', 'b0'):
    print (f'{k}: {params[k]:.3f}')
