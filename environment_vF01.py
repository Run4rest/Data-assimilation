
# Environment for searching uknown params: gb, Ia, vb (gd), Ib (Ia)  

import numpy as np
from scipy.integrate import odeint
from collections import OrderedDict as OD

class BoundError  (Exception): pass
class SmoothError (Exception): pass

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

    # Restrictions on absolute values and derivatives. 
    # These restrictions are soft enougth since are met by the global process (see below) 

    if s < 20.0 or x > 60.0 or y < 0.0 or a < 0.0 or b < 0.0: raise BoundError
    if np.abs (db) > 0.05 or np.abs (dy) > 0.05 or np.abs (da) > 0.05: raise SmoothError(db, dy, da, t)

    return [ds, dx, dy, da, db]


# Filtering out white noise to get a low frequancy one

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


from __main__ import EPSILON, RNG

GLOBAL_PERIOD = 10000 # period (in hours) of a global process. The process is built to select a relatively 
                      # short interval of a stable process without transition processes caused by boudary 
                      # conditions randomly selected. The selected short stable process will be 
                      # ground truth of our experiment.

START = 4320; # time of the beginning of the selected process 

PERIOD = 40.0; # duration of the selected process 

INTERVAL = 0.25 #Note: let introduce a term 'point' to specify a vector of params and 
                # whereas a term 'time point' will mean a time of X and S measurement  

# Correct params of the process

PARAMS = OD ((
  ('dt', 0.01), ('D', 0.25), ('si', 100), ('gamma', 1.0), 
  ('ga', 0.3), ('gb', 3.0), ('va', 0.0001), 
  ('mu0', 0.3), ('mua', 0.1), ('mub', 0.7),
  ('Ks', 10.0), ('Ka', 1.0),  ('Ia', 100.0),
  ('y0', 0.0), ('a0', 0.0), ('b0', 0.0)
  ))

DT = PARAMS['dt']
gtime = np.arange (0.0, GLOBAL_PERIOD, DT)

# The dilution rate is the sum of a constant and low frequency noise.

PARAMS['D'] = PARAMS['D'] + 0.03*PARAMS['D']*get_noise (gtime.size + 10, 1/DT, RNG)
ginistate = OD ((('s0', 67.2180), ('x0', 30.9196), ('y0', 0.355404), ('a0', 0.011528), ('b0', 0.0557069)))

# Calculation of the global process. 

global_graths = odeint (ode_system, tuple (ginistate.values()), gtime, tuple (PARAMS.values()), tfirst = True)


i0 = int ((START + EPSILON)//DT); i1 = i0 + int ((PERIOD + EPSILON)//DT)    
PARAMS['D']  = PARAMS['D'][i0:i1 + 100]

TIME = np.arange (0.0, PERIOD + EPSILON, DT)
GRATHS = odeint (ode_system, global_graths[i0], TIME, tuple (PARAMS.values()), tfirst = True)

LABELS = GRATHS[0 : int ((PERIOD + EPSILON)//DT) + 1 : int ((INTERVAL + EPSILON)//DT), 0:2]

del gtime, global_graths 

# class ODEEnv provides calculation of rewards for a given point and contains all data needed 

class ODEEnv:

  def __init__ (self, params = PARAMS):

    self.order_of_params = tuple (params.keys())

    # self.given_params is a dictionary of known params

    self.given_params = {
        'dt': params['dt'],  'D': params['D'], 
        'si': params['si'],  'gamma': params['gamma'], 
        'ga': params['ga'],  'va': params['va'],
        'mu0':params['mu0'], 'mua': params['mua'], 'mub': params['mub'], 
        'Ks': params['Ks'],  'Ka': params['Ka'] 
    }

    # self.params is a dictionary of all params

    self.params = OD ((k, None) for k in self.order_of_params)
    self.params.update (self.given_params)

    # self.params_space defines acceptable ranges of change for unknown params

    self.params_space = {
      'gb':(1.0, 10.0), 'Ia':(0.0001, 1000.0),
      'y0':(0.0, 2.0), 'a0':(0.0, 0.5), 'b0':(0.0, 0.5)
      }         

    # self.labels  defines acceptable ranges of change for unknown params

    self.labels = LABELS

    #self.number_of_steps = PERIOD
    self.number_of_steps = int ((PERIOD + EPSILON)//INTERVAL) 

    self.time = np.arange (0.0, INTERVAL + EPSILON, self.params['dt'])

    self.span_of_params = OD (
      (k, (self.params_space[k][1] - self.params_space[k][0])) for k in self.params if k in self.params_space
      )


  def step (self, point):

    calculations = [];  
    d2b = np.empty ((0,), dtype = np.float64)
    d2a = np.empty ((0,), dtype = np.float64)

    bounds = self.labels[0, 0:2].tolist() + [0.0]*3

    try:
      
      for step_num in range (self.number_of_steps):

        updated = {(k, self.params_space[k][0] + 0.5*self.span_of_params[k]*(1 + v)) for k, v in zip (self.span_of_params, point)}

        params = self.params.copy(); params.update (updated)
        params['D'] = self.params['D'][(self.time.size - 1)*step_num : (self.time.size - 1)*(step_num + 1) + 100]

        graths = odeint (ode_system, bounds, self.time, tuple (params.values()), tfirst = True)

        bounds = graths[-1, :].tolist()

        if graths[-1, 0] < 10 or graths[-1, 1] > 50 or graths[-1, 2] > 10.0: raise BoundError 

        calculations.append (graths [-1, 0:2])

        d1b = np.diff (graths[:, 4], n = 1, axis = 0)/params['dt']
        d2b = np.concatenate ((d2b, np.diff (d1b, n = 1, axis = 0)/params['dt']), axis = 0, dtype = np.float64)
        d1a = np.diff (graths[:, 3], n = 1, axis = 0)/params['dt']
        d2a = np.concatenate ((d2a, np.diff (d1a, n = 1, axis = 0)/params['dt']), axis = 0, dtype = np.float64)
    
    except BoundError: pass
    except SmoothError: calculations = []
           
    if len (calculations) > 0:
        
        values = np.zeros ((self.number_of_steps, 2))
        values[0 : len (calculations), 0:2] = np.array (calculations)
        
        reward = 100.*np.exp (-100.*np.sqrt (((LABELS[1:, 0:2] - values)**2).mean (axis = 0))/LABELS[1:, 0:2].mean (axis = 0)).mean()
        reward0 = reward

#        kd2b = np.exp (-10.*(np.abs (d2b).max()))
        kd2b = np.exp (-1.*(np.abs (d2b).max()))
        reward = reward*kd2b
#        kd2a = np.exp (-50.*(np.abs (d2a).max()))
        kd2a = np.exp (-10.*(np.abs (d2a).max()))
        reward = reward*kd2a

    else:
        reward = reward0 = 0.

    return reward, reward0
