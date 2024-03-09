import tensorflow as tf
import numpy as np
import logging

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from scipy.integrate import odeint
from collections import OrderedDict as OD

class StopException (Exception): pass
class SmoothError (Exception): pass
class BoundError (Exception): pass

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

    if s < 20.0 or x > 60.0 or y < 0.0 or a < 0.0 or b < 0.0: raise BoundError (s, x, y, a, b)

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


from __main__ import EPSILON, RNG, PERIOD, INTERVAL

GLOBAL_PERIOD = 10000

START = 4320

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


class ODEEnv (py_environment.PyEnvironment):

  def __init__ (self, params = PARAMS):

    py_environment.PyEnvironment.__init__(self)

    self.order_of_params = tuple (PARAMS.keys())

    self.given_params = {
        'dt': params['dt'],  'D': params['D'], 
        'si': params['si'],  'gamma': params['gamma'], 
        'ga': params['ga'],  'va': params['va'],
        'mu0':params['mu0'], 'mua': params['mua'], 'mub': params['mub'], 
        'Ks': params['Ks'],  'Ka': params['Ka'] 
    }

    self.params = OD ((k, None) for k in self.order_of_params)
    self.params.update (self.given_params)

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
    
    self._action_spec = array_spec.BoundedArraySpec (
                          shape = (5,), # 5 params 
                          dtype = np.float64, 
                          minimum = -1.0*np.ones ((5,), dtype = np.float64), 
                          maximum = np.ones ((5,), dtype = np.float64), 
                          name = 'action'
                          )

    self._observation_spec = array_spec.BoundedArraySpec (
                              shape = (16,), # 5 params + 5 bounds for s0, x0, y0, a0, b0 + 5 ends for s, x, y, a, b + step     
                              dtype = np.float64, 
                              minimum = -1.0*np.ones ((16,), dtype = np.float64), 
                              maximum = np.ones ((16,), dtype = np.float64), 
                              name = 'observation'
                              )
    self.bounds = [0.0]*3


  def _step (self, action):

    try:
        step_num = int (self._current_time_step.observation[-1] + EPSILON)
    
        updated = {(k, self.params_space[k][0] + 0.5*self.span_of_params[k]*(1 + v)) for k, v in zip (self.span_of_params, action[0:len(self.params_space)])}

        params = self.params.copy(); params.update (updated)
        
        params['D'] = self.params['D'][(self.time.size - 1)*step_num : (self.time.size - 1)*(step_num + 1) + 100]
                                
        if step_num == 0: self.bounds = [0.0]*3

        bounds = [self.labels[step_num, 0], self.labels[step_num, 1]] + self.bounds

        lbounds = [self.labels[step_num, 0]/100., self.labels[step_num, 1]/50.] + action[-3:].tolist()

        graths = odeint (ode_system, bounds, self.time, tuple (params.values()), tfirst = True)

        self.bounds = graths[-1, 2:].tolist()

        d1b = np.diff (graths[:, 4], n = 1, axis = 0)/params['dt']
        d2b = np.diff (d1b, n = 1, axis = 0)/params['dt']
        d1a = np.diff (graths[:, 3], n = 1, axis = 0)/params['dt']
        d2a = np.diff (d1a, n = 1, axis = 0)/params['dt']

        if step_num == self.number_of_steps - 1: self._episode_ended = True 

        reward = np.exp (-100.*np.sum (np.abs (self.labels[step_num + 1, 0:2] - graths [-1, 0:2])/(self.labels[step_num + 1, 0:2] + 1.0e-8)))

        kd2b = np.exp (-10.*(np.abs (d2b).max()))
        reward = reward*kd2b
        kd2a = np.exp (-10.*(np.abs (d2a).max()))
        reward = reward*kd2a

        rbounds = [graths[-1, 0]/100., graths[-1, 1]/50.]
        rbounds += [(graths[-1, 2] - (self.params_space['y0'][0] + 0.5*self.span_of_params['y0']))/(0.5*self.span_of_params['y0'])]
        rbounds += [(graths[-1, 3] - (self.params_space['a0'][0] + 0.5*self.span_of_params['a0']))/(0.5*self.span_of_params['a0'])]
        rbounds += [(graths[-1, 4] - (self.params_space['b0'][0] + 0.5*self.span_of_params['b0']))/(0.5*self.span_of_params['b0'])]

        obs = np.concatenate ((action[:len(self.params_space)], lbounds, rbounds, [step_num + 1]))

    except BoundError:

       self._episode_ended = True; reward = 0.
       obs = np.concatenate ((action[:len(self.params_space)], lbounds, [0.]*5, [step_num + 1]))
    
    if self._episode_ended:
      tmsp = ts.termination (obs, reward = reward)
    else:
      tmsp = ts.transition (obs, reward = reward, discount = 1.0)

    return tmsp


  def _reset (self):
    self._episode_ended = False
    obs = np.ones (self._observation_spec.shape, dtype = np.float64); obs[-1] = 0    
    return ts.restart (obs)

  def action_spec (self): return self._action_spec

  def observation_spec (self): return self._observation_spec

  def random_action (self): return tf.convert_to_tensor (RNG.uniform (self.action_spec().minimum, self.action_spec().maximum))
