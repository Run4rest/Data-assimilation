
# Main script for searching a profile of D. 
# D is approximated by two hyperbolas - one per a half of 
# the interval - with conditions of continuity for D and its first derivative 

import numpy as np
from collections import OrderedDict as OD

class StopException (Exception): pass

SEED = 42;  
RNG = np.random.default_rng (SEED)
EPSILON = 1.0e-10; 

from replay_buffer_vF00 import ReplayBuffer
from environment_vF11 import ODEEnv

DBFILE = '.\\log\\db_vF11.npy'
CMFILE = '.\\log\\cm_vF11.npy'
LOG = '.\\log\\log_vF11.log'

def set_boundaries (point):
    point = np.where (point >  1.0,  1.0, point) 
    point = np.where (point < -1.0, -1.0, point)
    return point


FIRST_START = 1   # this flag indicates 2 options:
                  # 1 - starting a new optimization
                  # 0 - continuing the previous one  
TRIALS  = 20000   # this is number of points with uniform PDF to calculate a reward, but after 
                  # selecting points belonged spherical neighborhood of a maximum, real number 
                  # of trials is about 100 for point_dim = 9
SAMPLES = 1000    # number of points with MND PDF
RADIUS  = 0.03    # an initial radius of neighborhood of a maximum
CM_BUFFER_SIZE = 200 # number of top-N points with largest rewards to calculate a covariance matrix 

open (LOG, 'w')

env = ODEEnv() # instance of our environment

point_dim = len (env.params_space) 

rb = ReplayBuffer (point_dim, DBFILE, FIRST_START) # instance of data base with history of maximums  

# Repeat until convergence

try:

    res = []; points = []; radius_hist = []; cm_changed = False; CM = False#; cm_idling = 0

    trials_num = TRIALS; samples_num = SAMPLES; radius = RADIUS; 

    if FIRST_START:

        reward = reward0 = 0.; step = 0

        # a search cycle of a starting point 

        while reward0 < 0.001:

            point = RNG.uniform (-1., 1., point_dim)

            # assigning initial values to params of D's approximation to 
            # reduce search time of a starting point. These values correspond 
            # a constant D = 0.25    
            point[-6:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
            reward, reward0 = env.step (point)

            step += 1
            
            print (f'step: {step}  reward0: {reward0:.6f} reward: {reward:.6f}')            

        rb.store (point, reward, reward0)

        points.append (point); res.append ((reward, reward0))

    else:

        # loading a covariance matrix, points and rewards saved earlier

        with open (LOG, 'a') as flog:
            print ('\n ... STARTING WITH COVARIANCE MATRIX: ' + CMFILE + '...', file = flog)

        with open (CMFILE, 'rb') as fcm:
            cm = np.load (fcm)
            points = list (np.load (fcm))
            res    = np.load (fcm).tolist()

            print ('res: \n', res)

        CM = True

    # endless cycle of searching params
    # use Ctrl C interruption to stop the process

    while True:

        rewards = np.array ([r[0] for r in res])
        ix_max = np.argpartition (rewards, -1)[-1]        
        max_point   = points[ix_max] 
        max_reward  = res[ix_max][0]
        max_reward0 = res[ix_max][1]

        with open (LOG, 'a') as flog:
            with np.printoptions (precision = 6, suppress = True, linewidth = 150):          
                print (f'\nMAX POINT:  {max_point}', file = flog)
                print (f'MAX REWARD: {max_reward:.6f}  REWARD0: {max_reward0}\n', file = flog)

        if not CM: 

            # this block of code implements a uniform PDF of trial points in 
            # the vicinity of a current maximum

            radius = RADIUS 

            misses = len (points)

            if misses > 2*TRIALS/point_dim: radius = RADIUS/3.  #0.01
            if misses > 4*TRIALS/point_dim: radius = RADIUS/10. #0.003
            if misses > 6*TRIALS/point_dim: radius = RADIUS/30  #0.001

            if len (radius_hist) > 10: 
                if np.any (np.array (radius_hist[-10:]) <= RADIUS/10.): radius = RADIUS/10.
                if np.any (np.array (radius_hist[-10:]) == RADIUS/30.): radius = RADIUS/30. 

            if len (res) > 0: 
                if res[ix_max][1] > 90.:  radius = RADIUS/10. if radius > RADIUS/10. else radius
                if res[ix_max][1] > 95.:  radius = RADIUS/30.

            new_ps = RNG.uniform (low = set_boundaries (max_point - radius), high = set_boundaries (max_point + radius), size = (trials_num, point_dim))
            new_ps = [new_p for new_p in list (new_ps) if np.sum ((new_p - max_point)**2) < radius**2]
            points = points + new_ps

            for p in new_ps:
                reward, reward0 = env.step (p)
                res.append ((reward, reward0))

            rewards = np.array ([r[0] for r in res])
            ix_max  = np.argpartition (rewards, -1)[-1]

        if CM:

            # this block calculates a covariance matrix of multivariable normal distribution (MND) and 
            # sample points based on the distribution

            points = np.array (points)

            ix = np.argpartition (rewards, -CM_BUFFER_SIZE)[-CM_BUFFER_SIZE:]

            radius = RADIUS/300.

            points = points[(ix,)]; rewards = rewards[(ix,)]

            print (f'ix: {sorted (list (ix))}')

            with open (LOG, 'a') as flog:
                with np.printoptions (precision = 6, suppress = True, linewidth = 150):
                    print (f'ix: {sorted (list (ix))}', file = flog)
                    print (f'\npoints & rewards: \n{np.concatenate ((points, np.expand_dims (rewards, axis = 1)), axis = 1)[-10:]}', file = flog)

            res = [r for i, r in enumerate (res) if i in list (ix)]

            cm = np.cov (points.T)
            k = np.diagonal (cm).max()
            cm = radius*cm/k

            cm_changed = True

            with open (LOG, 'a') as flog:
                with np.printoptions (precision = 10, suppress = True, linewidth = 150):          
                    print (f'\nCM: \n{cm}', file = flog)
             
            samples = RNG.multivariate_normal (max_point, cm, samples_num).astype (np.float64)

            samples = set_boundaries (samples)
                       
            for s in samples:
                reward, reward0 = env.step (s)
                res.append ((reward, reward0))

            with open (LOG, 'a') as flog:
                with np.printoptions (precision = 6, suppress = True, linewidth = 150):          
                    print (f'\nSAMPLES: \n{np.concatenate ((samples[-10:], np.array ([r[0] for r in res[-10:]], ndmin = 2).T), axis = 1)}', file = flog)

            points = list (points) + list (samples)

            rewards = np.array ([r[0] for r in res])

            ix_max = np.argpartition (rewards, -1)[-1]

        if rewards[ix_max] > max_reward: 

            # calculating vector between new and previous maximums 
            # and checking rewards in its direction

            grad = (points[ix_max] - max_point)/20.

            reward_along_grad = [(max_reward, None), (res[ix_max][0], res[ix_max][1])]
            points_along_grad = [max_point, points[ix_max]]

            grad_step = 1

            while reward_along_grad[-1][0] > reward_along_grad[-2][0]:

                point = set_boundaries (points[ix_max] + grad_step*grad)
                reward, reward0 = env.step (point)
                reward_along_grad.append ((reward, reward0))
                points_along_grad.append (point)
                grad_step += 1

            rb.store (points_along_grad[-2], reward_along_grad[-2][0], reward_along_grad[-2][1])

            with open (LOG, 'a') as flog:
                with np.printoptions (precision = 6, suppress = True):          
                    print (f'radius: {radius:.6f}  grad_step: {grad_step - 1}', file = flog)

            points = points + points_along_grad[2:]
            res = res + reward_along_grad[2:]
            rewards = np.array ([r[0] for r in res])

            if not CM: 
                points = points[-2:]; res = res[-2:]; 
                radius_hist.append (radius)

            continue

        if len (points) > 1000: CM = True

except StopException:

    print ("STOP "*10)

finally:

    if cm_changed:

        # saving a current covariance matrix, points, rewards
        # this "finally" block is executed always even after Ctrl C 
        # interruption of the process 

        open (CMFILE, 'w')

        with open (CMFILE, 'ab') as fcm: 
            np.save (fcm, cm)
            np.save (fcm, points)
            np.save (fcm, res[:len (points)])
