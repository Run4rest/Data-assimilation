import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from tf_agents.trajectories import time_step as ts, StepType
from collections import OrderedDict as OD

SEED = 42; tf.random.set_seed (SEED); 
RNG = np.random.default_rng (SEED)
EPSILON = 1.0e-10; 

PERIOD = 5; INTERVAL = 0.25
BATCH_SIZE = 256
LEARNING_INTERVAL = 1
REPORT_INTERVAL   = 10
BUFFER_SIZE = 100000
VERSION = 1

from replay_buffer_vsac import ReplayBuffer

class SACVersionError (Exception): pass

if VERSION == 1:
    from environment_vsac1 import ODEEnv
    from sac_v1 import SoftActorCritic
elif VERSION == 2:   
    from environment_vsac2 import ODEEnv
    from sac_v2 import SoftActorCritic
else:
    raise SACVersionError

LOGFILE = '.\\log\\log vsac.log'
DBFILE  = '.\\log\\db vsac.npy'

open (LOGFILE, 'w')

logging.basicConfig (
    filename = LOGFILE,
    format = '%(asctime)s %(message)s',
    level  = logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

class StopException (Exception): pass

def print_results (replay_buffer, env, plosses, qlosses):

    NOS = int ((PERIOD + EPSILON)//INTERVAL)
    res_act = replay_buffer.data.acts[0][-NOS:, :]
    res_obs = replay_buffer.data.steps[0][-NOS:, :]
    # idx = np.where (res_obs[:, -1] == 0)
    # obs0 = res_obs[idx]
    # act0 = res_act[idx]
    avg_params = np.mean (res_act[:,:len(env.params_space)], axis = 0)
    avg_params = OD ([(k, env.params_space[k][0] + 0.5*env.span_of_params[k]*(1 + v)) for k, v in zip (env.span_of_params, avg_params)])

    with open (LOGFILE, 'a') as flog:
        with np.printoptions (precision = 3, suppress = True, linewidth = 150):          
            print ('', file = flog)
            for p in avg_params.items():
                print (f'{p[0]} = {p[1]:.6f}', file = flog)

#            print (f'obs0: {obs0[:, -2*len(env.bounds_space) - 1 : -len(env.bounds_space) - 1]}  act0: {act0[:, -len(env.bounds_space):]}', file = flog)
            print (f'\nP losses: {plosses}', file = flog)
            print (f'\nQ losses: {qlosses}', file = flog)
            print (f'\nacts: \n{res_act}', file = flog)
            print (f'\nobss: \n{res_obs}\n', file = flog)
            

tf.keras.backend.set_floatx ('float64')

# Instantiate the environment.
env = ODEEnv()
observ_dim = env.observation_spec().shape[0]
action_dim = env.action_spec().shape[0]

# Initialize Replay buffer.
replay_buffer = ReplayBuffer (observ_dim, action_dim, BUFFER_SIZE, DBFILE, 1)
#replay_buffer = ReplayBuffer (observ_dim, action_dim, BUFFER_SIZE, DBFILE)

# Initialize policy and Q-function parameters.
sac = SoftActorCritic ( action_dim, env, learning_rate = 0.0003, gamma = 1.0, smoothing = 0.995)

# Repeat until convergence
step_count = 1; episode = 0; AER = 0.0 

try:
    while True:
        # Observe state
        step = env.reset()
        episode_reward = 0

        while step.step_type != StepType.LAST:

            action = sac.sample_action (step.observation)

            # Execute action, observe next state and reward
            try:
                next_step = env.step (action)
            except Exception:
                logging.info (f"ERROR, action : {action}")
                raise

            episode_reward += next_step.reward

            end = 0 if next_step.step_type == StepType.LAST else 1

            # Store transition in replay buffer
#            print (f'action shape: {action.shape}')
            replay_buffer.store (step.observation, action, next_step.reward, next_step.observation, end)

            # Update current state
            step = next_step; step_count += 1

            if (step_count % LEARNING_INTERVAL) == 0:
                for _ in range (1): 
                    plosses, qlosses = sac.train (replay_buffer.data, BATCH_SIZE)

        aer_smoothing = 0.9
        episode += 1; AER = aer_smoothing*AER + (1 - aer_smoothing)*episode_reward

        logging.info (f"Ep. {episode}: AER: {AER:.4f}     LS: {step.observation[-1]}      APLHA: {sac.alpha.numpy()}")

        if (episode % REPORT_INTERVAL) == 0:
            print (f'episode: {episode}  DB file size: {replay_buffer.total_size}  DB size: {replay_buffer.data.steps[0].shape[0]}  P loss: {plosses.mean():.4f}  Q loss: {qlosses.mean():.4f}')
            print_results (replay_buffer, env, plosses, qlosses)

#        if step_count > 4096*120 + 512: raise StopException

except StopException:

    print_results (replay_buffer, env); print ("STOP "*5)
