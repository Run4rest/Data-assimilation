import numpy as np
import os
from collections import namedtuple

#LOGFILE = '.\\log\\log vsac1.log'

class ReplayBuffer:

    def __init__ (self, obs_dim, act_dim, max_size, fname, flag = 0):

        self.db = fname

        self.obs_dim = obs_dim; self.act_dim = act_dim;  

        Buffer = namedtuple ('Buffer', ['steps', 'acts', 'rewards', 'next_steps', 'ends'])
        
        self.data = Buffer (
            steps      = [np.empty ((0, obs_dim), dtype = np.float64)],
            acts       = [np.empty ((0, act_dim), dtype = np.float64)],
            rewards    = [np.empty ((0, 1), dtype = np.float64)],
            next_steps = [np.empty ((0, obs_dim), dtype = np.float64)], 
            ends       = [np.empty ((0, 1), dtype = np.float64)]        
        )

        self.total_size = 0; self.max_size = max_size

        try: 
            if flag: os.remove (self.db)
            else: self.load()
        except FileNotFoundError: pass   


    def store (self, *args): #step, act, reward, next_step, end

        for x, rec in zip (self.data, args):
            x[0] = np.append (x[0], np.array (rec, ndmin = 2), axis = 0)

        a = np.concatenate (tuple (x[0][-1, :] for x in self.data), axis = 0, dtype = np.float64)

        with open (self.db, 'ab') as fdb: np.save (fdb, a)

        if self.data.steps[0].shape[0] > self.max_size:

            # with open (LOGFILE, 'a') as flog:
            #     with np.printoptions (precision = 3, suppress = True, linewidth = 150):          
            #         print (f'steps[0]:  \n{self.data.steps[0][:2]}',  file = flog)
            #         print (f'steps[-1]: \n{self.data.steps[0][-2:]}', file = flog)
            #         print (f'steps.shape: \n{self.data.steps[0].shape}', file = flog)

            for x in self.data: x[0] = x[0][1:]

            # with open (LOGFILE, 'a') as flog:
            #     with np.printoptions (precision = 3, suppress = True, linewidth = 150):          
            #         print (f'steps[0]:  \n{self.data.steps[0][:2]}',  file = flog)
            #         print (f'steps[-1]: \n{self.data.steps[0][-2:]}', file = flog)
            #         print (f'steps.shape: \n{self.data.steps[0].shape}', file = flog)

        self.total_size += 1


    def load (self):

        with open (self.db, 'rb') as fdb:
            l = []
            while True:
                l.append (np.array (np.load (fdb), ndmin = 2))
                if fdb.read(1) == b'': break
                fdb.seek (-1, 1)

        na = np.concatenate (l, axis = 0)

        self.data.steps[0]      = na[:, : self.obs_dim]
        self.data.acts[0]       = na[:, self.obs_dim : self.obs_dim + self.act_dim]
        self.data.rewards[0]    = na[:, self.obs_dim + self.act_dim : self.obs_dim + self.act_dim + 1]
        self.data.next_steps[0] = na[:, self.obs_dim + self.act_dim + 1 : self.obs_dim + self.act_dim + 1 + self.obs_dim]
        self.data.ends[0]       = na[:, -1:] 

        self.total_size = self.data.steps[0].shape[0]

        # with open (LOGFILE, 'a') as flog:
        #     with np.printoptions (precision = 3, suppress = True, linewidth = 150):          
        #         print (f'steps[0]:  \n{self.data.steps[0][:2]}',  file = flog)
        #         print (f'steps[-1]: \n{self.data.steps[0][-2:]}', file = flog)
        #         print (f'steps.shape #1: \n{self.data.steps[0].shape}', file = flog)

        if self.data.steps[0].shape[0] > self.max_size:

            for x in self.data: x[0] = x[0][-self.max_size:]

        # with open (LOGFILE, 'a') as flog:
        #     with np.printoptions (precision = 3, suppress = True, linewidth = 150):          
        #         print (f'steps[0]:  \n{self.data.steps[0][:2]}',  file = flog)
        #         print (f'steps[-1]: \n{self.data.steps[0][-2:]}', file = flog)
        #         print (f'steps.shape #2: \n{self.data.steps[0].shape}', file = flog)

        return self
