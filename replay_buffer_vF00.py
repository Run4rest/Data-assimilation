import os
import numpy as np
from collections import namedtuple

class ReplayBuffer:

    def __init__ (self, point_dim, fname, flag = 0):

        Buffer = namedtuple ('Buffer', ['points', 'rewards', 'reward0'])
        
        self.data = Buffer (
            points  = [np.empty ((0, point_dim), dtype = np.float64)],
            rewards = [np.empty ((0, 1),         dtype = np.float64)],
            reward0 = [np.empty ((0, 1),         dtype = np.float64)],
        )

        self.total_size = 0

        self.db = fname; self.point_dim = point_dim

        try: 
            if flag: os.remove (self.db)
            else: self.load()
        except FileNotFoundError: pass   


    def store (self, *args): #point, reward

        for x, rec in zip (self.data, args):
            x[0] = np.append (x[0], np.array (rec, ndmin = 2), axis = 0)

        a = np.concatenate (tuple (x[0][-1, :] for x in self.data), axis = 0, dtype = np.float64)

        with open (self.db, 'ab') as fdb: np.save (fdb, a)

        self.total_size += 1


    def load (self):

        with open (self.db, 'rb') as fdb:
            l = []
            while True:
                l.append (np.array (np.load (fdb), ndmin = 2))
                if fdb.read(1) == b'': break
                fdb.seek (-1, 1)

        na = np.concatenate (l, axis = 0)

        self.data.points[0]  = na[:, : self.point_dim]
        self.data.rewards[0] = na[:, self.point_dim    : self.point_dim + 1]
        self.data.reward0[0] = na[:, self.point_dim +1 : self.point_dim + 2]

        self.total_size = self.data.points[0].shape[0]

        return self