import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras import layers
from tensorflow.keras import Model
from collections import namedtuple

tf.keras.backend.set_floatx ('float64')

EPSILON = 1e-16

NEURONS = 256

class Actor (Model):

    def __init__(self, act_dim):
        super().__init__()
        self.act_dim = act_dim
#        self.dense1_layer  = SpectralNormalization (layers.Dense (NEURONS, activation = tf.nn.relu, kernel_initializer = tf.keras.initializers.GlorotNormal()))
#        self.dense2_layer  = SpectralNormalization (layers.Dense (NEURONS, activation = tf.nn.relu, kernel_initializer = tf.keras.initializers.GlorotNormal()))
        self.dense1_layer  = layers.Dense (NEURONS, activation = tf.nn.relu, kernel_initializer = tf.keras.initializers.GlorotNormal())
        self.dense2_layer  = layers.Dense (NEURONS, activation = tf.nn.relu, kernel_initializer = tf.keras.initializers.GlorotNormal())
        self.mean_layer    = layers.Dense (self.act_dim)
        self.stdev_layer   = layers.Dense (self.act_dim)


    def call (self, state):

        # Get mean and standard deviation from the policy network
        a1 = self.dense1_layer (state)
        a2 = self.dense2_layer (a1)
        mu = self.mean_layer (a2)

        # Standard deviation is bounded by a constraint of being non-negative
        # therefore we produce log stdev as output which can be [-inf, inf]
        log_sigma = self.stdev_layer (a2)
        sigma = tf.tanh (log_sigma) + 1.0 + 1.0e-8
        mu = tf.tanh (mu)

        # Use re-parameterization trick to deterministically sample action from
        # the policy network. First, sample from a Normal distribution of
        # sample size as the action and multiply it with stdev
        dist = tfp.distributions.Normal (mu, sigma)
        a = dist.sample()

        # Apply the tanh squashing to keep the gaussian bounded in (-1,1)
        action = tf.tanh (a)

        # Calculate the log probability
        log_pi_ = dist.log_prob (a)
        # Change log probability to account for tanh squashing as mentioned in
        # Appendix C of the paper
        log_pi = log_pi_ - tf.reduce_sum (tf.math.log (1 - action**2 + EPSILON), axis = 1, keepdims = True)

        return action, log_pi

class Critic (Model):

    def __init__ (self):
        super().__init__()
#        self.dense1_layer = SpectralNormalization (layers.Dense (NEURONS, activation = tf.nn.relu))
#        self.dense2_layer = SpectralNormalization (layers.Dense (NEURONS, activation = tf.nn.relu))
        self.dense1_layer = layers.Dense (NEURONS, activation = tf.nn.relu)
        self.dense2_layer = layers.Dense (NEURONS, activation = tf.nn.relu)
        self.output_layer = layers.Dense (1)

    def call (self, state, action, training = False):
        state_action = tf.concat ([state, action], axis = 1)
        a1 = self.dense1_layer (state_action)
#        bn1 = self.bn (a1, training = training)
        a2 = self.dense2_layer (a1)
        q  = self.output_layer (a2)
        return q

class SoftActorCritic:

    def __init__(self, act_dim, env, learning_rate = 0.0003,
                 alpha = -1.61, gamma = 1.0, smoothing = 0.995):
        
        self.action_size = act_dim
        self.polnet = Actor (act_dim)
        self.q1 = Critic()
        self.q2 = Critic()
        self.target_q1 = Critic()
        self.target_q2 = Critic()

        self.env = env

        self.alpha  = tf.Variable (alpha, dtype = tf.float64)
        self.target_entropy = -tf.constant (act_dim*10, dtype = tf.float64)
        self.gamma = gamma
        self.smoothing = smoothing

        self.actor_optimizer   = tf.keras.optimizers.Adam (learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam (learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam (learning_rate)
        self.alpha_optimizer   = tf.keras.optimizers.Adam (learning_rate/10.0)


    def sample_action (self, obs):

        obs = np.array (obs, ndmin = 2)
        a = self.polnet(obs)[0].numpy()[0]

        obs = np.squeeze (obs)
        if obs[-1] > 0:
            a = np.concatenate ((a[:2], obs[-4:-1]), axis = 0)

        return a


    def update_q_network (self, data, bsize):

        steps, actions, rewards, next_steps, ends = (tf.convert_to_tensor (x[0]) for x in data)

        # Sample actions from the policy for next states
        a_next, log_pi = self.polnet (next_steps)

        a_next_with_replacement = tf.concat ((a_next[:, :2], next_steps[:, -4:-1]), 1)
        a_next = tf.where (next_steps[:, -1:] == 0, a_next, a_next_with_replacement)

        indices = tf.range (start = 0, limit = tf.shape (steps)[0], dtype = tf.int32)
        shuffled_indices = tf.random.shuffle (indices)

        datalist = []

        for x in [steps, actions, rewards, next_steps, ends, a_next, log_pi]:
            x = tf.gather (x, shuffled_indices)
            datalist.append (tf.data.Dataset.from_tensor_slices (x))

        dataset = tf.data.Dataset.zip (tuple (datalist)).batch (bsize)

        losses = []

        for bsteps, bactions, brewards, bnext_steps, bends, ba_next, blog_pi in dataset:

            with tf.GradientTape() as tape1:

                # Get Q value estimates, action used here is from the replay buffer
                q1 = self.q1 (bsteps, bactions, training = True)

                # Get Q value estimates from target Q network
                q1_target = self.target_q1 (bnext_steps, ba_next)
                q2_target = self.target_q2 (bnext_steps, ba_next)

                # Apply the clipped double Q trick
                # Get the minimum Q value of the 2 target networks
                min_q_target = tf.minimum (q1_target, q2_target)

                # Add the entropy term to get soft Q target
                soft_q_target = min_q_target - tf.math.exp (self.alpha) * blog_pi
                y = tf.stop_gradient (brewards + self.gamma * bends * soft_q_target)

                critic1_loss = tf.reduce_mean ((q1 - y)**2)

            with tf.GradientTape() as tape2:

                # Get Q value estimates, action used here is from the replay buffer
                q2 = self.q2 (bsteps, bactions, training = True)

                # Get Q value estimates from target Q network
                q1_target = self.target_q1 (bnext_steps, ba_next)
                q2_target = self.target_q2 (bnext_steps, ba_next)

                # Apply the clipped double Q trick
                # Get the minimum Q value of the 2 target networks
                min_q_target = tf.minimum (q1_target, q2_target)

                # Add the entropy term to get soft Q target
                soft_q_target = min_q_target - tf.math.exp (self.alpha) * blog_pi
                y = tf.stop_gradient (brewards + self.gamma * bends * soft_q_target)

                critic2_loss = tf.reduce_mean ((q2 - y)**2)

            grads1 = tape1.gradient (critic1_loss, self.q1.trainable_variables)
            self.critic1_optimizer.apply_gradients (zip (grads1, self.q1.trainable_variables))

            grads2 = tape2.gradient (critic2_loss, self.q2.trainable_variables)
            self.critic2_optimizer.apply_gradients (zip (grads2, self.q2.trainable_variables))

            for theta_target, theta in zip (self.target_q1.trainable_variables, self.q1.trainable_variables):
                theta_target = self.smoothing*theta_target + (1 - self.smoothing)*theta

            for theta_target, theta in zip (self.target_q2.trainable_variables, self.q2.trainable_variables):
                theta_target = self.smoothing*theta_target + (1 - self.smoothing)*theta

            losses.append (critic1_loss.numpy())

        return np.array (losses)


    def update_policy_network (self, steps, bsize):

        steps = tf.convert_to_tensor (steps)

        Pol = namedtuple ('Pol', ['net', 'data', 'optimizer'])

        pol  = Pol (net = self.polnet, data = steps, optimizer = self.actor_optimizer) 

        indices = tf.range (start = 0, limit = tf.shape (pol.data)[0], dtype = tf.int32)
        shuffled_indices = tf.random.shuffle (indices)
        d = tf.gather (pol.data, shuffled_indices)

        dataset = tf.data.Dataset.from_tensor_slices (d)

        losses = []

        for bsteps in dataset.batch (bsize):

            with tf.GradientTape() as tape:

                # Sample actions from the policy for current states
                a, log_pi = pol.net (bsteps, training = True)

                a_with_replacement = tf.concat ((a[:, :2], tf.stop_gradient (bsteps[:, -4:-1])), 1)
                a = tf.where (bsteps[:, -1:] == 0, a, a_with_replacement)

                # Get Q value estimates from target Q network
                q1 = self.q1 (bsteps, a)
                q2 = self.q2 (bsteps, a)

                # Apply the clipped double Q trick
                # Get the minimum Q value of the 2 target networks
                min_q = tf.minimum (q1, q2)

                soft_q = min_q - tf.math.exp (self.alpha) * log_pi

                loss = -1.0*tf.reduce_mean (soft_q)

            grads = tape.gradient (loss, pol.net.trainable_variables)
            pol.optimizer.apply_gradients (zip (grads, pol.net.trainable_variables))

            losses.append (loss.numpy())
        
        return np.array (losses)

    def train (self, data, bsize):

        # Update Q network weights
        qlosses = self.update_q_network (data, bsize)

        # Update policy network weights
        plosses = self.update_policy_network (data.steps[0], bsize)

#        self.update_alpha (data.steps[0], bsize)

        return plosses, qlosses


    def update_alpha (self, steps, bsize):

        dataset = tf.data.Dataset.from_tensor_slices (steps).batch (bsize)

        for btss in dataset:

            with tf.GradientTape() as tape:
                # Sample actions from the policy for current states
                pi_a, log_pi_a = self.polnet (btss)

                alpha_loss = tf.reduce_mean ( - tf.math.exp (self.alpha)*(log_pi_a + self.target_entropy))

            variables = [self.alpha]
            grads = tape.gradient (alpha_loss, variables)
            self.alpha_optimizer.apply_gradients (zip (grads, variables)) 
