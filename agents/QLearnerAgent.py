
from agents.AbstractAgent import AbstractAgent
from gym.spaces import Discrete
import numpy as np

class QLearnerFull(AbstractAgent):

    def __init__(self, action_space, q_table_size, decay, lr, eps, eps_decay):
        if not isinstance(action_space, Discrete):
            raise RuntimeError('QLearner only supports Discrete Action Space')
        self.action_space = action_space
        self.q_table_size = q_table_size
        self.decay = decay
        self.lr = lr
        self.q_val = np.zeros([q_table_size, action_space.n])
        self.q_table_size = q_table_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.step = 0

    def get_q_val(self, obs, act):
        if obs[0] >= self.q_table_size:
            print('error')
        return self.q_val[obs[0], act]

    def set_q_val(self, obs, act, val):
        self.q_val[obs[0], act] = val

    def deterministic_action(self, obs):
        action_values = [self.get_q_val(obs, act) for act in range(self.action_space.n)]
        max_val = np.max(action_values)
        max_val_acts = np.where(action_values == max_val)[0]
        return np.random.choice(max_val_acts)[None]

    def action(self, obs):
        if np.random.rand() < self.eps * self.eps_decay ** self.step:
            return np.array(self.action_space.sample())[None]
        return self.deterministic_action(obs)

    def target_action(self, obs):
        return self.action(obs)

    def train(self, obs, act, rew, new_obs):
        best_next_val = np.max([self.get_q_val(new_obs, act) for
                                act in range(self.action_space.n)])
        new_val = rew + self.decay * best_next_val
        new_val_alpha = (1 - self.lr) * self.get_q_val(obs, act) + self.lr * new_val
        self.set_q_val(obs, act, new_val_alpha)

    def add_transition(self, obs, act, rew, new_obs, done):
        self.train(obs, act, rew, new_obs)

    def update(self, agents, step):
        self.step = step

    def save(self, fp):
        pass

    def load(self, fp):
        pass

