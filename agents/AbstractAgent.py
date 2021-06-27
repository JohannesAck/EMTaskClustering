from abc import ABC, abstractmethod
from builtins import NotImplementedError

from common.replay_buffer import EfficientReplayBuffer

class AbstractAgent(ABC):
    def __init__(self, buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay=False,
                 alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6):
        self.batch_size = batch_size
        self.prioritized_replay_eps = prioritized_replay_eps
        self.act_shape_n = act_shape_n
        self.obs_shape_n = obs_shape_n
        if prioritized_replay:
            raise NotImplementedError()
        else:
            self.replay_buffer = EfficientReplayBuffer(int(buff_size), len(obs_shape_n),
                                                       obs_shape_n, act_shape_n)
        self.prioritized_replay = prioritized_replay

    @abstractmethod
    def action(self, obs):
        pass

    @abstractmethod
    def target_action(self, obs):
        pass

    def add_transition(self, obs_n, act_n, rew, new_obs_n, done_n):
        """
        Adds a transition to the replay buffer.
        """
        self.replay_buffer.add(obs_n, act_n, rew, new_obs_n, float(done_n))

    @abstractmethod
    def update(self, agents, step):
        """
        Updates the Agent according to its implementation, i.e. performs (a) learning step.
        :return: Q_loss, Policy_loss
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, fp):
        """
        Saves the Agent to the specified directory.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, fp):
        """
        Loads weights from a given file.
        Has to be called on a fitting agent, that was created with the same hyperparams.
        """
        raise NotImplementedError()
