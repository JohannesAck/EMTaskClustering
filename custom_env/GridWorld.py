"""
Simple gridworld implementation.
"""

import numpy as np
from gym import Env
from gym.spaces import Discrete


class GridWorld(Env):

    def __init__(self, size=5, goal_position=(0, 0), sparse=True):
        """
        Goal position specified as location on x axis
        :param size:
        :param goal_position:
        """
        self.size = size
        self.goal_position = np.array([self.size - 1, self.size - 1])
        # self.start_position = np.array([size//2, size//2])
        self.start_position = np.array([0, 0])
        self.sparse = sparse
        self.observation_space = Discrete(self.size ** 2)
        self.action_space = Discrete(4)
        self.reward_threshold = 10.0 if sparse else 14.0  # todo this 14 is obv incorrect

        self.agent_position = self.start_position.copy()


    def my_init(self, params):
        if 'size' in params.keys():
            self.size = params['size']
        else:
            raise RuntimeError('Size not specified in Task')

        self.start_position = np.array([0, 0])
        # self.start_position = np.array([self.size//2, self.size//2])
        self.observation_space = Discrete(self.size ** 2)
        self.action_space = Discrete(4)
        self.goal_position = np.array([self.size - 1, self.size - 1])

        self.agent_position = self.start_position.copy()


    def reset(self):
        self.agent_position = self.start_position.copy()
        return self.observation()

    def observation(self):
        return np.array([self.agent_position[0] + self.size * self.agent_position[1]])

    def step(self, action):
        if action == 0:
            self.agent_position[0] += 1
        elif action == 1:
            self.agent_position[1] += 1
        elif action == 2:
            self.agent_position[0] -= 1
        elif action == 3:
            self.agent_position[1] -= 1

        if self.agent_position[0] == -1:
            self.agent_position[0] = 0
        elif self.agent_position[0] == self.size:
            self.agent_position[0] = self.size - 1
        if self.agent_position[1] == -1:
            self.agent_position[1] = 0
        elif self.agent_position[1] == self.size:
            self.agent_position[1] = self.size - 1

        return self.observation(), self.reward(), self.is_terminal(), {'goal' :self.goal_position}

    def is_terminal(self):
        if np.all(self.agent_position == self.goal_position):
            return True
        return False

    def reward(self):
        if self.sparse:
            if np.all(self.agent_position == self.goal_position):
                return 10.0
            else:
                return 0.0
        else:
            if np.all(self.agent_position == self.goal_position):
                return 10.0
            else:
                return 1.0 / np.sqrt(np.sum(np.square(self.agent_position - self.goal_position)))

    def render(self, mode='human'):
        print(self.__repr__())

    def __repr__(self):
        # this function is stupid
        out_str = ''
        for i in range(self.size):
            for j in range(self.size):
                if all(self.agent_position == np.array([i, j])):
                    out_str += 'A \t'
                elif all(self.goal_position == np.array([i, j])):
                    out_str += 'G \t'
                else:
                    out_str  += 'o \t'
            out_str +=  '\n'
        return out_str


class CornerGridWorld(GridWorld):

    def __init__(self, size=5, goal_position=(0, 0), sparse=False):
        """
        Gridworld with start in middle, and goal position in one of the corners, specified by
        index in order [0,0] [0,size-1] [size-1, size-1] [size-1, 0].
        Additionally an offset is specified, as [x y] toward the center.
        """
        super().__init__(size, goal_position, sparse)
        self.start_position = np.array([size//2, size//2])

    def my_init(self, params):
        if 'size' in params.keys():
            self.size = params['size']
        if 'goal_idx' in params.keys():
            self.goal_idx = params['goal_idx']
        if 'goal_os' in params.keys():
            self.goal_os = params['goal_os']
        else:
            raise RuntimeError('Goal offset not specified in CornerGridWorld')

        self.observation_space = Discrete(self.size ** 2)


        self.start_position = np.array([self.size//2, self.size//2])
        if self.goal_idx == 0:
            self.goal_position = np.array([0, 0])
        elif self.goal_idx == 1:
            self.goal_position = np.array([0, self.size - 1])
        elif self.goal_idx == 2:
            self.goal_position = np.array([self.size - 1, self.size - 1])
        elif self.goal_idx == 3:
            self.goal_position = np.array([self.size - 1, 0])

        # add offset
        if self.goal_position[0] == 0:
            self.goal_position[0] += self.goal_os[0]
        else:
            self.goal_position[0] -= self.goal_os[0]
        if self.goal_position[1] == 0:
            self.goal_position[1] += self.goal_os[1]
        else:
            self.goal_position[1] -= self.goal_os[1]

        self.agent_position = self.start_position.copy()



