"""
1D environments
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces import Discrete, Box

from custom_env.bipedal_walker_baseclass import BipedalWalker

class DiscreteLeftRightEnv(gym.Env):

    def __init__(self, size=11, right_goal=True, reward_density=0.0):
        """
        Discrete 1D environment.
        :param size: total number of fields
        :param right_goal: true if goal on the right end
        """
        self.size = size
        self.right_goal = right_goal
        self.goal_position = (size - 1) if right_goal else 0
        self.start_position = np.array([size // 2])
        self.reward_density = reward_density   # 0.0 : only at end, 0.1 every 10 steps, 0.2 every 5 etc

        self.observation_space = Discrete(size)
        self.action_space = Discrete(2)

        self.agent_position = self.start_position.copy()

    def my_init(self, params):
        params = deepcopy(params)
        if 'reward_density' in params.keys():
            self.reward_density = params['reward_density']
            del params['reward_density']
        if 'size' in params.keys():
            self.size = params['size']
            del params['size']
        if 'right_goal' in params.keys():
            self.right_goal = params['right_goal']
            del params['right_goal']
        del params['environment']
        if params.keys():
            raise RuntimeError('Unused keys {}'.format(params.keys()))

        self.start_position = np.array([self.size // 2])
        self.goal_position = (self.size - 1) if self.right_goal else 0
        self.observation_space = Discrete(self.size)

        self.reward_threshold = self.get_optimal_reward()

    def get_optimal_reward(self):
        total_reward = 0.0
        obs = self.reset()
        done = False
        while not done:
            act = int(self.right_goal)
            new_obs, rew, done, _ = self.step(act)
            obs = new_obs
            total_reward += rew
        return total_reward


    def reset(self):
        self.agent_position = self.start_position.copy()
        return self.observation()

    def observation(self):
        return self.agent_position.copy()

    def step(self, action):
        if action == 0:
            self.agent_position[0] -= 1
        elif action == 1:
            self.agent_position[0] += 1

        if self.agent_position[0] == -1:
            self.agent_position[0] = 0
        elif self.agent_position[0] == self.size:
            self.agent_position[0] = self.size - 1

        return self.observation(), self.reward(), self.is_terminal(), {'goal': np.array(self.goal_position)}

    def is_terminal(self):
        if np.all(self.agent_position == self.goal_position):
            return True
        return False

    def reward(self):
        rew = -0.1
        if np.all(self.agent_position == self.goal_position):
            return rew + 20.0
        if self.reward_density == 0.0:
            return rew
        else:
            reward_freq = int(1.0 / self.reward_density)
            if int(self.agent_position[0]) % reward_freq == 0:
                pos_rew = 1.0 / np.abs((self.goal_position - self.agent_position[0]))
                if ((self.agent_position < self.size // 2) and self.right_goal) or \
                    ((self.agent_position > self.size // 2) and not self.right_goal):
                    return rew - pos_rew
                else:
                    return rew + pos_rew
            else:
                return rew

    def render(self, mode='human'):
        plt.figure()
        plt.scatter(self.agent_position, [0], marker='o', s=400)
        plt.scatter(self.goal_position, [0], marker='x', s=400)
        plt.plot([0, self.size - 1], [0, 0], '--')
        plt.show()

    # def render(self, mode='human'):
    #     out_str = ''
    #     for i in range(self.size):
    #         if all(self.agent_position == np.array([i])):
    #             out_str += 'A \t'
    #         elif all(self.goal_position == np.array([i])):
    #             out_str += 'G \t'
    #         else:
    #             out_str  += 'o \t'
    #     print(out_str)


class ContinuousLeftRightEnv(gym.Env):
    def __init__(self, size=11, right_goal=True, reward_density=0.0):
        """
        Same as the discrete 1D environment, but state is continuous and the action
        is a force on the agent.
        :param size: total number of fields
        :param right_goal: true if goal on the right end
        """
        self.size = size
        self.right_goal = right_goal
        self.reward_density = reward_density
        self.goal_position = (size - 1) if right_goal else 0
        self.start_position = np.array([size / 2], dtype=np.float32)

        self.observation_space = Box(- size / 2.0, size / 2.0, [2])
        self.action_space = Box(-1.0, 1.0, [1])


        self.agent_position = self.start_position.copy()
        self.agent_velocity = np.array([0.0], dtype=np.float32)

    def my_init(self, params):
        if 'reward_density' in params.keys():
            self.reward_density = params['reward_density']
        if 'size' in params.keys():
            self.size = params['size']
        if 'right_goal' in params.keys():
            self.right_goal = params['right_goal']

    def reset(self):
        self.agent_position = self.start_position.copy()
        self.agent_velocity = np.array([0.0], dtype=np.float32)
        return self.observation()

    def observation(self):
        return np.concatenate([self.agent_position.copy(), self.agent_velocity.copy()])

    def step(self, action: np.ndarray):
        self.agent_velocity[0] += action[0]

        self.agent_position = self.agent_position + self.agent_velocity

        if self.agent_position[0] < 0:
            self.agent_position[0] = 0.0
            self.agent_velocity[0] = 0.0
        elif self.agent_position[0] > self.size - 1:
            self.agent_position[0] = self.size - 1.0
            self.agent_velocity[0] = 0.0

        return self.observation(), self.reward(), self.is_terminal(), {'goal': np.array(self.goal_position)}

    def is_terminal(self):
        if np.all(self.agent_position == self.goal_position):
            return True
        return False

    def reward(self):
        if np.all(self.agent_position == self.goal_position):
            return 2.0
        if self.reward_density == 0.0:
            return 0.0
        else:
            reward_freq = int(1.0 / self.reward_density)
            if int(self.agent_position[0]) % reward_freq == 0:
                return 1.0 / np.abs((self.goal_position - self.agent_position[0]))
            else:
                return 0.0


    def render(self, mode='human'):
        plt.figure()
        plt.scatter(self.agent_position, [0], marker='o', s=400)
        plt.scatter(self.goal_position, [0], marker='x', s=400)
        plt.plot([0, self.size - 1], [0, 0], '--')
        plt.show()


class BipedalLeftRightEnv(BipedalWalker):

    def __init__(self):
        """
        same as default environment, but flat
        """
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0

    def reward(self, state, action, pos, vel):
        reward = 0
        reward += 0.1 * vel.x

        # for a in action:
        #     # increased punishment for energy usage
        #     reward -= 0.00065 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            # reward = -100
            done = True
        if pos[0] > (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        stumps = self.stump_spacing is not None
        state = GRASS
        velocity = 0.0
        y = self.TERRAIN_HEIGHT
        counter = self.TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(self.TERRAIN_LENGTH):
            x = i * self.TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(self.TERRAIN_HEIGHT - y)
                if i > self.TERRAIN_STARTPAD:
                    velocity += self.TERRAIN_VARIANCE * (self.np_random.uniform(-1, 1) / self.SCALE)
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x, y),
                    (x + self.TERRAIN_STEP, y),
                    (x + self.TERRAIN_STEP, y - 4 * self.TERRAIN_STEP),
                    (x, y - 4 * self.TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [(p[0] + self.TERRAIN_STEP * counter, p[1]) for p
                                                  in poly]
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * self.TERRAIN_STEP

            elif state == STUMP and oneshot:
                poly = [
                    (x, y),
                    (x + self.TERRAIN_STEP, y),
                    (x + self.TERRAIN_STEP, y + self.stump_height),
                    (x, y + self.stump_height),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * self.TERRAIN_STEP,
                         y + (s * stair_height) * self.TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * self.TERRAIN_STEP,
                         y + (s * stair_height) * self.TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * self.TERRAIN_STEP,
                         y + (-1 + s * stair_height) * self.TERRAIN_STEP),
                        (x + (s * stair_width) * self.TERRAIN_STEP,
                         y + (-1 + s * stair_height) * self.TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon)
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * self.TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.randint(self.TERRAIN_GRASS / 2, self.TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                elif state == GRASS and stumps:
                    state = STUMP
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(self.TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()


if __name__ == '__main__':
    env = DiscreteLeftRightEnv()
    env.my_init({'size': 71,
                 'reward_density': 0.125,
                 'right_goal': False})
    print('rew_thresh', env.reward_threshold)
    obs = env.reset()

    episode_reward = 0
    while True:
        print(obs)
        env.render()
        action = float(input('Action [0,1]: '))
        obs, reward, terminal, goal_pos = env.step(action)
        episode_reward += reward
        print('obs', obs, 'reward', reward)
        if terminal:
            print('terminal, ', episode_reward)
            episode_reward = 0
            obs = env.reset()
