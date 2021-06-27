"""
Variations bsaed on the BipedalWalker base class, that represent different tasks inspired by
track and field competitions.
"""
import numpy as np

from custom_env.bipedal_walker_baseclass import BipedalWalker


class BipedalHighJump(BipedalWalker):
    def __init__(self):
        self.last_contact_y = None
        self.in_flight = False

        super().__init__()
        self.TERRAIN_VARIANCE = 0.0

    def additional_reset(self):
        self.last_contact_y = None
        self.in_flight = False

    def reward(self, state, action, pos, vel):
        reward = 0

        if not self.legs[1].ground_contact and not self.legs[3].ground_contact:
            self.in_flight = True
        else:
            if self.in_flight:  # just regained ground contact
                reward += pos.y - self.last_contact_y
            self.in_flight = False
            self.last_contact_y = pos.x

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward


class BipedalLongJump(BipedalWalker):
    def __init__(self):
        self.in_flight = False
        self.last_contact_x = None

        super().__init__()
        self.TERRAIN_VARIANCE = 0.0

    def additional_reset(self):
        self.in_flight = False
        self.last_contact_x = None

    def reward(self, state, action, pos, vel):
        reward = 0
        # reward += 0.01 * vel.x
        if not self.legs[1].ground_contact and not self.legs[3].ground_contact:
            self.in_flight = True
        else:
            if self.in_flight:
                reward += pos.x - self.last_contact_x
            self.in_flight = False
            self.last_contact_x = pos.x

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > 0.3 * (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward


class BipedalShortSprint(BipedalWalker):
    def __init__(self):
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0

    def reward(self, state, action, pos, vel):
        reward = 0
        reward += 0.1 * vel.x

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > 0.3 * (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward


class BipedalMediumRun(BipedalWalker):
    def __init__(self):
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0

    def reward(self, state, action, pos, vel):
        reward = 0
        reward += 0.1 * vel.x

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > 0.5 * (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward


class BipedalMarathon(BipedalWalker):
    def __init__(self):
        """
        same as default environment, but flat
        """
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0

    def reward(self, state, action, pos, vel):
        reward = 0
        reward += 0.1 * vel.x

        for a in action:
            # increased punishment for energy usage
            reward -= 0.00065 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward


class BipedalHurdles(BipedalWalker):
    def __init__(self):
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0
        self.stump_spacing = 4.0
        self.stump_height = 1.0
        self.my_init({'leg_length': 35, 'walker_type': 'default'})

    def reward(self, state, action, pos, vel):
        reward = 0
        reward += 0.1 * vel.x

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward

class BipedalSparseHurdles(BipedalWalker):
    def __init__(self):
        """
        This is pretty hard without transfer.
        """
        self.last_reward_pos = 0
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0
        self.stump_spacing = 4.0
        self.stump_height = 1.0
        self.my_init({'leg_length': 35, 'walker_type': 'default'})

    def additional_reset(self):
        self.last_reward_pos = 0

    def reward(self, state, action, pos, vel):
        reward = 0

        if pos.x > self.last_reward_pos + 4.0:
            reward += 10
            self.last_reward_pos = pos.x

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward


class BipedalHurdlesNoFall(BipedalWalker):
    def __init__(self):
        super().__init__()
        self.TERRAIN_VARIANCE = 0.0
        self.stump_spacing = 4.0
        self.stump_height = 1.0
        self.my_init({'leg_length': 35, 'walker_type': 'default'})

    def reward(self, state, action, pos, vel):
        reward = 0
        reward += 0.1 * vel.x

        # for a in action:
        #     reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        return reward

    def done(self, pos, reward):
        done = False
        if self.fell_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            done = True
        return done, reward



if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    # env = BipedalLo;ngJump()
    env = BipedalHighJump()
    env.my_init({ 'leg_length': 35, 'walker_type' : 'default'})
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4]]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9]]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
            print('pos', s[14:16])
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]: hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]: knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]: knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        env.render()
        if done: break
