"""
Implementation with the network structure from  "Sharing Knowledge in Multi-Task Deep Reinforcement
Learning" by Eramo et al. (ICLR 2020)

Instead of the MDDPG presented in their paper, this here implements a version of TD3
with the same network structure they proposed, which pretty much just means we use a
double critic and target policy smoothing, as well the decreased update rate for policy and
target networks.

This file is a bit less documented than train_em.py, but they are mostly the same so please check
there as well if something is unclear here.
"""

import os
import time
import pickle

import gym
from gym.spaces import Discrete
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


from agents.MTD3Agent import MTD3Agent
import custom_env
from common.sacred_util import get_run_path


train_ex = Experiment('Multi-Head')


@train_ex.config
def train_config():
    ### Logging
    display = False  # 'DISPLAY' in os.environ

    ### Environment
    #  'pendulum' for the pendulum taskset
    #  'corner-gridworld-v0' for the gridworld-corner taskset
    #  'leftright-discrete-v0' for the 1D-chain taskset
    #  'biped' for the BipedalWalker taskset with varying leg lengths and obstacles
    #  'track' for the BipedalWalker taskset with different rewards, inspired by track and field
    #           For Atari please see train_atari_em.py
    environment = 'track'

    checkpoint_freq = 1e6  # checkpoint every n training steps
    continue_cp = None  # path to a checkpoint.pkl file to continue training from

    task_list = []

    if environment == 'leftright-discrete-v0':
        traj_len = 51
        sizes = [51]
        reward_densities = [0.0625, 0.0, 0.5, 0.25, 0.125, ]
        right_goal = [True, False]
        iterations = 1
        for dens in reward_densities:
            for size in sizes:
                for it in range(iterations):
                    for rg in right_goal:
                        task_list.append({'environment': 'leftright-discrete-v0',
                                          'size': size,
                                          'reward_density': dens,
                                          'right_goal': rg})
        del size
        del it
        del rg
        if sizes[0] > traj_len:
            raise RuntimeError('Task impossible')
    elif environment == 'corner-gridworld-v0':
        traj_len = 7
        sizes = [7]
        goal_idxs = [0, 1, 2, 3]
        goal_oss = [[0, 0], [1, 0], [0, 1]]

        iterations = 1
        for size in sizes:
            for it in range(iterations):
                for goal_idx in goal_idxs:
                    for goal_os in goal_oss:
                        task_list.append({'environment': 'corner-gridworld-v0',
                                          'size': size,
                                          'goal_idx': goal_idx,
                                          'goal_os': goal_os})
        del size
        del it
        del goal_idx
        del goal_os
        if sizes[0] > traj_len:
            raise RuntimeError('Task impossible')
    if environment == 'biped':
        walker_types = ['default']
        leg_length = [25, 35, 45]
        stump_height = [0.0, 0.5]
        obstacle_spacing = [2.0, 4.0]
        task_list = []
        # manual task list generation to fit well with oracle (difficult with product):
        for t in walker_types:
            for l in leg_length:
                for h in stump_height:
                    if h == 0.0:
                        # spacing does not matter if height is 0
                        task_list.append({'environment': 'bipedal-walker-continuous-v0',
                                          'walker_type': t,
                                          'leg_length': l,
                                          'stump_height': h,
                                          'obstacle_spacing': obstacle_spacing[-1], })
                    else:
                        for s in obstacle_spacing:
                            task_list.append({'environment': 'bipedal-walker-continuous-v0',
                                              'walker_type': t,
                                              'leg_length': l,
                                              'stump_height': h,
                                              'obstacle_spacing': s})
        oracle_curriculum = list(np.arange(len(task_list)))
        target_task_id = 0
        traj_len = 2000
    elif environment == 'track':
        traj_len = 2000
        walker_types = ['default']
        leg_length = [45]
        env_name = ['bipedal-highjump-v0',
                    'bipedal-longjump-v0',
                    'bipedal-shortsprint-v0',
                    'bipedal-mediumrun-v0',
                    'bipedal-marathon-v0',
                    'bipedal-hurdles-v0',
                    ]
        task_list = []
        # manual task list generation to fit well with oracle (difficult with product):
        for t in walker_types:
            for l in leg_length:
                for n in env_name:
                    task_list.append({'environment': n,
                                      'leg_length': l,
                                      'walker_type': t,
                                      })

        del n
        del t
        del l
    elif environment == 'pendulum':
        # special set of tasks that are easy to distinguish based on dynamics
        task_list = [{'environment': 'pendulum-custom-v0', 'l': l, 'm': 1.0}
                     for l in np.arange(0.7, 1.4, 0.1)
                     ]
        oracle_curriculum = np.arange(len(task_list))
        traj_len = 200

    target_task_id = 0

    ### General Training Hyperparameters
    lr = 1e-3
    gamma = 0.99

    ## MLP Hyperparams
    rl_batch_size = 1000
    if environment == 'pendulum':
        num_units = [96, 64]
    else:
        num_units = [400, 400]
    update_rate = 5

    buff_size = 5e6
    tau = 0.005
    initial_random_steps = 1e5

    ## TD3

    policy_update_rate = 3

    if environment == 'pendulum':
        exploration_noise = 0.05
        target_policy_smoothing = 0.1
    else:
        exploration_noise = 0.1
        target_policy_smoothing = 0.2

    decode_reward = False
    input_reward = False

    total_steps_limit = 2e7  # total number of training steps, currently ignores eval steps


    eval_freq = 2e5                 # evaluates policies this often
    eval_num = 20                   # episodes per task for evaluation

    debug = False
    if debug:
        num_units = [32, 32]
        rl_batch_size = 100
        initial_random_steps = 3 * traj_len * len(task_list)
        eval_num = 1


@train_ex.main
def train_mtd3_style(task_list, target_task_id, _run,
                     lr, rl_batch_size, buff_size, tau, num_units, gamma,
                     exploration_noise, policy_update_rate,
                     target_policy_smoothing, environment, traj_len,
                     display,
                     total_steps_limit,
                     update_rate, initial_random_steps,
                     eval_freq, eval_num, continue_cp, checkpoint_freq):
    """
    Train MTD3 network. This code is a bit more complicated than necessary because it is based on
    the EM code.
    """
    print(_run.config)
    run_path = get_run_path(_run)

    env_list = []
    reward_thresholds = np.zeros(len(task_list))

    for idx, task in enumerate(task_list):
        env_list.append(gym.make(task['environment']))
        env_list[-1].env.my_init(task)
        reward_thresholds[idx] = env_list[-1].env.reward_threshold

    env = env_list[0]
    if isinstance(env.env.observation_space, Discrete):
        q_table_size = env.env.observation_space.n
    else:
        q_table_size = None

    agent = MTD3Agent(obs_space_n=[env.env.observation_space for env in env_list],
                      act_space_n=[env.env.action_space for env in env_list],
                      agent_index=0, batch_size=rl_batch_size, buff_size=buff_size,
                      lr=lr, num_units=num_units, gamma=gamma, tau=tau, num_tasks=len(task_list),
                      action_noise_value=exploration_noise, _run=_run,
                      policy_update_freq=policy_update_rate,
                      target_policy_smoothing_eps=target_policy_smoothing)

    total_steps = 0
    eval_steps = 0
    n_em_steps = int(total_steps_limit // eval_freq + 1)
    episode_rewards = []

    # assigned task probs for each agent
    rew_task_ag = np.zeros([n_em_steps, len(task_list), 1])

    if continue_cp is not None:
        # load a checkpoint to continue from if necessary
        with open(continue_cp, 'rb') as f:
            checkpoint_dict = pickle.load(f)
        
        total_steps = checkpoint_dict['total_steps']
        eval_steps = checkpoint_dict['eval_steps']
        rew_task_ag[:eval_steps] = checkpoint_dict['rew_task_ag'][:eval_steps]
        agent.load_checkpoint_dict(checkpoint_dict['agent'])

        ignore_list = ['continue_cp', 'seed', 'total_steps_limit']
        if 'config' in checkpoint_dict:
            for key in checkpoint_dict['config'].keys():
                if not key in _run.config.keys() or checkpoint_dict['config'][key] != _run.config[key]:
                    if not key in ignore_list:
                        raise RuntimeError(key, 'NOT MATCHING IN CHECKPOINT SAVED CONFIG')


    last_checkpoint_step = total_steps

    task_id = 0  # start at task zero and then swap to the next after each episode.
    # yes this is ordered but shouldn't really matter because all updates are random
    while total_steps < total_steps_limit:

        env = env_list[task_id]

        # do one episode
        obs = env.reset()

        episode_steps = 0
        episode_reward = 0

        terminal = False

        while not terminal:
            action = agent.action(obs, task_id)
            if isinstance(env.action_space, Discrete):
                if action.size > 1:
                    action = np.argmax(action)
            new_obs, reward, terminal, info = env.step(action)

            episode_reward += reward

            agent.add_transition([obs.copy()], [action.copy()], reward, [new_obs.copy()],
                                 terminal, task_id)

            obs = new_obs

            if total_steps % eval_freq == 0:
                # evaluate agents in all tasks
                for eval_id, task in enumerate(task_list):
                    env = env_list[eval_id]
                    rew = deterministic_episode_task_id(agent, env, traj_len, False, False,
                                                        n_episodes=eval_num, print_mean=False,
                                                        task_id=eval_id)
                    rew_task_ag[eval_steps, eval_id, 0] = rew

                print('total_steps {} reward on tasks {}'.format(total_steps,
                                                                 rew_task_ag[eval_steps]))
                _run.info['rew_task_ag'] = rew_task_ag
                eval_steps += 1

            episode_steps += 1
            total_steps += 1

        # episode finished
        episode_rewards.append(episode_reward)
        print('Step: {} Task: {} Reward: {}'.format(total_steps, task_id, episode_reward))

        if total_steps > initial_random_steps:
            # gets (episode_steps/n_tasks) steps per policy.
            # performs steps_per_policy / update_rate updates on each policy
            for i in range(episode_steps // (update_rate * len(task_list))):
                agent.update([agent], total_steps)  # updates agent on all tasks

        episode_steps = 0
        episode_reward = 0

        # checkpointin
        if total_steps - last_checkpoint_step > checkpoint_freq:
            checkpoint_dict = {}
            checkpoint_dict['agent'] = agent.get_checkpoint_dict()
            checkpoint_dict['rew_task_ag'] = rew_task_ag
            checkpoint_dict['total_steps'] = total_steps
            checkpoint_dict['eval_steps'] = eval_steps
            checkpoint_dict['config'] = _run.config


            fp = os.path.join(run_path,'checkpoint.pkl')
            with open(fp, 'wb') as f:
                pickle.dump(checkpoint_dict, f)
            print('Saved checkpoint to', fp)
            last_checkpoint_step = total_steps

        task_id = (task_id + 1) % len(task_list)


def deterministic_episode_task_id(agent, env, traj_len: int, display: bool, verbose: bool,
                                  n_episodes=10,
                                  print_mean=True, task_id=None):
    max_steps = traj_len
    episode_rewards = []

    for eps_idx in range(n_episodes):
        episode_step = 0
        episode_rewards.append(0)
        obs = env.reset()
        while episode_step < max_steps:
            action = agent.deterministic_action(obs, task_id)
            if isinstance(env.action_space, Discrete):
                if action.size > 1:  # deal with onehot output from mlps
                    action = np.argmax(action)
                new_obs, reward, terminal, info = env.step(action)
            else:
                new_obs, reward, terminal, info = env.step(action)
            episode_rewards[-1] += reward
            if display and eps_idx == 0:
                env.render()
                time.sleep(0.001)
                # test = input('frame ' + str(episode_step))

            obs = new_obs
            episode_step += 1

            if terminal:
                if verbose:
                    print(episode_rewards)
                break
    if print_mean:
        print('Deterministic Eval: ', np.mean(episode_rewards))
    return np.mean(episode_rewards)


def main():
    file_observer = FileStorageObserver(os.path.join('results', 'sacred'))
    train_ex.observers.append(file_observer)
    train_ex.run_commandline()


if __name__ == '__main__':
    main()
