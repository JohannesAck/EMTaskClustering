"""
Implementation of "Unsupervised Task Clustering for Multi-Task
Reinforcement Learning"
This file contains the code for EM, PPT and SP. For Multi-Head please use train_multi_head.py.
For all Atari experiments, please use train_atari_em.py.
"""
import os
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


from agents.AbstractAgent import AbstractAgent
# these imports are actually important because they register the envs
import custom_env
from training_snippets import get_agent, deterministic_episode
from common.sacred_util import get_run_path


train_ex = Experiment('EM')


# The following method is used to configure all the hyper-parameters of the experiment, as is
# usually done when using sacred.
# Configuration parameters can also be updated using the CLI provided by sacred.
@train_ex.config
def train_config():
    display = False                         # display the environment

    ### Environment
    #  'pendulum' for the pendulum taskset
    #  'corner-gridworld-v0' for the gridworld-corner taskset
    #  'leftright-discrete-v0' for the 1D-chain taskset
    #  'biped' for the BipedalWalker taskset with varying leg lengths and obstacles
    #  'track' for the BipedalWalker taskset with different rewards, inspired by track and field
    #           For Atari please see train_atari_em.py
    environment = 'biped'

    checkpoint_freq = 1e6             # checkpoint every n training steps
    continue_cp = None                # path to a checkpoint.pkl file to continue training from

    # generation of the different task sets
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
                                          'right_goal': rg,
                                          'swapped_actions': False})
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
        for t in walker_types:
            for l in leg_length:
                for n in env_name:
                    task_list.append({'environment': n,
                                      'leg_length': l,
                                      'walker_type': t,
                                      })

        del (n)
        del (t)
        del (l)
    elif environment == 'pendulum':
        task_list = [{'environment': 'pendulum-custom-v0', 'l': l, 'm': 1.0}
                     for l in np.arange(0.7, 1.4, 0.1)
                     ]
        oracle_curriculum = np.arange(len(task_list))
        traj_len = 200

    target_task_id = 0                              # this is just the task number used to
                                                    # build the network, does not really matter

    ### Agent Parameters

    if environment in ['corner-gridworld-v0', 'leftright-discrete-v0']:
        policy = 'qlearner'
    else:
        policy = 'td3'

    ## Q Learning
    eps = 0.2                                       # random action probability
    eps_decay = 0.999999                            # decay factor for rand. act. prob.

    initial_random_steps = 0                        # initial random steps to aid exploration

    ### General Training Hyperparameters

    ## MLP Hyperparams
    if environment in ['pendulum', 'mountaincar']:
        num_units = [64, 64]
        exploration_noise = 0.05
        target_policy_smoothing = 0.1
        rl_batch_size = 128
        buff_size = 2e6
        initial_random_steps = 1e4
        update_rate = 5                     # update policy after each x steps
        lr = 3e-3
    else:
        num_units = [400, 300]
        exploration_noise = 0.1
        target_policy_smoothing = 0.2
        rl_batch_size = 1000
        buff_size = 5e6
        initial_random_steps = 1e4
        update_rate = 3                     # update policy after each x steps
        lr = 1e-3

    if policy == 'qlearner':
        lr = 0.2
        gamma = 0.9
    else:
        gamma = 0.99

    tau = 0.005

    ## TD3
    if policy == 'td3':
        policy_update_rate = 3
    else:
        policy_update_rate = 1

    ## SAC
    entropy_coeff = 0.005


    # Expectation Maximization Parameters

    em_n_policies = 4                   # number of policies for our EM approach

    if environment in ['pendulum', 'mountaincar']:
        em_training_steps = 5e4         # training steps per M-step
        total_steps_limit = 2e6         # maximum total number of training steps per policy
        checkpoint_freq = None
    elif environment in ['gridworld-v0', 'corner-gridworld-v0', 'leftright-discrete-v0']:
        em_training_steps = 0.5e3
        total_steps_limit = 1e5
        checkpoint_freq = None
    else:
        em_training_steps = 2e5
        total_steps_limit = 2e7

    em_eval_assign_freq = 3                 # evaluate this times as often as assignments are made
    em_training_steps = em_training_steps // em_eval_assign_freq

    em_assign_strategy = 'greedy-uniform'   # only one in this code

    if 'atari' in environment:
        em_eval_trajs = 5                   # evaluation trajectories in the E-Step
    elif environment in ['gridworld-v0', 'corner-gridworld-v0', 'leftright-discrete-v0']:
        em_eval_trajs = 3
    else:
        em_eval_trajs = 20

    # baselines
    one_policy_per_task = False             # use the PPT baseline instead
    one_policy_all_tasks = False            # use the SP baseline instead
    randomly_assign_agents = False          # use randomly assigned tasks (for ablation)


    debug = False
    if debug:                       # simple settings for quick debugging
        em_n_policies = 1
        em_training_steps = 2e4
        initial_random_steps = 1e3
        em_eval_trajs = 2
        num_units = [32, 32]
        rl_batch_size = 32
        buff_size = 1e5


@train_ex.main
def train_em_style(task_list, target_task_id, _run,
                   lr, rl_batch_size, buff_size, tau, num_units, gamma,
                   entropy_coeff, policy,
                   exploration_noise, policy_update_rate,
                   target_policy_smoothing, environment, traj_len,
                   display,
                   total_steps_limit,
                   eps, eps_decay, update_rate, initial_random_steps,
                   em_n_policies, em_training_steps, em_assign_strategy, em_eval_trajs,
                   em_eval_assign_freq,
                   one_policy_per_task, one_policy_all_tasks,
                   continue_cp, checkpoint_freq, randomly_assign_agents, _log):
    """
        Trains in an Expectation Maximization style way.
        Starts of by randomly assigning tasks to agents, trains them on them and then
        evaluates all agents on all tasks.
        This generates a ranking of agent performance by task, which is used in the E-Step to
        reassign agents to tasks.

        This function takes care of the main training loop, it's a bit complex because it used to
        allow for paralellization of policy training, but that didn't work with tensorflow so it's
        gone now.
    """

    _log.info(str(_run.config))
    run_path = get_run_path(_run)

    env = gym.make(task_list[target_task_id]['environment'])
    env.env.my_init(task_list[target_task_id])

    # set up the agents
    if one_policy_per_task:
        em_n_policies = len(task_list)
    if one_policy_all_tasks:
        em_n_policies = 1
    if isinstance(env.env.observation_space, Discrete):
        q_table_size = env.env.observation_space.n
    else:
        q_table_size = None
    agent_args = [_run, buff_size, lr, entropy_coeff, env, exploration_noise, gamma, num_units,
                  policy, policy_update_rate, rl_batch_size, target_policy_smoothing, tau, eps,
                  eps_decay, q_table_size, 0]  # we need this in paralellization again
    agents = [get_agent(*agent_args)
              for _ in range(em_n_policies)]

    # logging
    total_steps = 0
    em_step = 0
    n_em_steps = int(total_steps_limit // em_training_steps + 1)
    probs_ag_task = np.zeros([n_em_steps, len(agents), len(task_list)])
    rew_task_ag = np.zeros([n_em_steps, len(task_list), len(agents)])

    # assign fixed tasks now if using random assignments
    if randomly_assign_agents:
        # first assign tasks to each agent including buffer ones and then remove impossible tasks
        # maybe there's a nicer solution but I don't really care too much
        task_num_with_buff = (np.ceil(len(task_list) / em_n_policies) * em_n_policies).astype(np.int32)
        per_agent = (task_num_with_buff / em_n_policies).astype(np.int32)
        def_tasks = np.random.choice(task_num_with_buff, (em_n_policies, per_agent),
                                     replace=False).tolist()
        for ag_idx in range(em_n_policies):
            invalid_ids = [el for el in def_tasks[ag_idx] if el >= len(task_list)]
            for inv_id in invalid_ids:
                def_tasks[ag_idx].remove(inv_id)
    else:
        def_tasks = None

    _log.info('Def Tasks' + str(def_tasks))

    if continue_cp is not None:
        # load a checkpoint to continue from if necessary
        with open(continue_cp, 'rb') as f:
            checkpoint_dict = pickle.load(f)

        for ag_idx in range(len(agents)):
            agents[ag_idx].load_checkpoint_dict(checkpoint_dict['agent' + str(ag_idx)])

        probs_ag_task = checkpoint_dict['probs_ag_task']
        rew_task_ag = checkpoint_dict['rew_task_ag']
        total_steps = checkpoint_dict['total_steps']
        em_step = checkpoint_dict['em_step']
        def_tasks = checkpoint_dict['def_tasks']

        ignore_list = ['continue_cp', 'seed', 'total_steps_limit', 'display']
        if 'config' in checkpoint_dict:
            for key in checkpoint_dict['config'].keys():
                if not key in _run.config.keys() or checkpoint_dict['config'][key] != _run.config[key]:
                    if not key in ignore_list:
                        raise RuntimeError(key, 'NOT MATCHING IN CHECKPOINT SAVED CONFIG')

    last_checkpoint_step = total_steps

    # train loop
    while total_steps < total_steps_limit:
        # evaluate agents in all tasks
        for task_idx, task in enumerate(task_list):
            env = gym.make(task['environment'])
            env.env.my_init(task)
            for ag_idx, agent in enumerate(agents):
                rew = deterministic_episode(agent, env, traj_len, display, False,
                                            n_episodes=em_eval_trajs, print_mean=False)
                rew_task_ag[em_step, task_idx, ag_idx] = rew

        # set task probabilities for each agent
        if em_step % em_eval_assign_freq == 0:
            probs = calculate_task_assignments(rew_task_ag[em_step], em_assign_strategy, def_tasks,
                                               one_policy_per_task, randomly_assign_agents)
            probs_ag_task[em_step] = probs
        else:  # reuse old if not assigning newly this step
            probs_ag_task[em_step] = probs_ag_task[em_step - 1]

        for ag_idx, agent in enumerate(agents):
            _log.info('agent {} reward on tasks {}'.format(ag_idx, rew_task_ag[em_step, :, ag_idx]))

        # train each agent on assigned tasks
        kwargs_list = []
        for ag_idx, agent in enumerate(agents):

            _log.info('training agent {} with tasks {}'.format(ag_idx, probs_ag_task[em_step, ag_idx]))

            # again a bit complex becaus of former paralellization
            kwargs = {'agent': agent, 'task_freqs': probs_ag_task[em_step, ag_idx],
                      'n_steps': em_training_steps, 'start_step': total_steps,
                      'update_rate': update_rate, 'task_list': task_list,
                      'initial_random_steps': initial_random_steps,
                      'agent_args': agent_args}
            kwargs_list.append(kwargs)

        for idx, kwargs in enumerate(kwargs_list):
            train_agent_on_task(**kwargs)

        # logging
        em_step += 1
        total_steps += em_training_steps
        _run.info['rew_task_ag'] = rew_task_ag
        _run.info['probs_ag_task'] = probs_ag_task

        # checkpointing
        if checkpoint_freq is not None and total_steps - last_checkpoint_step > checkpoint_freq:
            checkpoint_dict = {}
            for ag_idx, agent in enumerate(agents):
                checkpoint_dict['agent' + str(ag_idx)] = agent.get_checkpoint_dict()
            checkpoint_dict['probs_ag_task'] = probs_ag_task
            checkpoint_dict['rew_task_ag'] = rew_task_ag
            checkpoint_dict['total_steps'] = total_steps
            checkpoint_dict['em_step'] = em_step
            checkpoint_dict['def_tasks'] = def_tasks
            checkpoint_dict['config'] = _run.config

            fp = os.path.join(run_path, 'checkpoint.pkl')
            with open(fp, 'wb') as f:
                pickle.dump(checkpoint_dict, f)
            _log.info('Saved checkpoint to' + fp)
            last_checkpoint_step = total_steps


def calculate_task_assignments(rew_task_ag, em_assign_strategy, def_tasks,
                               one_policy_per_task=False, randomly_assign_agents=False):
    """
    Determines the assignments to tasks, based on the received rewards and assignment_strategy
    """
    n_tasks = rew_task_ag.shape[0]
    n_agents = rew_task_ag.shape[1]

    probs_ag_task = np.zeros([n_agents, n_tasks])
    for task_idx in range(n_tasks):
        if one_policy_per_task:
            for ag_idx in range(n_agents):
                probs_ag_task[ag_idx, task_idx] = 1.0 if ag_idx == task_idx else 0.0
        elif randomly_assign_agents:
            for ag_idx in range(n_agents):
                assigned = task_idx in def_tasks[ag_idx]
                probs_ag_task[ag_idx, task_idx] = 1.0 if assigned else 0.0
        elif em_assign_strategy in ['greedy-uniform']:
            best_agent = np.argmax(rew_task_ag[task_idx])
            for ag_idx in range(n_agents):
                probs_ag_task[ag_idx, task_idx] = 1.0 if ag_idx == best_agent else 0.0
        else:
            raise NotImplementedError('Unknown assignment strategy.')

    # assign all tasks if none were assigned, otherwise normalize
    for ag_idx in range(n_agents):
        if np.all(probs_ag_task[ag_idx] == 0.0):
            if em_assign_strategy == 'greedy-uniform':
                # in this case the agent is just uniformly trained on all tasks.
                probs_ag_task[ag_idx] = 1.0 / n_tasks
        else:  # normalize if not all 0
            probs_ag_task[ag_idx] /= np.sum(probs_ag_task[ag_idx])

    return probs_ag_task


@train_ex.capture
def train_agent_on_task(agent, task_freqs, n_steps, start_step, update_rate, task_list,
                        initial_random_steps, agent_args, _log):
    """
    Train given agent on task_list, with episodes being sampled with frequencies specified
    in task_freqs, for n_steps. Lets the agent finish the episode, if the step limit is reached.
    """
    env_list = []
    for idx, task in enumerate(task_list):
        if task_freqs[idx] == 0.0:
            env_list.append(None)
        else:
            env_list.append(gym.make(task_list[idx]['environment']))
            env_list[-1].env.my_init(task)

    agent_steps = 0
    episode_rewards = []

    while agent_steps < n_steps:
        # sample a task and train
        task_idx = np.random.choice(np.arange(len(task_list)), p=task_freqs, replace=False)
        env = env_list[task_idx]
        obs = env.reset()

        episode_steps = 0
        episode_reward = 0

        terminal = False
        while not terminal:
            # train on the current task for one episode
            agent.step = agent_steps + start_step  # update agent step for lr-schedule
            action = agent.action(obs)
            if isinstance(env.action_space, Discrete):
                if action.size > 1:  # in case we output action values
                    action = np.argmax(action)
            new_obs, reward, terminal, info = env.step(action)

            episode_reward += reward

            if isinstance(agent, AbstractAgent):
                agent.add_transition([obs.copy()], [action.copy()], reward, [new_obs.copy()],
                                     terminal)

            obs = new_obs
            agent_steps += 1
            episode_steps += 1

            if terminal:
                obs = env.reset()
                episode_rewards.append(episode_reward)
                _log.info('steps {} reward {}'.format(start_step + agent_steps, episode_reward))

                # performs all updates bundled at the end of the episode
                if isinstance(agent, AbstractAgent):
                    if agent_steps + start_step > initial_random_steps:
                        for i in range(episode_steps // update_rate):
                            # performs episode_step / update_rate updates, i.e.
                            # an update for every update_rate samples that belong to this policy
                            agent.update([agent], start_step + agent_steps)

                episode_steps = 0
                episode_reward = 0
    return agent


def main():
    file_observer = FileStorageObserver(os.path.join('results', 'sacred'))
    train_ex.observers.append(file_observer)
    train_ex.run_commandline()


if __name__ == '__main__':
    main()
