import time

import gym
from gym.spaces import Discrete

import numpy as np

from agents.QLearnerAgent import QLearnerFull
from agents.TD3Agent import TD3Agent

def get_agent(_run, buff_size, critic_lr, entropy_coeff, env, exploration_noise, gamma, num_units,
              policy, policy_update_rate, rl_batch_size, target_policy_smoothing, tau, eps,
              eps_decay, q_table_size, num_uncertainty_nets):
    if policy == 'td3':
        agent = TD3Agent([env.observation_space], [env.action_space], 0, batch_size=rl_batch_size,
                         buff_size=buff_size, lr=critic_lr, num_units=num_units,
                         gamma=gamma, tau=tau, action_noise_value=exploration_noise,
                         policy_update_freq=policy_update_rate,
                         target_policy_smoothing_eps=target_policy_smoothing, _run=_run,
                         num_uncertainty_nets=num_uncertainty_nets)
    elif policy == 'qlearner':
        agent = QLearnerFull(env.env.action_space, q_table_size, gamma, lr=critic_lr,
                             eps=eps, eps_decay=eps_decay)
    else:
        raise RuntimeError("policy not implemented")
    return agent


def deterministic_episode(agent, env, traj_len: int, display: bool, verbose: bool, n_episodes=10,
                          print_mean=True):
    max_steps = traj_len
    episode_rewards = []

    for eps_idx in range(n_episodes):
        episode_step = 0
        episode_rewards.append(0)
        obs = env.reset()
        while episode_step < max_steps:
            action = agent.deterministic_action(obs)
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

