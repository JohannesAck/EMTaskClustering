# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents.

Modified by modified by the authors of the ICLR submission
"Unsupervised Taks Clustering for Multi-Task Reinforcement Learning".
The main changes occur in the runner, as it schedules things.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.implicit_quantile import implicit_quantile_mt_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
# from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
# from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
# from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
# from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent

import numpy as np
import tensorflow as tf

import gin.tf
from train_em import calculate_task_assignments


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

    Args:
      gin_files: list, of paths to the gin configuration files for this
        experiment.
      gin_bindings: list, of gin parameter bindings to override the values in
        the config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


# #gin.configurable
# def create_agent(sess, envs, agent_name=None, summary_writer=None, num_outputs=1,
#                  debug_mode=False):
#     """Creates an agent.
#
#     Args:
#       sess: A `tf.compat.v1.Session` object for running associated ops.
#       environment: A gym environment (e.g. Atari 2600).
#       agent_name: str, name of the agent to create.
#       summary_writer: A Tensorflow summary writer to pass to the agent
#         for in-agent training statistics in Tensorboard.
#       debug_mode: bool, whether to output Tensorboard summaries. If set to true,
#         the agent will output in-episode statistics to Tensorboard. Disabled by
#         default as this results in slower training.
#
#     Returns:
#       agent: An RL agent.
#
#     Raises:
#       ValueError: If `agent_name` is not in supported list.
#     """
#     assert agent_name is not None
#     if not debug_mode:
#         summary_writer = None
#     if agent_name == 'dqn':
#         return dqn_agent.DQNAgent(sess, num_actions=envs[0].action_space.n,
#                                   summary_writer=summary_writer, tf_device='/device:gpu:1')
#     elif agent_name == 'rainbow':
#         return rainbow_agent.RainbowAgent(
#             sess, num_actions=envs[0].action_space.n,
#             summary_writer=summary_writer, tf_device='/device:gpu:1')
#     elif agent_name == 'implicit_quantile':
#         num_actions_task = [env.action_space.n for env in envs]
#         if num_outputs == 1:
#             return implicit_quantile_agent.ImplicitQuantileAgent(
#                 sess, num_actions_task=num_actions_task, num_outputs=num_outputs,
#                 summary_writer=summary_writer, tf_device='/device:gpu:1')
#         else:
#             return implicit_quantile_mt_agent.ImplicitQuantileMTAgent(
#                 sess, num_actions_task=num_actions_task, num_outputs=num_outputs,
#                 summary_writer=summary_writer, tf_device='/device:gpu:1', name=agent_name)
#
#
#     else:
#         raise ValueError('Unknown agent: {}'.format(agent_name))


# #@gin.configurable
# def create_runner(base_dir, schedule='continuous_train_and_eval'):
#     """Creates an experiment Runner.
#
#     Args:
#       base_dir: str, base directory for hosting all subdirectories.
#       schedule: string, which type of Runner to use.
#
#     Returns:
#       runner: A `Runner` like object.
#
#     Raises:
#       ValueError: When an unknown schedule is encountered.
#     """
#     assert base_dir is not None
#     # Continuously runs training and evaluation until max num_iterations is hit.
#     if schedule == 'continuous_train_and_eval':
#         return Runner(base_dir, create_agent)
#     # Continuously runs training until max num_iterations is hit.
#     else:
#         raise ValueError('Unknown schedule: {}'.format(schedule))


#@gin.configurable
# def create_em_runner(base_dir, schedule='continuous_train_and_eval'):
#     """Creates an experiment Runner.
#
#     Args:
#       base_dir: str, base directory for hosting all subdirectories.
#       schedule: string, which type of Runner to use.
#
#     Returns:
#       runner: A `Runner` like object.
#
#     Raises:
#       ValueError: When an unknown schedule is encountered.
#     """
#     assert base_dir is not None
#     # Continuously runs training and evaluation until max num_iterations is hit.
#     if schedule == 'continuous_train_and_eval':
#         return Runner(base_dir, create_agent)
#     # Continuously runs training until max num_iterations is hit.
#     elif schedule == 'continuous_train':
#         return TrainRunner(base_dir, create_agent)
#     else:
#         raise ValueError('Unknown schedule: {}'.format(schedule))


#@gin.configurable
class Runner(object):
    """
    Runner for the EM, as well as Multi-Head experiments.
    """

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 create_environment_fn=atari_lib.create_atari_environment,
                 checkpoint_file_prefix='ckpt',
                 logging_file_prefix='log',
                 log_every_n=1,
                 num_iterations=200,
                 training_steps=250000,       # em training steps
                 evaluation_steps=54000,      # em eval steps
                 max_steps_per_episode=27000,
                 em_n_agents=2,
                 multi_head=False,
                 env_list=('Pong', 'MsPacman'),
                 continue_fp=None,
                 _run=None,
                 _log=None,
                 buff_size=1e6,
                 pretraining_its=0,
                 large_network=False,
                 policy_per_task=False,
                 network_scaling_factor=1.0):
        """Initialize the Runner object in charge of running a full experiment.

        Args:
          base_dir: str, the base directory to host all required sub-directories.
          create_agent_fn: A function that takes as args a Tensorflow session and an
            environment, and returns an agent.
          create_environment_fn: A function which receives a problem name and
            creates a Gym environment for that problem (e.g. an Atari 2600 game).
          checkpoint_file_prefix: str, the prefix to use for checkpoint files.
          logging_file_prefix: str, prefix to use for the log files.
          log_every_n: int, the frequency for writing logs.
          num_iterations: int, the iteration number threshold (must be greater than
            start_iteration).
          training_steps: int, the number of training steps to perform.
          evaluation_steps: int, the number of evaluation steps to perform.
          max_steps_per_episode: int, maximum number of steps after which an episode
            terminates.

        This constructor will take the following actions:
        - Initialize an environment.
        - Initialize a `tf.compat.v1.Session`.
        - Initialize a logger.
        - Initialize an agent.
        - Reload from the latest checkpoint, if available, and initialize the
          Checkpointer object.
        """
        assert base_dir is not None
        tf.compat.v1.disable_v2_behavior()

        self._run = _run
        self._log = _log
        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        self._num_iterations = num_iterations
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = max_steps_per_episode
        self._base_dir = base_dir
        self._create_directories()
        self._summary_writer = tf.compat.v1.summary.FileWriter(self._base_dir)
        self.multi_head = multi_head
        self.buff_size = buff_size
        self.pretraining_its = int(pretraining_its)
        self.policy_per_task = policy_per_task
        self.network_scaling_factor = network_scaling_factor

        if self.policy_per_task:
            em_n_agents = len(env_list)

        self.rew_task_ag = []
        self.probs_ag_task = []

        self._envs = []
        for name in env_list:
            environment = create_environment_fn(game_name=name)
            self._envs.append(environment)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        # Allocate only subset of the GPU memory as needed which allows for running
        # multiple agents/workers on the same GPU.
        config.gpu_options.allow_growth = True
        # Set up a session and initialize variables.
        self._sess = tf.compat.v1.Session('', config=config)
        self._agents = []
        num_actions_task = [env.action_space.n for env in self._envs]
        if self.multi_head:
            num_outputs = len(self._envs)
        else:
            num_outputs = 1
        for idx in range(em_n_agents):
            if policy_per_task or num_outputs == 1:
                agent = implicit_quantile_agent.ImplicitQuantileAgent(
                    self._sess, num_actions_task=num_actions_task, num_outputs=num_outputs,
                    summary_writer=self._summary_writer, tf_device='/device:gpu:1', name=str(idx),
                    buff_size=buff_size, network_scaling_factor=network_scaling_factor)
            else:
                agent = implicit_quantile_mt_agent.ImplicitQuantileMTAgent(
                    self._sess, num_actions_task=num_actions_task, num_outputs=num_outputs,
                    summary_writer=self._summary_writer, tf_device='/device:gpu:1', name=str(idx),
                    buff_size=buff_size, large_network=large_network,
                    network_scaling_factor=network_scaling_factor)

            self._agents.append(agent)


        self._summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
        self._sess.run(tf.compat.v1.global_variables_initializer())

        restored = self._initialize_checkpointer_and_maybe_resume(continue_fp)
        if restored:
            self._checkpointer = checkpointer.Checkpointer(base_dir)

        self.checkpoint_file_prefix = checkpoint_file_prefix

    def _create_directories(self):
        """Create necessary sub-directories."""
        self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
        self._logger = self._log #logger.Logger(os.path.join(self._base_dir, 'logs'))

    def _initialize_episode(self):
        """Initialization for a new episode.

        Returns:
          action: int, the initial action chosen by the agent.
        """
        initial_observation = self._environment.reset()
        return self._agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.

        Args:
          action: int, the action to perform in the environment.

        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        observation, reward, is_terminal, _ = self._environment.step(action)
        return observation, reward, is_terminal

    def _end_episode(self, reward):
        """Finalizes an episode run.

        Args:
          reward: float, the last reward from the environment.
        """
        self._agent.end_episode(reward)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        self._agent.num_actions = self._environment.action_space.n
        self._agent.task_id = self.env_idx

        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1

            # Perform reward clipping.
            reward = np.clip(reward, -1, 1)

            if (self._environment.game_over or
                    step_number == self._max_steps_per_episode):
                # or step_number > 210): #todo remove this line
                # Stop the run loop once we reach the true end of episode.
                # print('debugging stop')
                # self._log.warning('debug episode end triggered')
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._agent.end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)

        self._end_episode(reward)

        return step_number, total_reward

    def _run_one_phase(self, min_steps, statistics, run_mode_str, task_probs):
        """Runs the agent/environment loop until a desired number of steps.

        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.

        Args:
          min_steps: int, minimum number of steps to generate in this phase.
          statistics: `IterationStatistics` object which records the experimental
            results.
          run_mode_str: str, describes the run mode for this agent.

        Returns:
          Tuple containing the number of steps taken in this phase (int), the sum of
            returns (float), and the number of episodes performed (int).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        # pick first task at random, then sample such as to have the same number of experience
        # from each assigned task, assumes probabilities of all tasks are the same
        step_per_env = np.zeros(len(task_probs), np.int32)
        self.env_idx = np.random.choice(len(task_probs), p=task_probs)
        self._environment = self._envs[self.env_idx]

        assigned_filter = task_probs > 0.0  # assigned tasks
        step_per_env[~assigned_filter] = min_steps + 1  # max val so they are never chosen
        # check that all are 0 or same val.
        assert all(list(task_probs[assigned_filter] == task_probs[assigned_filter][0]))

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            step_per_env[self.env_idx] += episode_length
            num_episodes += 1
            # We use sys.stdout.write instead of logging so as to flush frequently
            # without generating a line break.

            # sys.stdout.write('Env: {} Steps executed: {} '.format(env_idx, step_count) +
            #                  'Episode length: {} '.format(episode_length) +
            #                  'Return: {}\r'.format(episode_return))
            # sys.stdout.flush()

            if not run_mode_str == 'eval':
                self._log.info('Env: {} Steps executed: {} '.format(self.env_idx, step_count) +
                                 'Episode length: {} '.format(episode_length) +
                                 'Return: {}\r'.format(episode_return))


            # choose next environment as assigned task with fewest steps in this phase
            least_picked_assigned_task = np.argmin(step_per_env)
            self.env_idx = int(least_picked_assigned_task)
            self._environment = self._envs[self.env_idx]
        return step_count, sum_returns, num_episodes

    def _run_train_phase(self, statistics, agent, task_probs):
        """Run training phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
          average_steps_per_second: float, The average number of steps per second.
        """
        # Perform the training phase, during which the agent learns.
        self._agent = agent
        self._agent.eval_mode = False
        self._agent.task_probs = task_probs
        start_time = time.time()
        number_steps, sum_returns, num_episodes = self._run_one_phase(
            self._training_steps, statistics, 'train', task_probs)
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        self._agent.task_probs = None  # sanity check
        time_delta = time.time() - start_time
        average_steps_per_second = number_steps / time_delta
        statistics.append(
            {'train_average_steps_per_second': average_steps_per_second})
        self._log.info('Average undiscounted return per training episode: %.2f',
                     average_return)
        self._log.info('Average training steps per second: %.2f',
                     average_steps_per_second)
        return num_episodes, average_return, average_steps_per_second

    def _run_eval_phase(self, statistics, agent, env_idx):
        """Run evaluation phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent = agent
        self._agent.eval_mode = True
        task_probs = np.zeros([len(self._envs)])
        task_probs[env_idx] = 1.0
        _, sum_returns, num_episodes = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval', task_probs)
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        var_return = np.var(sum_returns) if num_episodes > 0 else 0.0
        # self._log.info('Average undiscounted return per evaluation episode: %.2f',
        #              average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return, var_return

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).

        Args:
          iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.

        Returns:
          A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()
        self._log.info('Starting iteration %d', iteration)

        #evaluate all agents on all tasks, to calculate assignments and for logging
        rew_task_ag = np.zeros([len(self._envs), len(self._agents)])
        if self.policy_per_task:
            for ag_idx, agent in enumerate(self._agents):
                env_idx = ag_idx
                env = self._envs[env_idx]
                num_episodes_eval, average_reward_eval, var_return_eval = \
                    self._run_eval_phase(statistics, agent, env_idx)
                rew_task_ag[env_idx][ag_idx] = average_reward_eval
                self._log.info('Evaluation Agent {} on {} : {:.2f}, var {:.2f}'
                               .format(ag_idx, env.environment.spec.id[:-14], average_reward_eval,
                                       var_return_eval))
        else:
            for ag_idx, agent in enumerate(self._agents):
                for env_idx, env in enumerate(self._envs):
                    num_episodes_eval, average_reward_eval, var_return_eval = \
                        self._run_eval_phase(statistics, agent, env_idx)
                    rew_task_ag[env_idx][ag_idx] = average_reward_eval
                    self._log.info('Evaluation Agent {} on {} : {:.2f}, var {:.2f}'
                                   .format(ag_idx, env.environment.spec.id[:-14], average_reward_eval,
                                           var_return_eval))

        self._log.info('rew task ag' + str(rew_task_ag))

        if iteration >= self.pretraining_its:
            if self.policy_per_task:
                assert len(self._agents) == len(self._envs)
                probs_ag_task = np.zeros([len(self._agents), len(self._envs)])
                for ag_idx in range(len(self._agents)):
                    probs_ag_task[ag_idx, ag_idx] = 1.0
            else:
                probs_ag_task = \
                    calculate_task_assignments(rew_task_ag, 'greedy-uniform', np.arange(len(self._agents)))
        else:
            # uniform pretraining
            probs_ag_task = np.ones([len(self._agents), len(self._envs)]) / len(self._envs)

        statistics.append({'rew_task_ag' : rew_task_ag,
                           'probs_ag_task': probs_ag_task})
        self._log.info('probs_ag_task' + str(probs_ag_task))

        # train agens on assigned tgasks
        for ag_idx, agent in enumerate(self._agents):
            num_episodes_train, average_reward_train, average_steps_per_second = \
                self._run_train_phase(statistics, agent, probs_ag_task[ag_idx])

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                         average_reward_train, num_episodes_eval,
                                         average_reward_eval,
                                         average_steps_per_second)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    num_episodes_eval,
                                    average_reward_eval,
                                    average_steps_per_second):
        """Save statistics as tensorboard summaries.

        Args:
          iteration: int, The current iteration number.
          num_episodes_train: int, number of training episodes run.
          average_reward_train: float, The average training reward.
          num_episodes_eval: int, number of evaluation episodes run.
          average_reward_eval: float, The average evaluation reward.
          average_steps_per_second: float, The average number of steps per second.
        """
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag='Train/NumEpisodes', simple_value=num_episodes_train),
            tf.compat.v1.Summary.Value(
                tag='Train/AverageReturns', simple_value=average_reward_train),
            tf.compat.v1.Summary.Value(
                tag='Train/AverageStepsPerSecond',
                simple_value=average_steps_per_second),
            tf.compat.v1.Summary.Value(
                tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
            tf.compat.v1.Summary.Value(
                tag='Eval/AverageReturns', simple_value=average_reward_eval)
        ])
        self._summary_writer.add_summary(summary, iteration)

    def _log_experiment(self, iteration, statistics):
        """Records the results of the current iteration.

        Args:
          iteration: int, iteration number.
          statistics: `IterationStatistics` object containing statistics to log.
        """
        step = (iteration + 1) * self._training_steps
        mean_rew_ag = np.mean(statistics['rew_task_ag'][0], 0)
        for ag_idx in range(len(mean_rew_ag)):
            self._run.log_scalar('ag{}meanrew'.format(ag_idx), mean_rew_ag[ag_idx], step)
        self.rew_task_ag.append(statistics['rew_task_ag'][0])
        self.probs_ag_task.append(statistics['probs_ag_task'][0])

        self._run.info['rew_task_ag'] = np.array(self.rew_task_ag)
        self._run.info['probs_ag_task'] = np.array(self.probs_ag_task)

        # self._logger['iteration_{:d}'.format(iteration)] = statistics
        # if iteration % self._log_every_n == 0:
        #     self._logger.log_to_file(self._logging_file_prefix, iteration)

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data.

        Args:
          iteration: int, iteration number for checkpointing.
        """
        self._checkpointer = checkpointer.Checkpointer(self._base_dir,
                                                       self.checkpoint_file_prefix)
        experiment_data = []
        for ag_idx, agent in enumerate(self._agents):
            bundle = self._agent.bundle_and_checkpoint(self._checkpoint_dir, iteration,
                                                       name='agent' + str(ag_idx))
            experiment_data.append(bundle)
            if experiment_data[-1]:
                experiment_data[-1]['current_iteration'] = iteration
                # experiment_data[-1]['logs'] = self._logger.data
                experiment_data[-1]['rew_task_ag'] = self.rew_task_ag
                experiment_data[-1]['prob_ag_task'] = self.probs_ag_task
        self._checkpointer.save_checkpoint(iteration, experiment_data)

    def _initialize_checkpointer_and_maybe_resume(self, continue_fp):
        """Reloads the latest checkpoint if it exists.

        This method will first create a `Checkpointer` object and then call
        `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
        checkpoint in self._checkpoint_dir, and what the largest file number is.
        If a valid checkpoint file is found, it will load the bundled data from this
        file and will pass it to the agent for it to reload its data.
        If the agent is able to successfully unbundle, this method will verify that
        the unbundled data contains the keys,'logs' and 'current_iteration'. It will
        then load the `Logger`'s data from the bundle, and will return the iteration
        number keyed by 'current_iteration' as one of the return values (along with
        the `Checkpointer` object).

        Args:
          checkpoint_file_prefix: str, the checkpoint file prefix.

        Returns:
          start_iteration: int, the iteration number to start the experiment from.
          experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._start_iteration = 0
        if continue_fp is None:
            return
        self._checkpointer = checkpointer.Checkpointer(continue_fp)
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(continue_fp)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            for ag_idx, agent in enumerate(self._agents):
                # sometimes the agents are at different checkpoint steps.
                # this kind of changes results a bit but shouldn't matter too much
                # self._log.warning('Differt checkpoint version!!!')
                # agent_checkpoint_version = get
                if agent.unbundle(
                        continue_fp, latest_checkpoint_version, experiment_data,
                        name='agent' + str(ag_idx)):
                    if experiment_data is not None:
                        assert 'current_iteration' in experiment_data[-1]
            # self._logger.data = experiment_data[-1]['logs']
            self._start_iteration = experiment_data[-1]['current_iteration'] + 1
            self.rew_task_ag = experiment_data[-1]['rew_task_ag']
            self.probs_ag_task = experiment_data[-1]['prob_ag_task']

            self._log.info('Reloaded checkpoint and will start from iteration %d',
                             self._start_iteration)
            return True
        else:
            return RuntimeError('Coninue_fp specified but checkpoint not found in {}'
                                .format(continue_fp)
                                )
        return False

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        self._log.info('Beginning training...')
        if self._num_iterations <= self._start_iteration:
            self._log.warning('num_iterations (%d) < start_iteration(%d)',
                            self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration)
