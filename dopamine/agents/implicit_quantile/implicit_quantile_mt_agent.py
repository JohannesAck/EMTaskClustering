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
"""The implicit quantile networks (IQN) agent.

The agent follows the description given in "Implicit Quantile Networks for
Distributional RL" (Dabney et. al, 2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
#import gin.tf

from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import circular_replay_buffer


# @gin.configurable
class ImplicitQuantileMTAgent(rainbow_agent.RainbowAgent):
    """An extension of Rainbow to perform implicit quantile regression."""

    def __init__(self,
                 sess,
                 num_actions_task,
                 num_outputs=1,
                 network=atari_lib.ImplicitQuantileNetwork,
                 kappa=1.0,
                 num_tau_samples=32,
                 num_tau_prime_samples=32,
                 num_quantile_samples=32,
                 quantile_embedding_dim=64,
                 double_dqn=False,
                 summary_writer=None,
                 summary_writing_frequency=500,
                 tf_device='/gpu:*',
                 name='',
                 buff_size=1e6,
                 large_network=False,
                 network_scaling_factor=1.0):
        """Initializes the agent and constructs the Graph.

        Most of this constructor's parameters are IQN-specific hyperparameters whose
        values are taken from Dabney et al. (2018).

        MODIFIED FOR MULTIPLE HEADS. BASICALLY JUST TURNED EVERYTHING INTO LISTS, PRETTY UGLY
        BUT SHOULD WORK I HOPE.

        Args:
          sess: `tf.compat.v1.Session` object for running associated ops.
          num_actions: a list of the number of actions per task.
          network: tf.Keras.Model, expects three parameters:
            (num_actions, quantile_embedding_dim, network_type). This class is used
            to generate network instances that are used by the agent. Each
            instantiation would have different set of variables. See
            dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
          kappa: float, Huber loss cutoff.
          num_tau_samples: int, number of online quantile samples for loss
            estimation.
          num_tau_prime_samples: int, number of target quantile samples for loss
            estimation.
          num_quantile_samples: int, number of quantile samples for computing
            Q-values.
          quantile_embedding_dim: int, embedding dimension for the quantile input.
          double_dqn: boolean, whether to perform double DQN style learning
            as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
          summary_writer: SummaryWriter object for outputting training statistics.
            Summary writing disabled if set to None.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
        """
        self.kappa = kappa
        # num_tau_samples = N below equation (3) in the paper.
        self.num_tau_samples = num_tau_samples
        # num_tau_prime_samples = N' below equation (3) in the paper.
        self.num_tau_prime_samples = num_tau_prime_samples
        # num_quantile_samples = k below equation (3) in the paper.
        self.num_quantile_samples = num_quantile_samples
        # quantile_embedding_dim = n above equation (4) in the paper.
        self.quantile_embedding_dim = quantile_embedding_dim
        # option to perform double dqn.
        self.double_dqn = double_dqn
        self.buff_size = int(buff_size)
        self.network_scaling_factor = network_scaling_factor

        assert isinstance(num_actions_task, list)
        assert num_outputs != 1
        self.num_actions_task = num_actions_task  # actions per task
        self.num_actions = np.max(num_actions_task)
        self.task_id = 0
        self.base_name = name
        self.num_outputs = num_outputs
        assert not (network_scaling_factor != 1.0 and large_network)
        if large_network:
            print('USING LARGE NETWORK')
            network = atari_lib.ImplicitQuantileMTLastLayerNetworkLarge
        elif not network_scaling_factor == 1.0:
            print('Using scalable network')
            network = atari_lib.ImplicitQuantileMTLastLayerNetworkScalable
        else:
            print('USING SMALL NETWORK')
            network = atari_lib.ImplicitQuantileMTLastLayerNetwork
        self.train_id = 0  # rotates batches for each task

        super(ImplicitQuantileMTAgent, self).__init__(
            sess=sess,
            num_actions=self.num_actions,
            network=network,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency, tf_device=tf_device)

    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent.

        Args:
          use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.

        Returns:
          A WrapperReplayBuffer object.
        """
        return [circular_replay_buffer.WrappedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            name='ag' + self.base_name + 'buff' + str(idx),
            observation_dtype=self.observation_dtype.as_numpy_dtype,
            extra_storage_types=[circular_replay_buffer.ReplayElement('task_id', (), np.int32),
                                 circular_replay_buffer.ReplayElement('valid_act_mask', (18,),
                                                                      np.bool)
                                ],
            replay_capacity=self.buff_size,
        ) for idx in range(self.num_outputs)]

    def _store_transition(self,
                          last_observation,
                          action,
                          reward,
                          is_terminal,
                          priority=None):
        """Stores a transition when in training mode.

        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer (last_observation, action, reward,
        is_terminal, priority).

        Args:
          last_observation: Last observation, type determined via observation_type
            parameter in the replay_memory constructor.
          action: An integer, the action taken.
          reward: A float, the reward.
          is_terminal: Boolean indicating if the current state is a terminal state.
          priority: Float. Priority of sampling the transition. If None, the default
            priority will be used. If replay scheme is uniform, the default priority
            is 1. If the replay scheme is prioritized, the default priority is the
            maximum ever seen [Schaul et al., 2015].
        """
        if not self.eval_mode:
            action_mask = np.zeros([18]).astype(np.bool)
            action_mask[:self.num_actions] = True
            self._replay[self.task_id].add(last_observation, action, reward, is_terminal,
                                           self.task_id, action_mask)

    def end_episode(self, reward):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          reward: float, the last reward from the environment.
        """
        if not self.eval_mode:
            self._store_transition(self._observation, self.action, reward, True)
        self.task_id = None  # ensure that this is always set at the start of an episode

    def _create_network(self, name):
        r"""Builds an Implicit Quantile ConvNet.

        Args:
          name: str, this name is passed to the tf.keras.Model and used to create
            variable scope under the hood by the tf.keras.Model.
        Returns:
          network: tf.keras.Model, the network instantiated by the Keras model.
        """
        if self.num_outputs == 1:
            if self.network_scaling_factor == 1.0:
                network = self.network(self.num_actions, self.quantile_embedding_dim,
                                       name=name)
            else:
                network = self.network(self.num_actions, self.quantile_embedding_dim,
                                       network_scaling_factor=self.network_scaling_factor,
                                       name=name)
        else:
            if self.network_scaling_factor == 1.0:
                network = self.network(self.num_actions_task, self.quantile_embedding_dim,
                                       name=name)
            else:
                network = self.network(self.num_actions_task, self.quantile_embedding_dim,
                                       network_scaling_factor=self.network_scaling_factor,
                                       name=name)
            return network

    def _build_networks(self):
        """Builds the IQN computations needed for acting and training.

        These are:
          self.online_convnet: For computing the current state's quantile values.
          self.target_convnet: For computing the next state's target quantile
            values.
          self._net_outputs: The actual quantile values.
          self._q_argmax: The action maximizing the current state's Q-values.
          self._replay_net_outputs: The replayed states' quantile values.
          self._replay_next_target_net_outputs: The replayed next states' target
            quantile values.
        """

        self.online_convnet = self._create_network(name='Online')
        self.target_convnet = self._create_network(name='Target')
        self._q_values = []
        self._q_argmax = []
        self._replay_net_target_quantile_values = []
        self._replay_net_quantile_values = []
        self._replay_net_quantiles = []
        self._replay_next_qt_argmax = []

        for out_idx in range(self.num_outputs):
            # Compute the Q-values which are used for action selection in the current
            # state.
            self._net_outputs = self.online_convnet(self.state_ph,
                                                    self.num_quantile_samples)
            # Shape of self._net_outputs.quantile_values:
            # num_quantile_samples x num_actions.
            # e.g. if num_actions is 2, it might look something like this:
            # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
            #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
            # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
            self._q_values.append(
                tf.reduce_mean(self._net_outputs[out_idx].quantile_values, axis=0))
            self._q_argmax.append(tf.argmax(self._q_values[out_idx], axis=0))

            _replay_net_outputs = self.online_convnet(self._replay[out_idx].states,
                                                      self.num_tau_samples)
            # Shape: (num_tau_samples x batch_size) x num_actions.
            self._replay_net_quantile_values.append(_replay_net_outputs[out_idx].quantile_values)
            self._replay_net_quantiles.append(_replay_net_outputs[out_idx].quantiles)

            # Do the same for next states in the replay buffer.
            replay_net_target_outputs = self.target_convnet(
                self._replay[out_idx].next_states, self.num_tau_prime_samples)
            # Shape: (num_tau_prime_samples x batch_size) x num_actions.
            vals = replay_net_target_outputs[out_idx].quantile_values
            self._replay_net_target_quantile_values.append(vals)

            # Compute Q-values which are used for action selection for the next states
            # in the replay buffer. Compute the argmax over the Q-values.
            if self.double_dqn:
                outputs_action = self.online_convnet(self._replay[out_idx].next_states,
                                                     self.num_quantile_samples)
            else:
                outputs_action = self.target_convnet(self._replay[out_idx].next_states,
                                                     self.num_quantile_samples)

            # Shape: (num_quantile_samples x batch_size) x num_actions.
            target_quantile_values_action = outputs_action[out_idx].quantile_values
            # Shape: num_quantile_samples x batch_size x num_actions.
            target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                                       [self.num_quantile_samples,
                                                        self._replay[out_idx].batch_size,
                                                        self.num_actions_task[out_idx]])
            # Shape: batch_size x num_actions_task.
            _replay_net_target_q_values = tf.squeeze(tf.reduce_mean(
                target_quantile_values_action, axis=0))

            # valid_act_mask = self._replay.transition['valid_act_mask']
            # self._replay_net_target_q_values_masked = tf.where(valid_act_mask,
            #                 self._replay_net_target_q_values, tf.float32.min)

            self._replay_next_qt_argmax.append(tf.argmax(_replay_net_target_q_values, axis=1))

    def _build_target_quantile_values_op(self):
        """Build an op used as a target for return values at given quantiles.

        Returns:
          An op calculating the target quantile return.
        """
        out_list = []

        for out_idx in range(self.num_outputs):
            batch_size = tf.shape(self._replay[out_idx].rewards)[0]
            # Shape of rewards: (num_tau_prime_samples x batch_size) x 1.
            rewards = self._replay[out_idx].rewards[:, None]
            rewards = tf.tile(rewards, [self.num_tau_prime_samples, 1])

            is_terminal_multiplier = 1. - tf.cast(self._replay[out_idx].terminals, tf.float32)
            # Incorporate terminal state to discount factor.
            # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
            gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
            gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                          [self.num_tau_prime_samples, 1])

            # Get the indices of the maximium Q-value across the action dimension.
            # Shape of replay_next_qt_argmax: (num_tau_prime_samples x batch_size) x 1.

            replay_next_qt_argmax = tf.tile(
                self._replay_next_qt_argmax[out_idx][:, None], [self.num_tau_prime_samples, 1])

            # Shape of batch_indices: (num_tau_prime_samples x batch_size) x 1.
            batch_indices = tf.cast(tf.range(
                self.num_tau_prime_samples * batch_size)[:, None], tf.int64)

            # Shape of batch_indexed_target_values:
            # (num_tau_prime_samples x batch_size) x 2.
            batch_indexed_target_values = tf.concat(
                [batch_indices, replay_next_qt_argmax], axis=1)

            # Shape of next_target_values: (num_tau_prime_samples x batch_size) x 1.
            target_quantile_values = tf.gather_nd(
                self._replay_net_target_quantile_values[out_idx],
                batch_indexed_target_values)[:, None]

            out_list.append(rewards + gamma_with_terminal * target_quantile_values)

        return out_list

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        target_quantile_values = self._build_target_quantile_values_op() # stop graident afterwards

        train_ops = []

        for out_idx in range(self.num_outputs):
            batch_size = tf.shape(self._replay[out_idx].rewards)[0]

            # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
            # the manner in which the target_quantile_values are tiled.
            target_quantile_values_res = tf.reshape(tf.stop_gradient(target_quantile_values[out_idx]),
                                                    [self.num_tau_prime_samples,
                                                     batch_size, 1])
            # Transpose dimensions so that the dimensionality is batch_size x
            # self.num_tau_prime_samples x 1 to prepare for computation of
            # Bellman errors.
            # Final shape of target_quantile_values:
            # batch_size x num_tau_prime_samples x 1.
            target_quantile_values_res = tf.transpose(target_quantile_values_res, [1, 0, 2])

            # Shape of indices: (num_tau_samples x batch_size) x 1.
            # Expand dimension by one so that it can be used to index into all the
            # quantiles when using the tf.gather_nd function (see below).
            indices = tf.range(self.num_tau_samples * batch_size)[:, None]

            # Expand the dimension by one so that it can be used to index into all the
            # quantiles when using the tf.gather_nd function (see below).
            reshaped_actions = self._replay[out_idx].actions[:, None]
            reshaped_actions = tf.tile(reshaped_actions, [self.num_tau_samples, 1])
            # Shape of reshaped_actions: (num_tau_samples x batch_size) x 2.
            reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

            chosen_action_quantile_values = tf.gather_nd(
                self._replay_net_quantile_values[out_idx], reshaped_actions)
            # Reshape to self.num_tau_samples x batch_size x 1 since this is the manner
            # in which the quantile values are tiled.
            chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                                       [self.num_tau_samples,
                                                        batch_size, 1])
            # Transpose dimensions so that the dimensionality is batch_size x
            # self.num_tau_samples x 1 to prepare for computation of
            # Bellman errors.
            # Final shape of chosen_action_quantile_values:
            # batch_size x num_tau_samples x 1.
            chosen_action_quantile_values = tf.transpose(
                chosen_action_quantile_values, [1, 0, 2])

            # Shape of bellman_erors and huber_loss:
            # batch_size x num_tau_prime_samples x num_tau_samples x 1.
            bellman_errors = target_quantile_values_res[
                             :, :, None, :] - chosen_action_quantile_values[:, None, :, :]
            # The huber loss (see Section 2.3 of the paper) is defined via two cases:
            # case_one: |bellman_errors| <= kappa
            # case_two: |bellman_errors| > kappa
            huber_loss_case_one = (
                    tf.cast(tf.abs(bellman_errors) <= self.kappa, tf.float32) *
                    0.5 * bellman_errors ** 2)
            huber_loss_case_two = (
                    tf.cast(tf.abs(bellman_errors) > self.kappa, tf.float32) *
                    self.kappa * (tf.abs(bellman_errors) - 0.5 * self.kappa))
            huber_loss = huber_loss_case_one + huber_loss_case_two

            # Reshape replay_quantiles to batch_size x num_tau_samples x 1
            replay_quantiles = tf.reshape(
                self._replay_net_quantiles[out_idx], [self.num_tau_samples, batch_size, 1])
            replay_quantiles = tf.transpose(replay_quantiles, [1, 0, 2])

            # Tile by num_tau_prime_samples along a new dimension. Shape is now
            # batch_size x num_tau_prime_samples x num_tau_samples x 1.
            # These quantiles will be used for computation of the quantile huber loss
            # below (see section 2.3 of the paper).
            replay_quantiles = tf.cast(
                tf.tile(replay_quantiles[:, None, :, :],
                        [1, self.num_tau_prime_samples, 1, 1]), tf.float32)
            # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
            quantile_huber_loss = (tf.abs(replay_quantiles - tf.stop_gradient(
                tf.cast(bellman_errors < 0, tf.float32))) * huber_loss) / self.kappa
            # Sum over current quantile value (num_tau_samples) dimension,
            # average over target quantile value (num_tau_prime_samples) dimension.
            # Shape: batch_size x num_tau_prime_samples x 1.
            loss = tf.reduce_sum(quantile_huber_loss, axis=2)
            # Shape: batch_size x 1.
            loss = tf.reduce_mean(loss, axis=1)

            # TODO(kumasaurabh): Add prioritized replay functionality here.
            update_priorities_op = tf.no_op()
            with tf.control_dependencies([update_priorities_op]):
                if self.summary_writer is not None:
                    with tf.compat.v1.variable_scope('Losses'):
                        tf.compat.v1.summary.scalar('QuantileLoss', tf.reduce_mean(loss))
                train_ops.append(
                    (self.optimizer.minimize(tf.reduce_mean(loss)), tf.reduce_mean(loss))
                )
        return train_ops

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
           int, the selected action.
        """
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(
                self.epsilon_decay_period,
                self.training_steps,
                self.min_replay_history,
                self.epsilon_train)
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            return random.randint(0, self.num_actions - 1)
        else:
            # Choose the action with highest Q-value at the current state.
            q_values = self._sess.run(self._q_values[self.task_id],
                                      {self.state_ph: self.state})
            action = np.argmax(q_values[..., :self.num_actions], -1)
            # this limit of the action space is not reflected in the update equation
            return action

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        train_id = np.random.choice(self.num_outputs, p=self.task_probs)
        if self._replay[train_id].memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                # try:
                self._sess.run(self._train_op[train_id]) # this samples from 0th buffer for train_id 1, fixme
                # except RuntimeError as err:
                #     print('update failed for ag ' + self.base_name + ' task ' + str(train_id))
                #     print(err)

                # if (self.summary_writer is not None and
                #         self.training_steps > 0 and
                #         self.training_steps % self.summary_writing_frequency == 0):
                #     summary = self._sess.run(self._merged_summaries)  # todo investigate why this crashes in the replay buffer somehow
                #     self.summary_writer.add_summary(summary, self.training_steps)

            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self.training_steps += 1

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number, name=''):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
          checkpoint_dir: str, directory where TensorFlow objects will be saved.
          iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
          A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not tf.io.gfile.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, name, 'tf_ckpt'),
            global_step=iteration_number)
        # Checkpoint the out-of-graph replay buffer.
        for idx in range(len(self._replay)):
            replay_fp = os.path.join(checkpoint_dir, name, 'replay' + str(idx))
            os.makedirs(replay_fp, exist_ok=True)
            self._replay[idx].save(replay_fp, iteration_number)

        bundle_dictionary = {}
        bundle_dictionary['state'] = self.state
        bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary, name=''):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
          agent's state.

        Args:
          checkpoint_dir: str, path to the checkpoint saved by tf.Save.
          iteration_number: int, checkpoint version, used when restoring the replay
            buffer.
          bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
          bool, True if unbundling was successful.
        """
        try:
            # self._replay.load() will throw a NotFoundError if it does not find all
            # the necessary files.
            for idx in range(len(self._replay)):
                replay_fp = os.path.join(checkpoint_dir, 'checkpoints', name, 'replay' + str(idx))
                self._replay[idx].load(replay_fp, iteration_number)
        except tf.errors.NotFoundError:
            if not self.allow_partial_reload:
                # If we don't allow partial reloads, we will return False.
                return False
            self._log.warning('Unable to reload replay buffer!')
        if bundle_dictionary is not None:
            for key in self.__dict__:
                if key in bundle_dictionary:
                    self.__dict__[key] = bundle_dictionary[key]
        elif not self.allow_partial_reload:
            return False
        else:
            self._log.warning("Unable to reload the agent's parameters!")
        # Restore the agent's TensorFlow graph.
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,'checkpoints', name,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True
