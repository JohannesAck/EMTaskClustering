"""
Agent like described in the MDDPG paper but with TD3 instead.
Assumes all tasks have the same Action and State space because this is our setting.
"""

import numpy as np
import tensorflow as tf
from functools import partial

from gym import Space
from gym.spaces import Discrete, Box

from agents.AbstractAgent import AbstractAgent
from common.util import clip_by_local_norm
from common.replay_buffer import EfficientReplayBuffer


def space_n_to_shape_n(space_n):
    return np.array([space_to_shape(space) for space in space_n])


def space_to_shape(space):
    if type(space) is Box:
        return space.shape
    if type(space) is Discrete:
        return [space.n]


class MTD3Agent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_units,
                 gamma,
                 tau, num_tasks, action_noise_value=0.1, prioritized_replay=False, alpha=0.6,
                 max_step=None,
                 initial_beta=0.6, prioritized_replay_eps=1e-6,
                 policy_update_freq=2, target_policy_smoothing_eps=0.2, _run=None,
                 num_uncertainty_nets=0):
        """

        """
        self._run = _run
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        self.act_space_n = act_space_n
        self.num_tasks = num_tasks

        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha,
                         max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        del self.replay_buffer
        self.replay_buffer_n = []
        for idx in range(self.num_tasks):
            self.replay_buffer_n.append(EfficientReplayBuffer(int(buff_size), 1,
                                                              obs_shape_n, act_shape_n))

        act_type = type(act_space_n[0])
        self.critic_1 = MuliTaskCriticNetwork(num_units, lr, obs_shape_n, act_shape_n,
                                              act_type, agent_index, num_tasks)
        self.critic_1_target = MuliTaskCriticNetwork(num_units, lr, obs_shape_n, act_shape_n,
                                                     act_type, agent_index, num_tasks)
        self.critic_1_target.model.set_weights(self.critic_1.model.get_weights())

        self.critic_2 = MuliTaskCriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type,
                                              agent_index, num_tasks)
        self.critic_2_target = MuliTaskCriticNetwork(num_units, lr, obs_shape_n, act_shape_n,
                                                     act_type, agent_index, num_tasks)
        self.critic_2_target.model.set_weights(self.critic_2.model.get_weights())

        self.policy = MuliTaskPolicyNetwork(num_units, lr, obs_shape_n, act_shape_n[agent_index],
                                            act_type, 1,
                                            self.critic_1, agent_index, num_tasks)
        self.policy_target = MuliTaskPolicyNetwork(num_units, lr, obs_shape_n,
                                                   act_shape_n[agent_index], act_type, 1,
                                                   self.critic_1, agent_index, num_tasks)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.uncert_est_nets = []  # uncertainty estimation nets
        self.uncert_est_target_nets = []

        self.batch_size = batch_size
        self.action_noise_value = action_noise_value
        self.decay = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.target_policy_smoothing_eps = target_policy_smoothing_eps
        self.update_counter = 0
        self.agent_index = agent_index
        self.num_uncertainty_nets = num_uncertainty_nets

        # logging
        self.q_losses0 = []
        self.q_losses1 = []
        self.policy_losses = []
        self.log_counter = 0
        self.log_rate = 2000

    def add_transition(self, obs_n, act_n, rew, new_obs_n, done_n, task_id):
        """
        Adds a transition to the replay buffer.
        """
        self.replay_buffer_n[task_id].add(obs_n, act_n, rew, new_obs_n, float(done_n))

    @tf.function
    def get_gauss_action(self, obs, task_id):
        action = self.policy.get_action(obs[None], task_id)[0]
        noise = tf.random.normal(tf.shape(action), 0, self.action_noise_value)
        clipped_noise = tf.clip_by_value(noise, -0.5, 0.5)
        action_noisy = tf.clip_by_value(action + clipped_noise, self.act_space_n[0].low,
                                        self.act_space_n[0].high)
        return action_noisy

    def action(self, obs, task_id):
        """
        Get an action from the non-target policy
        """
        if self.policy.use_gumbel:
            action = self.policy.get_action(obs[None], task_id)[0].numpy()
            output = np.zeros(action.shape)
            output[np.argmax(action)] = 1.0
            return output
        else:
            return self.get_gauss_action(obs, task_id).numpy()

    def deterministic_action(self, obs, task_id):
        if self.policy.use_gumbel:
            action = self.policy.get_action(obs[None], task_id)[0].numpy()
            output = np.zeros(action.shape)
            output[np.argmax(action)] = 1.0
            return output
        else:
            return self.policy.get_action(obs[None], task_id)[0].numpy()

    def target_action(self, obs, task_id):
        """
        Get an action from the non-target policy
        """
        if self.policy.use_gumbel:
            action = self.policy_target.get_action(obs[None], task_id)[0].numpy()
            output = np.zeros(action.shape)
            output[np.argmax(action, 1)] = 1.0
            return output
        else:
            action = self.policy_target.get_action(obs[None], task_id)[0].numpy()
            noise = np.random.normal(0, self.action_noise_value, size=action.shape).astype(np.float32)
            clipped_noise = np.clip(noise, -0.5, 0.5)
            action_noisy = (action + clipped_noise).clip(self.act_space_n[0].low,
                                                         self.act_space_n[0].high)
            return action_noisy

    def preupdate(self):
        pass

    def update_target_networks(self, tau):
        """
        Implements the updates of the target networks, which slowly follow the real network.
        """

        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        for task_id in range(self.num_tasks):
            update_target_network(self.critic_1.model_n[task_id],
                                  self.critic_1_target.model_n[task_id])
            update_target_network(self.critic_2.model_n[task_id],
                                  self.critic_2_target.model_n[task_id])
            update_target_network(self.policy.model_n[task_id],
                                  self.policy_target.model_n[task_id])

        for idx in range(len(self.uncert_est_nets)):
            update_target_network(self.uncert_est_nets[idx].model,
                                  self.uncert_est_target_nets[idx].model)

    def update(self, agents, step, force_policy_update=False):
        """
        Update the agent, by first updating the two critics and then the policy.
        Requires the list of the other agents as input, to determine the target actions.
        """
        assert agents[self.agent_index] is self
        self.update_counter += 1

        for task_id in range(self.num_tasks):
            obs_n, acts_n, rew_n, next_obs_n, done_n = \
                self.replay_buffer_n[task_id].sample(self.batch_size)

            # Train the critic, using the target actions in the target critic network, to determine the
            # training target (i.e. target in MSE loss) for the critic update.
            target_act_next = [ag.target_action(obs, task_id) for ag, obs in
                               zip(agents, next_obs_n)]
            noise = np.random.normal(0, self.target_policy_smoothing_eps,
                                     size=target_act_next[self.agent_index].shape).astype(np.float32)
            noise = np.clip(noise, -0.5, 0.5)
            target_act_next[self.agent_index] += noise

            critic_outputs = np.empty([2, self.batch_size],
                                      dtype=np.float32)  # this is a lot faster than python list plus minimum
            critic_outputs[0] = self.critic_1_target.predict(next_obs_n, target_act_next, task_id)[
                                :, 0]
            critic_outputs[1] = self.critic_2_target.predict(next_obs_n, target_act_next, task_id)[
                                :, 0]
            target_q_next = np.min(critic_outputs, 0)[:, None]

            target_q_next[done_n == 1.0] = 0

            q_train_target = rew_n[:, None] + self.decay * target_q_next

            td_loss = np.empty([2, self.batch_size], dtype=np.float32)
            loss_0, q_gradients = self.critic_1.train_step(obs_n, acts_n, q_train_target, task_id)
            td_loss[0] = loss_0.numpy()[:, 0]
            td_loss[1] = \
                self.critic_2.train_step(obs_n, acts_n, q_train_target, task_id)[0].numpy()[:, 0]
            max_loss = np.max(td_loss, 0)

            if self.update_counter % self.policy_update_freq == 0 or force_policy_update:  # delayed policy updates
                # Train the policy.
                policy_loss, p_gradients = self.policy.train(obs_n, acts_n, task_id)
                # Update target networks.
            else:
                policy_loss = None
                p_gradients = None

        if self.update_counter % self.policy_update_freq == 0 or force_policy_update:  # delayed policy updates
            self.update_target_networks(self.tau)

        self.log_values(td_loss[0], td_loss[1], policy_loss, step)

        return [td_loss, policy_loss, q_gradients, p_gradients]

    # def get_gradients_for_batch(self, obs_n, acts_n, rew_n, next_obs_n, done_n):
    #     """
    #     Performs an update with given batch and returns the gradients.
    #     WARNING: ALSO APPLIES TRAINS THE POLICY, DOESNT JUST GET THE GRADIENT!
    #     """
    #     obs_n = [np.array(obs_n[0])]
    #     acts_n = [np.array(acts_n[0], np.float32)]  # this is necessary because reasons
    #     rew_n = np.array(rew_n, np.float32)
    #     next_obs_n = [np.array(next_obs_n[0])]
    #     done_n = np.array(done_n)
    #
    #     target_act_next = [self.target_action(next_obs_n[0])]
    #     noise = np.random.normal(0, self.target_policy_smoothing_eps,
    #                              size=target_act_next[self.agent_index].shape)
    #     noise = np.clip(noise, -0.5, 0.5)
    #     target_act_next[self.agent_index] += noise
    #
    #     critic_outputs = np.empty([2, len(obs_n[0])],
    #                               dtype=np.float32)  # this is a lot faster than python list plus minimum
    #     critic_outputs[0] = self.critic_1_target.predict(next_obs_n, target_act_next)[:, 0]
    #     critic_outputs[1] = self.critic_2_target.predict(next_obs_n, target_act_next)[:, 0]
    #     target_q_next = np.min(critic_outputs, 0)[:, None]
    #
    #     target_q_next[done_n == 1.0] = 0
    #     q_train_target = rew_n[:, None] + self.decay * target_q_next
    #     _, q_gradients = self.critic_1.train_step(obs_n, acts_n, q_train_target,
    #                                               tf.ones_like(rew_n))
    #
    #     # Train the policy.
    #     policy_loss, p_gradients = self.policy.train(obs_n, acts_n)
    #
    #     return [q_gradients, p_gradients]

    def log_values(self, q_loss1, q_loss2, policy_loss, step):
        """
        This function is called after every training to log losses, which are then sent to the
        sacred logging database every 1000 or so updates. Otherwise the log file becomes to large
        and sacred crashes.
        """
        self.q_losses0.append(np.mean(q_loss1))
        self.q_losses1.append(np.mean(q_loss2))
        if policy_loss is not None:
            self.policy_losses.append(np.mean(policy_loss))
        self.log_counter += 1
        if self.log_counter > self.log_rate:
            self._run.log_scalar('policy_loss', np.mean(self.policy_losses), step)
            self._run.log_scalar('q_loss0', np.mean(self.q_losses0), step)
            self._run.log_scalar('q_loss1', np.mean(self.q_losses1), step)
            self.q_losses0 = []
            self.q_losses1 = []
            self.policy_losses = []
            self.entropies = []
            self.log_counter = 0

    def save(self, fp):
        self.critic_1.model.save_weights(fp + 'critic_1.h5', )
        self.critic_1_target.model.save_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.save_weights(fp + 'critic_2.h5', )
        self.critic_2_target.model.save_weights(fp + 'critic_2_target.h5')

        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        self.critic_1.model.load_weights(fp + 'critic_1.h5', )
        self.critic_1_target.model.load_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.load_weights(fp + 'critic_2.h5', )
        self.critic_2_target.model.load_weights(fp + 'critic_2_target.h5')

        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')

    def get_checkpoint_dict(self):
        checkpoint_dict = {'weights': self.get_weights(),
                           'replaybuffer_n': self.replay_buffer_n,}
        return checkpoint_dict

    def load_checkpoint_dict(self, checkpoint_dict):
        self.replay_buffer_n = checkpoint_dict['replaybuffer_n']
        # first perform one update step and then set the weights, as this initializes the
        # optimizers properly. We overwrite their state afterwards.
        self.update([self], 0, force_policy_update=True)
        self.set_weights(checkpoint_dict['weights'])

    def get_weights(self):
        """
        returns all weights from all functions including optimizer weights(state).
        """
        return [self.policy.model.get_weights(), self.policy.optimizer.get_weights(),
                self.policy_target.model.get_weights(), self.policy_target.optimizer.get_weights(),
                self.critic_1.model.get_weights(), self.critic_1.optimizer.get_weights(),
                self.critic_1_target.model.get_weights(), self.critic_1_target.optimizer.get_weights(),
                self.critic_2.model.get_weights(), self.critic_2.optimizer.get_weights(),
                self.critic_2_target.model.get_weights(), self.critic_2_target.optimizer.get_weights(),
                ]

    def set_weights(self, weights):
        """
        Sets all weights, should be in order from fucntion self.get_weights
        """
        self.policy.model.set_weights(weights[0])
        self.policy.optimizer.set_weights(weights[1])
        self.policy_target.model.set_weights(weights[2])
        self.policy_target.optimizer.set_weights(weights[3])
        self.critic_1.model.set_weights(weights[4])
        self.critic_1.optimizer.set_weights(weights[5]),
        self.critic_1_target.model.set_weights(weights[6])
        self.critic_1_target.optimizer.set_weights(weights[7]),
        self.critic_2.model.set_weights(weights[8])
        self.critic_2.optimizer.set_weights(weights[9])
        self.critic_2_target.model.set_weights(weights[10])
        self.critic_2_target.optimizer.set_weights(weights[11])



class MuliTaskPolicyNetwork(object):
    def __init__(self, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index, num_tasks):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        Units per layer is [units_task_specific_in, units_shared] and the output is the action_dim
        """
        assert len(units_per_layer) == 2
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network: MuliTaskCriticNetwork = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5
        self.num_tasks = num_tasks

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input_n = []
        for idx in range(self.num_tasks):
            self.obs_input_n.append(tf.keras.layers.Input(shape=self.obs_shape_n[idx]))

        self.input_layers_n = []
        for idx in range(self.num_tasks):
            layer = tf.keras.layers.Dense(units_per_layer[0], activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.input_layers_n.append(layer)

        self.shared_layer = tf.keras.layers.Dense(units_per_layer[1], activation='relu',
                                                  name='ag{}_pol_shared'.format(agent_index))

        self.output_layer_n = []
        for idx in range(self.num_tasks):
            self.output_layer_n.append(tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                             name='ag{}pol_out{}'.format(
                                                                 agent_index, idx)))

        # connect layers
        self.model_n = []
        x_outs = []
        for idx in range(self.num_tasks):
            x = self.obs_input_n[idx]
            x = self.input_layers_n[idx](x)
            x = self.shared_layer(x)
            x = self.output_layer_n[idx](x)
            x_outs.append(x)

            model = tf.keras.Model(inputs=[self.obs_input_n[idx]], outputs=[x])
            self.model_n.append(model)

        self.model = tf.keras.Model(inputs=self.obs_input_n, outputs=x_outs)
        self.model.compile(self.optimizer, loss='mse')

        self.train_functions = []
        for idx in range(num_tasks):
            part = partial(self._train_internal, task_id=idx)
            self.train_functions.append(tf.function(part))

    @tf.function
    def get_action(self, obs, task_id):
        return self.model_n[task_id](obs)
        # x = obs
        # for idx in range(self.num_layers):
        #     x = self.hidden_layers[idx](x)
        # outputs = self.output_layer(
        #     x)  # log probabilities of the gumbel softmax dist are the output of the network
        #
        # if self.use_gumbel:
        #     samples = self.gumbel_softmax_sample(outputs)
        #     return samples
        # else:
        #     return outputs

    # tracing in constructor
    def train(self, obs_n, act_n, task_id):
        return self.train_functions[task_id](obs_n, act_n)

    def _train_internal(self, obs_n, act_n, task_id):
        with tf.GradientTape() as tape:
            x = obs_n[self.agent_index]
            x = self.input_layers_n[task_id](x)
            x = self.shared_layer(x)
            x = self.output_layer_n[task_id](x)
            act_n = tf.unstack(act_n)
            act_n[self.agent_index] = x
            q_value = self.q_network.predict(obs_n, act_n, task_id)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(
                q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss,
                                  self.model_n[
                                      task_id].trainable_variables)
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(
            zip(local_clipped, self.model_n[task_id].trainable_variables))
        return loss, gradients


class MuliTaskCriticNetwork(object):
    def __init__(self, units_per_layer, lr, obs_shape_n, act_shape_n, act_type, agent_index,
                 num_tasks):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        Units per layer is [units_task_specific_in, units_shared] and the output is 1
        """
        assert len(units_per_layer) == 2
        assert len(obs_shape_n) == len(act_shape_n) == num_tasks
        self.lr = lr
        self.obs_shape_n = obs_shape_n
        self.act_shape_n = act_shape_n
        self.act_type = act_type
        self.num_tasks = num_tasks

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        self.act_input_n = []
        self.input_concat_layer_n = []

        for idx in range(num_tasks):
            self.obs_input_n.append(
                tf.keras.layers.Input(shape=obs_shape_n[idx], name='obs_in' + str(idx)))
            self.act_input_n.append(
                tf.keras.layers.Input(shape=act_shape_n[idx], name='act_in' + str(idx)))
            self.input_concat_layer_n.append(tf.keras.layers.Concatenate())

        self.input_layer_n = []
        for idx in range(self.num_tasks):
            layer = tf.keras.layers.Dense(units_per_layer[0], activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.input_layer_n.append(layer)

        self.shared_layer = tf.keras.layers.Dense(units_per_layer[1], activation='sigmoid',
                                                  name='ag{}crit_shared'.format(agent_index))

        self.output_layer_n = []
        for idx in range(self.num_tasks):
            layer = tf.keras.layers.Dense(1, activation='linear',
                                          name='ag{}crit_out{}'.format(agent_index, idx))
            self.output_layer_n.append(layer)

        self.model_n = []
        x_out_n = []
        # connect layers
        for idx in range(num_tasks):
            x = self.input_concat_layer_n[idx]([self.obs_input_n[idx], self.act_input_n[idx]])
            x = self.input_layer_n[idx](x)
            x = self.shared_layer(x)
            x = self.output_layer_n[idx](x)
            x_out_n.append(x)

            model = tf.keras.Model(inputs=[self.obs_input_n[idx], self.act_input_n[idx]],
                                   outputs=[x], name='task{}critic'.format(idx))
            model.compile(self.optimizer, loss='mse')
            self.model_n.append(model)
        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,
                                    outputs=x_out_n)
        self.model.compile(self.optimizer, loss='mse')

        self.train_functions = []
        for idx in range(num_tasks):
            part = partial(self._train_step_internal, task_id=idx)
            self.train_functions.append(tf.function(part))

    def predict(self, obs_n, act_n, task_id):
        """
        Predict the value of the input.
        """
        return self.model_n[task_id](obs_n + act_n)

    def train_step(self, obs_n, act_n, target_q, task_id):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        assert isinstance(task_id, int)
        return self.train_functions[task_id](obs_n + act_n, target_q)

    # @tf.function
    def _train_step_internal(self, concatenated_input, target_q, task_id):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer_n[task_id](concatenated_input)
            x = self.input_layer_n[task_id](x)
            x = self.shared_layer(x)
            q_pred = self.output_layer_n[task_id](x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss)

        gradients = tape.gradient(loss, self.model_n[task_id].trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped,
                                           self.model_n[task_id].trainable_variables))

        return td_loss, gradients
