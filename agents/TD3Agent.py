"""
Simple implementation of TD3.
"""

import pickle

import numpy as np
import tensorflow as tf
from gym import Space
from gym.spaces import Discrete, Box

from agents.AbstractAgent import AbstractAgent
from common.util import clip_by_local_norm

def space_n_to_shape_n(space_n):
    return np.array([space_to_shape(space) for space in space_n])

def space_to_shape(space):
    if type(space) is Box:
        return space.shape
    if type(space) is Discrete:
        return [space.n]


class TD3Agent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_units, gamma,
                 tau, action_noise_value=0.1, prioritized_replay=False, alpha=0.6, max_step=None,
                 initial_beta=0.6, prioritized_replay_eps=1e-6,
                 policy_update_freq=2, target_policy_smoothing_eps=0.2, _run=None,
                 num_uncertainty_nets=0):
        """
        An object containing critic, actor and training functions.
        :param num_layer:
        """
        self._run = _run
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        self.act_space_n = act_space_n
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic_1 = CriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target = CriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target.model.set_weights(self.critic_1.model.get_weights())

        self.critic_2 = CriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target = CriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target.model.set_weights(self.critic_2.model.get_weights())

        self.policy = PolicyNetwork(num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                    self.critic_1, agent_index)
        self.policy_target = PolicyNetwork(num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                           self.critic_1, agent_index)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.uncert_est_nets = []   # uncertainty estimation nets
        self.uncert_est_target_nets = []

        for net_idx in range(num_uncertainty_nets):
            self.uncert_est_nets.append(
                CriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index))
            self.uncert_est_target_nets.append(
                CriticNetwork(num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index))
            self.uncert_est_target_nets[-1].model.set_weights(
                self.uncert_est_nets[-1].model.get_weights()
            )


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

    @tf.function
    def get_gauss_action(self, obs):
        action = self.policy.get_action(obs[None])[0]
        noise = tf.random.normal(tf.shape(action), 0, self.action_noise_value)
        clipped_noise = tf.clip_by_value(noise, -0.5, 0.5)
        action_noisy = tf.clip_by_value(action + clipped_noise, self.act_space_n[0].low,
                                                     self.act_space_n[0].high)
        return action_noisy

    def action(self, obs):
        """
        Get an action from the non-target policy
        """
        if self.policy.use_gumbel:
            action = self.policy.get_action(obs[None])[0].numpy()
            output = np.zeros(action.shape)
            output[np.argmax(action)] = 1.0
            return output
        else:
            return self.get_gauss_action(obs).numpy()

    def deterministic_action(self, obs):
        if self.policy.use_gumbel:
            action = self.policy.get_action(obs[None])[0].numpy()
            output = np.zeros(action.shape)
            output[np.argmax(action)] = 1.0
            return output
        else:
            return self.policy.get_action(obs[None])[0].numpy()

    def target_action(self, obs):
        """
        Get an action from the non-target policy
        """
        if self.policy.use_gumbel:
            action = self.policy_target.get_action(obs[None])[0].numpy()
            output = np.zeros(action.shape)
            output[np.argmax(action, 1)] = 1.0
            return output
        else:
            action = self.policy_target.get_action(obs[None])[0].numpy()
            noise = np.random.normal(0, self.action_noise_value, size=action.shape)
            clipped_noise = np.clip(noise, -0.5, 0.5)
            action_noisy = (action + clipped_noise).clip(self.act_space_n[0].low, self.act_space_n[0].high)
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

        update_target_network(self.critic_1.model, self.critic_1_target.model)
        update_target_network(self.critic_2.model, self.critic_2_target.model)
        update_target_network(self.policy.model, self.policy_target.model)

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

        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
            # self._run.log_scalar('agent_{}.train.mean_weight'.format(self.agent_index), np.mean(weights), step)
            # self._run.log_scalar('agent_{}.train.max_weight'.format(self.agent_index), np.max(weights), step)
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)

        # Train the critic, using the target actions in the target critic network, to determine the
        # training target (i.e. target in MSE loss) for the critic update.
        target_act_next = [ag.target_action(obs) for ag, obs in zip(agents, next_obs_n)]
        noise = np.random.normal(0, self.target_policy_smoothing_eps, size=target_act_next[self.agent_index].shape)
        noise = np.clip(noise, -0.5, 0.5)
        target_act_next[self.agent_index] += noise

        critic_outputs = np.empty([2, self.batch_size], dtype=np.float32)  # this is a lot faster than python list plus minimum
        critic_outputs[0] = self.critic_1_target.predict(next_obs_n, target_act_next)[:, 0]
        critic_outputs[1] = self.critic_2_target.predict(next_obs_n, target_act_next)[:, 0]
        target_q_next = np.min(critic_outputs, 0)[:, None]

        target_q_next[done_n == 1.0] = 0

        q_train_target = rew_n[:, None] + self.decay * target_q_next

        td_loss = np.empty([2, self.batch_size], dtype=np.float32)
        loss_0, q_gradients = self.critic_1.train_step(obs_n, acts_n, q_train_target, weights)
        td_loss[0] = loss_0.numpy()[:, 0]
        td_loss[1] = self.critic_2.train_step(obs_n, acts_n, q_train_target, weights)[0].numpy()[:, 0]
        max_loss = np.max(td_loss, 0)


        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, max_loss + self.prioritized_replay_eps)

        if self.update_counter % self.policy_update_freq == 0 or force_policy_update:  # delayed policy updates
            # Train the policy.
            policy_loss, p_gradients = self.policy.train(obs_n, acts_n)
            # Update target networks.
        else:
            policy_loss = None
            p_gradients = None

        # separate batches for each uncertainty net
        for uncert_net, uncert_target_net in zip(self.uncert_est_nets, self.uncert_est_target_nets):
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            target_act_next = [ag.target_action(obs) for ag, obs in zip(agents, next_obs_n)]
            target_q_next = self.critic_1_target.predict(next_obs_n, target_act_next).numpy()[:, 0]
            target_q_next[done_n == 1.0] = 0
            q_train_target = rew_n[:, None] + self.decay * target_q_next
            uncert_net.train_step(obs_n, acts_n, q_train_target, weights)

        if self.update_counter % self.policy_update_freq == 0 or force_policy_update:  # delayed policy updates
            self.update_target_networks(self.tau)

        self.log_values(td_loss[0], td_loss[1], policy_loss, step)

        return [td_loss, policy_loss, q_gradients, p_gradients]

    @tf.function(experimental_relax_shapes=True)
    def get_q_value_disagreement(self, obs_n, acts_n):
        """
        Calculates disagreement between q value certainty estimation functions (not used for
        training the policy).

        Calculates stdev per transition and then returns the mean over these values
        """
        # predictions = tf.zeros([len(self.uncert_est_nets), obs_n.shape[0]])
        preds = []
        for idx, net in enumerate(self.uncert_est_nets):
            preds.append(net.predict([obs_n], [acts_n])[:, 0])
        disagreement = tf.math.reduce_std(preds, 0)
        mean_disagreement = tf.math.reduce_mean(disagreement)
        return mean_disagreement

    def get_uncertainty_traj(self, trajectory):
        """
        Get uncertainty for an input trajectory
        """
        obs_n = []
        acts_n = []
        for trans in trajectory:
            obs_n.append(trans[0])
            acts_n.append(trans[1])
        obs_n = np.array(obs_n)
        acts_n = np.array(acts_n)

        return self.get_q_value_disagreement(obs_n, acts_n).numpy()

    def get_uncertainty_buffer(self, trajectory):
        """
        Get uncertainty for each task from the replay buffer.
        """
        raise NotImplementedError()


    def get_gradients_for_batch(self, obs_n, acts_n, rew_n, next_obs_n, done_n):
        """
        Performs an update with given batch and returns the gradients.
        WARNING: ALSO APPLIES TRAINS THE POLICY, DOESNT JUST GET THE GRADIENT!
        """
        obs_n = [np.array(obs_n[0])]
        acts_n = [np.array(acts_n[0], np.float32)]  # this is necessary because reasons
        rew_n = np.array(rew_n, np.float32)
        next_obs_n = [np.array(next_obs_n[0])]
        done_n = np.array(done_n)

        target_act_next = [self.target_action(next_obs_n[0])]
        noise = np.random.normal(0, self.target_policy_smoothing_eps, size=target_act_next[self.agent_index].shape)
        noise = np.clip(noise, -0.5, 0.5)
        target_act_next[self.agent_index] += noise

        critic_outputs = np.empty([2, len(obs_n[0])], dtype=np.float32)  # this is a lot faster than python list plus minimum
        critic_outputs[0] = self.critic_1_target.predict(next_obs_n, target_act_next)[:, 0]
        critic_outputs[1] = self.critic_2_target.predict(next_obs_n, target_act_next)[:, 0]
        target_q_next = np.min(critic_outputs, 0)[:, None]

        target_q_next[done_n == 1.0] = 0
        q_train_target = rew_n[:, None] + self.decay * target_q_next
        _, q_gradients = self.critic_1.train_step(obs_n, acts_n, q_train_target, tf.ones_like(rew_n))

        # Train the policy.
        policy_loss, p_gradients = self.policy.train(obs_n, acts_n)

        return [q_gradients, p_gradients]

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
            if '_run' in self.__dict__.keys() and self._run is not None:
                # after pickling this is false
                self._run.log_scalar('policy_loss', np.mean(self.policy_losses), step)
                self._run.log_scalar('q_loss0', np.mean(self.q_losses0), step)
                self._run.log_scalar('q_loss1', np.mean(self.q_losses1), step)
            self.q_losses0 = []
            self.q_losses1 = []
            self.policy_losses = []
            self.entropies = []
            self.log_counter = 0

    def save(self, fp):
        self.critic_1.model.save_weights(fp + 'critic_1.h5',)
        self.critic_1_target.model.save_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.save_weights(fp + 'critic_2.h5',)
        self.critic_2_target.model.save_weights(fp + 'critic_2_target.h5')

        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        self.critic_1.model.load_weights(fp + 'critic_1.h5',)
        self.critic_1_target.model.load_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.load_weights(fp + 'critic_2.h5',)
        self.critic_2_target.model.load_weights(fp + 'critic_2_target.h5')

        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')

    def get_checkpoint_dict(self):
        checkpoint_dict = {'weights': self.get_weights(),
                           'replaybuffer': self.replay_buffer,}
        return checkpoint_dict

    def load_checkpoint_dict(self, checkpoint_dict):
        self.replay_buffer = checkpoint_dict['replaybuffer']
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

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['_run']
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)


class PolicyNetwork(object):
    def __init__(self, units_per_layer, lr, obs_shape_n, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)
        self.lr = lr
        self.obs_shape_n = obs_shape_n
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_shape_n[agent_index])

        self.hidden_layers = []
        for idx in range(self.num_layers):
            layer = tf.keras.layers.Dense(units_per_layer[idx], activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    @tf.function
    def get_action(self, obs):
        x = obs
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network

        if self.use_gumbel:
            samples = self.gumbel_softmax_sample(outputs)
            return samples
        else:
            return outputs

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            x = obs_n[self.agent_index]
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            x = self.output_layer(x)
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x  # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
                # act_n = tf.stack(act_n)
            else:
                act_n[self.agent_index] = x
            q_value = self.q_network._predict_internal(obs_n + act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss, gradients

    # def __getstate__(self):
    #     copy_elements = ['units_per_layer', 'lr', 'obs_shape_n', 'act_shape', 'act_type',
    #                      'gumbel_temperature', 'q_network', 'agent_index']
    #     state = {}
    #     for name in copy_elements:
    #         state[name] = self.__dict__[name]
    #     state['network_config'] = self.model.get_config()
    #     state['network_weights'] = self.model.get_weights()
    #     state['optimizer_weights'] = self.optimizer.get_weights()
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__init__(state['units_per_layer'], state['lr'], state['obs_shape_n'], state['act_shape'],
    #                   state['act_type'], state['gumbel_temperature'], state['q_network'],
    #                   state['agent_index'])
    #     self.model.set_weights(state['network_weights'])
    #     self.optimizer.set_weights(state['optimizer_weights'])


class CriticNetwork(object):
    def __init__(self, units_per_layer, lr, obs_shape_n, act_shape_n, act_type, agent_index):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)
        self.lr = lr
        self.obs_shape_n = obs_shape_n
        self.act_shape_n = act_shape_n
        self.act_type = act_type
        self.agent_index = agent_index

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(self.num_layers):
            layer = tf.keras.layers.Dense(units_per_layer[idx], activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}crit_out{}'.format(agent_index, idx))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n, act_n):
        """
        Predict the value of the input.
        """
        return self._predict_internal(obs_n + act_n)

    @tf.function
    def _predict_internal(self, concatenated_input):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        loss, gradients = self._train_step_internal(obs_n + act_n, target_q, weights)
        return loss, gradients

    @tf.function
    def _train_step_internal(self, concatenated_input, target_q, weights):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss, gradients

    # def __getstate__(self):
    #     copy_elements = ['units_per_layer', 'lr', 'obs_shape_n', 'act_shape_n', 'act_type',
    #                      'agent_index']
    #     state = {}
    #     for name in copy_elements:
    #         state[name] = self.__dict__[name]
    #     state['network_weights'] = self.model.get_weights()
    #     state['optimizer_weights'] = self.optimizer.get_weights()
    #
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__init__(state['units_per_layer'], state['lr'], state['obs_shape_n'], state['act_shape_n'],
    #                   state['act_type'], state['agent_index'])
    #     self.model.set_weights(state['network_weights'])
    #     self.optimizer.set_weights(state['optimizer_weights'])



def main():
    import pickle
    shape = Box(np.array([-1.,-1.]),np.array([1., 1.]))
    crit = CriticNetwork([32, 32], 1e-4, [shape.shape], [shape.shape], [Box],0)
    policy = PolicyNetwork([32, 32], 1e-4, [shape.shape], 2, [Box], 0, crit, 0)

    pick = pickle.dumps(crit)
    rest_crit = pickle.loads(pick)

    for w1, w2 in zip(crit.model.get_weights(), rest_crit.model.get_weights()):
        tf.debugging.assert_equal(w1, w2)
    for w1, w2 in zip(crit.optimizer.get_weights(), rest_crit.optimizer.get_weights()):
        tf.debugging.assert_equal(w1, w2)


    pol_pick = pickle.dumps(policy)
    rest_pol = pickle.loads(pol_pick)
    for w1, w2 in zip(policy.model.get_weights(), rest_pol.model.get_weights()):
        tf.debugging.assert_equal(w1, w2)
    for w1, w2 in zip(policy.optimizer.get_weights(), rest_pol.optimizer.get_weights()):
        tf.debugging.assert_equal(w1, w2)



if __name__ == '__main__':
    main()
