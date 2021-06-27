# Implementation of the Atari experiments for
# "Unsupervised Task Clustering for Multi-Task Reinforcement Learning".
# This part is closely based on the Dopamine implementation, please credit them appropriately if
# you are to reuse any of this code.
# We have disabled most of the gin functionality in favor of sacred and made some changes to the
# network structures and training algorithm as detailed in the paper.

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
r"""The entry point for running a Dopamine agent.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import flags
from absl import logging
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver

from dopamine.discrete_domains import em_run_experiment
from common.sacred_util import get_run_path

# SETTINGS.CONFIG.READ_ONLY_CONFIG = False

train_ex = Experiment('Atari-EM')


@train_ex.config
def train_config():
    env_names = ['MsPacman', 'Alien','BankHeist', 'WizardOfWor',
                 'SpaceInvaders', 'Assault', 'DemonAttack', 'Phoenix',
                 'ChopperCommand', 'Jamesbond', 'Zaxxon', 'Riverraid']
    em_n_agents = 4                 # number em agents
    training_steps = 250000         # training steps per M step
    evaluation_steps = 27000        # evaluation steps per E step
    num_iterations = 200            # number of em iterations
    multi_head = True               # different final layer per task
    buff_size = 3e5                 # size of the replay buffer
    continue_fp = None              # continue from this checkpoint
    pretraining_its = 0             # iterations trained on uniformly at the start
    large_network = False           # 4x parameters compared to normal
    policy_per_task = False         # train a separate policy on each task
    network_scaling_factor = 1.0    # adjusts parameter count in the network compared to Nature DQN

    debug = True
    if debug:
        env_names = ['Pong', 'Amidar', 'MsPacman', 'SpaceInvaders',]
        em_n_agents = 2          # number em agents
        training_steps = 3000    # em training steps
        evaluation_steps = 100          # em evaluation steps
        num_iterations = 200                # num em_steps
        buff_size = 1e4
        large_network = False
        policy_per_task = True
        multi_head = False
        network_scaling_factor = 1 / len(env_names)
        continue_fp = None



@train_ex.main
def internal_main(env_names, em_n_agents, training_steps, evaluation_steps,
                  num_iterations, _run, _log, multi_head, continue_fp, buff_size,
                  pretraining_its, large_network, policy_per_task, network_scaling_factor):
  """
  Main method, it essentially just starts the experiment runner in which most of the important
  things happen.
  """
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.disable_v2_behavior()
  flags.DEFINE_string('base_dir', None,
                      'Base directory to host all required sub-directories.')
  flags.DEFINE_multi_string(
      'gin_files', [], 'List of paths to gin configuration files (e.g.'
                       '"dopamine/agents/dqn/dqn.gin").')
  flags.DEFINE_multi_string(
      'gin_bindings', [],
      'Gin bindings to override the values set in the config files '
      '(e.g. "DQNAgent.epsilon_train=0.1",'
      '      "create_environment.game_name="Pong"").')

  FLAGS = flags.FLAGS
  if policy_per_task:
      assert not large_network
      assert not multi_head

  runner = em_run_experiment.Runner(base_dir=get_run_path(_run),
                                    create_agent_fn=None,  # not using this for our purposes
                                    num_iterations=num_iterations,
                                    training_steps=training_steps,
                                    evaluation_steps=evaluation_steps,
                                    em_n_agents=em_n_agents,
                                    env_list=env_names,
                                    _run=_run,
                                    _log=_log,
                                    multi_head=multi_head,
                                    continue_fp=continue_fp,
                                    buff_size=buff_size,
                                    pretraining_its=pretraining_its,
                                    large_network=large_network,
                                    policy_per_task=policy_per_task,
                                    network_scaling_factor=network_scaling_factor
                                    )
  runner.run_experiment()


def main():
    file_observer = FileStorageObserver(os.path.join('results', 'sacred'))
    train_ex.observers.append(file_observer)
    train_ex.run_commandline()


if __name__ == '__main__':
    main()
  # flags.mark_flag_as_required('base_dir')
  # app.run(main)
