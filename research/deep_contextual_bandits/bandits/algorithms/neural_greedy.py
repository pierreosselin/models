# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class NeuralGreedy(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, hparams, optimizer='RMS'):

    self.name = name
    self.eps = 0.9
    self.decay = 0.99 # computed for 10,000 steps
    self.hparams = hparams


    # Regression and NN Update Frequency
    self.update_freq_lr = hparams.training_freq
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.optimizer_n = optimizer

    self.num_epochs = hparams.training_epochs
    self.data_h = ContextualDataset(hparams.context_dim,
                                    hparams.num_actions,
                                    intercept=False)
    self.bnn = NeuralBanditModel(optimizer, hparams, '{}-greedy'.format(name))

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    #if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      #return self.t % self.hparams.num_actions ## No need with greedy

    if np.random.random() < self.eps:
        return np.random.choice(range(self.hparams.num_actions))
    else:
        with self.bnn.graph.as_default():
          c = context.reshape((1, self.hparams.context_dim))
          output = self.bnn.sess.run(self.bnn.y_pred, feed_dict={self.bnn.x: c})
          return np.argmax(output)

  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1
    self.eps *= self.decay
    self.data_h.add(context, action, reward)
    c = context.reshape((1, self.hparams.context_dim))

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:

      if self.hparams.reset_lr:
        self.bnn.assign_lr()
      self.bnn.train(self.data_h, self.num_epochs)

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
