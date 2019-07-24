#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.nn.modules.rnn import RNNCellBase


class AbstractRNNCell(RNNCellBase):
  def __init__(self, func, *args, **kwargs):
    super(AbstractRNNCell, self).__init__()
    self._weights = self.weights(*args, **kwargs)
    self._biases = self.biases(*args, **kwargs)

    [ self.register_parameter('w_'+str(i), x) for i, x in enumerate(self._weights) ]
    [ self.register_parameter('b_'+str(i), x) for i, x in enumerate(self._biases) ]

    self.F = func
    self.reset_parameters()

  def __init(self, params):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in params:
      weight.data.uniform_(-stdv, stdv)

  def reset_parameters(self, fn=self.__init):
    fn(self.parameters())

  def weights(self, *args, **kwargs):
    raise NotImplementedError

  def biases(self, *args, **kwargs):
    raise NotImplementedError

  def forward(self, input, hx, weights, biases):
    return self.F(input, hx, weights, biases)

