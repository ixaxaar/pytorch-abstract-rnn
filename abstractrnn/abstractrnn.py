#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import copy

from .cell import AbstractRNNCell
from .stack import StackedAbstractRNN
from .unroll import Unroll


class AbstractRNN(nn.Module):
  def __init__(self, cell, input_size, hidden_size,
               dropout=0, bidirectional=False, train=True,
               num_layers=1, batch_first=False, **kwargs):
    super(AbstractRNN, self).__init__()

    self.cells = (cell, reverse(cell)) if bidirectional else (cell, )

    w = [ c._weights for c in cells ]
    b = [ c._biases for c in cells ]
    # weights and biases for every layer
    self.weights = [ copy.deepcopy(w) for _ in range(num_layers) ]
    self.biases = [ copy.deepcopy(b) for _ in range(num_layers) ]

    self.F = Unroll(StackedAbstractRNN(self.cells, num_layers, dropout, train), batch_first)

  def forward(self, input, hidden):
    if not batch_first:
      input = input.transpose(0, 1)

    nexth, output = self.F(input, hidden, self.weights, self.biases)

    if not batch_first:
      output = output.transpose(0, 1)

    return output, nexth