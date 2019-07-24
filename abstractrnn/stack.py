#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .cell import AbstractRNNCell


def StackedAbstractRNN(cell, num_layers, dropout=0, train=True):

  num_directions = len(cells)
  # total_layers = num_layers * num_directions

  def forward(input, hidden, weights, biases):
    assert(len(weights) == num_layers)
    next_hidden = []

    for i in range(num_layers):
      # all_output = []
      # for j, cell in enumerate(cells):
      hy, output = cell(input, hidden[i], weights[i], biases[i])
      next_hidden.append(hy)
      all_output.append(output)

      # TODO: verify
      input = torch.cat(all_output, input.dim() - 1)

      if dropout != 0 and i < num_layers - 1:
        input = F.dropout(input, p=dropout, training=train, inplace=False)

    return next_hidden, input

  return forward

