#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def Unroll(rnn, batch_first=False, bidirectional=False):

  def forward(input, hidden, weights, biases):
    if not batch_first:
      input = input.transpose(0, 1)

    outputs = []
    steps = input.size(1)
    for step in range(steps):
      hidden, o = rnn(input[:, step, :], hidden, weights, biases)
      outputs.append(o)
    outputs = T.stack(outputs, dim=1)

    if not batch_first:
      outputs = outputs.transpose(0, 1)

    return hidden, outputs

