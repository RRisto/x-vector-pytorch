#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""

import torch.nn as nn

from x_vectors.models.LDE import LDE
from x_vectors.models.angleloss import AngleLinear
from x_vectors.models.tdnn import TDNN
import torch


class X_vector(nn.Module):
    def __init__(self, input_dim=40, num_classes=8, pooling='stat', use_angular=True):
        super(X_vector, self).__init__()

        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3, dropout_p=0.5)
        self.pooling = pooling
        if pooling == 'lde':
            self.lde = LDE(2, 512, distance_type='sqr', network_type='lde')
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.use_angluar=use_angular
        if self.use_angluar:
            self.fc2 = AngleLinear(num_classes, num_classes)

    def forward(self, inputs):

        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        if self.pooling == 'stat':
            mean = torch.mean(tdnn5_out, 1)
            std = torch.var(tdnn5_out, 1)
            stat_pooling = torch.cat((mean, std), 1)
        elif self.pooling == 'lde':
            stat_pooling = self.lde(tdnn5_out)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        if self.use_angluar:
            predictions=self.fc2(predictions)
        return predictions, x_vec
