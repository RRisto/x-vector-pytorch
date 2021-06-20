import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, device='cpu'):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.rand(in_features, out_features, device=device)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # self.weight.data.uniform_(-1, 1).renorm_(2, -1, 1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos().clone().detach().requires_grad_(True)
        k = (self.m * theta / np.pi).floor()
        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * cos_m_theta - 2 * k

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, test=False, mode='mean'):
        super(AngleLoss, self).__init__()
        self.it = 0
        self.LambdaMin = 0.1
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.test = test
        self.mode = mode

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = index.clone().detach().bool()

        output = cos_theta * 1.0  # size=(B,Classnum)
        if self.test:
            output[index] = phi_theta[index]
        else:
            self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.2 * self.it))
            output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
            output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)  # .clamp(-2e9, 2e9)

        loss = -logpt
        if self.mode == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss
