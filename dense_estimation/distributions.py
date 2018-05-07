## Experimental research code - not part of the depth prediction re-implementation

import torch
from torch.autograd import Variable
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import math

from pytorch.contrib.distributions import BaseDistribution, MultivariateDiag


class GaussianScaleMixture(BaseDistribution):
    def __init__(self, mean, variances, weights):
        self._mean = mean
        self._variances = variances
        self._weights = weights

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return torch.sum(self._variances * self._weights, dim=1)

    @property
    def averages(self):
        variances_lst = torch.split(self._variances, 1, dim=1)
        weights_lst = torch.split(self._weights, 1, dim=1)
        avgs = [torch.squeeze(torch.mean(w)) for w in variances_lst]
        avgs += [torch.squeeze(torch.mean(v)) for v in weights_lst]
        return torch.cat(avgs)

    def log_prob(self, x):
        return math.log(1. / (math.sqrt(2 * math.pi))) - self.log_loss(x)

    def log_loss(self, x):
        variances_lst = torch.split(self._variances, 1, dim=1)
        weights_lst = torch.split(self._weights, 1, dim=1)

        out = Variable(torch.zeros(*self._mean.size()))
        if x.is_cuda:
            out = out.cuda()

        exponent_lst = []
        for var, weight in zip(variances_lst, weights_lst):
            exponent = - ((x - self._mean) ** 2) / (2 * var)
            exponent_lst.append(exponent)
        # Assuming # channels = 1, we can concat and take max along dim 1
        a = torch.max(torch.cat(exponent_lst, dim=1), dim=1)[0]

        for var, weight, exponent in zip(variances_lst, weights_lst, exponent_lst):
            exp = torch.exp(exponent - a)
            out += weight * exp / torch.sqrt(var)
        return - a - torch.log(out)

    @staticmethod
    def plot(averages, label):
        averages = averages.cpu().data.numpy()
        num_gaussians = int(len(averages) / 2)
        means = np.zeros((num_gaussians))
        stdevs = np.sqrt(averages[:num_gaussians])
        weights = averages[num_gaussians:]
        x = np.arange(-5., 5., 0.01)

        pdfs = [p * ss.norm.pdf(x, mu, sd) for mu, sd, p in zip(means, stdevs, weights)]

        density = np.sum(np.array(pdfs), axis=0)
        plt.plot(x, density, label=label)


class PowerExponential(BaseDistribution):
    def __init__(self, mean, variance, k=0.5, eps=1e-6):
        self._mean = mean
        self._variance = variance
        self._dim = int(mean.size()[1])
        self._k = k
        self._eps = eps

    @property
    def mean(self):
        return self._mean

    @property
    def averages(self):
        avgs = [torch.mean(self._variance)]
        return torch.cat(avgs)

    @property
    def variance(self):
        return self._variance

    def log_loss(self, x):
        vr = self._variance
        u = x - self._mean

        t2 = torch.sum(torch.log(vr), dim=1)
        t3 = (torch.sum((u ** 2) / vr, dim=1) + self._eps) ** self._k
        return t2 + t3

    @staticmethod
    def plot(averages, label):
        pass
