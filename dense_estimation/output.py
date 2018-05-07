import torch
import torch.nn as nn
import numpy as np


class GaussianScaleMixtureOutput(nn.Module):
    def __init__(self, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.num_channels = 2 * num_gaussians + 1
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        assert x.size(1) == self.num_channels

        weights, variances, mean  = torch.split(x, self.num_gaussians, dim=1)
        variances = torch.exp(variances)
        weights = self.softmax(weights)
        return mean, variances, weights


class PowerExponentialOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 2
        self.relu = nn.ReLU()

    def forward(self, x):
        assert x.size(1) == 2

        mean, variance = torch.split(x, 1, dim=1)
        #mean = self.relu(mean)

        variance = torch.exp(variance)
        return mean, variance
