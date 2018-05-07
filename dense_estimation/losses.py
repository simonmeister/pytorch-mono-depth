import numpy as np
import torch
import torch.nn as nn
from math import log


def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).data[0]
    else:
        count = np.prod(input.size(), dtype=np.float32).item()
    return input, count


class BerHuLoss(nn.Module):
    def forward(self, input, target, mask=None):
        x = input - target
        abs_x = torch.abs(x)
        c = torch.max(abs_x).data[0] / 5
        leq = (abs_x <= c).float()
        l2_losses = (x ** 2 + c ** 2) / (2 * c)
        losses = leq * abs_x + (1 - leq) * l2_losses
        losses, count = _mask_input(losses, mask)
        return torch.sum(losses) / count


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(size_average=False)

    def forward(self, input, target, mask=None):
        if mask is not None:
            loss = self.loss(input * mask, target * mask)
            count = torch.sum(mask).data[0]
            return loss / count

        count = np.prod(input.size(), dtype=np.float32).item()
        return self.loss(input, target) / count


class DistributionLogLoss(nn.Module):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def forward(self, input, target, mask=None):
        d = self.distribution(*input)
        loss = d.log_loss(target)
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class RMSLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.pow(input - target, 2)
        loss, count = _mask_input(loss, mask)
        return torch.sqrt(torch.sum(loss) / count)


class RelLoss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.abs(input - target) / target
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class Log10Loss(nn.Module):
    def forward(self, input, target, mask=None):
        loss = torch.abs((torch.log(target) - torch.log(input)) / log(10))
        loss, count = _mask_input(loss, mask)
        return torch.sum(loss) / count


class TestingLosses(nn.Module):
    def __init__(self, scalar_losses):
        super().__init__()
        self.scalar_losses = nn.ModuleList(scalar_losses)

    def forward(self, input, target):
        scalars = [m(input, target) for m in self.scalar_losses]
        return torch.cat(scalars)
