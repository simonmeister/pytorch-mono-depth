import argparse
import os
import shutil
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dense_estimation.resnet import resnet50
from dense_estimation.output import GaussianScaleMixtureOutput, PowerExponentialOutput
from dense_estimation.losses import (BerHuLoss, RMSLoss, RelLoss, TestingLosses, HuberLoss,
                                     Log10Loss, DistributionLogLoss)
from dense_estimation.distributions import GaussianScaleMixture, PowerExponential
from dense_estimation.datasets.nyu_depth_v2 import NYU_Depth_V2
from dense_estimation.data import get_testing_loader
from dense_estimation.app.experiment import get_experiment
from dense_estimation.app.gui import display
from dense_estimation.logger import DistributionVisualizer, BasicVisualizer, visuals_to_numpy

parser = argparse.ArgumentParser(description='testing script')
parser.add_argument('--no_cuda', action='store_true', help='use cpu')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
parser.add_argument('--ex', type=str, default='default',
                    help='comma separated names of experiments to compare; use name:epoch to specify epoch to load')
parser.add_argument('--gpu', type=str, default='0', help='cuda device to use if using --cuda')
parser.add_argument('--max', type=int, default=20, help='max number of examples to visualize')
parser.add_argument('--samples', type=int, default=1, help='number of monte carlo dropout samples (sampling enabled if > 1)')
opt = parser.parse_args()


cuda = not opt.no_cuda
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

out_size = (208, 256)
transf_size = (out_size[1], out_size[0])

raw_root = '/home/smeister/datasets'
testing_loader = get_testing_loader(NYU_Depth_V2, raw_root, 1, transf_size,
                                    opt.threads, debug=False)

class BasicDist():
    def __init__(self, mean, var):
        self.mean = mean
        self.variance = var


def _test(ex, epoch):
    results = []
    with open('./log/{}/opts.txt'.format(ex), 'r') as f:
        ex_opt = json.load(f)

    dist_map = {
        'gsm': (GaussianScaleMixture, lambda: GaussianScaleMixtureOutput(ex_opt['num_gaussians'])),
        'exp': (PowerExponential, lambda: PowerExponentialOutput()),
    }

    if ex_opt['dist'] != '':
        distribution, output_unit = dist_map[ex_opt['dist'] ]
        model = resnet50(output=output_unit(), fpn=ex_opt['fpn'], dropout_active=False)
        visualizer = DistributionVisualizer(distribution)
        dropout_active = False
    else:
        output_unit = None
        dropout_active = opt.samples > 1
        model = resnet50(fpn=ex_opt['fpn'], dropout_active=dropout_active)
        if dropout_active:
            distribution = BasicDist
            visualizer = DistributionVisualizer(BasicDist)
        else:
            distribution = None
            visualizer = BasicVisualizer()

    losses_clses = [RMSLoss(), RelLoss(), Log10Loss()]
    #if distribution is not None:
    #    losses_clses += [DistributionLogLoss(distribution)]

    testing_multi_criterion = TestingLosses(losses_clses)

    if cuda:
        model = model.cuda()
        testing_multi_criterion = testing_multi_criterion.cuda()

    _, _, restore_path, _ = get_experiment(ex, False, epoch=epoch)
    state_dict = torch.load(restore_path)
    model.load_state_dict(state_dict)

    loss_names = [m.__class__.__name__
                  for m in testing_multi_criterion.scalar_losses]
    losses = np.zeros(len(loss_names))
    model.eval()
    prob = 0

    num = opt.max if opt.max != -1 else len(testing_loader)

    averages = []

    for i, batch in enumerate(testing_loader):
        print(i)
        if i > num: break

        input = torch.autograd.Variable(batch[0], volatile=True)
        target = torch.autograd.Variable(batch[1], volatile=True)
        if cuda:
            input = input.cuda()
            target = target.cuda()

        # Predictions are computed at half resolution
        upsample = nn.UpsamplingBilinear2d(size=target.size()[2:])

        samples = []
        if dropout_active:
            for _ in range(opt.samples):
                sample = model(input)
                samples.append(sample)
            stacked = torch.cat(samples, dim=1)
            mean = torch.mean(stacked, dim=1)
            var = torch.var(stacked, dim=1)
            output = [mean, var]
        else:
            output = model(input)

        if isinstance(output, list):
            output = [upsample(x) for x in output]
            cpu_outputs = [x.cpu().data for x in output]
            d = distribution(*output)
            output = d.mean
            if output_unit:
                prob += torch.mean(d.prob(target[:, 0:1, :, :])).cpu().data[0]
                averages.append(d.averages)
        else:
            output = upsample(output)
            cpu_outputs = [output.cpu().data]

        losses += testing_multi_criterion(output, target).cpu().data.numpy()

        viz_pt = visualizer(input.cpu().data, cpu_outputs, target.cpu().data)
        images = visuals_to_numpy(viz_pt)
        results.append(images)

    losses /= len(testing_loader)
    loss_strings = ["{}: {:.4f}".format(n, l)
                    for n, l in zip(loss_names, losses)]

    print("===> [{}] Testing {}"
          .format(ex, ', '.join(loss_strings)))

    if output_unit:
        averages = torch.squeeze(torch.mean(torch.stack(averages, dim=1), dim=1))
        prob /= len(testing_loader)
        print("===> [{}] Avg. Likelihood {}".format(ex, prob))
        print("===> [{}] Dist. Averages {}"
              .format(ex, averages.cpu().data.numpy()))
        distribution.plot(averages, label=ex)

    return results, visualizer.names


if __name__ == '__main__':
    results = []
    plt.figure()
    for spec in opt.ex.split(','):
        splits = spec.split(':')
        ex = splits[0]
        epoch = int(splits[1]) if len(splits) == 2 else None

        result, image_names = _test(ex, epoch)
        results.append(result)
    plt.legend()
    plt.show()
    display(results, image_names)
