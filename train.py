import argparse
import os
import shutil
import json

import torch
import torch.nn as nn

from dense_estimation.densenet import DenseNet
from dense_estimation.resnet import resnet50
from dense_estimation.output import GaussianScaleMixtureOutput, PowerExponentialOutput
from dense_estimation.losses import (BerHuLoss, RMSLoss, RelLoss, TestingLosses, HuberLoss,
                                     Log10Loss, DistributionLogLoss)
from dense_estimation.distributions import GaussianScaleMixture, PowerExponential
from dense_estimation.datasets.nyu_depth_v2 import NYU_Depth_V2
from dense_estimation.data import get_testing_loader, get_training_loader
from dense_estimation.trainer import Trainer
from dense_estimation.logger import TensorBoardLogger, BasicVisualizer, DistributionVisualizer
from dense_estimation.app.experiment import get_experiment


parser = argparse.ArgumentParser(description='Monocular Depth Prediction + Uncertainty')
parser.add_argument('--batch', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
parser.add_argument('--debug', action='store_true', help='load random fake data to run quickly')
parser.add_argument('--overfit', action='store_true', help='train on testing set to check model')
parser.add_argument('--ex', type=str, default='default',
                    help='name of experiment (continue training if existing)')
parser.add_argument('--ow', action='store_true', help='overwrite existing experiment')
parser.add_argument('--gpu', type=str, default='0', help='cuda device to use if using --cuda')
parser.add_argument('--dist', type=str, default='', help='gsm or exp')
parser.add_argument('--num_gaussians', type=int, default=2, help='number of gaussians for gsm distribution')
parser.add_argument('--limit', action='store_true', help='limit number of training examples per epoch')
parser.add_argument('--fpn', action='store_true', help='use resnet upsampling style from FPN paper')
parser.add_argument('--densenet', action='store_true', help='use DenseNet instead of ResNet')
#parser.add_argument('--ckpt', type=str, help='checkpoint epoch to run from if --ex is given and name exists')
opt = parser.parse_args()
print(opt)




cuda = opt.cuda
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

out_size = (208, 256)
transf_size = (out_size[1], out_size[0])

dist_map = {
    'gsm': (GaussianScaleMixture, lambda: GaussianScaleMixtureOutput(opt.num_gaussians)),
    'exp': (PowerExponential, lambda: PowerExponentialOutput())
}

if opt.dist != '':
    distribution, output_unit = dist_map[opt.dist]
    model = resnet50(output=output_unit(), fpn=opt.fpn)
    visualizer = DistributionVisualizer(distribution)
    training_criterion = DistributionLogLoss(distribution)
else:
    distribution = None
    if opt.densenet:
        model = DenseNet()
    else:
        model = resnet50(fpn=opt.fpn)
    visualizer = BasicVisualizer()
    training_criterion = BerHuLoss()


dset_root = './datasets'
raw_root = '/home/smeister/datasets'


log_dir, save_dir, restore_path, starting_epoch = get_experiment(opt.ex,
                                                                 opt.ow)
with open('./log/{}/opts.txt'.format(opt.ex), 'w') as f:
    json.dump({k:getattr(opt,k) for k in opt.__dict__}, f)
    
print("Training from epoch {}".format(starting_epoch))

testing_multi_criterion = TestingLosses([RMSLoss(), RelLoss(), Log10Loss()])


if opt.overfit:
    training_loader = get_testing_loader(NYU_Depth_V2, raw_root, opt.batch, transf_size,
                                         opt.threads,debug=opt.debug, shuffle=True,
                                         training=True)
else:
    training_loader = get_training_loader(NYU_Depth_V2, raw_root, opt.batch, transf_size,
                                          opt.threads, limit=30 if opt.limit else None)

testing_loader = get_testing_loader(NYU_Depth_V2, raw_root, opt.batch, transf_size,
                                    opt.threads, debug=opt.debug)

logger = TensorBoardLogger(log_dir, visualizer,
                           testing_loss_names=['RMS', 'Rel', 'Log10'],
                           max_testing_images=9,
                           run_options=str(opt),
                           starting_epoch=starting_epoch)

trainer = Trainer(model, training_criterion, testing_multi_criterion,
                  training_loader, testing_loader, save_dir=save_dir,
                  cuda=opt.cuda, display_interval=10, logger=logger,
                  logging_interval=10, lr=opt.lr, distribution=distribution)
trainer.train(opt.epochs, restore_path=restore_path, starting_epoch=starting_epoch)

# TODO config file for directories & environment setup
