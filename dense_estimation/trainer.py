import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .losses import DistributionLogLoss


class Trainer():
    def __init__(self, model, training_criterion, testing_multi_criterion,
                 training_loader, testing_loader, display_interval=100, cuda=True,
                 save_dir=None, logger=None, logging_interval=100, lr=0.001,
                 distribution=None):
        if cuda:
            model = model.cuda()
            training_criterion = training_criterion.cuda()
            testing_multi_criterion = testing_multi_criterion.cuda()

        assert distribution is None or isinstance(training_criterion,
                                                  DistributionLogLoss)

        self.cuda = cuda
        self.model = model
        self.training_criterion = training_criterion
        self.testing_multi_criterion = testing_multi_criterion
        self.testing_loader = testing_loader
        self.training_loader = training_loader
        self.display_interval = display_interval
        self.save_dir = save_dir
        self.logger = logger
        self.logging_interval = logging_interval
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                   weight_decay=1e-4)
        #self.optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
        self.distribution = distribution
        self.lr = lr

    def train(self, epochs, restore_path=None, starting_epoch=0):
        if restore_path:
            self.restore(restore_path)

        for epoch in range(epochs):
            self.train_epoch(epoch, starting_epoch)
            self.test(epoch, starting_epoch)
            if self.save_dir:
                self.save(epoch + starting_epoch)

    def train_epoch(self, epoch, starting_epoch):
        log_epoch = epoch + starting_epoch

        self.model.train()
        num_iterations = len(self.training_loader)

        epoch_loss = 0

        for i, batch in enumerate(self.training_loader):
            input = Variable(batch[0])
            target = Variable(batch[1])
            if self.cuda:
                input = input.cuda()
                target = target.cuda()

            output = self.model(input)

            if self.distribution is not None:
                pred_channels = self.distribution(*output).mean.size(1)
            else:
                pred_channels = output.size(1)

            if pred_channels != target.size(1):
                target, mask = torch.split(target, pred_channels, dim=1)
                loss = self.training_criterion(output, target, mask=mask)
            else:
                loss = self.training_criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.data[0]

            if i % self.display_interval == 0 or i == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}"
                      .format(log_epoch,
                              i + starting_epoch,
                              len(self.training_loader),
                              loss.data[0]))

            if self.logger is not None and (i % self.logging_interval == 0 or i == 0):
                criterion_name = self.training_criterion.__class__.__name__
                self.logger.log_training_loss(log_epoch * num_iterations + i,
                                              loss.data[0],
                                              self.lr)

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}"
              .format(log_epoch, epoch_loss / num_iterations))


    def test(self, epoch, starting_epoch):
        log_epoch = epoch + starting_epoch

        loss_names = [m.__class__.__name__
                      for m in self.testing_multi_criterion.scalar_losses]
        losses = np.zeros(len(loss_names))
        self.model.eval()

        for i, batch in enumerate(self.testing_loader):
            input = Variable(batch[0], volatile=True)
            target = Variable(batch[1], volatile=True)
            if self.cuda:
                input = input.cuda()
                target = target.cuda()

            # Predictions are computed at half resolution
            upsample = nn.UpsamplingBilinear2d(size=target.size()[2:])
            output = self.model(input)

            if self.distribution is not None:
                cpu_outputs = [x.cpu().data for x in output]
                output = self.distribution(*output).mean
            else:
                cpu_outputs = [output.cpu().data]

            output = upsample(output)

            losses += self.testing_multi_criterion(output, target).cpu().data.numpy()

            if self.logger is not None and i == 0:
                self.logger.log_testing_images(log_epoch, input.cpu().data,
                                               cpu_outputs,
                                               target.cpu().data)

        losses /= len(self.testing_loader)
        loss_strings = ["{}: {:.4f}".format(n, l)
                        for n, l in zip(loss_names, losses)]

        if self.logger is not None:
            self.logger.log_testing_losses(log_epoch, losses)

        print("===> Avg. Testing {}"
              .format(', '.join(loss_strings)))

    def save(self, epoch):
        assert self.save_dir is not None
        checkpoint_name = "model_{}.pth".format(epoch)
        save_path = os.path.join(self.save_dir, checkpoint_name)
        torch.save(self.model.state_dict(), save_path)
        print("Checkpoint saved to {}".format(save_path))

    def restore(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("Restored checkpoint from {}".format(path))
