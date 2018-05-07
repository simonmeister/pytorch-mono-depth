import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import math


class _BN_ReLU_Conv(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, dropout_p,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.add_module('bn', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=kernel_size, stride=stride, padding=padding))
        if dropout_p > 0:
            self.add_module('dropout', nn.Dropout(dropout_p))


class _TransitionDown(nn.Sequential):
    def __init__(self, num_features, dropout_p):
        super().__init__()
        self.add_module('bn_relu_conv', _BN_ReLU_Conv(num_features, num_features, dropout_p,
                        kernel_size=1, padding=0))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2))


class _TransitionUp(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(num_features, num_features,
                                         kernel_size=3, stride=2, padding=1)

    def forward(self, x, skip):
        self.deconv.padding = (
            ((x.size(2) - 1) * self.deconv.stride[0] - skip.size(2)
             + self.deconv.kernel_size[0] + 1) // 2,
            ((x.size(3) - 1) * self.deconv.stride[1] - skip.size(3)
             + self.deconv.kernel_size[1] + 1) // 2)
        up = self.deconv(x, output_size=skip.size())
        return torch.cat([up, skip], 1)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, dropout_p):
        super().__init__()
        self.add_module('bn_relu_conv', _BN_ReLU_Conv(num_input_features,
                        num_output_features, dropout_p))

    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, dropout_p):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                dropout_p)
            self.add_module('denselayer{}'.format(i + 1), layer)


class SoftmaxLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size(3)

        x = x.transpose(1, 3)
        x = x.view(-1, c)
        x = F.softmax(x)
        x = t.view(b, w, h, c)
        return x.transpose(1, 3)


class DenseNet(nn.Module):
    """Fully Convolutional DenseNet described in <https://arxiv.org/pdf/1611.09326v1.pdf>
    Args:
        num_input_features (int) - number of module input features
        num_output_features (int) - number of module output features
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        num_transitions (int) - number of transition down = number of transition up
        block_config (int or list of size 2 * num_transitions + 1) - how many layers in each block
        num_init_features (int) - number of filters to learn in the first convolution layer
        dropout_p (float) - dropout rate after each dense layer
    """
    def __init__(self, num_input_features=3, num_output_features=1,
                 growth_rate=16, num_transitions=5,
                 block_config=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
                 num_init_features=48, dropout_p=0.2):

        block_config_size = 2 * num_transitions + 1
        if isinstance(block_config, list):
            assert len(block_config) == block_config_size
        else:
            block_config = [block_config] * block_config_size

        super().__init__()

        self.block_config = block_config
        self.growth_rate = growth_rate
        self.num_transitions = num_transitions

        self.downsampling_blocks = nn.ModuleList()
        self.downsampling_transitions = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.upsampling_transitions = nn.ModuleList()
        self.relu = torch.nn.ReLU()

        num_features = num_init_features
        num_features_skip = []

        self.init_conv = nn.Conv2d(num_input_features, num_features, kernel_size=3, padding=1)

        for i in range(num_transitions):
            num_layers = block_config[i]
            dense = _DenseBlock(num_layers, num_features, growth_rate, dropout_p)
            num_features += growth_rate * num_layers
            transition = _TransitionDown(num_features, dropout_p)
            self.downsampling_blocks.append(dense)
            self.downsampling_transitions.append(transition)
            num_features_skip.append(num_features)

        num_layers = block_config[num_transitions]
        self.bottleneck_block = _DenseBlock(num_layers, num_features, growth_rate, dropout_p)
        num_features_skip = num_features_skip[::-1]

        for i in range(num_transitions):
            num_features_last_block = growth_rate * block_config[num_transitions + i]
            transition = _TransitionUp(num_features_last_block)
            num_features = num_features_last_block + num_features_skip[i]

            num_layers = block_config[num_transitions + 1 + i]
            dense = _DenseBlock(num_layers, num_features, growth_rate, dropout_p)
            self.upsampling_blocks.append(dense)
            self.upsampling_transitions.append(transition)
            num_features += growth_rate * num_layers

        self.output_conv = nn.Conv2d(num_features, num_output_features, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # We have to initialize the output weights to keep predicted variance low, as it will be
        # exponentiated
        # self.output_conv.weight.data.normal_(0, 0.01)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_uniform(m.weight.data)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        features = self.init_conv(x)
        skip_features = []

        for dense, transition in zip(self.downsampling_blocks, self.downsampling_transitions):
            block_features = dense(features)
            skip_features.append(block_features)
            features = transition(block_features)

        features = self.bottleneck_block(features)
        for i, block in enumerate(zip(self.upsampling_transitions, self.upsampling_blocks)):
            skip = skip_features.pop()
            num_features_last_block = self.growth_rate * self.block_config[self.num_transitions + i]
            transition, dense = block
            up_features = transition(features[:, -num_features_last_block:], skip)
            features = dense(up_features)

        return self.output_conv(features)
