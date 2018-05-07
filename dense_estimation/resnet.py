import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from torchvision.models.resnet import Bottleneck, model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class _ProjectUp(nn.Module):
    def __init__(self, num_input_features):
        super().__init__()
        num_output_features = int(num_input_features / 2)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_output_features, num_output_features,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_output_features)
        self.conv_proj = nn.Conv2d(num_input_features, num_output_features,
                                   kernel_size=5, padding=2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self._unpool_masks = dict()

    def _get_unpool_mask(self, x):
        size = x.size()
        n, c, h, w = size
        key = tuple(size)
        if not key in self._unpool_masks:
            unpool_mask = [[0.0 if x % 2 == 0 and y % 2 == 0 else 1.0 for x in range(w)]
                           for y in range(h)]
            unpool_mask = np.tile(unpool_mask, (n, c, 1, 1))
            unpool_mask = torch.Tensor(unpool_mask).byte()
            if x.is_cuda:
                unpool_mask = unpool_mask.cuda()
            self._unpool_masks[key] = unpool_mask
        return self._unpool_masks[key]

    def forward(self, x, skip):
        x = self.upsample(x)
        unpool_mask = self._get_unpool_mask(x)
        x = x.masked_fill(Variable(unpool_mask), 0.0)

        proj = self.conv_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += proj
        out = self.relu(out)

        return out


class _FPNUp(nn.Module):
    def __init__(self, num_input_features, skip_channel_adjust=True):
        super().__init__()
        self.conv_channel_adjust = nn.Conv2d(num_input_features, 256,
                                             kernel_size=1)
        self.conv_fusion = nn.Conv2d(256, 256,
                                     kernel_size=3, padding=1)

    def forward(self, x, skip):
        upsample = nn.UpsamplingBilinear2d(size=skip.size()[2:])
        x = upsample(x)
        skip = self.conv_channel_adjust(skip)
        fused = self.conv_fusion(x + skip)
        return fused


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, output=None, fpn=False,
                 dropout_active=True):
        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout_active = dropout_active

        # -- Adapted for fully convolutional operation
        Up = _FPNUp if fpn else _ProjectUp
        self.fpn = fpn
        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        if fpn:
            self.conv_init_fpn = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=True)
            out_channels = 256
        else:
            self.conv_up = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=True)
            self.bn_up = nn.BatchNorm2d(1024)
            out_channels = 64

        self.output = output
        if output is not None:
            num_classes = output.num_channels

        self.conv_out = nn.Conv2d(out_channels, num_classes, kernel_size=3, stride=1, padding=0,
                                  bias=True)
        self._upsamplings = dict()

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
        self.conv_out.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_upsample(self, x):
        h, w = x.size()[2:]
        key = (h, w)
        if not key in self._upsamplings:
            upsample = nn.UpsamplingBilinear2d((h, w))
            self._upsamplings[key] = upsample
        return self._upsamplings[key]

    def forward(self, x):
        upsample = self._get_upsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        l0 = self.relu(x)
        x = self.maxpool(l0)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = F.dropout(l4, training=self.dropout_active)

        # -- Adapted for fully convolutional operation

        if self.fpn:
            x = self.conv_init_fpn(l4)
        else:
            x = self.conv_up(l4)
            x = self.bn_up(x)


        x = self.up1(x, l3)
        x = self.up2(x, l2)
        x = self.up3(x, l1)
        if not self.fpn:
            x = self.up4(x, l0)

        x = self.conv_out(x)
        x = upsample(x)

        if self.output is not None:
            x = self.output(x)
        else:
            x = self.relu(x)

        return x


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state = model.state_dict()
        state.update(model_zoo.load_url(model_urls['resnet50']))
        del state['fc.weight']
        del state['fc.bias']
        model.load_state_dict(state)
    return model
