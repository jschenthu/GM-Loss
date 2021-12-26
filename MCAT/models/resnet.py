"""
ResNet.
Take from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
"""
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import common.torch
from .classifier import Classifier
from .resnet_block import ResNetBlock


class LGM_Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LGM_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

    def forward(self, input):
        xy = torch.matmul(input, self.mu.permute(1,0))
        xx = torch.sum(input * input, dim=1, keepdim=True)
        yy = torch.sum(self.mu * self.mu, dim=1, keepdim=True).permute(1,0)
        out = -0.5 * (xx - 2.0 * xy + yy)
        return out



class ResNet(Classifier):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], normalization=True, channels=64, gm=1, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        """

        super(ResNet, self).__init__(N_class, resolution, **kwargs)

        self.blocks = blocks
        """ ([int]) Blocks. """

        self.channels = channels
        """ (int) Channels. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.inplace = False
        """ (bool) Inplace. """

        conv1 = torch.nn.Conv2d(self.resolution[0], self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        self.append_layer('conv1', conv1)

        if self.normalization:
            norm1 = torch.nn.BatchNorm2d(self.channels)
            torch.nn.init.constant_(norm1.weight, 1)
            torch.nn.init.constant_(norm1.bias, 0)
            self.append_layer('norm1', norm1)

        relu = torch.nn.ReLU(inplace=self.inplace)
        self.append_layer('relu1', relu)

        downsampled = 1
        for i in range(len(self.blocks)):
            in_planes = (2 ** max(0, i - 1)) * self.channels
            out_planes = (2 ** i) * self.channels
            layers = self.blocks[i]
            stride = 2 if i > 0 else 1

            downsample = None
            if stride != 1 or in_planes != out_planes:
                conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

                if self.normalization:
                    bn = torch.nn.BatchNorm2d(out_planes)
                    torch.nn.init.constant_(bn.weight, 1)
                    torch.nn.init.constant_(bn.bias, 0)
                    downsample = torch.nn.Sequential(*[conv, bn])
                else:
                    downsample = torch.nn.Sequential(*[conv])

            sequence = []
            sequence.append(ResNetBlock(in_planes, out_planes, stride=stride, downsample=downsample, normalization=self.normalization))
            for _ in range(1, layers):
                sequence.append(ResNetBlock(out_planes, out_planes, stride=1, downsample=None, normalization=self.normalization))

            self.append_layer('block%d' % i, torch.nn.Sequential(*sequence))
            downsampled *= stride

        representation = out_planes
        pool = torch.nn.AvgPool2d((self.resolution[1] // downsampled, self.resolution[2] // downsampled), stride=1)
        self.append_layer('avgpool', pool)

        view = common.torch.View(-1, representation)
        self.append_layer('view', view)

        gain = torch.nn.init.calculate_gain('relu')
        if gm == 0:
            logits = torch.nn.Linear(representation, self._N_output)
            torch.nn.init.kaiming_normal_(logits.weight, gain)
            torch.nn.init.constant_(logits.bias, 0)
            self.append_layer('logits', logits)
        else:
            logits = LGM_Linear(representation, self._N_output)
            torch.nn.init.kaiming_normal_(logits.mu, gain)
            self.append_layer('logits', logits)


