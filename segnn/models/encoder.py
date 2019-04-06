from functools import partial

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchvision.models import resnet

# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(resnet.ResNet):
    def __init__(self, *args, dilation=True):
        super().__init__(*args)
        if dilation:
            self.layer3.apply(partial(self.set_dilation, dilation=2))
            self.layer4.apply(partial(self.set_dilation, dilation=4))

    @staticmethod
    def set_dilation(m, dilation):
        classname = m.__class__.__name__
        if 'Conv' in classname:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation//2, dilation//2)
                    m.padding = (dilation//2, dilation//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)

    def forward(self, x):
        conv_out = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        return conv_out


def resnet18(pretrained=False, **kwargs):
    model = ResNet(resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
