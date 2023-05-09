# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : densenet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""

from collections import OrderedDict
from eca import eca_block
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from SE_model import SELayer
from torch.hub import load_state_dict_from_url
from Nonlocal import _NonLocalBlockND
model_urls = {
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
}

#BN-ReLU-3×3 conv之前进行了BN-ReLU-1×1 conv.
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        :param num_input_features:  输入特征图的数量
        :param growth_rate:         在论文中为32，每个DenseLayer产生k个特征图，这里的k即growth_rate
        :param bn_size:             让1x1卷积产生4k个特征图，达到降维的作用
        :param drop_rate:           DropOut层的丢弃概率
        """
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        #self.add_module("selayer", SELayer(channel=num_input_features)),
        # 论文中Composite function定义为bn -> relu -> conv
        self.add_module("norm1", nn.BatchNorm1d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1", nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )

        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.add_module("Non", _NonLocalBlockND(growth_rate))
        self.add_module("SE", SELayer(channel=growth_rate))
        #self.add_module("eca",eca_block(growth_rate))
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x

        # 这一操作实现了Dense连接操作
        concated_features = torch.cat(prev_features, 1)
        # 降维
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        #bottleneck_output = self.Non(bottleneck_output)
        # 提取特征
        #bottleneck_output = self.eca(bottleneck_output)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        #new_features = self.eca(new_features)
        new_features = self.SE(new_features)
        #new_features = self.Non(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

#***************************************************************************************************************************
class _DenseLayer1(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        :param num_input_features:  输入特征图的数量
        :param growth_rate:         在论文中为32，每个DenseLayer产生k个特征图，这里的k即growth_rate
        :param bn_size:             让1x1卷积产生4k个特征图，达到降维的作用
        :param drop_rate:           DropOut层的丢弃概率
        """
        super(_DenseLayer1, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        #self.add_module("selayer", SELayer(channel=num_input_features)),
        # 论文中Composite function定义为bn -> relu -> conv
        self.add_module("norm1", nn.BatchNorm1d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1", nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=5, stride=1, padding=2, bias=False)
        )
        self.add_module("Non", _NonLocalBlockND(growth_rate))
        self.add_module("SE", SELayer(channel=growth_rate))
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x

        # 这一操作实现了Dense连接操作
        concated_features = torch.cat(prev_features, 1)

        # 降维
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        # 提取特征
        #bottleneck_output = self.eca(bottleneck_output)
        #bottleneck_output = self.Non(bottleneck_output)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        new_features = self.SE(new_features)
        #new_features = self.eca(new_features)
        #  new_features = self.Non(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
#**********************************************************************************************************************
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
#*************************************************************************************************************************
class _DenseBlock1(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock1, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer1(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        features1 = [x]
        for name, layer in self.items():
            new_features = layer(features1)
            features1.append(new_features)
        return torch.cat(features1, 1)
#****************************************************************************************************************
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))
class _Transition1(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition1, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))
#******************************************************************************************************************************
class DenseNet(nn.Module):
    def __init__(
            self,
            growth_rate=32,
            block_config=(1, 1, 1, 1),
            block_config1=(1, 1, 1, 1),
            num_init_features=64,
            num_init_features1=64,
            bn_size=4,
            drop_rate=0.0,
            num_classes=6
    ):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ('conv0', nn.Conv1d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("conv1", nn.Conv1d(num_init_features, num_init_features, kernel_size=3, stride=2, bias=False)),
                    ('norm0', nn.BatchNorm1d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        self.features1 = nn.Sequential(
            OrderedDict(
                [
                    ('conv0', nn.Conv1d(1, num_init_features1, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("conv1", nn.Conv1d(num_init_features, num_init_features, kernel_size=3, stride=2, bias=False)),
                    ('norm0', nn.BatchNorm1d(num_init_features1)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        # Each denseblock
        num_features = num_init_features
        num_features1 = num_init_features1
#***************************************************************************************************************************
        for i, num_layers in enumerate(block_config):
            # Add a SELayer
            # self.features.add_module("SELayer_%da" % (i + 1), SELayer(channel=num_features))
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )

                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

                #*********************************************************************************************************
        for i, num_layers in enumerate(block_config1):
            # Add a SELayer
            # self.features.add_module("SELayer_%da" % (i + 1), SELayer(channel=num_features))

            block1 = _DenseBlock1(
                num_layers=num_layers,
                num_input_features=num_features1,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )

            self.features1.add_module('denseblock%d' % (i + 1), block1)
            num_features1 = num_features1 + num_layers * growth_rate
            if i != len(block_config) - 1:

                trans1 = _Transition1(
                    num_input_features=num_features1,
                    num_output_features=num_features1 // 2
                )

                self.features1.add_module('transition%d' % (i + 1), trans1)
                num_features1 = num_features1 // 2
            #*********************************************************************************************************************
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.features1.add_module('norm6', nn.BatchNorm1d(num_features1))
        # Linear layer
        self.classifier = nn.Linear(num_features*2, num_classes)
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 提取特征、激活、池化、摊平、分类
        x1 = x
        x2 = x

        features = self.features(x1)
        features1 = self.features1(x2)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, output_size=1)


        out1 = F.relu(features1, inplace=True)
        out1 = F.adaptive_avg_pool1d(out1, output_size=1)


        out = torch.cat((out,out1), dim=1)
        out_features = torch.flatten(out, start_dim=1)
        out = self.classifier(out_features)
        #out1 = self.classifier(out1)
        #out = torch.cat((out,out1), dim=1)
        return out, out_features

# def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs):
#     model = DenseNet(growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch], progress)
#         model.load_state_dict(state_dict)
#     return model
#
# #预训练权重，其中第一个参数'densenet121'代表densenet的模型名称，32代表每一层添加32个特征图，(6, 12, 24, 16)表示4个denselayer重复的次数，64表示初始特征数
# def densenet121_d(pretrained=False, progress=True, **kwargs):
#     return _densenet(
#         arch='densenet121',
#         growth_rate=32,
#         #block_config=(6, 12, 24, 16),#121
#         block_config=(1, 1, 1, 1),  # 19
#         block_config1=(1, 1, 1, 1),  # 19
#
#         num_init_features=64,
#         pretrained=pretrained,
#         progress=progress,
#         **kwargs
#     )
#
#
# def densenet169(pretrained=False, progress=True, **kwargs):
#     return _densenet(
#         arch='densenet161',
#         growth_rate=32,
#         block_config=(6, 12, 32, 32),
#         num_init_features=64,
#         pretrained=pretrained,
#         progress=progress,
#         **kwargs
#     )
#
#
# def densenet201(pretrained=False, progress=True, **kwargs):
#     return _densenet(
#         arch='densenet201',
#         growth_rate=32,
#         block_config=(6, 12, 48, 32),
#         num_init_features=64,
#         pretrained=pretrained,
#         progress=progress,
#         **kwargs
#     )


if __name__ == '__main__':
    inputs = torch.randn(8, 3, 224, 224)
    model = densenet121_d(num_classes=6)
    out = model(inputs)
    print(out.shape)