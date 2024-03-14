# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch.nn as nn

from nanotrack.core.xcorr import xcorr_pixelwise


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        if self.in_channels != self.out_channels:
            x = self.downsample(x)

        # todo: ?
        if x.size(3) < 16:
            l = 2
            r = l + 4
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                out.append(adj_layer(features[i]))
            return out


class cls_tools(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cls_tools, self).__init__()
        cls_tool = []

        # 1 layer
        cls_tool.append(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False))
        cls_tool.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        cls_tool.append(nn.BatchNorm2d(out_channels))
        cls_tool.append(nn.ReLU6(inplace=True))

        # 2 layer
        cls_tool.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False))
        cls_tool.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        cls_tool.append(nn.BatchNorm2d(out_channels))
        cls_tool.append(nn.ReLU6(inplace=True))

        # 3 layer
        cls_tool.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False))
        cls_tool.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        cls_tool.append(nn.BatchNorm2d(out_channels))
        cls_tool.append(nn.ReLU6(inplace=True))

        # 4 layer
        cls_tool.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False))
        cls_tool.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        # cls_tool.append(nn.BatchNorm2d(128))
        # cls_tool.append(nn.ReLU6(inplace=True))

        self.add_module('cls_tool', nn.Sequential(*cls_tool))

        for modules in self.cls_tool:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.cls_tool(x)
        return x


class reg_tools(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(reg_tools, self).__init__()
        reg_tool = []

        # 1 layer
        reg_tool.append(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False))
        reg_tool.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        reg_tool.append(nn.BatchNorm2d(out_channels))
        reg_tool.append(nn.ReLU6(inplace=True))

        # 2 layer
        reg_tool.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False))
        reg_tool.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        reg_tool.append(nn.BatchNorm2d(out_channels))
        reg_tool.append(nn.ReLU6(inplace=True))

        # 3 layer
        reg_tool.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False))
        reg_tool.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        reg_tool.append(nn.BatchNorm2d(out_channels))
        reg_tool.append(nn.ReLU6(inplace=True))

        # 4 layer
        reg_tool.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False))
        reg_tool.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        # reg_tool.append(nn.BatchNorm2d(128))
        # reg_tool.append(nn.ReLU6(inplace=True))

        self.add_module('reg_tool', nn.Sequential(*reg_tool))

        for modules in self.reg_tool:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.reg_tool(x)
        return x


class multi_neck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(multi_neck, self).__init__()
        self.reg_tools = reg_tools(in_channels, out_channels)
        self.cls_tools = cls_tools(in_channels, out_channels)
        self.loc_adjust = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x, z):
        x_reg = self.reg_tools(x)
        z_reg = self.reg_tools(z)

        x_cls = self.cls_tools(x)
        z_cls = self.cls_tools(z)
        cls = xcorr_pixelwise(x_cls, z_cls)
        loc = xcorr_pixelwise(x_reg, z_reg)
        # cls = xcorr_fast(x_cls, z_cls)
        # loc = self.loc_adjust(xcorr_fast(x_reg, z_reg))
        return cls, loc

