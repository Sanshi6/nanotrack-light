import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nanotrack.core.xcorr import xcorr_pixelwise
from nanotrack.super_models.head.connect import PWCA


class SeparableConv2d_BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_BNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.ReLU(self.BN(x))
        return x


class MC_BN(nn.Module):
    """2020.10.14 Batch Normalization with Multiple input Channels"""

    def __init__(self, inp_c=(40, 80, 96)):
        super(MC_BN, self).__init__()
        self.BN_z = nn.ModuleList()  # BN for the template branch
        self.BN_x = nn.ModuleList()  # BN for the search branch
        for idx, channel in enumerate(inp_c):
            self.BN_z.append(nn.BatchNorm2d(channel))
            self.BN_x.append(nn.BatchNorm2d(channel))

    def forward(self, kernel, search, index=None):
        if index is None:
            index = 0
        return self.BN_z[index](kernel), self.BN_x[index](search)


'''2020.10.09 Simplify prvious model'''


class Point_Neck_Mobile_simple(nn.Module):
    def __init__(self, inchannels=1280, num_kernel=16, cat=False, BN_choice='before', matrix=True):
        super(Point_Neck_Mobile_simple, self).__init__()
        self.BN_choice = BN_choice
        if self.BN_choice == 'before':
            '''template and search use separate BN'''
            self.BN_adj_z = nn.BatchNorm2d(inchannels)
            self.BN_adj_x = nn.BatchNorm2d(inchannels)
        '''Point-wise Correlation'''
        self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)

    def forward(self, kernel, search):
        """input: features of the template and the search region
           output: correlation features of cls and reg"""
        oup = {}
        if self.BN_choice == 'before':
            kernel, search = self.BN_adj_z(kernel), self.BN_adj_x(search)
        corr_feat = self.pw_corr([kernel], [search])
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


'''2020.10.15 DP version'''


class Point_Neck_Mobile_simple_DP(nn.Module):
    def __init__(self, num_kernel_list=(256, 64), cat=False, matrix=True, adjust=True, adj_channel=128):
        super(Point_Neck_Mobile_simple_DP, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = torch.nn.ModuleList()
        self.adj_layer = torch.nn.ModuleList()
        for num_kernel in num_kernel_list:
            self.pw_corr.append(PWCA(num_kernel, cat=cat, CA=True, matrix=matrix))
            self.adj_layer.append(nn.Conv2d(num_kernel, adj_channel, 1))

    def forward(self, kernel, search, stride_idx=None):
        """stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16"""
        if stride_idx is None:
            stride_idx = -1
        oup = {}
        corr_feat = self.pw_corr[stride_idx]([kernel], [search])
        if self.adjust:
            corr_feat = self.adj_layer[stride_idx](corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class PixelwiseXCorr(nn.Module):
    def __init__(self):
        super(PixelwiseXCorr, self).__init__()

        self.CA_layer = CAModule(channels=64)

    def forward(self, kernel, search):
        feature = xcorr_pixelwise(search, kernel)  #

        corr = self.CA_layer(feature)

        return corr


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


'''2020.09.06 head supernet with mobile settings'''


class tower_subnet_singlechannel(nn.Module):
    """
    tower's supernet
    """

    def __init__(self, inchannels=256, outchannels=256, towernum=8,
                 base_op=SeparableConv2d_BNReLU, kernel_list=[3, 5, 7, 0], path=None):
        super(tower_subnet_singlechannel, self).__init__()
        if 0 in kernel_list:
            assert (kernel_list[-1] == 0)
        self.kernel_list = kernel_list
        self.num_choice = len(self.kernel_list)
        self.tower = nn.ModuleList()

        # tower
        for i in range(towernum):
            kernel_size = self.kernel_list[path[i]]
            if kernel_size == 0:
                continue
            padding = (kernel_size - 1) // 2
            self.tower.append(base_op(inchannels, outchannels, kernel_size=kernel_size,
                                      stride=1, padding=padding))
            inchannels = outchannels
        self.tower = nn.Sequential(*self.tower)

    def forward(self, x):
        x = self.tower(x)
        return x


class reg_pred_head(nn.Module):
    def __init__(self, inchannels=256, linear_reg=False, stride=16):
        super(reg_pred_head, self).__init__()
        self.stride = stride
        # reg head
        self.bbox_pred = nn.Conv2d(inchannels, 4, kernel_size=1, stride=1, padding=0)
        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, x):
        x = self.adjust * self.bbox_pred(x) + self.bias
        x = torch.exp(x)
        return x


class cls_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(cls_pred_head, self).__init__()
        self.cls_pred = nn.Conv2d(inchannels, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """mode should be in ['all', 'cls', 'reg']"""
        x = self.cls_pred(x)
        return x


class sub_connect(nn.Module):
    def __init__(self, channel_list=[112, 256, 512], kernel_list=[3, 5, 7, 0], inchannels=64, towernum=8,
                 linear_reg=False, base_op_name='SeparableConv2d_BNReLU'):
        super(sub_connect, self).__init__()

        self.cand_path = {'cls': [1, [0, 1, 0, 1, 2, 3, 3, 3]], 'reg': [1, [1, 2, 0, 0, 3, 3, 3, 3]]}

        if base_op_name == 'SeparableConv2d_BNReLU':
            base_op = SeparableConv2d_BNReLU
        else:
            raise ValueError('Unsupported OP')

        self.num_cand = len(channel_list)
        self.cand_tower_cls = nn.ModuleList()
        self.cand_head_cls = nn.ModuleList()
        self.cand_tower_reg = nn.ModuleList()
        self.cand_head_reg = nn.ModuleList()
        self.tower_num = towernum

        cls_outchannel = channel_list[self.cand_path['cls'][0]]
        reg_outchannel = channel_list[self.cand_path['reg'][0]]

        self.corr_pw = PixelwiseXCorr()

        # cls  TODO
        self.cand_tower_cls.append(tower_subnet_singlechannel(inchannels=inchannels, outchannels=cls_outchannel,
                                                              towernum=towernum, base_op=base_op,
                                                              kernel_list=kernel_list,
                                                              path=self.cand_path['cls'][1]))
        self.cand_head_cls.append(cls_pred_head(inchannels=cls_outchannel))
        # reg

        self.cand_tower_reg.append(tower_subnet_singlechannel(inchannels=inchannels, outchannels=reg_outchannel,
                                                              towernum=towernum, base_op=base_op,
                                                              kernel_list=kernel_list,
                                                              path=self.cand_path['reg'][1]))
        self.cand_head_reg.append(reg_pred_head(inchannels=reg_outchannel, linear_reg=linear_reg))

        self.cand_tower_cls = nn.Sequential(*self.cand_tower_cls)
        self.cand_head_cls = nn.Sequential(*self.cand_head_cls)
        self.cand_tower_reg = nn.Sequential(*self.cand_tower_reg)
        self.cand_head_reg = nn.Sequential(*self.cand_head_reg)

    def forward(self, z_f, x_f):
        x_cls_reg = self.corr_pw(z_f, x_f)
        # cls
        cls_feat = self.cand_tower_cls(x_cls_reg)
        logits = self.cand_head_cls(cls_feat)

        # reg
        reg_feat = self.cand_tower_cls(x_cls_reg)
        bbox_reg = self.cand_head_reg(reg_feat)
        return logits, bbox_reg


if __name__ == '__main__':
    net = sub_connect()
    print(net)
