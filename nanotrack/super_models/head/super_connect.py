import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from nanotrack.core.xcorr import xcorr_pixelwise
from nanotrack.super_models.head.connect import *


class SeparableConv2d_BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_BNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU6()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.ReLU(self.BN(x))
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


'''2020.09.06 head supernet with mobile settings'''


class tower_supernet_singlechannel(nn.Module):
    """
    tower's supernet
    """

    def __init__(self, inchannels=256, outchannels=256, towernum=8,
                 base_op=SeparableConv2d_BNReLU, kernel_list=[3, 5, 7, 0]):
        super(tower_supernet_singlechannel, self).__init__()
        if 0 in kernel_list:
            assert (kernel_list[-1] == 0)
        self.kernel_list = kernel_list
        self.num_choice = len(self.kernel_list)

        self.tower = nn.ModuleList()

        # tower
        for i in range(towernum):
            '''the first layer, we don't use identity'''
            if i == 0:
                op_list = nn.ModuleList()
                if self.num_choice == 1:
                    kernel_size = self.kernel_list[-1]
                    padding = (kernel_size - 1) // 2
                    op_list.append(base_op(inchannels, outchannels, kernel_size=kernel_size, stride=1, padding=padding))
                else:
                    for choice_idx in range(self.num_choice - 1):
                        kernel_size = self.kernel_list[choice_idx]
                        padding = (kernel_size - 1) // 2
                        op_list.append(base_op(inchannels, outchannels, kernel_size=kernel_size,
                                               stride=1, padding=padding))
                self.tower.append(op_list)

            else:
                op_list = nn.ModuleList()
                for choice_idx in range(self.num_choice):
                    kernel_size = self.kernel_list[choice_idx]
                    if kernel_size != 0:
                        padding = (kernel_size - 1) // 2
                        op_list.append(base_op(outchannels, outchannels, kernel_size=kernel_size,
                                               stride=1, padding=padding))
                    else:
                        op_list.append(Identity())
                self.tower.append(op_list)

    def forward(self, x, arch_list):

        for archs, arch_id in zip(self.tower, arch_list):
            x = archs[arch_id](x)

        return x


class reg_pred_head(nn.Module):
    def __init__(self, inchannels=256, linear_reg=False, stride=16):
        super(reg_pred_head, self).__init__()
        self.linear_reg = linear_reg
        self.stride = stride
        # reg head
        self.bbox_pred = nn.Conv2d(inchannels, 4, kernel_size=3, stride=1, padding=1)
        # adjust scale
        if not self.linear_reg:
            self.adjust = nn.Parameter(0.1 * torch.ones(1))
            self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, x):
        if self.linear_reg:
            x = nn.functional.relu(self.bbox_pred(x)) * self.stride
        else:
            x = self.adjust * self.bbox_pred(x) + self.bias
            x = torch.exp(x)
        return x


class PixelwiseXCorr(nn.Module):
    def __init__(self, CA_channels=64):
        super(PixelwiseXCorr, self).__init__()

        self.CA_layer = CAModule(channels=CA_channels)

    def forward(self, kernel, search):
        feature = xcorr_pixelwise(search, kernel)  #

        corr = self.CA_layer(feature)

        return corr


class FeatureFusion(nn.Module):
    def __init__(self, in_channels_stride_8=256, in_channels_stride_16=64, out_channels=64):
        super(FeatureFusion, self).__init__()
        self.down_sample_stride_8 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1x1_stride_8 = nn.Conv2d(in_channels_stride_8, out_channels, kernel_size=1)
        # self.conv1x1_stride_16 = nn.Conv2d(in_channels_stride_16, out_channels, kernel_size=1)

    def forward(self, feature_stride_8, feature_stride_16):
        # 对stride=8的特征图进行下采样
        downsampled_feature_stride_8 = self.down_sample_stride_8(feature_stride_8)

        # 调整stride=8特征图的通道数
        adjusted_feature_stride_8 = self.conv1x1_stride_8(downsampled_feature_stride_8)

        # 调整stride=16特征图的通道数
        # adjusted_feature_stride_16 = self.conv1x1_stride_16(feature_stride_16)

        # 拼接调整后的特征图
        fused_feature = torch.cat((adjusted_feature_stride_8, feature_stride_16), dim=1)

        return fused_feature


class head_supernet(nn.Module):
    def __init__(self, in_channel=64, channel_list=[64, 96, 128], kernel_list=[3, 5, 7, 0], towernum=8, linear_reg=False):
        super(head_supernet, self).__init__()
        base_op = SeparableConv2d_BNReLU
        self.cand_path = None
        self._get_path_back()

        self.correlation1024 = PixelwiseXCorr(CA_channels=256)
        self.correlation64 = PixelwiseXCorr(CA_channels=64)
        self.fusion = FeatureFusion()

        self.num_cand = len(channel_list)
        self.cand_tower_cls = nn.ModuleList()
        self.cand_head_cls = nn.ModuleList()
        self.cand_tower_reg = nn.ModuleList()
        self.cand_head_reg = nn.ModuleList()
        self.tower_num = towernum

        # cls  TODO
        for outchannel in channel_list:
            self.cand_tower_cls.append(tower_supernet_singlechannel(inchannels=in_channel, outchannels=outchannel,
                                                                    towernum=towernum, base_op=base_op,
                                                                    kernel_list=kernel_list))
            self.cand_head_cls.append(cls_pred_head(inchannels=outchannel))

        # reg
        for outchannel in channel_list:
            self.cand_tower_reg.append(tower_supernet_singlechannel(inchannels=in_channel, outchannels=outchannel,
                                                                    towernum=towernum, base_op=base_op,
                                                                    kernel_list=kernel_list))
            self.cand_head_reg.append(reg_pred_head(inchannels=outchannel, linear_reg=linear_reg))

        for modules in [self.cand_tower_cls, self.cand_head_cls,
                        self.cand_tower_reg, self.cand_head_reg]:
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

    def _get_path_back(self):
        cls_total_path = []
        channel_choice_path = np.random.choice(3)
        kernel_choice_path = np.random.choice(4, 8).tolist()
        kernel_choice_path = [x for x in kernel_choice_path if x != 3] + [3] * kernel_choice_path.count(3)
        if kernel_choice_path[0] == 3: kernel_choice_path[0] = np.random.choice(3)
        cls_total_path.append(channel_choice_path)
        cls_total_path.append(kernel_choice_path)

        reg_total_path = []
        channel_choice_path = np.random.choice(3)
        kernel_choice_path = np.random.choice(4, 8).tolist()
        kernel_choice_path = [x for x in kernel_choice_path if x != 3] + [3] * kernel_choice_path.count(3)
        if kernel_choice_path[0] == 3: kernel_choice_path[0] = np.random.choice(3)
        reg_total_path.append(channel_choice_path)
        reg_total_path.append(kernel_choice_path)

        cand_h_dict = {'cls': cls_total_path, 'reg': reg_total_path}
        self.cand_path = cand_h_dict

    def forward(self, z_f, x_f):
        """cand_dict key: cls, reg
         [0/1/2, []]"""
        cand_dict = self.cand_path

        corr_pw1 = self.correlation1024(z_f[0], x_f[0])
        corr_pw2 = self.correlation64(z_f[1], x_f[1])
        # print(corr_pw1.shape, corr_pw2.shape)
        x_cls_reg = self.fusion(corr_pw1, corr_pw2)
        # print("out.shape: ", out.shape)

        # x_cls_reg = self.corr_pw(z_f, x_f)

        # cls
        cand_list_cls = cand_dict['cls']  # [0/1/2, []]
        cls_feat = self.cand_tower_cls[cand_list_cls[0]](x_cls_reg, cand_list_cls[1])
        logits = self.cand_head_cls[cand_list_cls[0]](cls_feat)

        # reg
        cand_list_reg = cand_dict['reg']  # [0/1/2, []]
        bbox_reg = self.cand_tower_cls[cand_list_reg[0]](x_cls_reg, cand_list_reg[1])
        bbox_reg = self.cand_head_reg[cand_list_reg[0]](bbox_reg)
        bbox_reg = torch.exp(bbox_reg)

        return logits, bbox_reg


class cls_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(cls_pred_head, self).__init__()
        self.cls_pred = nn.Conv2d(inchannels, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """mode should be in ['all', 'cls', 'reg']"""
        x = 0.1 * self.cls_pred(x)
        return x


if __name__ == '__main__':
    head = head_supernet().cuda()

    x = torch.randn(1, 512, 16, 16).cuda()
    z = torch.randn(1, 512, 8, 8).cuda()

    for i in range(1000):
        out = head(z, x)
        head._get_path_back()
        print(out[0].shape, out[1].shape)
