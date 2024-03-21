# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotrack.models.loss import select_cross_entropy_loss, select_iou_loss
from nanotrack.models.backbone import get_backbone
from nanotrack.models.head import get_ban_head
from nanotrack.models.neck import get_neck


class ModelBuilderTRT(nn.Module):
    def __init__(self, cfg):
        super(ModelBuilderTRT, self).__init__()
        self.cfg = cfg

        # build backbone
        self.backbone = None

        # build ban head
        self.ban_head = None

    def tensorrt_init(self, trt_net):
        """
        TensorRT init
        """
        self.backbone, self.ban_head = trt_net

    def template(self, z):
        zf = self.backbone(z)
        self.zf = zf

    def track(self, x):

        xf = self.backbone(x)

        cls, loc = self.ban_head(self.zf, xf)

        return {
            'cls': cls,
            'loc': loc,
        }
