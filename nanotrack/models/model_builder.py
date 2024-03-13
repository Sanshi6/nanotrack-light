# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

# from nanotrack.core.config import cfg
from nanotrack.models.loss import select_cross_entropy_loss, select_iou_loss
from nanotrack.models.backbone import get_backbone
from nanotrack.models.head import get_ban_head
from nanotrack.models.neck import get_neck

# test
from nanotrack.models.backbone.RLightTrack1 import mobileone
from nanotrack.models.head.ban_v1 import multi_head


class ModelBuilder(nn.Module):
    def __init__(self, cfg):
        super(ModelBuilder, self).__init__()
        self.cfg = cfg

        # build backbone
        self.backbone = mobileone(inference_mode=False, variant='s0')
        # self.backbone = get_backbone(cfg.BACKBONE.TYPE,
        #                              **cfg.BACKBONE.KWARGS)
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        # build ban head
        if cfg.BAN.BAN:
            self.ban_head = get_ban_head(cfg.BAN.TYPE,
                                         **cfg.BAN.KWARGS)
            self.multi_head = multi_head()

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

    def log_softmax(self, cls):

        if self.cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()

            cls = F.log_softmax(cls, dim=3)

        return cls

        #  forward

    def forward(self, data):
        """ only used in training
        """
        # train mode
        if len(data) >= 4:
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            label_loc = data['label_loc'].cuda()

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)

            # ban model
            cls, reg = self.multi_head(xf, zf)
            cls, loc = self.ban_head(cls, reg)

            # cls, loc = self.ban_head(zf, xf)

            # cls loss with cross entropy loss , TODO
            cls = self.log_softmax(cls)

            cls_loss = select_cross_entropy_loss(cls, label_cls)

            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc, label_cls)
            outputs = {}

            outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            return outputs

            # test speed
        else:

            xf = self.backbone(data)
            cls, loc = self.ban_head(self.zf, xf)

            return {
                'cls': cls,
                'loc': loc,
            }
