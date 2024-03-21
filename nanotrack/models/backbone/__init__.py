# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nanotrack.models.backbone.mobile_v3 import mobilenetv3_small, mobilenetv3_small_v3
from nanotrack.models.backbone.alexnet import AlexNet
from nanotrack.super_models.backbone.supernet import SuperNet
from nanotrack.sub_models.backbone.subnet import SubNet
from nanotrack.models.backbone.RLightTrack1 import mobileone


BACKBONES = {
    'mobilenetv3_small': mobilenetv3_small,
    'mobilenetv3_small_v3': mobilenetv3_small_v3,
    'AlexNet': AlexNet,
    'supernet': SuperNet,
    'subnet': SubNet,
    'mobileone': mobileone,
}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
