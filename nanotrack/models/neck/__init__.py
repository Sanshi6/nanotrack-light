# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotrack.models.neck.neck import AdjustLayer, AdjustAllLayer
from nanotrack.super_models.neck.neck import multi_neck


NECKS = {
    'AdjustLayer': AdjustLayer,
    'AdjustAllLayer': AdjustAllLayer,
    'multi_neck': multi_neck,
}


def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
