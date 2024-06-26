from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# # NanoTrackV1
from nanotrack.models.head.ban_v1 import UPChannelBAN, DepthwiseBAN, new_head

# # NanoTrackV2
# from nanotrack.models.head.ban_v2 import UPChannelBAN, DepthwiseBAN

# NanoTrackV3
# from nanotrack.models.head.ban_v3 import UPChannelBAN, DepthwiseBAN

# RepHead, LightTrack
# from nanotrack.models.head.RHead import UPChannelBAN, DepthwiseBAN

from nanotrack.super_models.head.super_connect import head_supernet
from nanotrack.sub_models.head.sub_connect import sub_connect

BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'new_head': new_head,
        'head_supernet': head_supernet,
        'sub_connect': sub_connect,
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)

