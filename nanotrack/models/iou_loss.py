import torch
from torch import nn

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        EPS = 1e-7
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        w_intersect = torch.relu(w_intersect)

        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        g_w_intersect = torch.relu(g_w_intersect)

        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        h_intersect = torch.relu(h_intersect)

        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        g_h_intersect = torch.relu(g_h_intersect)

        ac_uion = g_w_intersect * g_h_intersect + EPS
        area_intersect = w_intersect * h_intersect + EPS


        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + EPS) / (area_union + EPS)
        ious = torch.clamp(ious, min=EPS, max=1.0)  # 限制 IoU 值范围

        gious = ious - (ac_uion - area_union) / ac_uion
        gious = torch.clamp(gious, min=EPS, max=1.0)  # 限制 IoU 值范围

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


linear_iou = IOULoss(loc_loss_type='linear_iou')
