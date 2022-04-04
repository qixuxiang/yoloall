from __future__ import division
import pdb
import time

import torch
import torch.nn as nn
import math
from yolodet.utils.general import bbox_iou


def compute_loss_v4(self, predictions, targets):
    # pdb.set_trace()

    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if targets.is_cuda else torch.FloatTensor
    device = targets.device

    self.bce_loss = self.bce_loss.to(device)
    if self.box_loss_func is not None:
        self.box_loss_func = self.box_loss_func.to(device)

    # anchors
    anchors_org  = FloatTensor(self.anchors_org)
    scaled_anchors_org = FloatTensor(self.scaled_anchors_org)

    if len(targets) > 0:
        iou_anchor_gt = wh_iou(scaled_anchors_org, targets[:, 4:6]) # anchors num * targets num
        best_ious_all, best_n_all = iou_anchor_gt.max(0)
        # pdb.set_trace()

    # loss
    loss, obj_loss, noobj_loss, conf_loss, cls_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    
    if self.box_loss_func is None:
        iou_loss = torch.zeros(1, device=device)
    else:
        x_loss, y_loss, w_loss, h_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # yolo outputs
    logstrs = []

    for p, prediction in enumerate(predictions):

        # gt nums
        p_count = p_recall = p_recall75 = 0

        # cur anchor
        anchors_mask = self.anchors_masks[p]
        anchors = anchors_org[anchors_mask]
        l_n = len(anchors_mask)
        anchor_w = anchors[:, 0:1].view((1, l_n, 1, 1))
        anchor_h = anchors[:, 1:2].view((1, l_n, 1, 1))

        # cur feature size
        l_batch = prediction.size(0)
        l_h = prediction.size(2)
        l_w = prediction.size(3)

        pred_x = torch.sigmoid(prediction[..., 0])
        pred_y = torch.sigmoid(prediction[..., 1])
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_obj_sigmoid = torch.sigmoid(prediction[..., 4])
        pred_cls_sigmoid = torch.sigmoid(prediction[..., 5:])

        # compute new offsets
        grid_x = torch.arange(l_w).repeat(l_h, 1).view([1, 1, l_h, l_w]).type(FloatTensor)
        grid_y = torch.arange(l_h).repeat(l_w, 1).t().view([1, 1, l_h, l_w]).type(FloatTensor)

        # box pre
        boxe_pred = FloatTensor(prediction[..., :4].shape)

        # scale 
        boxe_pred[..., 0] = (pred_x + grid_x) / l_w
        boxe_pred[..., 1] = (pred_y + grid_y) / l_h
        boxe_pred[..., 2] = torch.exp(pred_w) * anchor_w / self.net_w
        boxe_pred[..., 3] = torch.exp(pred_h) * anchor_h / self.net_h

        # mask
        obj_mask = torch.zeros(size=(l_batch, l_n, l_h, l_w), requires_grad=False, device=device)
        noobj_mask = torch.ones(size=(l_batch, l_n, l_h, l_w), requires_grad=False, device=device)
        tcls = torch.zeros(size=(l_batch, l_n, l_h, l_w, self.num_classes), requires_grad=False, device=device)
        class_mask = torch.zeros(size=(l_batch, l_n, l_h, l_w), requires_grad=False, device=device)
            
        for l_b in range(l_batch):
            # btime = time.time()

            b_gt_index = targets[:, 0] == l_b

            b_targets = targets[b_gt_index]
            nt = b_targets.size(0)
            if nt == 0:
                continue

            ##########
            # best_match_iou > l.ignore_thresh and class > 0.25
            tbox = b_targets[:, 2:6]
            pbox = boxe_pred[l_b].view(-1, 4)

            b_iou_pre_gt = bbox_ious_target(pbox, tbox).view(l_n, l_h, l_w, nt)

            max_iou_pg = (b_iou_pre_gt.max(-1)[0]) > self.ignore_thresh

            class_id_match = (pred_cls_sigmoid[l_b] > 0.25).float().sum(axis=-1) >= 1
            
            mask = max_iou_pg & class_id_match

            # ignore some noobj_mask
            noobj_mask[l_b][mask] = 0

            ##########
            # loss with bset match anchor
            best_n_mask = [(ii in anchors_mask) for ii in best_n_all[b_gt_index]]
            targets_cur = b_targets[best_n_mask]
            best_gt_num = len(targets_cur)

            if best_gt_num > 0:
                a = best_n_all[b_gt_index][best_n_mask]
                for i in range(best_gt_num):
                    a[i] = anchors_mask.index(a[i])

                gt_range = torch.arange(nt)[best_n_mask]

                if not self.use_all_anchors:
                    b, c = targets_cur[:, :2].long().t()
                    gx = targets_cur[:, 2] * l_w
                    gy = targets_cur[:, 3] * l_h
                    gi = gx.long()
                    gj = gy.long()

                    # cal tcls
                    if self.label_smooth_eps:
                        tcls[b, a, gj, gi, :] = self.y_false
                        tcls[b, a, gj, gi, c] = self.y_true
                    else:
                        tcls[b, a, gj, gi, c] = 1.0

                    # cal obj_mask , noobj_mask, and class_mask
                    obj_mask[b, a, gj, gi] = 1.0
                    noobj_mask[b, a, gj, gi] = 0.0
                    # class_mask[b, a, gj, gi] = (pred_cls_sigmoid[b, a, gj, gi].argmax(-1) == c).float()

                    # cal bbox Loss
                    if self.box_loss_func is None:
                        tbox = targets_cur[:, 2:6]

                        if self.box_loss_type == 'giou':
                            iou_tmp = bbox_iou(boxe_pred[b, a, gj, gi].view(-1, 4).t(), tbox, x1y1x2y2=False, GIoU=True)

                        elif self.box_loss_type == 'diou':
                            iou_tmp = bbox_iou(boxe_pred[b, a, gj, gi].view(-1, 4).t(), tbox, x1y1x2y2=False, DIoU=True)

                        elif self.box_loss_type == 'ciou':
                            iou_tmp = bbox_iou(boxe_pred[b, a, gj, gi].view(-1, 4).t(), tbox, x1y1x2y2=False, CIoU=True)

                        
                        iou_loss += (1.0 - iou_tmp).sum()

                    else:
                        gw = targets_cur[:, 4] * self.net_w
                        gh = targets_cur[:, 5] * self.net_h

                        tx = gx - gx.floor()
                        ty = gy - gy.floor()
                        tw = torch.log(gw / anchors[a][:, 0])
                        th = torch.log(gh / anchors[a][:, 1])

                        tcoord_weight = (2 - targets_cur[:, 4]*targets_cur[:, 5])

                        # iou_normalizer * coord_scale
                        x_loss += (tcoord_weight * self.box_loss_func(pred_x[b, a, gj, gi], tx)).sum() 
                        y_loss += (tcoord_weight * self.box_loss_func(pred_y[b, a, gj, gi], ty)).sum()

                        w_loss_tmp = tcoord_weight * self.box_loss_func(pred_w[b, a, gj, gi], tw)
                        h_loss_tmp = tcoord_weight * self.box_loss_func(pred_h[b, a, gj, gi], th)

                        w_loss += w_loss_tmp.sum()
                        h_loss += h_loss_tmp.sum()

                    p_count += best_gt_num
                    recall_iou = b_iou_pre_gt[a, gj, gi, gt_range]
                    p_recall += (recall_iou > 0.5).float().sum()
                    p_recall75 += (recall_iou > 0.75).float().sum()

            ##########
            # loss with iou_anchor_gt > l.iou_thresh, and add best anchor
            if self.use_all_anchors:
                j = iou_anchor_gt[anchors_mask][:, b_gt_index] > self.iou_thresh
                # add best anchor
                if best_gt_num > 0:
                    j[a, gt_range] = True
                j = j.view(-1)

                a2 = torch.arange(l_n).view(-1, 1).repeat(1, nt).view(-1)
                targets_cur2 = b_targets.repeat(l_n, 1)
                gt_range2 = torch.arange(nt).view(-1, 1).repeat(l_n, 1).view(-1)
                # print(nt, a2.shape, targets_cur2.shape, gt_range2.shape, j.shape)

                # reject
                targets_cur2, a2, gt_range2 = targets_cur2[j], a2[j], gt_range2[j]

                match_gt_num = len(targets_cur2)

                if match_gt_num > 0:
                    b2, c2 = targets_cur2[:, :2].long().t()

                    gx2 = targets_cur2[:, 2] * l_w
                    gy2 = targets_cur2[:, 3] * l_h
                    gi2 = gx2.long()
                    gj2 = gy2.long()

                    # cal tcls
                    if self.label_smooth_eps:
                        tcls[b2, a2, gj2, gi2, :] = self.y_false
                        tcls[b2, a2, gj2, gi2, c2] = self.y_true
                    else:
                        tcls[b2, a2, gj2, gi2, c2] = 1.0
                                
                    # cal obj_mask , noobj_mask, and class_mask
                    obj_mask[b2, a2, gj2, gi2] = 1.0
                    noobj_mask[b2, a2, gj2, gi2] = 0.0
                    # class_mask[b2, a2, gj2, gi2] = (pred_cls_sigmoid[b2, a2, gj2, gi2].argmax(-1) == c2).float()

                    # cal coord Loss
                    if self.box_loss_func is None:
                        tbox2 = targets_cur2[:, 2:6]

                        if self.box_loss_type == 'giou':
                            iou_tmp2 = bbox_iou(boxe_pred[b2, a2, gj2, gi2].view(-1, 4).t(), tbox2, x1y1x2y2=False, GIoU=True)

                        elif self.box_loss_type == 'diou':
                            iou_tmp2 = bbox_iou(boxe_pred[b2, a2, gj2, gi2].view(-1, 4).t(), tbox2, x1y1x2y2=False, DIoU=True)

                        elif self.box_loss_type == 'ciou':
                            iou_tmp2 = bbox_iou(boxe_pred[b2, a2, gj2, gi2].view(-1, 4).t(), tbox2, x1y1x2y2=False, CIoU=True)

                        iou_loss += (1.0 - iou_tmp2).sum()

                    else:
                        gw2 = targets_cur2[:, 4] * self.net_w
                        gh2 = targets_cur2[:, 5] * self.net_h

                        tx2 = gx2 - gx2.floor()
                        ty2 = gy2 - gy2.floor()
                        tw2 = torch.log(gw2 / anchors[a2][:, 0])
                        th2 = torch.log(gh2 / anchors[a2][:, 1])

                        tcoord_weight2 = (2 - targets_cur2[:, 4]*targets_cur2[:, 5])

                        # iou_normalizer * coord_scale
                        x_loss += (tcoord_weight2 * self.box_loss_func(pred_x[b2, a2, gj2, gi2], tx2)).sum() 
                        y_loss += (tcoord_weight2 * self.box_loss_func(pred_y[b2, a2, gj2, gi2], ty2)).sum()
                        w_loss_tmp = tcoord_weight2 * self.box_loss_func(pred_w[b2, a2, gj2, gi2], tw2)
                        h_loss_tmp = tcoord_weight2 * self.box_loss_func(pred_h[b2, a2, gj2, gi2], th2)

                        w_loss += w_loss_tmp.sum()
                        h_loss += h_loss_tmp.sum()

                    p_count += match_gt_num
                    recall_iou = b_iou_pre_gt[a2, gj2, gi2, gt_range2]
                    p_recall += (recall_iou > 0.5).float().sum()
                    p_recall75 += (recall_iou > 0.75).float().sum()

        # cal conf_loss and cls_loss
        bce_conf = self.bce_loss(prediction[..., 4], obj_mask.float())
        obj_loss += (obj_mask * bce_conf).sum()
        noobj_loss += (noobj_mask * bce_conf).sum()
        

        if len(targets) == 0 or p_count == 0:
            logstr = f"s: {self.layer_stride[p]}"
        else:

            cls_loss += (obj_mask * (self.bce_loss(prediction[..., 5:], tcls)).sum(dim=-1)).sum()

            # logging 
            conf_obj = pred_obj_sigmoid[obj_mask.bool()].mean()
            conf_noobj = pred_obj_sigmoid[noobj_mask.bool()].mean()
            class_prob = pred_cls_sigmoid[tcls==self.y_true].mean()

            logstr = "s: {:2d}, class_prob: {:.3f}, recall: {:.3f}, recall75: {:.3f}, conf_obj: {:.4f}, conf_noobj: {:.4f}, count: {:d}" \
            .format(self.layer_stride[p], class_prob.data, (1.0 * p_recall / p_count).data, (1.0 * p_recall75 / p_count).data, conf_obj.data, conf_noobj.data, p_count)

        logstrs.append(logstr)


    if self.box_loss_func is None:
        iou_loss *= self.coord_scale * self.iou_normalizer
        conf_loss = self.object_scale * self.cls_normalizer * obj_loss + self.noobject_scale * self.cls_normalizer * noobj_loss
        cls_loss *= self.class_scale

        total_loss = iou_loss + conf_loss + cls_loss
        
        return total_loss, (torch.cat((iou_loss, conf_loss, cls_loss, total_loss)) / l_batch).detach()#, logstrs
    else:
        xy_loss = self.coord_scale * self.iou_normalizer * (x_loss + y_loss)
        wh_loss = self.coord_scale * self.iou_normalizer * (w_loss + h_loss)
        conf_loss = self.object_scale * self.cls_normalizer * obj_loss + self.noobject_scale * self.cls_normalizer * noobj_loss
        cls_loss *= self.class_scale

        total_loss = xy_loss + wh_loss + conf_loss + cls_loss
        return total_loss, (torch.cat((xy_loss+wh_loss, conf_loss, cls_loss, total_loss)) / l_batch).detach()#, logstrs
        

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        # loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.alpha = alpha
        self.gamma = gamma
        # self.reduction = reduction

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        pt = torch.exp(-loss)
        loss *= self.alpha * (1 - pt) ** self.gamma

        return loss


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ious_target(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
      boxes1 (torch.Tensor): List of bounding boxes
      boxes2 (torch.Tensor): List of bounding boxes

    Note:
      List format: [[xc, yc, w, h],...]
    """

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions