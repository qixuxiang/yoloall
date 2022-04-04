# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

# from ..builder import BBOX_ASSIGNERS
from yolodet.models.two_stage_utils.iou2d_calculator import bbox_overlaps
from yolodet.models.two_stage_utils.assigners import AssignResult, BaseAssigner

# @BBOX_ASSIGNERS.register_module()
class SimOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """

    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(self,
               pred_scores,
               priors,
               decoded_bboxes,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Assign gt to priors using SimOTA. It will switch to CPU mode when
        GPU is out of memory.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            assign_result (obj:`AssignResult`): The assigned result.
        """
        try:
            assign_result = self._assign(pred_scores, priors, decoded_bboxes,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore, eps)
            return assign_result
        except RuntimeError:
            origin_device = pred_scores.device
            warnings.warn('OOM RuntimeError is raised due to the huge memory '
                          'cost during label assignment. CPU mode is applied '
                          'in this batch. If you want to avoid this issue, '
                          'try to reduce the batch size or image size.')
            torch.cuda.empty_cache()

            pred_scores = pred_scores.cpu()
            priors = priors.cpu()
            decoded_bboxes = decoded_bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu().float()
            gt_labels = gt_labels.cpu()

            assign_result = self._assign(pred_scores, priors, decoded_bboxes,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore, eps)
            assign_result.gt_inds = assign_result.gt_inds.to(origin_device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(
                origin_device)
            assign_result.labels = assign_result.labels.to(origin_device)

            return assign_result

    def _assign(self,
                pred_scores,    #分类和框置信度的乘积
                priors,         #anchor的(中心点xy, 宽度wh)
                decoded_bboxes, #anchor和预测偏置框(左上点，右下点)
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None,
                eps=1e-7):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info( #gt框的真实中心点落在
            priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                          -1,
                                                          dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps) # iou的loss

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1) #对候选区域计算cost（损失）
        cls_cost = F.binary_cross_entropy_with_logits( # Lcls  binary_cross_entropy #binary_cross_entropy_with_logits
            valid_pred_scores.to(torch.float).sqrt_(), gt_onehot_label,
            reduction='none').sum(-1)

        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight + #λ*Lreg,实际代码中把λ设置为了3
            (~is_in_boxes_and_center) * INF) #把不在考虑范围内的anchor置为很大的数值

        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, offset_priors, gt_bboxes):#gt_bboxes ->为（左上点和右下点）
        #offset_priors (anchor的左上点和右下点) priors (anchor的中心点和wh)
        priors = torch.stack([(offset_priors[:, 0] + offset_priors[:, 2]) * 0.5, 
                              (offset_priors[:, 1] + offset_priors[:, 3]) * 0.5,
                              (offset_priors[:, 2] - offset_priors[:, 0]),
                              (offset_priors[:, 3] - offset_priors[:, 1])], dim=-1)
        #1、初筛：anchor的中心点落在gt内的anchor框
        num_gt = gt_bboxes.size(0)
        #priors （x_c, y_c, w,h） 以下为中心点和wh
        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        # repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        # repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]#预设anchor框中心点和真实框左上点和右下点的差值  这步操作为找到落在目标框内的点（即找预设框的中心点落在真实框内部）
        l_ = repeated_x - gt_bboxes[:, 0] 
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y
        # 上述中l_、t_、r_、b_均大于0表示中心点落在目标框内部，那么下面就开始找满足这样条件的框
        deltas = torch.stack([l_, t_, r_, b_], dim=1) #取每个l_, t_, r_, b_按照列进行cat在一起，shape为预设框priors的数量：8400，（l_, t_, r_, b_）4，gt框坐标长度-> 得到的为 预设框中心点xy分别相对于真实框的左上点右下点的距离
        is_in_gts = deltas.min(dim=1).values > 0 #找距离gt目标框的中心点最近的点 必须取min, min使这个点趋近于真实的中心点， 而且value>0, 同时这步会导致gt框周围的点也可以满足该条件
        is_in_gts_all = is_in_gts.sum(dim=1) > 0 #在行上取sum 只要存在一个为True 就说明改点为目标框周围(附近的点)


        #2、细筛在符合的目标框内左上点和右下点距离最小
        #prior_bboxes = ffset_priors[is_in_gts_all]
        min_dis = is_in_gts_all.new_full((is_in_gts_all.size(0), ),
                                    False,
                                    dtype=torch.bool)
        repeated_bbox_x1 = offset_priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_bbox_y1 = offset_priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_bbox_x2 = offset_priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_bbox_y2 = offset_priors[:, 3].unsqueeze(1).repeat(1, num_gt)
        x1 = torch.abs(gt_bboxes[:,0] - repeated_bbox_x1)
        y1 = torch.abs(gt_bboxes[:,1] - repeated_bbox_y1)
        x2 = torch.abs(gt_bboxes[:,2] - repeated_bbox_x2)
        y2 = torch.abs(gt_bboxes[:,3] - repeated_bbox_y2)
        
        for res in range(num_gt):
            diff = torch.cat([x1[:,res].unsqueeze(1), y1[:,res].unsqueeze(1), x2[:,res].unsqueeze(1), y2[:,res].unsqueeze(1)], dim=-1)
            min_dis[torch.topk(diff.sum(1),3, largest=False)[1]] = True


        # # is prior centers in gt centers 扩大中心点落下的范围
        # gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        # gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        # ct_box_l = gt_cxs - self.center_radius * repeated_stride_x #扩大左上点和右下点的网格范围 （问题:边缘目标网格容易画到外面去）
        # ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        # ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        # ct_box_b = gt_cys + self.center_radius * repeated_stride_y
        # #同上面一样进行编码
        # cl_ = repeated_x - ct_box_l
        # ct_ = repeated_y - ct_box_t
        # cr_ = ct_box_r - repeated_x
        # cb_ = ct_box_b - repeated_y

        # ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        # is_in_cts = ct_deltas.min(dim=1).values > 0 #找距离gt目标框的中心点最近的点 必须取min, min使这个点趋近于真实的中心点， 而且value>0, 同时这步会导致gt框周围的点也可以满足该条件
        # is_in_cts_all = is_in_cts.sum(dim=1) > 0 #-> 这样满足的点更多，因为中心点框的范围更广了

        # # in boxes or in centers, shape: [num_priors]
        # is_in_gts_or_centers = is_in_gts_all | is_in_cts_all #预测的和真实的均作为正样本

        # both in boxes and centers, shape: [num_fg, num_gt]
        # is_in_boxes_and_centers = (
        #     is_in_gts[is_in_gts_or_centers, :]
        #     & is_in_cts[is_in_gts_or_centers, :])

        is_in_gts_or_centers = is_in_gts_all #& min_dis
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)# 取预测值与gt拥有最大iou前10名的iou总和作为dynamic_k
        # calculate dynamic k for each gt  # min=1,即把dynamic_ks限制最小为1，保证一个gt至少有一个正样本 # 刚开始训练时候，由于预测基本不准，导致dynamic_k基本上都是1
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk( # 取cost排名最小的前dynamic_k个anchor作为postive 即选取损失最小的前k个作为postive
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx
        ## 针对一个anchor匹配了2个gt情况进行处理
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
