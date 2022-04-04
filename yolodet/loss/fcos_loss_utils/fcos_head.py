# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Scale
# from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmdet.core import multi_apply, reduce_mean
# from ..builder import HEADS, build_loss
# from .anchor_free_head import AnchorFreeHead
from yolodet.models.two_stage_utils.coder import DistancePointBBoxCoder
from yolodet.loss.mmdet_loss_utils import CrossEntropyLoss, MSELoss, CIoULoss, FocalLoss, GIoULoss, IoULoss
from yolodet.models.two_stage_utils import MlvlPointGenerator
from abc import ABCMeta
import logging
logger = logging.getLogger(__name__)
INF = 1e8


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


def _bbox_post_process(mlvl_scores,
                        mlvl_labels,
                        mlvl_bboxes,
                        mlvl_score_factors,
                        scale_factor=None,
                        cfg={'max_per_img': 100, 'min_bbox_size': 0, 'nms': {'iou_threshold': 0.5, 'type': 'nms'}, 'nms_pre': 1000, 'score_thr': 0.05},
                        rescale=False,
                        with_nms=True,
                        **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        #mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        # mlvl_scores = torch.cat(mlvl_scores)
        # mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            #mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg['nms'])
            det_bboxes = det_bboxes[:cfg['max_per_img']]
            det_labels = mlvl_labels[keep_idxs][:cfg['max_per_img']]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

# @HEADS.register_module()
class FCOSHead(nn.Module, metaclass=ABCMeta):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    # def __init__(self,
    #              num_classes,
    #              in_channels,
    #              regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
    #                              (512, INF)),
    #              center_sampling=False,
    #              center_sample_radius=1.5,
    #              norm_on_bbox=False,
    #              centerness_on_reg=False,
    #              loss_cls=dict(
    #                  type='FocalLoss',
    #                  use_sigmoid=True,
    #                  gamma=2.0,
    #                  alpha=0.25,
    #                  loss_weight=1.0),
    #              loss_bbox=dict(type='IoULoss', loss_weight=1.0),
    #              loss_centerness=dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=True,
    #                  loss_weight=1.0),
    #              norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #              init_cfg=dict(
    #                  type='Normal',
    #                  layer='Conv2d',
    #                  std=0.01,
    #                  override=dict(
    #                      type='Normal',
    #                      name='conv_cls',
    #                      std=0.01,
    #                      bias_prob=0.01)),
    #              **kwargs):
    def __init__(self,
                num_classes=None,
                anchor_generator=None,
                featmap_strides=None,
                add=None,
                box_loss_type='iou',
                one_hot_smoother=0.,
                show_pos_bbox=None,
                loss_cls_weight=None,
                loss_conf_weight=None,
                loss_reduction=None,
                topk=None,
                area_scale=None,
                object_scale=None,
                noobject_scale=None,
                loss_iou_weight=None,
                regress_ranges=None,
                center_sampling=False,
                center_sample_radius=1.5,
                norm_on_bbox=False,
                centerness_on_reg=False,
                **kwargs):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes
        self.per_anchor = len(anchor_generator[0])
        self.add = add
        self.box_loss_type = box_loss_type
        self.show_pos_bbox = show_pos_bbox
        self.featmap_strides = featmap_strides
        self.area_scale = area_scale
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.loss_iou_weight = loss_iou_weight
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        self.loss_cls = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=1) #FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0) #build_loss(loss_cls)
        self.loss_bbox =  CIoULoss(reduction=loss_reduction, loss_weight=loss_iou_weight) #build_loss(loss_bbox)
        self.loss_centerness = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_conf_weight)
        self.bbox_coder = DistancePointBBoxCoder()#build_bbox_coder(bbox_coder)

        self.prior_generator = MlvlPointGenerator(featmap_strides)#, anchors=anchor_generator

        # In order to keep a more general interface and be consistent with
        # anchor_head. We can think of point like one anchor
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        # self.scales = Scale(1.0)

        tags = [
            'num_classes',
            'featmap_strides',
            'anchor_generator',
            'add',
            'box_loss_type',
            'show_pos_bbox',
            'loss_cls_weight',
            'loss_conf_weight',
            'loss_reduction',
            'area_scale',
            'object_scale',
            'noobject_scale',
            'loss_iou_weight',
            'regress_ranges',
            'center_sampling',
            'center_sample_radius',
            'norm_on_bbox',
            'centerness_on_reg'
            ]
        logger.info('\nmmdet yolo params:')
        for tag in tags:
            logger.info('   - {:20}: {}'.format(tag, eval(tag)))

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _get_box_single(self,
                        pred_map,
                        num_imgs=None,
                        level_idx=None):
        # 下面是mmdet写法
        # batched_list = []
        # featmap_sizes = pred_map.size()[-2:]
        # pred_map = pred_map.permute(0,2,3,1).contiguous()
        # flatten_bbox_preds = pred_map[..., :4].reshape(num_imgs, -1, 4) #self.scales().float()
        # flatten_centerness = pred_map[..., 4].reshape(num_imgs, -1).sigmoid()
        # flatten_cls_scores = pred_map[..., 5:].reshape(num_imgs, -1, self.num_classes).sigmoid()
        # flatten_level_points = self.prior_generator.grid_priors(
        #     featmap_sizes,
        #     dtype=pred_map[0].dtype,
        #     device=pred_map[0].device,
        #     level_id=level_idx)
        # #all_level_points = flatten_level_points.repeat(num_imgs, 1)
        # for img_idx in range(num_imgs):
        #     results = filter_scores_and_topk(
        #             flatten_cls_scores[img_idx], 0.05, 1000,
        #             dict(bbox_pred=flatten_bbox_preds[img_idx], priors=flatten_level_points)) 
        #     scores, labels, keep_idxs, filtered_results = results

        #     bbox_pred = filtered_results['bbox_pred']
        #     priors = filtered_results['priors']
            
        #     score_factor = flatten_centerness[img_idx][keep_idxs]
        #     boxes_pred = self.bbox_coder.decode(priors, bbox_pred) #max_shape=[960, 576]
            
        #     out_put = _bbox_post_process(scores, labels, boxes_pred, score_factor)
        #     batch_num = torch.full([out_put[0].shape[0],1],img_idx).to(bbox_pred.device)
        #     out_puts = torch.cat([batch_num, out_put[0], out_put[1].unsqueeze(-1)],-1)
        #     batched_list.append(out_puts)

        # return batched_list #torch.cat([result[0],result[1].unsqueeze(-1)],-1) #torch.cat([boxes_pred, flatten_centerness.unsqueeze(-1), flatten_cls_scores], 1).reshape(num_imgs, -1, self.num_attrib)

        featmap_sizes = pred_map.size()[-2:]
        pred_map = pred_map.permute(0,2,3,1).contiguous()
        flatten_bbox_preds = pred_map[..., :4].reshape(-1, 4) #self.scales().float()
        flatten_centerness = pred_map[..., 4].reshape(-1).sigmoid()
        flatten_cls_scores = pred_map[..., 5:].reshape(-1, self.num_classes).sigmoid()
        flatten_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=pred_map[0].dtype,
            device=pred_map[0].device,
            level_id=level_idx)
        all_level_points = flatten_level_points.repeat(num_imgs, 1)
        boxes_pred = self.bbox_coder.decode(all_level_points, flatten_bbox_preds)
        return torch.cat([boxes_pred, flatten_centerness.unsqueeze(-1), flatten_cls_scores], 1).reshape(num_imgs, -1, self.num_attrib)

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def compute_loss_fcos(self,
             pred_maps,
            #  cls_scores,
            #  bbox_preds,
            #  centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        #sum_loss_bbox, sum_loss_centerness, sum_loss_cls = 0,0,0
        num_imgs = pred_maps[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in pred_maps]
        pred_maps = [pred_map.permute(0,2,3,1).contiguous() for pred_map in pred_maps]

        flatten_bbox_preds = [level_maps[..., :4].reshape(-1, 4) for level_maps in pred_maps]
        flatten_centerness = [level_maps[..., 4].reshape(-1) for level_maps in pred_maps]
        flatten_cls_scores = [level_maps[..., 5:].reshape(-1, self.num_classes) for level_maps in pred_maps] #cls不能加sigmoid()

        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=pred_maps[0].dtype,
            device=pred_maps[0].device) #这里和其他算法得到的anchor不一样这里是 point,另外该point就是目标的中心点，其他算法为左上角点，具体做法为，在特征图层上的每个点上（+0.5）*stride
        labels_list, bbox_targets_list = self.get_targets(all_level_points, gt_bboxes, #返回各个不同level下的label、bbox_targets的
                                                gt_labels)
        #show org_img
        if self.show_pos_bbox:
            import cv2
            for info in range(len(img_metas)):
                img_shape = img_metas[info]['img_shape'] #(h,w)
                pad_shape = img_metas[info]['pad_shape']
                t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
                l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
                img = cv2.imread(img_metas[info]['filename'])
                img = cv2.resize(img,(img_shape[1],img_shape[0]))
                index_lable = ((labels_list[info] >= 0) & (labels_list[info] < self.num_classes)).nonzero()
                labels = labels_list[info][index_lable].squeeze(-1)
                gt_boxes = bbox_targets_list[info][index_lable].squeeze(1)
                anchor_point = torch.cat(all_level_points,0)[index_lable].squeeze(1)
                gt_boxes_decoded = self.bbox_coder.decode(anchor_point, gt_boxes)

                pred_boxes = torch.cat([flatten_bbox_preds[i].reshape(num_imgs,-1,4)[info] for i in range(len(pred_maps))],0)[index_lable].squeeze(1)
                pred_boxes_decoded = self.bbox_coder.decode(anchor_point, pred_boxes)

                for idx in range(len(gt_boxes_decoded)):
                    pos_gt_bbox = gt_boxes_decoded[idx]
                    pos_bbox = pred_boxes_decoded[idx]
                    cv2.rectangle(img,(int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)),(int(pos_gt_bbox[2]-l_pad),int(pos_gt_bbox[3]-t_pad)),(0,255,0),1)
                    cv2.putText(img, str(labels[idx].item()), (int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)+5), 0, 1, (255,0,0), 1)
                    cv2.rectangle(img,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
                
                cv2.imshow("pos_bbox",img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    raise Exception

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in all_level_points]
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(len(num_points)):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.featmap_strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
     
        # for i in range(len(num_points)):
        #     _loss_bbox, _loss_centerness, _loss_cls = self.single_loss(num_imgs, all_level_points[i], concat_lvl_labels[i], concat_lvl_bbox_targets[i], 
        #                                                             flatten_cls_scores[i], flatten_bbox_preds[i], flatten_centerness[i])
        #     sum_loss_cls += _loss_cls
        #     sum_loss_centerness += _loss_centerness
        #     sum_loss_bbox += _loss_bbox
        
        # loss = sum_loss_cls + sum_loss_centerness + sum_loss_bbox
        # return loss, torch.stack((sum_loss_bbox,sum_loss_centerness,sum_loss_cls,loss)).detach()


        # flatten cls_scores, bbox_preds and centerness
        # flatten_cls_scores = [
        #     cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        #     for cls_score in cls_scores
        # ]
        # flatten_bbox_preds = [
        #     bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        #     for bbox_pred in bbox_preds
        # ]
        # flatten_centerness = [
        #     centerness.permute(0, 2, 3, 1).reshape(-1)
        #     for centerness in centernesses
        # ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(concat_lvl_labels)
        flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=pred_maps[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
        
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        loss = loss_cls + loss_centerness + loss_bbox
        return loss, torch.stack((loss_bbox,loss_centerness,loss_cls,loss)).detach()

    def single_loss(self, num_imgs, every_level_point, every_level_gt_label, every_level_bbox_target, every_level_pred_label, every_level_bbox_pred, every_level_centerness):
        
        # flatten_points = torch.cat(
        #     [points.repeat(num_imgs, 1) for points in every_level_point])
        flatten_points = every_level_point.repeat(num_imgs, 1)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((every_level_gt_label >= 0)
                    & (every_level_gt_label < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=every_level_gt_label.device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            every_level_pred_label, every_level_gt_label, avg_factor=num_pos)

        pos_bbox_preds = every_level_bbox_pred[pos_inds]
        pos_centerness = every_level_centerness[pos_inds]
        pos_bbox_targets = every_level_bbox_target[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
        
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return loss_bbox, loss_centerness, loss_cls


    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges) #这里设置[(-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000000.0)]这是为了将gt框按照该尺寸
        num_levels = len(points)                       #来区分对用的gt应该分配到那个特征图上进行预测，在不在个尺寸范围的gt作为背景
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0) #每特征层上gt框大小的限制
        concat_points = torch.cat(points, dim=0) #每特征层上的所有中心点

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        return labels_list, bbox_targets_list
        # split to per img, per level
        # labels_list = [labels.split(num_points, 0) for labels in labels_list]
        # bbox_targets_list = [
        #     bbox_targets.split(num_points, 0)
        #     for bbox_targets in bbox_targets_list
        # ]
        
        # concat per level image
        # concat_lvl_labels = []
        # concat_lvl_bbox_targets = []
        # for i in range(num_levels):
        #     concat_lvl_labels.append(
        #         torch.cat([labels[i] for labels in labels_list]))
        #     bbox_targets = torch.cat(
        #         [bbox_targets[i] for bbox_targets in bbox_targets_list])
        #     if self.norm_on_bbox:
        #         bbox_targets = bbox_targets / self.featmap_strides[i]
        #     concat_lvl_bbox_targets.append(bbox_targets)
        #return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]) # gt框的面积
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1) #将所有gt框的面积进行重复num_points次 行:所有的num_point，列为每个gt的area的面积
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1) #当left，top，right，bottom均大于0时表示该点落在gt框内部

        if self.center_sampling:#中心点采样方式
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.featmap_strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], # 左上点 在gt内
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], #左上点 在gt内
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], #右下点 在gt内
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], #右下点 在gt内
                                             gt_bboxes[..., 3], y_maxs)
            #上一步相当于缩小gt中心点的范围在原中心点上分别移动8,16,32等步长的框，下面条件为中心点必须落在该设置中心点的步长范围内，相当于局限了中心点落的范围。
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox  bbox_targets为gt框上的各个点的坐标相对于anchor_point的位置差值，及确定anchor_point中哪些点是落在gt框内的
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0 #min 某一行内的最小值都大于0，那么该行必定均大于0。就表示该点落在了gt框内

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0] #距离最大差值   按照regress设定的区间可以自己学或者
        inside_regress_range = ( #按照regress设定的区间 找到落在区间范围内的点 应该是距离中心点的偏置点上下左右落在设置的regress的区间范围内。
            (max_regress_distance >= regress_ranges[..., 0]) #regress_ranges[..., 0] 0为区间范围下限，1为区间范围上限
            & (max_regress_distance <= regress_ranges[..., 1]))
        # regress_ranges 的意思是对于不同层仅仅负责指定范围大小的gt bbox，例如最浅层输出，其范围是0-64，表示对于该特征图上面任何一点，假设其负责的gt bbox的label值是left、top、right和bottom，那么这4个值里面取max必须在0-64范围内，否则就算背景样本。
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF 
        areas[inside_regress_range == 0] = INF #这两句为了获取面积小于INF的行索引号
        min_area, min_area_inds = areas.min(dim=1) #根据inside_regress_range分配在self.regress_ranges区间的

        labels = gt_labels[min_area_inds] #获取label
        labels[min_area == INF] = self.num_classes  # set as BG 将INF的设置为背景
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        
        #vild_index = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1) #得到的为目标框的index
        #pos_boxes = bbox_targets[vild_index] #采样结果的box 这边可以进行可视化
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points
