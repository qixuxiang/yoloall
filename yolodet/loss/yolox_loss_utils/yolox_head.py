# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops.nms import batched_nms
from mmdet.core import multi_apply, reduce_mean

from .point_generator import MlvlPointGenerator
from yolodet.models.two_stage_utils.transforms import bbox_xyxy_to_cxcywh
from yolodet.models.two_stage_utils.assigners import SimOTAAssigner
from yolodet.models.two_stage_utils.samplers import PseudoSampler
from yolodet.loss.mmdet_loss_utils import CrossEntropyLoss, IoULoss, L1Loss
import logging
logger = logging.getLogger(__name__)

from abc import ABCMeta

# @HEADS.register_module()
class YOLOXHead(nn.Module, metaclass=ABCMeta):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    # def __init__(self,
    #              num_classes,
    #              in_channels,
    #              feat_channels=256,
    #              stacked_convs=2,
    #              strides=[8, 16, 32],
    #              use_depthwise=False,
    #              dcn_on_last_conv=False,
    #              conv_bias='auto',
    #              conv_cfg=None,
    #              norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    #              act_cfg=dict(type='Swish'),
    #              loss_cls=dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=True,
    #                  reduction='sum',
    #                  loss_weight=1.0),
    #              loss_bbox=dict(
    #                  type='IoULoss',
    #                  mode='square',
    #                  eps=1e-16,
    #                  reduction='sum',
    #                  loss_weight=5.0),
    #              loss_obj=dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=True,
    #                  reduction='sum',
    #                  loss_weight=1.0),
    #              loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    #              train_cfg=None,
    #              test_cfg=None,
    #              init_cfg=dict(
    #                  type='Kaiming',
    #                  layer='Conv2d',
    #                  a=math.sqrt(5),
    #                  distribution='uniform',
    #                  mode='fan_in',
    #                  nonlinearity='leaky_relu')):
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
                area_scale=None,
                object_scale=None,
                noobject_scale=None,
                loss_iou_weight=None,
                simOTA=None,
                 **kwargs
    ):

        super().__init__()
        self.num_classes = num_classes
        self.per_anchor = len(anchor_generator[0])
        self.add = add
        self.show_pos_bbox = show_pos_bbox
        self.strides = featmap_strides
        self.area_scale = area_scale
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.loss_cls = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_cls_weight) #build_loss(loss_cls)
        self.loss_bbox = IoULoss(mode='square', eps=1e-16, reduction=loss_reduction, loss_weight=loss_iou_weight) #build_loss(loss_bbox)
        self.loss_obj = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_conf_weight) #build_loss(loss_obj)
        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_l1 = L1Loss(reduction=loss_reduction, loss_weight=1.0) #build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(featmap_strides, anchors=anchor_generator, offset=0)

        self.test_cfg = None #test_cfg
        # self.train_cfg = train_cfg

        self.sampling = False
        # if self.train_cfg:
        self.assigner = SimOTAAssigner(**simOTA) if simOTA != None else None #center_radius=2.5, candidate_topk=10, iou_weight=3.0, cls_weight=1.0) #build_assigner
        self.sampler = PseudoSampler()
    #     if simOTA != None:
    #         self.display_param()

    # def display_param(self):
        
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
            'simOTA'
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
                        level_idx=None,
                        #    cfg=None,
                        #    rescale=False,
                        ):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        featmap_size = pred_map.shape[2:4] #[cls_score.shape[2:] for cls_score in cls_scores]
        flatten_priors = self.prior_generator.grid_priors(
            featmap_size,
            dtype=pred_map.dtype, #cls_scores[0].dtype,
            device=pred_map.device,
            with_stride=True,
            level_id=level_idx)
        pred = pred_map.reshape(num_imgs,-1,self.num_attrib)
        # flatten_bbox_preds = pred[...,:4]
        # flatten_objectness = pred[...,4].sigmoid()
        # flatten_cls_preds = pred[...,5:].sigmoid()
        pred[...,:4] = self._bbox_decode(flatten_priors, pred[...,:4], train=False)
        pred[...,4] = pred[...,4].sigmoid()
        pred[...,5:] = pred[...,5:].sigmoid()
        return pred


        # assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        # cfg = self.test_cfg if cfg is None else cfg
        # scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        # num_imgs = len(img_metas)
        # featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        # mlvl_priors = self.prior_generator.grid_priors(
        #     featmap_sizes,
        #     dtype=cls_scores[0].dtype,
        #     device=cls_scores[0].device,
        #     with_stride=True)

        # # flatten cls_scores, bbox_preds and objectness
        # flatten_cls_scores = [
        #     cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
        #     for cls_score in cls_scores
        # ]
        # flatten_bbox_preds = [
        #     bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        #     for bbox_pred in bbox_preds
        # ]
        # flatten_objectness = [
        #     objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
        #     for objectness in objectnesses
        # ]

        # flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        # flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        # flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        # flatten_priors = torch.cat(mlvl_priors)

        # flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        # if rescale:
        #     flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
        #         scale_factors).unsqueeze(1)

        # result_list = []
        # for img_id in range(len(img_metas)):
        #     cls_scores = flatten_cls_scores[img_id]
        #     score_factor = flatten_objectness[img_id]
        #     bboxes = flatten_bboxes[img_id]

        #     result_list.append(
        #         self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))

        # return result_list

    def _bbox_decode(self, priors, bbox_preds, train=True):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]  #（框中心点x,y的偏移量 * stride = 中心点在原图上的偏移量）+ 网格左上角点坐标 = 真实中心点坐标
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:] #w,h.exp() * stride = 原图上的宽和高
        if train:
            tl_x = (xys[..., 0] - whs[..., 0] / 2) #左上角点x
            tl_y = (xys[..., 1] - whs[..., 1] / 2) #左上角点y
            br_x = (xys[..., 0] + whs[..., 0] / 2) #右下角点x
            br_y = (xys[..., 1] + whs[..., 1] / 2) #右下角点y
            decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1) #输出的为左上角点和右下角点的坐标
        else:
            decoded_bboxes = torch.cat([xys, whs], -1)
        return decoded_bboxes

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    #@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def compute_loss_yolox(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [pred.shape[2:] for pred in pred_maps] #[cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=pred_maps[0].dtype, #cls_scores[0].dtype,
            device=pred_maps[0].device,
            with_stride=True)

        preds = [pred_maps[i].view(num_imgs,self.per_anchor,self.num_attrib,featmap_sizes[i][0],featmap_sizes[i][1]).permute(0,1,3,4,2).contiguous().reshape(num_imgs,-1,self.num_attrib) for i in range(len(pred_maps))]
        flatten_bbox_preds = [pred[...,:4] for pred in preds]
        flatten_objectness = [pred[...,4] for pred in preds]
        flatten_cls_preds = [pred[...,5:] for pred in preds]
        
        # flatten_cls_preds = [
        #     cls_pred.permute(0,1,3,4,2).reshape(num_imgs, -1, self.num_classes)
        #     for cls_pred in cls_scores
        # ]
        # flatten_bbox_preds = [
        #     bbox_pred.permute(0,1,3,4,2).reshape(num_imgs, -1, 4)
        #     for bbox_pred in bbox_preds
        # ]
        # flatten_objectness = [
        #     objectness.permute(0,1,3,4,2).reshape(num_imgs, -1)
        #     for objectness in objectnesses
        # ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        if self.show_pos_bbox:
            for num_img in range(len(pos_masks)):
                import cv2
                img_meta = img_metas[num_img]
                img_shape = img_meta['img_shape'] #(h,w)
                pad_shape = img_meta['pad_shape']
                t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
                l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
                img = cv2.imread(img_meta['filename'])
                img = cv2.resize(img,(img_shape[1],img_shape[0]))

                gt_bbox = np.unique(bbox_targets[num_img].cpu().numpy(), axis=0)
                pred_box = flatten_bboxes[num_img][pos_masks[num_img]].cpu().detach().numpy()

                for idx in range(len(gt_bbox)):
                    pos_gt_bbox = gt_bbox[idx]
                    # pos_bbox = sampling_result.pos_bboxes[idx]
                    img = cv2.rectangle(img,(int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)),(int(pos_gt_bbox[2]-l_pad),int(pos_gt_bbox[3]-t_pad)),(0,255,0),1)
                    #img = cv2.rectangle(img,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)

                for idx in range(len(pred_box)):
                    pos_bbox = pred_box[idx]
                    img = cv2.rectangle(img,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)

                cv2.imshow("pos_bbox",img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    raise Exception

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples

        # loss_dict = dict(
        #     loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        loss = loss_cls + loss_obj + loss_bbox
        return loss, torch.stack((loss_bbox,loss_obj,loss_cls,loss)).detach()

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat( #左上角点 + (w/2, h/2) -> (中心点, (w,h))
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign( #置信度*预测分类分数-> 排名，偏置(anchor), 解码的预测框, 真实框、 label
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
