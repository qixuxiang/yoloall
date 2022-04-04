# Copyright (c) OpenMMLab. All rights reserved.

import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import reduce_mean #该函数需要优化

from yolodet.loss.mmdet_loss_utils import CrossEntropyLoss, MSELoss, CIoULoss
from yolodet.utils.general import bbox_iou
from yolodet.models.two_stage_utils.coder import YOLOBBoxCoder, YOLOIouBBoxCoder, YOLOV5BBoxCoder
from yolodet.models.two_stage_utils import YOLOAnchorGenerator
# from yolodet.models.two_stage_utils.assigners import GridAssigner
from .sim_ota_yzf_assigner import SimOTAAssigner
from yolodet.models.two_stage_utils.samplers import PseudoSampler
import logging
logger = logging.getLogger(__name__)
from abc import ABCMeta

from functools import partial
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


# @HEADS.register_module()
class YOLOXHead_yzf(nn.Module, metaclass=ABCMeta):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

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
                loss_xy_weight=None,
                loss_wh_weight=None,
                loss_reduction=None,
                pos_iou_thr=None,
                neg_iou_thr=None,
                area_scale=None,
                object_scale=None,
                noobject_scale=None,
                loss_iou_weight=None,
                simOTA=None
                ):
        super(YZFHead, self).__init__()
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
        self.num_levels = len(featmap_strides)
        self.one_hot_smoother = one_hot_smoother

        #self.assigner = GridAssigner(pos_iou_thr=pos_iou_thr,neg_iou_thr=neg_iou_thr,min_pos_iou=.0)
        self.sampler = PseudoSampler()
        #YOLOBBoxCoder
        self.bbox_coder = YOLOBBoxCoder()# if self.box_loss_type in ['iou','mse'] else YOLOIouBBoxCoder()
        self.prior_generator = YOLOAnchorGenerator(strides=featmap_strides, base_sizes = anchor_generator)
        
        self.assigner = SimOTAAssigner(**simOTA) if simOTA != None else None #center_radius=2.5, candidate_topk=10, iou_weight=3.0, cls_weight=1.0) #build_assigner

        self.loss_cls = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_cls_weight)#1.0
        self.loss_conf = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_conf_weight)#1.0
        self.loss_xy = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_xy_weight)#2.0
        self.loss_wh = MSELoss(reduction=loss_reduction, loss_weight=loss_wh_weight)#2.0
        self.iou_loss =  CIoULoss(reduction=loss_reduction, loss_weight=loss_iou_weight)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.use_l1 = False
        assert len(self.prior_generator.num_base_priors) == len(featmap_strides)

        tags = [
            'num_classes',
            'featmap_strides',
            'anchor_generator',
            'add',
            'box_loss_type',
            'show_pos_bbox',
            'loss_cls_weight',
            'loss_conf_weight',
            'loss_xy_weight',
            'loss_wh_weight',
            'loss_reduction',
            'pos_iou_thr',
            'neg_iou_thr',
            'area_scale',
            # 'ignore_iou',
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
        featmap_size = pred_map.shape[2:4]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_size, level=level_idx, dtype=pred_map.dtype, device=pred_map.device)

        pred = pred_map.reshape(num_imgs,-1,self.num_attrib)
        pred[...,:4] = self._bbox_decode(mlvl_anchors, pred[...,:4], stride=self.featmap_strides[level_idx], train=False)
        pred[...,4] = pred[...,4].sigmoid()
        pred[...,5:] = pred[...,5:].sigmoid()
        return pred


    def _bbox_decode(self, bboxes, pred_bboxes, stride, train=True):
        # assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        # xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (pred_bboxes[..., :2] - 0.5) * stride #anchor 的中心点 + pred预测中心点的偏移量
        # whs = (bboxes[..., 2:] - bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp() #anchor 的w,h * pred预测w,h的偏移量
        # if train:
        #     decoded_bboxes = torch.stack((xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] - whs[..., 1],
        #                                 xy_centers[..., 0] + whs[..., 0], xy_centers[..., 1] + whs[..., 1]), dim=-1) # -> 框坐标：左上点和右下点
        # else:
        #     decoded_bboxes = torch.cat([xy_centers,  whs], dim=-1)
  
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        x_center = (bboxes[..., 2] + bboxes[..., 0]) * 0.5
        y_center = (bboxes[..., 3] + bboxes[..., 1]) * 0.5
        w_anchor = bboxes[...,2] - bboxes[...,0]
        h_anchor = bboxes[...,3] - bboxes[...,1]
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
        w_pre = pred_bboxes[...,2].exp() * w_anchor
        h_pre = pred_bboxes[...,3].exp() * h_anchor
        if train:
            decoded_bboxes = torch.stack((x_center_pred - w_pre*0.5,  y_center_pred - h_pre * 0.5, 
                                          x_center_pred + w_pre*0.5,  y_center_pred + h_pre * 0.5), dim=-1)
        else:
            decoded_bboxes = torch.stack((x_center_pred,y_center_pred,w_pre,h_pre),dim=-1)

        return decoded_bboxes

    def compute_loss_yzf(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
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
        num_imgs = len(img_metas)
        device = pred_maps[0][0].device
        
        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        #anchor_list = [mlvl_anchors for _ in range(num_imgs)]

        preds = [pred_maps[i].view(num_imgs,self.per_anchor,self.num_attrib,featmap_sizes[i][0],featmap_sizes[i][1]).permute(0,1,3,4,2).contiguous().reshape(num_imgs,-1,self.num_attrib) for i in range(len(pred_maps))]
        flatten_bbox_preds = [pred[...,:4] for pred in preds]
        flatten_objectness = [pred[...,4] for pred in preds]
        flatten_cls_preds = [pred[...,5:] for pred in preds]
        
        # anchor和预测偏置的编码得到的目标框
        flatten_bboxes = []
        for mlvl_anchor, flatten_bbox, anchor_level in zip(mlvl_anchors, flatten_bbox_preds, self.featmap_strides):
            flatten_bboxes.append(self._bbox_decode(mlvl_anchor, flatten_bbox, anchor_level))

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_anchors)#.repeat(num_imgs, 1, 1)
        flatten_bboxes = torch.cat(flatten_bboxes, dim=1)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),# 共8张图片，每张图均按照这样进行画目标框
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


        loss_bbox = self.iou_loss(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_conf(flatten_objectness.view(-1, 1),
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



        # #新加
        # valid_flag_list = []
        # for img_id, img_meta in enumerate(img_metas):
        #     multi_level_flags = self.prior_generator.valid_flags(
        #         featmap_sizes, img_meta['pad_shape'], device)
        #     valid_flag_list.append(multi_level_flags)
        
        # responsible_flag_list = []
        # for img_id in range(len(img_metas)):
        #     responsible_flag_list.append(
        #         self.prior_generator.responsible_flags(featmap_sizes,
        #                                                gt_bboxes[img_id],
        #                                                device))
        # if self.box_loss_type == 'iou':
        #     target_maps_list, neg_maps_list = self.get_targets(
        #         anchor_list, responsible_flag_list, gt_bboxes, gt_labels, img_metas)

        #     losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
        #         self.loss_single, pred_maps, target_maps_list, neg_maps_list, featmap_sizes)
         
        #     losses_cls = sum(losses_cls)
        #     losses_conf = sum(losses_conf)
        #     losses_xywh = sum(losses_xy) + sum(losses_wh)

            
        #     if not self.add:
        #         losses_cls /= num_imgs
        #         losses_conf /= num_imgs
        #         losses_xywh /= num_imgs

        #     loss = losses_cls + losses_conf + losses_xywh

        #     return loss, torch.stack((losses_xywh,losses_conf,losses_cls,loss)).detach()
        # else:  # CIOU
        #     target_maps_list, neg_maps_list, gt_bbox_list, anchor_box_list = self.get_targets_ciou(
        #         anchor_list, responsible_flag_list, gt_bboxes, gt_labels, img_metas)
            
        #     losses_cls, losses_conf, loss_iou = multi_apply(
        #         self.loss_single_ciou, self.featmap_strides, pred_maps, target_maps_list, neg_maps_list, featmap_sizes, gt_bbox_list, anchor_box_list)


        #     losses_cls = sum(losses_cls)
        #     losses_conf = sum(losses_conf)
        #     loss_iou = sum(loss_iou)# * self.loss_iou_weight

        #     if not self.add:
        #         losses_cls /= num_imgs
        #         losses_conf /= num_imgs
        #         loss_iou /= num_imgs

        #     loss = losses_cls + losses_conf + loss_iou
        #     return loss, torch.stack((loss_iou,losses_conf,losses_cls,loss)).detach()


    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        
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
        # offset_priors = torch.cat( #中心点和w,h
        #     [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        # 中心点和w,h
        # offset_priors = torch.stack([(priors[:, 0] + priors[:, 2]) * 0.5, 
        #                              (priors[:, 1] + priors[:, 3]) * 0.5,
        #                              (priors[:, 2] - priors[:, 0]),
        #                              (priors[:, 3] - priors[:, 1])], dim=-1)
        offset_priors = priors

        assign_result = self.assigner.assign( #置信度*预测分类分数-> 排名，偏置(anchor), 解码的预测框, 真实框、 label
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(), #分类和框置信度乘积
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels) #decoded_bboxes ->(左上点， 右下点)

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


    def loss_single(self, pred_map, target_map, neg_map, freature_size):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        # pred_map = pred_map.permute(0, 2, 3,
        #                             1).reshape(num_imgs, -1, self.num_attrib)
        #self.na = 3,
        pred_map = pred_map.view(num_imgs,self.per_anchor,self.num_attrib,freature_size[0],freature_size[1]).permute(0,1,3,4,2).contiguous().reshape(num_imgs,-1,self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]

        pos_and_neg_mask = neg_mask * self.noobject_scale + pos_mask * self.object_scale
        pos_mask = pos_mask.unsqueeze(dim=-1)

        # if torch.max(pos_and_neg_mask) > 1.:
        #     warnings.warn('There is overlap between pos and neg sample.')
        #     pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        if torch.max(pos_and_neg_mask) > max(self.noobject_scale,self.object_scale):
             warnings.warn('There is overlap between pos and neg sample.')
             pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=max(self.noobject_scale, self.object_scale))

        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]

        if self.area_scale: #darknet的思想 大小目标宽高不一致、表现在数值上就会时大时小，对于l1或者l2来说，梯度就不一样大，这是不好的，因为极端情况就是网络只学习大物体，小物体由于梯度太小被忽略了。虽然前面引入了log，但是还可以进一步克服，具体就是在基于gt的宽高， 给大小物体引入一个不一样大的系数
            tcoord_weight = 2. - target_wh[..., 0:1] * target_wh[..., 1:2]
        else:
            tcoord_weight = 1.

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask * tcoord_weight)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask * tcoord_weight)

        return loss_cls, loss_conf, loss_xy, loss_wh

    def loss_single_ciou(self, stride, pred_map, target_map, neg_map, freature_size, gt_bbox, anchor_box):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        # pred_map = pred_map.permute(0, 2, 3,
        #                             1).reshape(num_imgs, -1, self.num_attrib)
        #self.na = 3,
        pred_map = pred_map.view(num_imgs,self.per_anchor,self.num_attrib,freature_size[0],freature_size[1]).permute(0,1,3,4,2).contiguous().reshape(num_imgs,-1,self.num_attrib)
        
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 0] # conf
        pos_and_neg_mask = neg_mask * self.noobject_scale + pos_mask * self.object_scale
        # pos_mask = pos_mask.unsqueeze(dim=-1)

        box_pos_mask = pos_mask.bool()
        pos_mask = box_pos_mask.unsqueeze(dim=-1)
        # if torch.max(pos_and_neg_mask) > 1.:
        #     warnings.warn('There is overlap between pos and neg sample.')
        #     pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        if torch.max(pos_and_neg_mask) > max(self.noobject_scale,self.object_scale):
             warnings.warn('There is overlap between pos and neg sample.')
             pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=max(self.noobject_scale, self.object_scale))
        #yolov5的写法:预测出框，用预测的框和gtbox做ciou
        anchor_x_center = anchor_box[..., 0]
        anchor_y_center = anchor_box[..., 1]
        anchor_w = anchor_box[..., 2]
        anchor_h = anchor_box[..., 3]
        pred_x = pred_map[..., 0].sigmoid() * stride + anchor_x_center
        pred_y = pred_map[..., 1].sigmoid() * stride + anchor_y_center
        pred_w = torch.exp(pred_map[..., 2]) * anchor_w
        pred_h = torch.exp(pred_map[..., 3]) * anchor_h
        pred_box = torch.stack((pred_x,pred_y,pred_w,pred_h),-1)

        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_conf = target_map[..., 0]
        target_label = target_map[..., 1:]

        # if self.area_scale: #darknet的思想 大小目标宽高不一致、表现在数值上就会时大时小，对于l1或者l2来说，梯度就不一样大，这是不好的，因为极端情况就是网络只学习大物体，小物体由于梯度太小被忽略了。虽然前面引入了log，但是还可以进一步克服，具体就是在基于gt的宽高， 给大小物体引入一个不一样大的系数
        #     tcoord_weight = 2. - target_wh[..., 0:1] * target_wh[..., 1:2]
        # else:
        #     tcoord_weight = 1.

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(pred_conf, target_conf, weight=pos_and_neg_mask)
        # loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask * tcoord_weight)
        # loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask * tcoord_weight)

        # iou loss
        # iou = bbox_iou(pred_box[box_pos_mask].T, gt_bbox[box_pos_mask], x1y1x2y2=False, CIoU=True)
        # loss_box = (1 - iou).sum()

        loss_box = self.iou_loss(pred = pred_box[box_pos_mask], target= gt_bbox[box_pos_mask], x1y1x2y2=True)


        return loss_cls, loss_conf, loss_box


    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list, img_metas):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list, img_metas)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels,img_meta):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):#anchors为各个下采样4种倍率的feature map上的绘制的框
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)#将所有(下采样4种倍率)feature_map上的框进行cat在一起
        concat_responsible_flags = torch.cat(responsible_flags)#将所有(下采样4种倍率)相应的index(位置标志) cat在一起

        anchor_strides = torch.cat(anchor_strides)#所有下采样倍率的anchor的框cat在一起
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,                 #assign_result类中包含num_gts：框个数、gt_inds：正样本的index、max_overlaps：(每个gt框和多个anchor框的交并比 取每列的最大交并比)
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors, 
                                              gt_bboxes)
        #sampling_result类中包含self.pos_inds(正样本索引)、self.neg_inds(负样本索引)、self.pos_bboxes(正样本索引)、self.neg_bboxes(负样本框)、self.pos_is_gt(正样本标志位)、self.num_gts(正样本数量)、self.pos_assigned_gt_inds(正样本目标框index)、self.pos_gt_bboxes(正样本目标框)
        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib) #构建target_map: 对所有的anchor框 (x1,y1,x2,y2,conf,n_class)
        #show org_img
        if self.show_pos_bbox:
            import cv2
            img_shape = img_meta['img_shape'] #(h,w)
            pad_shape = img_meta['pad_shape']
            t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
            l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
            img = cv2.imread(img_meta['filename'])
            img = cv2.resize(img,(img_shape[1],img_shape[0]))

            for idx in range(len(sampling_result.pos_gt_bboxes)):
                pos_gt_bbox = sampling_result.pos_gt_bboxes[idx]
                pos_bbox = sampling_result.pos_bboxes[idx]

                img = cv2.rectangle(img,(int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)),(int(pos_gt_bbox[2]-l_pad),int(pos_gt_bbox[3]-t_pad)),(0,255,0),1)
                img = cv2.rectangle(img,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
            
            cv2.imshow("pos_bbox",img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                raise Exception

        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels.long(), num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds] #将target后面的几个放置到target_map的后面 object
        
        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1   #无目标的no_object
        #show pre box 按照feature_map显示目标框
        if self.show_pos_bbox:
            import cv2
            import copy
            img_shape = img_meta['img_shape'] #(h,w)
            pad_shape = img_meta['pad_shape']
            t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
            l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
            img = cv2.imread(img_meta['filename'])
            img = cv2.resize(img,(img_shape[1],img_shape[0]))

            anchors_unique = anchor_strides[sampling_result.pos_inds].unique()
            bboxes_index = [[] for idx in range(len(anchors_unique))]
            for idx in range(len(anchors_unique)):
                bboxes_index[idx].append(sampling_result.pos_bboxes[anchor_strides[sampling_result.pos_inds] == anchors_unique[idx]])
            for anchors_map in range(len(anchors_unique)):
                image = copy.copy(img)
                for pos_bboxes in bboxes_index[anchors_map]:
                    for pos_bbox in pos_bboxes:
                        image = cv2.rectangle(image,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
                        image = cv2.putText(image, str(int(anchors_unique[anchors_map].item())), (int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)), 0, 1, (0,0,255), 1)
                cv2.imshow("pos_bbox",image)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    raise Exception
        return target_map, neg_map

    def get_targets_ciou(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                        gt_labels_list, img_metas):
            """Compute target maps for anchors in multiple images.

            Args:
                anchor_list (list[list[Tensor]]): Multi level anchors of each
                    image. The outer list indicates images, and the inner list
                    corresponds to feature levels of the image. Each element of
                    the inner list is a tensor of shape (num_total_anchors, 4).
                responsible_flag_list (list[list[Tensor]]): Multi level responsible
                    flags of each image. Each element is a tensor of shape
                    (num_total_anchors, )
                gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
                gt_labels_list (list[Tensor]): Ground truth labels of each box.

            Returns:
                tuple: Usually returns a tuple containing learning targets.
                    - target_map_list (list[Tensor]): Target map of each level.
                    - neg_map_list (list[Tensor]): Negative map of each level.
            """
            num_imgs = len(anchor_list)

            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

            results = multi_apply(self._get_targets_single_ciou, anchor_list,
                                  responsible_flag_list, gt_bboxes_list,
                                  gt_labels_list, img_metas)

            all_target_maps, all_neg_maps, gt_bbox_maps, anchor_bbox_maps = results
            assert num_imgs == len(all_target_maps) == len(all_neg_maps) == len(gt_bbox_maps) == len(anchor_bbox_maps)
            target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
            neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)
            gt_bbox_list = images_to_levels(gt_bbox_maps, num_level_anchors)
            anchor_box_list = images_to_levels(anchor_bbox_maps, num_level_anchors)

            return target_maps_list, neg_maps_list, gt_bbox_list, anchor_box_list

    def _get_targets_single_ciou(self, anchors, responsible_flags, gt_bboxes,
                                gt_labels,img_meta):
            """Generate matching bounding box prior and converted GT.

            Args:
                anchors (list[Tensor]): Multi-level anchors of the image.
                responsible_flags (list[Tensor]): Multi-level responsible flags of
                    anchors
                gt_bboxes (Tensor): Ground truth bboxes of single image.
                gt_labels (Tensor): Ground truth labels of single image.

            Returns:
                tuple:
                    target_map (Tensor): Predication target map of each
                        scale level, shape (num_total_anchors,
                        5+num_classes)
                    neg_map (Tensor): Negative map of each scale level,
                        shape (num_total_anchors,)
            """

            anchor_strides = []
            for i in range(len(anchors)):#anchors为各个下采样4种倍率的feature map上的绘制的框
                anchor_strides.append(
                    torch.tensor(self.featmap_strides[i],
                                device=gt_bboxes.device).repeat(len(anchors[i])))
            concat_anchors = torch.cat(anchors)#将所有(下采样4种倍率)feature_map上的框进行cat在一起
            concat_responsible_flags = torch.cat(responsible_flags)#将所有(下采样4种倍率)相应的index(位置标志) cat在一起

            anchor_strides = torch.cat(anchor_strides)#所有下采样倍率的anchor的框cat在一起
            assert len(anchor_strides) == len(concat_anchors) == \
                len(concat_responsible_flags)
            assign_result = self.assigner.assign(concat_anchors,                 #assign_result类中包含num_gts：框个数、gt_inds：正样本的index、max_overlaps：(每个gt框和多个anchor框的交并比 取每列的最大交并比)
                                                concat_responsible_flags,
                                                gt_bboxes)
            sampling_result = self.sampler.sample(assign_result, concat_anchors, 
                                                gt_bboxes)
            #sampling_result类中包含self.pos_inds(正样本索引)、self.neg_inds(负样本索引)、self.pos_bboxes(正样本索引)、self.neg_bboxes(负样本框)、self.pos_is_gt(正样本标志位)、self.num_gts(正样本数量)、self.pos_assigned_gt_inds(正样本目标框index)、self.pos_gt_bboxes(正样本目标框)
            target_map = concat_anchors.new_zeros(concat_anchors.size(0), self.num_attrib-4) #构建target_map: 对所有的anchor框 (x1,y1,x2,y2,conf,n_class)
            anchor_bbox = concat_anchors.new_zeros(concat_anchors.size(0), 4) #anchor_xywh
            gt_bbox = concat_anchors.new_zeros(concat_anchors.size(0), 4) #gt_xywh
            #show org_img
            if self.show_pos_bbox:
                import cv2
                img_shape = img_meta['img_shape'] #(h,w)
                pad_shape = img_meta['pad_shape']
                t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
                l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
                img = cv2.imread(img_meta['filename'])
                img = cv2.resize(img,(img_shape[1],img_shape[0]))

                for idx in range(len(sampling_result.pos_gt_bboxes)):
                    pos_gt_bbox = sampling_result.pos_gt_bboxes[idx]
                    pos_bbox = sampling_result.pos_bboxes[idx]

                    img = cv2.rectangle(img,(int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)),(int(pos_gt_bbox[2]-l_pad),int(pos_gt_bbox[3]-t_pad)),(0,255,0),1)
                    img = cv2.rectangle(img,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
                
                cv2.imshow("pos_bbox",img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    raise Exception

            gt_bbox[sampling_result.pos_inds, :4], anchor_bbox[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
                anchor_strides[sampling_result.pos_inds])

            # target_map[sampling_result.pos_inds, :4]
            target_map[sampling_result.pos_inds, 0] = 1

            gt_labels_one_hot = F.one_hot(
                gt_labels.long(), num_classes=self.num_classes).float()
            if self.one_hot_smoother != 0:  # label smooth
                gt_labels_one_hot = gt_labels_one_hot * (
                    1 - self.one_hot_smoother
                ) + self.one_hot_smoother / self.num_classes
            target_map[sampling_result.pos_inds, 1:] = gt_labels_one_hot[
                sampling_result.pos_assigned_gt_inds] #将target后面的几个放置到target_map的后面 object
            
            neg_map = concat_anchors.new_zeros(
                concat_anchors.size(0), dtype=torch.uint8)
            neg_map[sampling_result.neg_inds] = 1   #无目标的no_object
            #show pre box 按照feature_map显示目标框
            if self.show_pos_bbox:
                import cv2
                import copy
                img_shape = img_meta['img_shape'] #(h,w)
                pad_shape = img_meta['pad_shape']
                t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
                l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
                img = cv2.imread(img_meta['filename'])
                img = cv2.resize(img,(img_shape[1],img_shape[0]))

                anchors_unique = anchor_strides[sampling_result.pos_inds].unique()
                bboxes_index = [[] for idx in range(len(anchors_unique))]
                for idx in range(len(anchors_unique)):
                    bboxes_index[idx].append(sampling_result.pos_bboxes[anchor_strides[sampling_result.pos_inds] == anchors_unique[idx]])
                for anchors_map in range(len(anchors_unique)):
                    image = copy.copy(img)
                    for pos_bboxes in bboxes_index[anchors_map]:
                        for pos_bbox in pos_bboxes:
                            image = cv2.rectangle(image,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
                            image = cv2.putText(image, str(int(anchors_unique[anchors_map].item())), (int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)), 0, 1, (0,0,255), 1)
                    cv2.imshow("pos_bbox",image)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        raise Exception
            return target_map, neg_map, gt_bbox, anchor_bbox


    # def aug_test(self, feats, img_metas, rescale=False):
    #     """Test function with test time augmentation.

    #     Args:
    #         feats (list[Tensor]): the outer list indicates test-time
    #             augmentations and inner Tensor should have a shape NxCxHxW,
    #             which contains features for all images in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch. each dict has image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[ndarray]: bbox results of each class
    #     """
    #     return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

    # #@force_fp32(apply_to=('pred_maps'))
    # def onnx_export(self, pred_maps, img_metas, with_nms=True):
    #     num_levels = len(pred_maps)
    #     pred_maps_list = [pred_maps[i].detach() for i in range(num_levels)]

    #     cfg = self.test_cfg
    #     assert len(pred_maps_list) == self.num_levels

    #     device = pred_maps_list[0].device
    #     batch_size = pred_maps_list[0].shape[0]

    #     featmap_sizes = [
    #         pred_maps_list[i].shape[-2:] for i in range(self.num_levels)
    #     ]
    #     mlvl_anchors = self.prior_generator.grid_priors(
    #         featmap_sizes, device=device)
    #     # convert to tensor to keep tracing
    #     nms_pre_tensor = torch.tensor(
    #         cfg.get('nms_pre', -1), device=device, dtype=torch.long)

    #     multi_lvl_bboxes = []
    #     multi_lvl_cls_scores = []
    #     multi_lvl_conf_scores = []
    #     for i in range(self.num_levels):
    #         # get some key info for current scale
    #         pred_map = pred_maps_list[i]
    #         stride = self.featmap_strides[i]
    #         # (b,h, w, num_anchors*num_attrib) ->
    #         # (b,h*w*num_anchors, num_attrib)
    #         pred_map = pred_map.permute(0, 2, 3,
    #                                     1).reshape(batch_size, -1,
    #                                                self.num_attrib)
    #         # Inplace operation like
    #         # ```pred_map[..., :2] = \torch.sigmoid(pred_map[..., :2])```
    #         # would create constant tensor when exporting to onnx
    #         pred_map_conf = torch.sigmoid(pred_map[..., :2])
    #         pred_map_rest = pred_map[..., 2:]
    #         pred_map = torch.cat([pred_map_conf, pred_map_rest], dim=-1)
    #         pred_map_boxes = pred_map[..., :4]
    #         multi_lvl_anchor = mlvl_anchors[i]
    #         multi_lvl_anchor = multi_lvl_anchor.expand_as(pred_map_boxes)
    #         bbox_pred = self.bbox_coder.decode(multi_lvl_anchor,
    #                                            pred_map_boxes, stride)
    #         # conf and cls
    #         conf_pred = torch.sigmoid(pred_map[..., 4])
    #         cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
    #             batch_size, -1, self.num_classes)  # Cls pred one-hot.

    #         # Get top-k prediction
    #         from mmdet.core.export import get_k_for_topk
    #         nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
    #         if nms_pre > 0:
    #             _, topk_inds = conf_pred.topk(nms_pre)
    #             batch_inds = torch.arange(batch_size).view(
    #                 -1, 1).expand_as(topk_inds).long()
    #             # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
    #             transformed_inds = (
    #                 bbox_pred.shape[1] * batch_inds + topk_inds)
    #             bbox_pred = bbox_pred.reshape(-1,
    #                                           4)[transformed_inds, :].reshape(
    #                                               batch_size, -1, 4)
    #             cls_pred = cls_pred.reshape(
    #                 -1, self.num_classes)[transformed_inds, :].reshape(
    #                     batch_size, -1, self.num_classes)
    #             conf_pred = conf_pred.reshape(-1, 1)[transformed_inds].reshape(
    #                 batch_size, -1)

    #         # Save the result of current scale
    #         multi_lvl_bboxes.append(bbox_pred)
    #         multi_lvl_cls_scores.append(cls_pred)
    #         multi_lvl_conf_scores.append(conf_pred)

    #     # Merge the results of different scales together
    #     batch_mlvl_bboxes = torch.cat(multi_lvl_bboxes, dim=1)
    #     batch_mlvl_scores = torch.cat(multi_lvl_cls_scores, dim=1)
    #     batch_mlvl_conf_scores = torch.cat(multi_lvl_conf_scores, dim=1)

    #     # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
    #     from mmdet.core.export import add_dummy_nms_for_onnx
    #     conf_thr = cfg.get('conf_thr', -1)
    #     score_thr = cfg.get('score_thr', -1)
    #     # follow original pipeline of YOLOv3
    #     if conf_thr > 0:
    #         mask = (batch_mlvl_conf_scores >= conf_thr).float()
    #         batch_mlvl_conf_scores *= mask
    #     if score_thr > 0:
    #         mask = (batch_mlvl_scores > score_thr).float()
    #         batch_mlvl_scores *= mask
    #     batch_mlvl_conf_scores = batch_mlvl_conf_scores.unsqueeze(2).expand_as(
    #         batch_mlvl_scores)
    #     batch_mlvl_scores = batch_mlvl_scores * batch_mlvl_conf_scores
    #     if with_nms:
    #         max_output_boxes_per_class = cfg.nms.get(
    #             'max_output_boxes_per_class', 200)
    #         iou_threshold = cfg.nms.get('iou_threshold', 0.5)
    #         # keep aligned with original pipeline, improve
    #         # mAP by 1% for YOLOv3 in ONNX
    #         score_threshold = 0
    #         nms_pre = cfg.get('deploy_nms_pre', -1)
    #         return add_dummy_nms_for_onnx(
    #             batch_mlvl_bboxes,
    #             batch_mlvl_scores,
    #             max_output_boxes_per_class,
    #             iou_threshold,
    #             score_threshold,
    #             nms_pre,
    #             cfg.max_per_img,
    #         )
    #     else:
    #         return batch_mlvl_bboxes, batch_mlvl_scores