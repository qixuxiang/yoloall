# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
# from mmcv.cnn import ConvModule, Scale
# from mmcv.runner import force_fp32

# from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
#                         images_to_levels, multi_apply, reduce_mean, unmap)
# from ..builder import HEADS, build_loss
# from .anchor_head import AnchorHead
from mmdet.core import anchor_inside_flags, multi_apply, reduce_mean, unmap
from yolodet.models.two_stage_utils.coder import YOLOBBoxCoder, YOLOIouBBoxCoder, YOLOV5BBoxCoder, DeltaXYWHBBoxCoder
from yolodet.loss.mmdet_loss_utils import CrossEntropyLoss, MSELoss, CIoULoss, FocalLoss
from yolodet.models.two_stage_utils import YOLOAnchorGenerator
from abc import ABCMeta
from yolodet.models.two_stage_utils.samplers import PseudoSampler
from yolodet.models.two_stage_utils.assigners import ATSSAssigner
import logging
logger = logging.getLogger(__name__)


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
class ATSSHead(nn.Module, metaclass=ABCMeta):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    # def __init__(self,
    #              num_classes,
    #              in_channels,
    #              stacked_convs=4,
    #              conv_cfg=None,
    #              norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #              reg_decoded_bbox=True,
    #              loss_centerness=dict(
    #                  type='CrossEntropyLoss',
    #                  use_sigmoid=True,
    #                  loss_weight=1.0),
    #              init_cfg=dict(
    #                  type='Normal',
    #                  layer='Conv2d',
    #                  std=0.01,
    #                  override=dict(
    #                      type='Normal',
    #                      name='atss_cls',
    #                      std=0.01,
    #                      bias_prob=0.01)),
    #              **kwargs):
    #     self.stacked_convs = stacked_convs
    #     self.conv_cfg = conv_cfg
    #     self.norm_cfg = norm_cfg
    #     super(ATSSHead, self).__init__(
    #         num_classes,
    #         in_channels,
    #         reg_decoded_bbox=reg_decoded_bbox,
    #         init_cfg=init_cfg,
    #         **kwargs)

    #     self.sampling = False
    #     if self.train_cfg:
    #         self.assigner = build_assigner(self.train_cfg.assigner)
    #         # SSD sampling=False so use PseudoSampler
    #         sampler_cfg = dict(type='PseudoSampler')
    #         self.sampler = build_sampler(sampler_cfg, context=self)
    #     self.loss_centerness = build_loss(loss_centerness)
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
                topk=None,
                area_scale=None,
                object_scale=None,
                noobject_scale=None,
                loss_iou_weight=None,
                 **kwargs):
        super(ATSSHead, self).__init__()
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

        self.assigner = ATSSAssigner(topk=topk)
        self.sampler = PseudoSampler()
        self.sampling = False
        #self.pos_weight = -1
        #YOLOBBoxCoder
        self.bbox_coder = YOLOBBoxCoder() #YOLOIouBBoxCoder() #YOLOV5BBoxCoder() if self.box_loss_type in ['iou','mse'] else YOLOIouBBoxCoder() #DeltaXYWHBBoxCoder()#
        self.prior_generator = YOLOAnchorGenerator(strides=featmap_strides, base_sizes=anchor_generator)

        self.loss_cls = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        #CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_cls_weight)#1.0

        self.reg_decoded_bbox = True
        # self.loss_cls = FocalLoss(use_sigmoid=True,
        #                           gamma=2.0,
        #                           alpha=0.25,
        #                           #reduction=loss_reduction,
        #                           loss_weight=1.0)
        # self.loss_bbox = GIoULoss(#reduction=loss_reduction,
        #                           loss_weight=1.0)
        self.loss_centerness = CrossEntropyLoss(use_sigmoid=True, reduction=loss_reduction, loss_weight=loss_conf_weight)#1.0
        # self.loss_xy = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_xy_weight)#2.0
        # self.loss_wh = MSELoss(reduction=loss_reduction, loss_weight=loss_wh_weight)#2.0
        self.iou_loss =  CIoULoss(reduction=loss_reduction, loss_weight=loss_iou_weight)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        #assert len(self.prior_generator.num_base_priors) == len(featmap_strides)

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
            'area_scale',
            # 'ignore_iou',
            'object_scale',
            'noobject_scale',
            'loss_iou_weight'
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
        batch_size = len(pred_map)
        featmap_size = pred_map.shape[2:4] #[cls_score.shape[2:] for cls_score in cls_scores]
        flatten_priors = self.prior_generator.grid_priors(
            featmap_size,
            dtype=pred_map.dtype, #cls_scores[0].dtype,
            device=pred_map.device,
            #with_stride=True,
            level=level_idx)

        anchors = flatten_priors.reshape(-1, 4).repeat(batch_size,1)
        cls_score = pred_map[..., 5:].reshape(-1, self.num_classes).sigmoid()
        bbox_pred = pred_map[..., :4].reshape(-1, 4)
        centerness = pred_map[..., 4].reshape(-1).sigmoid()
        
        pos_decode_bbox_pred = self.bbox_coder.decode(
                anchors, bbox_pred, self.featmap_strides[level_idx])

        pred_maps = torch.cat([pos_decode_bbox_pred, centerness.unsqueeze(-1), cls_score], -1).reshape(batch_size, -1, self.num_attrib)
        return pred_maps


    # def forward(self, feats):
    #     """Forward features from the upstream network.

    #     Args:
    #         feats (tuple[Tensor]): Features from the upstream network, each is
    #             a 4D-tensor.

    #     Returns:
    #         tuple: Usually a tuple of classification scores and bbox prediction
    #             cls_scores (list[Tensor]): Classification scores for all scale
    #                 levels, each is a 4D-tensor, the channels number is
    #                 num_anchors * num_classes.
    #             bbox_preds (list[Tensor]): Box energies / deltas for all scale
    #                 levels, each is a 4D-tensor, the channels number is
    #                 num_anchors * 4.
    #     """
    #     return multi_apply(self.forward_single, feats, self.scales)

    # def forward_single(self, x, scale):
    #     """Forward feature of a single scale level.

    #     Args:
    #         x (Tensor): Features of a single scale level.
    #         scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
    #             the bbox prediction.

    #     Returns:
    #         tuple:
    #             cls_score (Tensor): Cls scores for a single scale level
    #                 the channels number is num_anchors * num_classes.
    #             bbox_pred (Tensor): Box energies / deltas for a single scale
    #                 level, the channels number is num_anchors * 4.
    #             centerness (Tensor): Centerness for a single scale level, the
    #                 channel number is (N, num_anchors * 1, H, W).
    #     """
    #     cls_feat = x
    #     reg_feat = x
    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)
    #     cls_score = self.atss_cls(cls_feat)
    #     # we just follow atss, not apply exp in bbox_pred
    #     bbox_pred = scale(self.atss_reg(reg_feat)).float()
    #     centerness = self.atss_centerness(reg_feat)
    #     return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, pred_maps, strides, featmap_sizes,#cls_score, bbox_pred, centerness, 
                    labels,label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(pred_maps)
        pred_maps = pred_maps.view(num_imgs,self.per_anchor,self.num_attrib,featmap_sizes[0],featmap_sizes[1]).permute(0,1,3,4,2).contiguous().reshape(num_imgs,-1,self.num_attrib)

        cls_score = pred_maps[..., 5:].reshape(-1, self.num_classes)
        bbox_pred = pred_maps[..., :4].reshape(-1, 4)
        centerness = pred_maps[..., 4].reshape(-1)

        anchors = anchors.reshape(-1, 4)
        # cls_score = cls_score.permute(0, 2, 3, 1).reshape(
        #     -1, self.cls_out_channels).contiguous()
        # bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        # centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred, strides)

            # regression loss
            loss_bbox = self.iou_loss(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    #@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def compute_loss_atss(self,
             #cls_scores, 
             #bbox_preds,
             #centernesses,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in pred_maps]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = pred_maps[0].device

        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(len(img_metas))]
        #新加
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        
        # anchor_list, valid_flag_list = self.get_anchors(
        #     featmap_sizes, img_metas, device=device)
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=self.num_classes)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                pred_maps,
                self.featmap_strides,
                featmap_sizes,
                # cls_scores,
                # bbox_preds,
                # centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        losses_conf = sum(loss_centerness)
        losses_cls = sum(losses_cls)
        loss_iou = sum(losses_bbox)
        loss = losses_conf + losses_cls + loss_iou
        return loss, torch.stack((loss_iou,losses_conf,losses_cls,loss)).detach()

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           -1)#self.train_cfg.allowed_border = -1
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        
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

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            #if self.train_cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
            # else:
            #     label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
