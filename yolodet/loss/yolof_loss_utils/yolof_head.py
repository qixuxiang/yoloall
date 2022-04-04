import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
# from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
#                       normal_init)
# from mmcv.runner import force_fp32

from mmdet.core import anchor_inside_flags, multi_apply, reduce_mean, unmap
# from ..builder import HEADS
# from .anchor_head import AnchorHead
from yolodet.loss.mmdet_loss_utils import CrossEntropyLoss, MSELoss, CIoULoss, GIoULoss, FocalLoss
from yolodet.utils.general import bbox_iou
from yolodet.models.two_stage_utils.coder import YOLOBBoxCoder, YOLOIouBBoxCoder, YOLOV5BBoxCoder, DeltaXYWHBBoxCoder
from yolodet.models.two_stage_utils import YOLOAnchorGenerator
from yolodet.models.two_stage_utils.assigners import UniformAssigner
from yolodet.models.two_stage_utils.samplers import PseudoSampler
import logging
logger = logging.getLogger(__name__)

INF = 1e8


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:# t 的原shape:[32, 720, 18, 30] -> [32, 18, 30, 720] -> [32, 540, 720]
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


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


class YOLOFHead(nn.Module, metaclass=ABCMeta):
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Default 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Default 4.
        norm_cfg (dict): Dictionary to construct and config norm layer.
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
                pos_ignore_thr=None,
                neg_ignore_thr=None,
                area_scale=None,
                object_scale=None,
                noobject_scale=None,
                loss_iou_weight=None,
                 **kwargs):
        super(YOLOFHead, self).__init__()
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
        #pos_iou_thr  neg_iou_thr
        self.assigner = UniformAssigner(pos_ignore_thr=pos_ignore_thr,neg_ignore_thr=neg_ignore_thr)
        self.sampler = PseudoSampler()
        self.sampling = False
        self.pos_weight = -1
        #YOLOBBoxCoder
        self.bbox_coder = YOLOIouBBoxCoder() #YOLOV5BBoxCoder() if self.box_loss_type in ['iou','mse'] else YOLOIouBBoxCoder() #DeltaXYWHBBoxCoder()#
        self.prior_generator = YOLOAnchorGenerator(strides=featmap_strides, base_sizes=anchor_generator)

        self.loss_cls = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_cls_weight)#1.0
        # self.loss_cls = FocalLoss(use_sigmoid=True,
        #                           gamma=2.0,
        #                           alpha=0.25,
        #                           #reduction=loss_reduction,
        #                           loss_weight=1.0)
        # self.loss_bbox = GIoULoss(#reduction=loss_reduction,
        #                           loss_weight=1.0)
        self.loss_conf = CrossEntropyLoss(use_sigmoid=True, reduction=loss_reduction, loss_weight=loss_conf_weight)#1.0
        self.loss_xy = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_xy_weight)#2.0
        self.loss_wh = MSELoss(reduction=loss_reduction, loss_weight=loss_wh_weight)#2.0
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
            'pos_ignore_thr',
            'neg_ignore_thr',
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

    #@force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def compute_loss_yolof(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

            #  cls_scores,
            #  bbox_preds,
            #  gt_bboxes,
            #  gt_labels,
            #  img_metas,
            #  gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (batch, num_anchors * num_classes, h, w)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (batch, num_anchors * 4, h, w)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(pred_maps) == 1
        assert self.prior_generator.num_levels == 1
        # 改变特征图的size
        # N, _, H, W = pred_maps[-1].shape
        # pred_result = pred_maps[-1].view(N, -1, H, W,  self.num_attrib) # -> [32, 9, 18, 30, 85]
        # bbox_reg = pred_result[...,:4].reshape(N, -1, H, W) # [32, 36, 18, 30]
        # normalized_cls_score = pred_result[...,5:].reshape(N, -1, H, W) # [32, 729, 18, 30]

        # conf = pred_result[...,4].reshape(N,-1, H, W)

        # objectness = pred_result[...,4].reshape(N,-1, 1, H, W)
        # cls_score = pred_result[...,5:].reshape(N, -1, self.num_classes, H, W)
        # normalized_cls_score = cls_score + objectness - torch.log(
        #     1. + torch.clamp(cls_score.exp(), max=INF) +
        #     torch.clamp(objectness.exp(), max=INF))
        # normalized_cls_score = normalized_cls_score.view(N, -1, H, W)


        # cls_scores = [normalized_cls_score] #原来 [32, 720, 18, 30]
        # bbox_preds = [bbox_reg] #原来 [32, 36, 18, 30]
        
        # conf_scores = [conf]
        num_imgs = len(img_metas)

        device = pred_maps[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in pred_maps]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(len(img_metas))]
        #新加
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        # The output level is always 1
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        #cls_scores_list = levels_to_images(cls_scores) #shape -> [feature_map_h * feature_map_w, anchor_num * class_num]
        #bbox_preds_list = levels_to_images(bbox_preds) #shape -> [feature_map_h * feature_map_w, anchor_num * 4(wywh)]

        label_channels = self.num_classes#self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            #cls_scores_list,
            #bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        target_maps_list, neg_maps_list, gt_bbox_list, anchor_box_list = cls_reg_targets

        losses_cls, losses_conf, loss_iou = multi_apply(
                self.loss_single_ciou, self.featmap_strides, pred_maps, target_maps_list, neg_maps_list, featmap_sizes, gt_bbox_list, anchor_box_list)

        losses_cls = sum(losses_cls)
        losses_conf = sum(losses_conf)
        loss_iou = sum(loss_iou)# * self.loss_iou_weight

        if not self.add:
            losses_cls /= num_imgs
            losses_conf /= num_imgs
            loss_iou /= num_imgs

        loss = losses_cls + losses_conf + loss_iou
        return loss, torch.stack((loss_iou,losses_conf,losses_cls,loss)).detach()


        # (batch_labels, batch_label_weights, num_total_pos, num_total_neg,
        #  batch_confs, batch_bbox_weights, batch_pos_predicted_boxes,
        #  batch_target_boxes) = cls_reg_targets

        # flatten_labels = batch_labels.reshape(-1)

        # batch_label_weights = batch_label_weights.reshape(-1)
        # batch_confs = batch_confs.reshape(-1)

        # cls_score = cls_scores[0].permute(0, 2, 3,  #现在才用cls_scores
        #                                   1).reshape(-1, self.num_classes)

        # conf_score = conf_scores[0].permute(0, 2, 3,  #现在才用cls_scores
        #                                   1).reshape(-1, 1)

        # num_total_samples = (num_total_pos +
        #                      num_total_neg) if self.sampling else num_total_pos
        # num_total_samples = reduce_mean(
        #     cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()

        # # classification loss
        # loss_cls = self.loss_cls(
        #     cls_score,
        #     flatten_labels,
        #     batch_label_weights,
        #     avg_factor=num_total_samples)

        # loss_conf = self.loss_conf(
        #     conf_score,
        #     batch_confs.type(torch.long),

        # )
        # # regression loss
        # if batch_pos_predicted_boxes.shape[0] == 0:
        #     # no pos sample
        #     loss_bbox = batch_pos_predicted_boxes.sum() * 0
        # else:
        #     loss_bbox = self.loss_bbox(
        #         batch_pos_predicted_boxes,
        #         batch_target_boxes,
        #         batch_bbox_weights.float(),
        #         avg_factor=num_total_samples)
        
        # loss_total = loss_cls + loss_bbox + loss_conf
        # return loss_total, torch.stack((loss_bbox, loss_conf, loss_cls, loss_total)).detach()#dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

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


    def get_targets(self,
                    #cls_scores_list,
                    #bbox_preds_list,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores_list (list[Tensor])： Classification scores of
                each image. each is a 4D-tensor, the shape is
                (h * w, num_anchors * num_classes).
            bbox_preds_list (list[Tensor])： Bbox preds of each image.
                each is a 4D-tensor, the shape is (h * w, num_anchors * 4).
            anchor_list (list[Tensor]): Anchors of each image. Each element of
                is a tensor of shape (h * w * num_anchors, 4).
            valid_flag_list (list[Tensor]): Valid flags of each image. Each
               element of is a tensor of shape (h * w * num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - batch_labels (Tensor): Label of all images. Each element \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - batch_label_weights (Tensor): Label weights of all images \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        # anchor number of multi levels
        num_level_anchors = [anchor_list[0].size()[0]]
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            #bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        
        all_target_maps, all_neg_maps, gt_bbox_maps, anchor_bbox_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps) == len(gt_bbox_maps) == len(anchor_bbox_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)
        gt_bbox_list = images_to_levels(gt_bbox_maps, num_level_anchors)
        anchor_box_list = images_to_levels(anchor_bbox_maps, num_level_anchors)
        return target_maps_list, neg_maps_list, gt_bbox_list, anchor_box_list


        # (all_labels, all_label_weights, pos_inds_list, neg_inds_list,
        #  sampling_results_list) = results[:5]
        # rest_results = list(results[5:])  # user-added return values
        # # no valid anchors
        # if any([labels is None for labels in all_labels]):
        #     return None
        # # sampled anchors of all images
        # num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        # num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # #按照batch进行cat
        # batch_labels = torch.stack(all_labels, 0)
        # batch_label_weights = torch.stack(all_label_weights, 0)
        # #按照batch进行cat
        # res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        # for i, rests in enumerate(rest_results):  # user-added return values
        #     rest_results[i] = torch.cat(rests, 0)

        # return res + tuple(rest_results)

    def _get_targets_single(self,
                            #bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            bbox_preds (Tensor): Bbox prediction of the image, which
                shape is (h * w ,4)
            flat_anchors (Tensor): Anchors of the image, which shape is
                (h * w * num_anchors ,4)
            valid_flags (Tensor): Valid flags of the image, which shape is
                (h * w * num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels (Tensor): Labels of image, which shape is
                    (h * w * num_anchors, ).
                label_weights (Tensor): Label weights of image, which shape is
                    (h * w * num_anchors, ).
                pos_inds (Tensor): Pos index of image.
                neg_inds (Tensor): Neg index of image.
                sampling_result (obj:`SamplingResult`): Sampling result.
                pos_bbox_weights (Tensor): The Weight of using to calculate
                    the bbox branch loss, which shape is (num, ).
                pos_predicted_boxes (Tensor): boxes predicted value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
                pos_target_boxes (Tensor): boxes target value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           -1)#self.train_cfg.allowed_border
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        target_map = anchors.new_zeros(anchors.size(0), self.num_attrib-4) #构建target_map: 对所有的anchor框 (x1,y1,x2,y2,conf,n_class)
        anchor_bbox = anchors.new_zeros(anchors.size(0), 4)
        gt_bbox = anchors.new_zeros(anchors.size(0), 4)
        neg_map = anchors.new_zeros(anchors.size(0), dtype=torch.uint8)

        # decoded bbox
        #decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds, self.featmap_strides[-1] if isinstance(self.featmap_strides, list) else self.featmap_strides)
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore,   #decoder_bbox_preds
            None if self.sampling else gt_labels)

        pos_bbox_weights = assign_result.get_extra_property('pos_idx')
        pos_predicted_boxes = assign_result.get_extra_property('pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes) #做object_loss 取：sampling_result.pos_inds 为对应anchor的index  conf_bboxes[sampling_result.pos_inds, 0] = 1
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
                self.featmap_strides[-1] if isinstance(self.featmap_strides, list) else self.featmap_strides)
            
        target_map[sampling_result.pos_inds, 0] = 1
        gt_labels_one_hot = F.one_hot(
                gt_labels.long(), num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
                gt_labels_one_hot = gt_labels_one_hot * (
                    1 - self.one_hot_smoother
                ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 1:] = gt_labels_one_hot[
                sampling_result.pos_assigned_gt_inds] #将target后面的几个放置到target_map的后面 object
        
        neg_map[sampling_result.neg_inds] = 1   #无目标的no_object

        return target_map, neg_map, gt_bbox, anchor_bbox

        
        #无用参数
        # bbox_preds = bbox_preds.reshape(-1, 4)
        # bbox_preds = bbox_preds[inside_flags, :]
        # pos_conf_bboxes = bbox_preds.new_zeros(bbox_preds.size(0),1) #
        # neg_conf_bboxes = bbox_preds.new_zeros(bbox_preds.size(0), dtype=torch.uint8)


        # num_valid_anchors = anchors.shape[0]
        # labels = anchors.new_full((num_valid_anchors, ),
        #                           self.num_classes,
        #                           dtype=torch.long)
        # label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds
        # if len(pos_inds) > 0:
        #     if gt_labels is None:
        #         # Only rpn gives gt_labels as None
        #         # Foreground is the first class since v2.5.0 #这边是将正负样本的index的label_weight设置为1，即mask用于计算object和noobeject(前后背景)的损失
        #         labels[pos_inds] = 0
        #     else:
        #         labels[pos_inds] = gt_labels[
        #             sampling_result.pos_assigned_gt_inds]
        #     if self.pos_weight <= 0:#self.train_cfg.pos_weight
        #         label_weights[pos_inds] = 1.0 
        #     else:
        #         label_weights[pos_inds] = self.pos_weight #self.train_cfg.pos_weight
        # if len(neg_inds) > 0:
        #     label_weights[neg_inds] = 1.0
        # pos_conf_bboxes[sampling_result.pos_inds] = 1 #有目标框置信度
        # neg_conf_bboxes[sampling_result.neg_inds] = 1 #无目标置信度
        # # map up to original set of anchors
        # if unmap_outputs:
        #     num_total_anchors = flat_anchors.size(0)
        #     labels = unmap(
        #         labels, num_total_anchors, inside_flags,
        #         fill=self.num_classes)  # fill bg label
        #     label_weights = unmap(label_weights, num_total_anchors,
        #                           inside_flags)

        # return (labels, label_weights, pos_inds, neg_inds, sampling_result,
        #         pos_conf_bboxes, pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)
