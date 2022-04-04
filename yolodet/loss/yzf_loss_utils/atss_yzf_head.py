import torch
import torch.nn as nn
import torch.nn.functional as F
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
class ATSSHead_yzf(nn.Module, metaclass=ABCMeta):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
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
                topk=None,
                area_scale=None,
                object_scale=None,
                noobject_scale=None,
                loss_iou_weight=None,
                 **kwargs):
        super(ATSSHead_yzf, self).__init__()
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
        self.bbox_coder = YOLOIouBBoxCoder() #YOLOV5BBoxCoder() if self.box_loss_type in ['iou','mse'] else YOLOIouBBoxCoder() #DeltaXYWHBBoxCoder()#
        self.prior_generator = YOLOAnchorGenerator(strides=featmap_strides, base_sizes=anchor_generator)

        self.loss_cls = CrossEntropyLoss(use_sigmoid=True,reduction=loss_reduction,loss_weight=loss_cls_weight)#1.0
        #FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)

        #self.reg_decoded_bbox = True
        # self.loss_cls = FocalLoss(use_sigmoid=True,
        #                           gamma=2.0,
        #                           alpha=0.25,
        #                           #reduction=loss_reduction,
        #                           loss_weight=1.0)
        # self.loss_bbox = GIoULoss(#reduction=loss_reduction,
        #                           loss_weight=1.0)
        self.loss_conf = CrossEntropyLoss(use_sigmoid=True, reduction=loss_reduction, loss_weight=loss_conf_weight)#1.0
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


    # def loss_single(self, anchors, pred_maps, strides, featmap_sizes,#cls_score, bbox_pred, centerness, 
    #                 labels,label_weights, bbox_targets, num_total_samples):
    #     """Compute loss of a single scale level.

    #     Args:
    #         cls_score (Tensor): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W).
    #         bbox_pred (Tensor): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         anchors (Tensor): Box reference for each scale level with shape
    #             (N, num_total_anchors, 4).
    #         labels (Tensor): Labels of each anchors with shape
    #             (N, num_total_anchors).
    #         label_weights (Tensor): Label weights of each anchor with shape
    #             (N, num_total_anchors)
    #         bbox_targets (Tensor): BBox regression targets of each anchor
    #             weight shape (N, num_total_anchors, 4).
    #         num_total_samples (int): Number os positive samples that is
    #             reduced over all GPUs.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     num_imgs = len(pred_maps)
    #     pred_maps = pred_maps.view(num_imgs,self.per_anchor,self.num_attrib,featmap_sizes[0],featmap_sizes[1]).permute(0,1,3,4,2).contiguous().reshape(num_imgs,-1,self.num_attrib)

    #     cls_score = pred_maps[..., 5:].reshape(-1, self.num_classes)
    #     bbox_pred = pred_maps[..., :4].reshape(-1, 4)
    #     centerness = pred_maps[..., 4].reshape(-1)

    #     anchors = anchors.reshape(-1, 4)
    #     # cls_score = cls_score.permute(0, 2, 3, 1).reshape(
    #     #     -1, self.cls_out_channels).contiguous()
    #     # bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     # centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)

    #     # classification loss
    #     loss_cls = self.loss_cls(
    #         cls_score, labels, label_weights, avg_factor=num_total_samples)

    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((labels >= 0)
    #                 & (labels < bg_class_ind)).nonzero().squeeze(1)

    #     if len(pos_inds) > 0:
    #         pos_bbox_targets = bbox_targets[pos_inds]
    #         pos_bbox_pred = bbox_pred[pos_inds]
    #         pos_anchors = anchors[pos_inds]
    #         pos_centerness = centerness[pos_inds]

    #         centerness_targets = self.centerness_target(
    #             pos_anchors, pos_bbox_targets)
    #         pos_decode_bbox_pred = self.bbox_coder.decode(
    #             pos_anchors, pos_bbox_pred, strides)

    #         # regression loss
    #         loss_bbox = self.iou_loss(
    #             pos_decode_bbox_pred,
    #             pos_bbox_targets,
    #             weight=centerness_targets,
    #             avg_factor=1.0)

    #         # centerness loss
    #         loss_centerness = self.loss_centerness(
    #             pos_centerness,
    #             centerness_targets,
    #             avg_factor=num_total_samples)

    #     else:
    #         loss_bbox = bbox_pred.sum() * 0
    #         loss_centerness = centerness.sum() * 0
    #         centerness_targets = bbox_targets.new_tensor(0.)

    #     return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()
    def loss_single(self, stride, pred_map, target_map, neg_map, freature_size, gt_bbox, anchor_box):
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



    def compute_loss_atss_yzf(self,
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
        num_imgs = len(img_metas)
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

        target_maps_list, neg_maps_list, gt_bbox_list, anchor_box_list = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=self.num_classes)
        
        losses_cls, losses_conf, loss_iou = multi_apply(
                self.loss_single, self.featmap_strides, pred_maps, target_maps_list, neg_maps_list, featmap_sizes, gt_bbox_list, anchor_box_list)

        losses_cls = sum(losses_cls)
        losses_conf = sum(losses_conf)
        loss_iou = sum(loss_iou)# * self.loss_iou_weight

        if not self.add:
            losses_cls /= num_imgs
            losses_conf /= num_imgs
            loss_iou /= num_imgs

        loss = losses_cls + losses_conf + loss_iou
        return loss, torch.stack((loss_iou,losses_conf,losses_cls,loss)).detach()


        # if cls_reg_targets is None:
        #     return None

        # (anchor_list, labels_list, label_weights_list, bbox_targets_list,
        #  bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        # num_total_samples = reduce_mean(
        #     torch.tensor(num_total_pos, dtype=torch.float,
        #                  device=device)).item()
        # num_total_samples = max(num_total_samples, 1.0)

        # losses_cls, losses_bbox, loss_centerness,\
        #     bbox_avg_factor = multi_apply(
        #         self.loss_single,
        #         anchor_list,
        #         pred_maps,
        #         self.featmap_strides,
        #         featmap_sizes,
        #         # cls_scores,
        #         # bbox_preds,
        #         # centernesses,
        #         labels_list,
        #         label_weights_list,
        #         bbox_targets_list,
        #         num_total_samples=num_total_samples)

        # bbox_avg_factor = sum(bbox_avg_factor)
        # bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        # losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        # losses_conf = sum(loss_centerness)
        # losses_cls = sum(losses_cls)
        # loss_iou = sum(losses_bbox)
        # loss = losses_conf + losses_cls + loss_iou
        # return loss, torch.stack((loss_iou,losses_conf,losses_cls,loss)).detach()

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
        results = multi_apply(
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
        
        all_target_maps, all_neg_maps, gt_bbox_maps, anchor_bbox_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps) == len(gt_bbox_maps) == len(anchor_bbox_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)
        gt_bbox_list = images_to_levels(gt_bbox_maps, num_level_anchors)
        anchor_box_list = images_to_levels(anchor_bbox_maps, num_level_anchors)

        return target_maps_list, neg_maps_list, gt_bbox_list, anchor_box_list

        # # no valid anchors
        # if any([labels is None for labels in all_labels]):
        #     return None
        # # sampled anchors of all images
        # num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        # num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # # split targets to a list w.r.t. multiple levels
        # anchors_list = images_to_levels(all_anchors, num_level_anchors)
        # labels_list = images_to_levels(all_labels, num_level_anchors)
        # label_weights_list = images_to_levels(all_label_weights,
        #                                       num_level_anchors)
        # bbox_targets_list = images_to_levels(all_bbox_targets,
        #                                      num_level_anchors)
        # bbox_weights_list = images_to_levels(all_bbox_weights,
        #                                      num_level_anchors)
        # return (anchors_list, labels_list, label_weights_list,
        #         bbox_targets_list, bbox_weights_list, num_total_pos,
        #         num_total_neg)

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
        anchor_strides = []
        for i in range(len(num_level_anchors)):#anchors为各个下采样4种倍率的feature map上的绘制的框
            anchor_strides.append(torch.tensor(self.featmap_strides[i]).repeat(num_level_anchors[i]))
        anchor_strides = torch.cat(anchor_strides).to(gt_bboxes.device)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
                #sampling_result类中包含self.pos_inds(正样本索引)、self.neg_inds(负样本索引)、self.pos_bboxes(正样本索引)、self.neg_bboxes(负样本框)、self.pos_is_gt(正样本标志位)、self.num_gts(正样本数量)、self.pos_assigned_gt_inds(正样本目标框index)、self.pos_gt_bboxes(正样本目标框)
        target_map = flat_anchors.new_zeros(flat_anchors.size(0), self.num_attrib-4) #构建target_map: 对所有的anchor框 (x1,y1,x2,y2,conf,n_class)
        anchor_bbox = flat_anchors.new_zeros(flat_anchors.size(0), 4) #anchor_xywh
        gt_bbox = flat_anchors.new_zeros(flat_anchors.size(0), 4) #gt_xywh

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
        
        neg_map = flat_anchors.new_zeros(
            flat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1   #无目标的no_object

        return target_map, neg_map, gt_bbox, anchor_bbox


        # num_valid_anchors = anchors.shape[0]
        # bbox_targets = torch.zeros_like(anchors)
        # bbox_weights = torch.zeros_like(anchors)
        # labels = anchors.new_full((num_valid_anchors, ),
        #                           self.num_classes,
        #                           dtype=torch.long)
        # label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds
        # if len(pos_inds) > 0:
        #     if self.reg_decoded_bbox:
        #         pos_bbox_targets = sampling_result.pos_gt_bboxes
        #     else:
        #         pos_bbox_targets = self.bbox_coder.encode(
        #             sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

        #     bbox_targets[pos_inds, :] = pos_bbox_targets
        #     bbox_weights[pos_inds, :] = 1.0
        #     if gt_labels is None:
        #         # Only rpn gives gt_labels as None
        #         # Foreground is the first class since v2.5.0
        #         labels[pos_inds] = 0
        #     else:
        #         labels[pos_inds] = gt_labels[
        #             sampling_result.pos_assigned_gt_inds]
        #     #if self.train_cfg.pos_weight <= 0:
        #     label_weights[pos_inds] = 1.0
        #     # else:
        #     #     label_weights[pos_inds] = self.train_cfg.pos_weight
        # if len(neg_inds) > 0:
        #     label_weights[neg_inds] = 1.0

        # # map up to original set of anchors
        # if unmap_outputs:
        #     num_total_anchors = flat_anchors.size(0)
        #     anchors = unmap(anchors, num_total_anchors, inside_flags)
        #     labels = unmap(
        #         labels, num_total_anchors, inside_flags, fill=self.num_classes)
        #     label_weights = unmap(label_weights, num_total_anchors,
        #                           inside_flags)
        #     bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        #     bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        # return (anchors, labels, label_weights, bbox_targets, bbox_weights,
        #         pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
