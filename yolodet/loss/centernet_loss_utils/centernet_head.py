# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
# from mmcv.cnn import bias_init_with_prob, normal_init
# from mmcv.ops import batched_nms
# from mmcv.runner import force_fp32

from mmdet.core import multi_apply
# from mmdet.models import HEADS, build_loss
from .gaussian_target import gaussian_radius, gen_gaussian_target, get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat
# from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
#                                      transpose_and_gather_feat)
# from .base_dense_head import BaseDenseHead
# from .dense_test_mixins import BBoxTestMixin

from ..mmdet_loss_utils import GaussianFocalLoss, L1Loss
from yolodet.utils.general import xywh2xyxy
from abc import ABCMeta
import logging
logger = logging.getLogger(__name__)


# @HEADS.register_module()
class CenterNetHead(nn.Module, metaclass=ABCMeta):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # def __init__(self,
    #              in_channel,
    #              feat_channel,
    #              num_classes,
    #              loss_center_heatmap=dict(
    #                  type='GaussianFocalLoss', loss_weight=1.0),
    #              loss_wh=dict(type='L1Loss', loss_weight=0.1),
    #              loss_offset=dict(type='L1Loss', loss_weight=1.0),
    #              train_cfg=None,
    #              test_cfg=None,
    #              init_cfg=None):
    #     super(CenterNetHead, self).__init__(init_cfg)
    def __init__(self,
                num_classes=None,
                featmap_strides=None,
                loss_weight_heatmap=None,
                loss_wh_weight=None,
                loss_offset_weight=None,
                show_pos_bbox=None,
                **kwargs):
        super(CenterNetHead, self).__init__()
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.show_pos_bbox = show_pos_bbox
        # self.heatmap_head = self._build_head(in_channel, feat_channel,
        #                                      num_classes)
        # self.wh_head = self._build_head(in_channel, feat_channel, 2)
        # self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = GaussianFocalLoss(loss_weight=1.0)#build_loss(loss_center_heatmap)
        self.loss_wh = L1Loss(loss_weight=0.1) #build_loss(loss_wh)
        self.loss_offset = L1Loss(loss_weight=1.0) #build_loss(loss_offset)

        self.train_cfg = dict(topk=100, local_maximum_kernel=3, max_per_img=100)
        # self.test_cfg = test_cfg
        # self.fp16_enabled = False

    # def _build_head(self, in_channel, feat_channel, out_channel):
    #     """Build head for each branch."""
    #     layer = nn.Sequential(
    #         nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(feat_channel, out_channel, kernel_size=1))
    #     return layer

    # def init_weights(self):
    #     """Initialize weights of the head."""
    #     bias_init = bias_init_with_prob(0.1)
    #     self.heatmap_head[-1].bias.data.fill_(bias_init)
    #     for head in [self.wh_head, self.offset_head]:
    #         for m in head.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid() #类别预测
        wh_pred = self.wh_head(feat) #中心点x,y的预测
        offset_pred = self.offset_head(feat) #偏置距离框边据距离（w/2，h/2）
        return center_heatmap_pred, wh_pred, offset_pred

    #@force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def compute_loss_center(self,
             pred_maps,
            #  center_heatmap_preds,
            #  wh_preds,
            #  offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        loss_item = []
        num_imgs = pred_maps[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in pred_maps]

        pred_maps = [torch.split(pred_map,[2,2,self.num_classes],1) for pred_map in pred_maps] 

        for level in range(len(pred_maps)):

            wh_pred, offset_pred,center_heatmap_pred = pred_maps[level]
            # assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1 #feature_map
            # center_heatmap_pred = center_heatmap_preds[0]
            # wh_pred = wh_preds[0]
            # offset_pred = offset_preds[0]

            target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                        center_heatmap_pred.shape,
                                                        img_metas[0]['pad_shape'])

            center_heatmap_target = target_result['center_heatmap_target'] #获取class的信息
            wh_target = target_result['wh_target'] #
            offset_target = target_result['offset_target']
            wh_offset_target_weight = target_result['wh_offset_target_weight']

            #show_pos_bbox
            if self.show_pos_bbox:
                import cv2
                for info in range(len(img_metas)):
                    img_shape = img_metas[info]['img_shape'] #(h,w)
                    pad_shape = img_metas[info]['pad_shape']
                    t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
                    l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
                    img = cv2.imread(img_metas[info]['filename'])
                    img = cv2.resize(img,(img_shape[1],img_shape[0]))
                    
                    xywh_info = wh_target.nonzero()[wh_target.nonzero()[:,0] == info] #中心点一定不为(0,0)
                    #assert xywh_info.shape[0] // 2 == gt_bboxes[info].shape[0], 'target number not equal!'
                    gt_boxes = []
                    for _index in range(len(xywh_info) // 2):
                        center_x = (xywh_info[_index][-1] + offset_target[info, 0, xywh_info[_index][-2].item(), xywh_info[_index][-1].item()]) * self.featmap_strides[level]
                        center_y = (xywh_info[_index][-2] + offset_target[info, 1, xywh_info[_index][-2].item(), xywh_info[_index][-1].item()]) * self.featmap_strides[level]
                        w = wh_target[info, 0, xywh_info[_index][-2].item(), xywh_info[_index][-1].item()] * self.featmap_strides[level]
                        h = wh_target[info, 1, xywh_info[_index][-2].item(), xywh_info[_index][-1].item()] * self.featmap_strides[level]
                        gt_boxes.append(torch.stack([center_x, center_y, w, h],-1))
                    gt_boxes = xywh2xyxy(torch.stack(gt_boxes, 0))

                    for idx in range(len(gt_boxes)):
                        pos_gt_bbox = gt_boxes[idx]
                        #pos_bbox = gt_bboxes[info][idx]
                        cv2.rectangle(img,(int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)),(int(pos_gt_bbox[2]-l_pad),int(pos_gt_bbox[3]-t_pad)),(0,255,0),1)
                        #cv2.putText(img, str(labels[idx].item()), (int(pos_gt_bbox[0]-l_pad),int(pos_gt_bbox[1]-t_pad)+5), 0, 1, (255,0,0), 1)
                        #cv2.rectangle(img,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
                    
                    cv2.imshow("pos_bbox",img)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        raise Exception


            # Since the channel of wh_target and offset_target is 2, the avg_factor
            # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
            loss_center_heatmap = self.loss_center_heatmap(
                center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
            loss_wh = self.loss_wh(
                wh_pred,
                wh_target,
                wh_offset_target_weight,
                avg_factor=avg_factor * 2)
            loss_offset = self.loss_offset(
                offset_pred,
                offset_target,
                wh_offset_target_weight,
                avg_factor=avg_factor * 2)
            
            loss_item.append((loss_center_heatmap,loss_wh,loss_offset))

        loss_wh_sum = sum([i[1] for i in loss_item])
        loss_offset_sum = sum([i[2] for i in loss_item])
        loss_center_heatmap_sum = sum([i[0] for i in loss_item])
        loss_sum = loss_wh_sum + loss_offset_sum + loss_center_heatmap_sum
        return loss_sum, torch.stack((loss_wh_sum, loss_offset_sum, loss_center_heatmap_sum,loss_sum)).detach()

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w]) #分类的feature_map
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w]) #中心点相对于边缘框的偏移即 最左边和最上边的距离
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w]) # 
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):#对每张图片进行遍历
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2 #gt框的中心点在下采样n倍率下的坐标x
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2 #gt框的中心点在下采样n倍率下的坐标y
            gt_centers = torch.cat((center_x, center_y), dim=1) 

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int() #gt的中心点
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio #在该stride下的feature_map下的h
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio #在该stride下的feature_map下的w
                radius = gaussian_radius([scale_box_h, scale_box_w], #该函数的功能为：假定最小值交叠的IOU为0.7，计算3种情况下的预测框与gt框左上角和右下角最小的圆半径
                                         min_overlap=0.3) 
                radius = max(0, int(radius))
                ind = gt_label[j] #选取当前gt框对应的label
                gen_gaussian_target(center_heatmap_target[batch_id, ind], #选取对应的图像上的对应的特征图 -> 在对应类别的特征图上建二维的高斯模，以此来得到heat_map图，最亮的点就是目标的中心点
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w #gt框的w,h在该feature_map下的尺寸大小，把它附在了该特征图层的中心点上
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h 

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int #在当前下采样倍率下中心点所在框的左上角点的偏移(返回时偏移量必须乘以下采样倍率)
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1 # 中心点的权重

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    # @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def _get_box_single(self,
                   pred_map,
                #    center_heatmap_preds,
                #    wh_preds,
                #    offset_preds,
                   #img_metas,
                   num_imgs=None,
                   level_idx=None,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        input_shape = [self.featmap_strides[level_idx] * pred_map.shape[-2], self.featmap_strides[level_idx] * pred_map.shape[-1]]
        pred_map = torch.split(pred_map,[2,2,self.num_classes],1)   #[ for every_level_map in pred_map]
        # assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        result_list = []
        for img_id in range(num_imgs):
            wh_pred, offset_pred, center_heatmap_pred = pred_map
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_pred[img_id:img_id + 1, ...],
                    wh_pred[img_id:img_id + 1, ...],
                    offset_pred[img_id:img_id + 1, ...],
                    input_shape,
                    rescale=rescale,
                    with_nms=with_nms))
        # bbox = [i[0] for i in result_list]
        # label = [i[1] for i in result_list]
        #result = torch.cat([torch.stack(bbox, 0),torch.stack(label, 0).unsqueeze(-1)], -1)
        return result_list #torch.stack(bbox, 0), torch.stack(label, 0).unsqueeze(-1)

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
                           input_shape,
                           rescale=False,
                           with_nms=False):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            input_shape,
            k=100,
            kernel=3)#self.test_cfg.local_maximum_kernel

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)
        bbox_info = torch.cat([det_bboxes,det_labels.unsqueeze(-1)], 1)
        # batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
        #                                                          [2, 0, 2, 0]]
        # det_bboxes[..., :4] -= batch_border

        # if rescale:
        #     det_bboxes[..., :4] /= det_bboxes.new_tensor(
        #         img_meta['scale_factor'])

        # if with_nms:
        #     det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
        #                                               self.test_cfg)
        return bbox_info #det_bboxes, det_labels #

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels
