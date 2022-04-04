# Copyright (c) OpenMMLab. All rights reserved.
import torch

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
# from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from yolodet.models.two_stage_utils.transforms import bbox2result, bbox2roi #, build_assigner, build_sampler
from yolodet.models.two_stage_utils.roi_extractors import SingleRoIExtractor
from yolodet.models.two_stage_utils.roi_heads import Shared2FCBBoxHead, Shared4Conv1FCBBoxHead
from yolodet.models.two_stage_utils.assigners import MaxIoUAssigner
from yolodet.models.two_stage_utils.samplers import RandomSampler

# @HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):#
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BaseRoIHead, self).__init__(init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # if shared_head is not None:
        #     shared_head.pretrained = pretrained
        #     self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = MaxIoUAssigner(pos_iou_thr=self.train_cfg['assigner']['pos_iou_thr'],
                                                neg_iou_thr=self.train_cfg['assigner']['neg_iou_thr'],
                                                min_pos_iou=self.train_cfg['assigner']['min_pos_iou'],
                                                ignore_iof_thr=self.train_cfg['assigner']['ignore_iof_thr'],
                                                match_low_quality=self.train_cfg['assigner']['match_low_quality'])
            self.bbox_sampler = RandomSampler(num = self.train_cfg['sampler']['num'],
                                            pos_fraction = self.train_cfg['sampler']['pos_fraction'],
                                            neg_pos_ub= self.train_cfg['sampler']['neg_pos_ub'],
                                            add_gt_as_proposals= self.train_cfg['sampler']['add_gt_as_proposals'])

            # self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            # self.bbox_sampler = build_sampler(
            #     self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        # self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # self.bbox_head = build_head(bbox_head)
        #**bbox_roi_extractor
        self.bbox_roi_extractor = SingleRoIExtractor(roi_layer = bbox_roi_extractor['roi_layer'],
                                                     out_channels = bbox_roi_extractor['out_channels'],
                                                     featmap_strides = bbox_roi_extractor['featmap_strides'])

        self.bbox_head = Shared2FCBBoxHead(in_channels=bbox_head['in_channels'],
                                           fc_out_channels=bbox_head['fc_out_channels'],
                                           roi_feat_size=bbox_head['roi_feat_size'],
                                           num_classes=bbox_head['num_classes'],
                                           bbox_coder=bbox_head['bbox_coder'],
                                           reg_class_agnostic=bbox_head['reg_class_agnostic'],
                                           loss_cls=bbox_head['loss_cls'],
                                           loss_bbox=bbox_head['loss_bbox'])

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feature_level = len(x)
        #map_list = [[] for i in range(feature_level)]
        num_imgs = len(img_metas)
        each_level = [[] for i in range(feature_level)]
        for idx in range(feature_level):
            for idy in range(len(proposal_list[idx])):
                each_level[idx].append(proposal_list[idx][idy].shape[0])
        # for img_num in range(num_imgs):
        #     h,w,_ = img_metas[img_num]['pad_shape']
        #     for level in range(feature_level):
        #         tensor_tmp = proposal_list[img_num][proposal_list[img_num][:,5] == level][:,:4]
        #         each_level[level].append(tensor_tmp.shape[0])
        #         map_list[level].append(torch.cat([tensor_tmp[:,0].clamp(0,w)[None].T,tensor_tmp[:,1].clamp(0,h)[None].T,tensor_tmp[:,2].clamp(0,w)[None].T,tensor_tmp[:,3].clamp(0,h)[None].T],1))
        roi_num = [sum(level) for level in each_level]
        # assign gts and sample proposals
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        # if self.with_bbox or self.with_mask:
        losses = [dict() for i in range(feature_level)]
        for map_level in range(feature_level):
            proposal_level_list = proposal_list[map_level] #map_list[map_level]
            sampling_results = []
            for i in range(num_imgs):# 对每张图片的proposal（1000个框）与gt_bboxes进行iou匹配，并随机采样512样本（正负样本比例按照1:3进行采样）
                assign_result = self.bbox_assigner.assign(
                    proposal_level_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_level_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[x[map_level][i][None]]#[lvl_feat[i][None] for lvl_feat in x] #传入单个feature_map
                )
                sampling_results.append(sampling_result)

            
            # bbox head forward and loss
            if self.with_bbox:
                bbox_results = self._bbox_forward_train([x[map_level]], sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas, map_level)
                losses[map_level].update(bbox_results['loss_bbox'])
                losses[map_level]['roi_num'] = roi_num[map_level]
            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(x, sampling_results,
                                                        bbox_results['bbox_feats'],
                                                        gt_masks, img_metas)
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, level):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois, level=level) #最后一层不进行，只有前四层
        # bbox_feats = self.bbox_roi_extractor(x, rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)# 进行分类和回归

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas,map_level):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results]) #将正样本和负样本进行cat在一起 并将所有的图片采样框cat在一起
        bbox_results = self._bbox_forward(x, rois, level=map_level)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        levels = len(x)
        img_nums = len(img_metas)
        bbox_batch_results = []
        for img_num in range(img_nums):
            det_bbox = []
            det_label = []
            for level in range(levels):
                pre_bbox, pre_label = self.simple_test_bboxes(
                    [x[level][img_num][None]], [img_metas[img_num]], [proposal_list[level][img_num]], self.test_cfg, rescale=rescale,level=level)
                det_bbox.extend(pre_bbox)
                det_label.extend(pre_label)
            
            det_bboxes = torch.cat(det_bbox)
            det_labels = torch.cat(det_label)

            if len(det_bboxes) and len(det_labels):
                bbox_results = [
                    bbox2result(det_bboxes[i], det_labels[i],
                                self.bbox_head.num_classes)
                    for i in range(len([det_bboxes]))
                ]
                bbox_batch_results.extend(bbox_results)
        
        if not self.with_mask:
            return bbox_batch_results
        else:
            # 添加mask需要更改下面代码
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
