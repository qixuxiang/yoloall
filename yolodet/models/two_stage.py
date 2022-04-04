# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
# from ..builder import DETECTORS, build_backbone, build_head, build_neck
from yolodet.models.two_stage_utils.roi_heads import StandardRoIHead
from yolodet.models.two_stage_utils.base import BaseDetector
from abc import ABCMeta#, abstractmethod
from mmcv.runner import BaseModule #, auto_fp16

from yolodet.models.common_py import Conv
from yolodet.utils.general import tsh_batch_non_max_suppression

# @DETECTORS.register_module()
class TwoStageDetector(BaseModule, metaclass=ABCMeta):#BaseDetector
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 one_stage,
                 opt,
                 neck=None,
                 rpn_head=None,
                 roi_head=dict(
                            bbox_roi_extractor=dict(
                                type='SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign', output_size=(7,7), sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[8, 16, 32]),

                            bbox_head=dict(
                                type='Shared2FCBBoxHead',
                                in_channels=256,
                                fc_out_channels=256,#1024
                                roi_feat_size=7,
                                num_classes=14, #aicity 14类
                                bbox_coder=dict(
                                    type='DeltaXYWHBBoxCoder',
                                    target_means=[0., 0., 0., 0.],
                                    #target_stds=[0.1, 0.1, 0.2, 0.2]
                                    target_stds=[1, 1, 1, 1]),
                                reg_class_agnostic=False,
                                loss_cls=dict(
                                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, reduction='sum'),
                                loss_bbox=dict(type='L1Loss', loss_weight=1.0, reduction='sum')),

                            train_cfg=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.5,
                                    neg_iou_thr=0.5,
                                    min_pos_iou=0.5,
                                    match_low_quality=False,
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=512,
                                    pos_fraction=0.25,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=False),# 不把gt_box作为候选区域
                                pos_weight=-1,
                                debug=False),

                            test_cfg=dict(
                                score_thr=0.05,
                                nms=dict(type='nms', iou_threshold=0.5),
                                    max_per_img=100)
                                ),
                #  train_cfg=None,
                #  test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone = one_stage
        self.feat_channels = opt.two_stage['feat_channels']
        try:
            self.backbone.model.detect.two_stage = True
        except:
            self.backbone.model[-1].two_stage = True

        # 通过前向或得参数
        with torch.no_grad():
            detections_list, pred_list, x_b_list  = self.backbone.forward(torch.zeros(1, 3, 256, 256))
            in_channels = [xb.shape[1] for xb in x_b_list]   
      
        self.neck = nn.ModuleList([Conv(ch, self.feat_channels, 1, 1, False) for ch in in_channels])

        if roi_head is not None:
            self.roi_head = StandardRoIHead(
                bbox_roi_extractor=roi_head['bbox_roi_extractor'],
                bbox_head=roi_head['bbox_head'],
                mask_roi_extractor=None,
                mask_head=None,
                shared_head=None,
                train_cfg=roi_head['train_cfg'],
                test_cfg=roi_head['test_cfg'],
                pretrained=None,
                init_cfg=None
            )

        self.train_cfg = roi_head['train_cfg']
        self.test_cfg = roi_head['test_cfg']
        self.loss_mean = opt.two_stage['loss_mean']
        self.stride = [8,16,32]

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img) #下采样[4,8,16,32] -> FPN 4个输出
        if self.with_neck:
            x = self.neck(x) # 5个输出
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
    
    # forward_once backbone提取特征
    def forward_once(self,
                     imgs,
                     augment=False,
                     epoch=0,
                     train=True
                        ):
        proposal, pred_list, feature_list = self.backbone(imgs, augment=augment)
        proposal_list = []
        if epoch > 2 and train:  #warm_up完后进入二阶段训练 
            for proposal_data in proposal:
                detections, detections_roi = tsh_batch_non_max_suppression(
                            prediction=proposal_data.detach(), 
                            conf_thres=0.3,#self.cls_conf_thresh, 
                            nms_thres=0.3,#self.iou_thres, 
                            all_nms=True,
                            )
                proposal_list.append(detections_roi)

                # level_list = torch.cat([torch.full([proposal[num_level].shape[1]],num_level) for num_level in range(len(proposal))],0)[None].T                     
                # proposals = torch.cat(proposal,1)# 将各个feature_map上预测的框和框的置信度进行cat
                # proposals = torch.cat((proposals[...,:5],level_list.repeat(len(img_metas),1,1).to(device)),-1)
                # # 将多有框进行iou剔除，返回的为框的index和iou的分数
                # proposal_box = [xywh2xyxy(proposals[num_img,...]) for num_img in range(len(img_metas))]
                # proposal_list =[proposal_box[img_idx][proposal_box[img_idx][:,4] > 0.3] for img_idx in range(len(proposal_box))]

                # from yolodet.utils.general import box_iou 
                # #gt_box进行对预测框iou匹配  
                # proposal_list = []                      
                # for gt_num in range(len(proposal_box)):
                #     if gt_boxes[gt_num].shape[0] == 0:
                #         proposal_list.append(proposal_box[gt_num][torch.tensor([random.random() < 1000/proposal_box[gt_num].shape[0] for i in range(proposal_box[gt_num].shape[0])]),:])
                #     else:
                #         overlaps = box_iou(proposal_box[gt_num], gt_boxes[gt_num]) #shape 为(peopsoal数量,gt_box) -> (18144,7)
                #         values,indexs = overlaps.max(1) #取每行的最大值
                #         proposal_list.append(torch.cat((proposal_box[gt_num][values > 0.3,:], values[values > 0.3][None].T),1))

        return proposal, proposal_list, pred_list, feature_list


    def forward(self,  #forward_train
                img,
                # feature_map,              
                # proposal_list,#img,
                img_metas,
                gt_bboxes=None,
                gt_labels=None,
                gt_bboxes_ignore=None,
                gt_masks=None,
                proposals=None,
                train=True,
                augment=False,
                epoch=0,
                **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        detections_proposal, proposal_list, pred_list, feature_map = self.forward_once(img, augment, epoch, train) #backbone提取特征
        if len(proposal_list) == 0 and train:
            return pred_list, None, None
        #x = self.extract_feat(img)
        # extract_feat = []
        # for level in range(len(feature_map)):
        #     extract_feat.append(self.neck[level](feature_map[level]))
        # x = extract_feat
        # x = [self.neck(feat) for feat in feature_map]      
        # losses = dict()

        # # RPN forward and loss
        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                       self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         gt_bboxes,
        #         gt_labels=None,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg,
        #         **kwargs)
        #     losses.update(rpn_losses)
        # else:
        #     proposal_list = proposals
        
        #show pre box 按照feature_map显示目标框
        if 0:
            import cv2
            import copy
            for img_num in range(len(img_metas)):
                img_shape = img_metas[img_num]['img_shape'] #(h,w)
                pad_shape = img_metas[img_num]['pad_shape']
                t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
                l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
                img = cv2.imread(img_metas[img_num]['filename'])
                img = cv2.resize(img,(img_shape[1],img_shape[0]))
                proposal_boxes = torch.cat([proposal_list[level][img_num][:,:4] for level in range(len(proposal_list))],0) #proposal_list[img_num][:,:4]
                image = copy.copy(img)
                for pos_bbox in proposal_boxes:             
                    # for pos_bboxes in bboxes_index[anchors_map]:
                    #     for pos_bbox in pos_bboxes:
                    image = cv2.rectangle(image,(int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)),(int(pos_bbox[2]-l_pad),int(pos_bbox[3]-t_pad)),(0,0,255),1)
                    #image = cv2.putText(image, str(int(anchors_unique[anchors_map].item())), (int(pos_bbox[0]-l_pad),int(pos_bbox[1]-t_pad)), 0, 1, (0,0,255), 1)
                cv2.imshow("pos_bbox",image)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    raise Exception

        if train:
            loss_roi = 0
            logstrs_roi = ['roi layer:']
            extract_feat = []
            for level in range(len(feature_map)):
                extract_feat.append(self.neck[level](feature_map[level]))
            x = extract_feat
            #train
            # 计算ROI 参数：x：原特征图层<所有图片合在一起的>，img_metas：图片信息，proposal_list：候选框区域<每个图片的候选区域>
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            #losses.update(roi_losses)

            num_imgs = len(img_metas)
            for every_stride in range(len(roi_losses)):
                try:
                    loss_roi += roi_losses[every_stride]['loss_bbox']
                    loss_roi += roi_losses[every_stride]['loss_cls']
                    logstrs_roi.append("s: {:2d}, loss_cls_mean: {:6.3f}, loss_bbox_mean: {:6.3f}, roi_nums: {:3d}, acc: {:6.4f}"
                        .format(int(self.stride[every_stride]), float(roi_losses[every_stride]['loss_cls'].cpu()) / num_imgs, float(roi_losses[every_stride]['loss_bbox'].cpu()) / num_imgs, roi_losses[every_stride]['roi_num'], float(roi_losses[every_stride]['acc'].cpu())
                        ))
                except:
                    pass

            if self.loss_mean:
                loss_roi = loss_roi / num_imgs

            return pred_list, loss_roi, logstrs_roi 
        #test
        else:
            return detections_proposal, None, None
            #return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=True)#rescale



    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )