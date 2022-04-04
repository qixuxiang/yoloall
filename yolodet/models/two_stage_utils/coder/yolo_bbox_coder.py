# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

# from ..builder import BBOX_CODERS
# from .base_bbox_coder import BaseBBoxCoder
from .base_bbox_coder import BaseBBoxCoder

# @BBOX_CODERS.register_module()
class YOLOBBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    # @mmcv.jit(coderize=True)
    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        w_target = torch.log((w_gt / w).clamp(min=self.eps))
        h_target = torch.log((h_gt / h).clamp(min=self.eps))
        # 放缩到原feature_map上，注意(x_center_gt - x_center) / stride 这一步已经将gt_box和anchor_box放缩到一个框内了取值范围为[-1~1],加0.5的作用是，anchor保存的是相对于grid cell中心点box,而网络预测的是相对于grid cell的左上点， 因此加上0.5
        # 这一步是相当于平移到中心点所再网格点的左上点上（xy编码就是gt中心基于左上角网格点的偏移）
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp( #向左边平移：左加右减 x_center_target相当于神经网络的输出:predict
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        encoded_bboxes = torch.stack(
            [x_center_target, y_center_target, w_target, h_target], dim=-1)
        return encoded_bboxes

    # @mmcv.jit(coderize=True)
    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5) * stride #anchor 的中心点 + pred预测中心点的偏移量
        whs = (bboxes[..., 2:] -
               bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp() #anchor 的w,h * pred预测w,h的偏移量
        decoded_bboxes = torch.stack(
            (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
             whs[..., 1], xy_centers[..., 0] + whs[..., 0],
             xy_centers[..., 1] + whs[..., 1]), # -> 框坐标：左上点和右下点
            dim=-1)
        return decoded_bboxes


class YOLOIouBBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    # @mmcv.jit(coderize=True)
    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        # gt_box
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]

        #x_c y_c
        x_center_pred = (bboxes[...,0] + bboxes[...,2]) * 0.5 - 0.5 * stride
        y_center_pred = (bboxes[...,1] + bboxes[...,3]) * 0.5 - 0.5 * stride
        w_pre = bboxes[...,2] - bboxes[...,0]
        h_pre = bboxes[...,3] - bboxes[...,1]

        encoded_gt_bboxes = torch.stack(
                [x_center_gt, y_center_gt, w_gt, h_gt], dim=-1)

        encoded_pred_bboxes = torch.stack(
                [x_center_pred, y_center_pred, w_pre, h_pre], dim=-1)
        return encoded_gt_bboxes,encoded_pred_bboxes

    # 解码
    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        x_center = (bboxes[..., 2] + bboxes[..., 0]) * 0.5
        y_center = (bboxes[..., 3] + bboxes[..., 1]) * 0.5
        w_anchor = bboxes[...,2] - bboxes[...,0]
        h_anchor = bboxes[...,3] - bboxes[...,1]
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
        w_pre = pred_bboxes[...,2].exp() * w_anchor
        h_pre = pred_bboxes[...,3].exp() * h_anchor
        decoded_bboxes = torch.stack((x_center_pred,y_center_pred,w_pre,h_pre),dim=-1)
        return decoded_bboxes


class YOLOV5BBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        # 所有值都是原图坐标值，故需要/ stride，变成特征图尺度
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # w h编码是 gt bbox的宽高/anchor宽高，加上log，保证大小bbox的宽高预测值不会差距很大(log可以拉近差距)，保证训练过程更加稳定而已
        w_target = torch.log((w_gt / w).clamp(min=self.eps))
        h_target = torch.log((h_gt / h).clamp(min=self.eps))
        # xy编码就是gt中心基于左上角网格点的偏移
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        encoded_bboxes = torch.stack(
            [x_center_target, y_center_target, w_target, h_target], dim=-1)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        #assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4

        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]

        # Get outputs x, y
        # 由于mmdetection的anchor已经偏移了0.5，故*2的操作要放在外面
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * 2 * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * 2 * stride + y_center
        # yolov5中正常情况应该是
        # x_center_pred = (pred_bboxes[..., 0] * 2. - 0.5 + grid[:, 0]) * stride  # xy
        # y_center_pred = (pred_bboxes[..., 1] * 2. - 0.5 + grid[:, 1]) * stride  # xy

        # wh也需要sigmoid，然后乘以4来还原
        w_pred = (pred_bboxes[..., 2].sigmoid() * 2) ** 2 * w
        h_pred = (pred_bboxes[..., 3].sigmoid() * 2) ** 2 * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)

        return decoded_bboxes
