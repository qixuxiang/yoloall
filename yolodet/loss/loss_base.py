# Loss functions

import torch
import torch.nn as nn
import numpy as np
from yolodet.utils.general import bbox_iou
from yolodet.utils.torch_utils import is_parallel
from yolodet.loss.yolov5_loss import compute_loss_v5
from yolodet.loss.yolo3_loss import compute_loss_v3
from yolodet.loss.darknet_loss import compute_loss_darknet
# from yolodet.loss.yolov4_loss import build_targets_v4
from yolodet.loss.mmdet_loss import YOLOV3Head


class ComputeLoss:
    # Compute losses
    def __init__(self, model, opt, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        self.hyp = model.hyp  # hyperparameters
        self.version = opt.train_cfg['version']
        self.model = model
        # self.add = False
        try:
            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        except:
            det = model.module.model.detect if is_parallel(model) else model.model.detect  # Detect() module
        #self.stride = list(np.array(det.stride))
        self.show_pos_img = opt.yolommdet['show_pos_bbox']
        self.nc = det.nc
        self.na = det.na
        self.anchors = det.anchors
        self.nl = det.nl

        if self.version in ['v3','v5']:

            # Define criteria
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))
            # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
            self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))  # positive, negative BCE targets

            # Focal loss
            g = self.hyp['fl_gamma']  # focal loss gamma
            if g > 0:
                BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, model.gr, autobalance
            self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
            self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
            # for k in 'na', 'nc', 'nl', 'anchors':
            #     setattr(self, k, getattr(det, k))

        elif self.version == 'mmdet':
            self.head_loss = YOLOV3Head(
                num_classes = self.nc,
                anchor_generator = [i.reshape(-1,2).tolist() for i in np.array(opt.train_cfg['anchors'])], 
                featmap_strides = list(np.array(det.stride)),
                add = not opt.yolommdet['loss_mean'],
                box_loss_type = opt.yolommdet['box_loss_type'],
                show_pos_img = opt.yolommdet['show_pos_bbox'],
                loss_cls_weight = opt.yolommdet['loss_cls_weight'],
                loss_conf_weight = opt.yolommdet['loss_conf_weight'],
                loss_xy_weight = opt.yolommdet['loss_xy_weight'],
                loss_wh_weight = opt.yolommdet['loss_wh_weight'],
                loss_reduction = opt.yolommdet['loss_reduction'],
                pos_iou_thr = opt.yolommdet['pos_iou_thr'],
                neg_iou_thr = opt.yolommdet['neg_iou_thr'],
                area_scale = opt.yolommdet['area_scale'],
                obj_scale = opt.yolommdet['mmdet_obj_scale'],
                noobj_scale = opt.yolommdet['mmdet_noobj_scale'],
                loss_iou_weight = opt.yolommdet['loss_iou_weight'],
            )
        else:
            print('not sport')
            raise "error"


    def __call__(self, p, targets, img_metas=None):  # predictions, targets, model

        if self.version == 'v5':
            return compute_loss_v5(self,p,targets)
        elif self.version == 'v3':
            return compute_loss_v3(self,p,targets)
        elif self.version == 'darknet':
            return compute_loss_darknet(self,p,targets)
        elif self.version == 'mmdet':
            return self.head_loss.compute_loss_mmdet(p,targets[0],targets[1],img_metas)#gt_bboxes,gt_labels,
        else:
            print('erro!')


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element
            
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss