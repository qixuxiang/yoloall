# Loss functions

import torch
import torch.nn as nn
import numpy as np
from yolodet.utils.general import bbox_iou
from yolodet.utils.torch_utils import is_parallel
#detection
from yolodet.loss.yolov5_loss import compute_loss_v5
from yolodet.loss.yolov3_loss import compute_loss_v3
from yolodet.loss.yolov4_loss import compute_loss_v4
from yolodet.loss.gaussian_loss_utils import compute_gaussian_yolo_loss
from yolodet.loss.darknet_loss import compute_loss_darknet
# from yolodet.loss.yolov4_loss import build_targets_v4
from yolodet.loss.mmdet_loss import YOLOV3Head
from yolodet.loss.yolof_loss_utils import YOLOFHead
from yolodet.loss.yolox_loss_utils import YOLOXHead
from yolodet.loss.atss_loss_utils import ATSSHead
from yolodet.loss.fcos_loss_utils import FCOSHead
from yolodet.loss.centernet_loss_utils import CenterNetHead
from yolodet.loss.gfocal_loss_utills import GFLHead

#distiller
from yolodet.loss.distiller_loss_utils import LDHead

#for test
from yolodet.loss.yzf_loss_utils import YOLOXHead_yzf, ATSSHead_yzf


class ComputeLoss:
    # Compute losses
    def __init__(self, model, opt, autobalance=False):
        super(ComputeLoss, self).__init__()
        device                  = next(model.parameters()).device  # get model device
        self.hyp                = model.hyp  # hyperparameters
        self.version            = opt.train_cfg['version']
        self.model              = model.module if is_parallel(model) else model
        self.distiller          = opt.distiller['distill_enable']
        try:
            det = self.model.model[-1] # if is_parallel(model) else model.model[-1]  # yaml
        except Exception:
            #det = self.model.detect # model.module.model.detect if is_parallel(model) else model.detect  # py
            if self.distiller:
                det = self.model.student.model.detect 
            else:
                det = self.model.model.detect 
        #self.stride     = list(det.stride.numpy())
        self.nc         = det.nc
        self.na         = det.na
        self.anchors    = det.anchors
        self.nl         = det.nl
        self.stride     = list(det.stride)

        if self.version in ['yolov3','yolov5','yolov3-gaussian']:

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
        elif self.version == 'yolov4':
            anchors = []
            [anchors.extend(ans) for ans in opt.train_cfg['anchors']]
            anchors = np.array(anchors).reshape(-1,2).tolist()
            anchors_mask = []

            ans_index = 0
            for n, ans in enumerate(opt.train_cfg['anchors']):
                anchors_mask.append([])
                for an in ans[::2]:
                    anchors_mask[n].append(ans_index)
                    ans_index += 1
            
            # logger.info("yolov4 anchors >>>")
            # logger.info(anchors)
            # logger.info("yolov4 anchors_mask >>>")
            # logger.info(anchors_mask)

            # img size
            self.net_w, self.net_h = opt.train_cfg['width'], opt.train_cfg['height']
            # class
            self.num_classes  = self.nc
            # anchors
            self.anchors_org  = anchors
            self.scaled_anchors_org = [(1.0 * a_w / self.net_w, 1.0 * a_h / self.net_h) for a_w, a_h in self.anchors_org]
            self.anchors_masks = anchors_mask
            # stride
            self.layer_stride = det.stride.int().numpy().tolist()
            # use match anchor not only best match
            self.use_all_anchors = opt.yolov4['use_all_anchors']

            # weight scale
            self.object_scale    = opt.yolov4['object_scale']
            self.noobject_scale  = opt.yolov4['noobject_scale']
            self.class_scale     = opt.yolov4['class_scale']
            self.coord_scale     = opt.yolov4['coord_scale']

            # delta params
            self.cls_normalizer   = opt.yolov4['cls_normalizer']
            self.iou_normalizer   = opt.yolov4['iou_normalizer']
            self.ignore_thresh    = opt.yolov4['ignore_thresh']
            self.iou_thresh       = opt.yolov4['iou_thresh']
            self.label_smooth_eps = opt.yolov4['label_smooth_eps']
            self.y_true = 1.0 -  0.5 * self.label_smooth_eps
            self.y_false = 0.5 * self.label_smooth_eps

            # loss func
            # self.bce_loss = nn.BCELoss(reduce=False)
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            if opt.yolov4['focalloss']:
                self.bce_loss = FocalLoss(self.bce_loss)

            self.box_loss_type = opt.yolov4['box_loss_type']
            if self.box_loss_type == 'mse':
                self.box_loss_func = nn.MSELoss(reduction='none')#(reduce=False)

            elif self.box_loss_type == 'smooth':
                self.box_loss_func = nn.SmoothL1Loss(reduction='none')#(reduce=False)

            elif 'iou' in self.box_loss_type:
                self.box_loss_func = None
        
        elif self.version == 'yolo-mmdet':#
            self.head_loss = YOLOV3Head(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = self.model.strides_list if opt.transformer['transformer_enabl'] else list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yolommdet['loss_mean'],
                box_loss_type = opt.yolommdet['box_loss_type'],
                show_pos_bbox = opt.yolommdet['show_pos_bbox'],
                loss_cls_weight = opt.yolommdet['loss_cls_weight'],
                loss_conf_weight = opt.yolommdet['loss_conf_weight'],
                loss_xy_weight = opt.yolommdet['loss_xy_weight'],
                loss_wh_weight = opt.yolommdet['loss_wh_weight'],
                loss_reduction = opt.yolommdet['loss_reduction'],
                pos_iou_thr = opt.yolommdet['pos_iou_thr'],
                neg_iou_thr = opt.yolommdet['neg_iou_thr'],
                area_scale = opt.yolommdet['area_scale'],
                object_scale = opt.yolommdet['mmdet_obj_scale'],
                noobject_scale = opt.yolommdet['mmdet_noobj_scale'],
                loss_iou_weight = opt.yolommdet['loss_iou_weight']
            )

        elif self.version == 'yolox-yzf-mmdet':
            self.head_loss = YOLOXHead_yzf(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = self.model.strides_list if opt.transformer['transformer_enabl'] else list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yzfmmdet['loss_mean'],
                box_loss_type = opt.yzfmmdet['box_loss_type'],
                show_pos_bbox = opt.yzfmmdet['show_pos_bbox'],
                loss_cls_weight = opt.yzfmmdet['loss_cls_weight'],
                loss_conf_weight = opt.yzfmmdet['loss_conf_weight'],
                loss_xy_weight = opt.yzfmmdet['loss_xy_weight'],
                loss_wh_weight = opt.yzfmmdet['loss_wh_weight'],
                loss_reduction = opt.yzfmmdet['loss_reduction'],
                pos_iou_thr = opt.yzfmmdet['pos_iou_thr'],
                neg_iou_thr = opt.yzfmmdet['neg_iou_thr'],
                area_scale = opt.yzfmmdet['area_scale'],
                object_scale = opt.yzfmmdet['mmdet_obj_scale'],
                noobject_scale = opt.yzfmmdet['mmdet_noobj_scale'],
                loss_iou_weight = opt.yzfmmdet['loss_iou_weight'],
                simOTA = opt.yzfmmdet['simOTA']
            )
        elif self.version == 'yolof-mmdet':
            self.head_loss = YOLOFHead(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yolofmmdet['loss_mean'],
                box_loss_type = opt.yolofmmdet['box_loss_type'],
                show_pos_bbox = opt.yolofmmdet['show_pos_bbox'],
                loss_cls_weight = opt.yolofmmdet['loss_cls_weight'],
                loss_conf_weight = opt.yolofmmdet['loss_conf_weight'],
                loss_xy_weight = opt.yolofmmdet['loss_xy_weight'],
                loss_wh_weight = opt.yolofmmdet['loss_wh_weight'],
                loss_reduction = opt.yolofmmdet['loss_reduction'],
                pos_ignore_thr = opt.yolofmmdet['pos_ignore_thr'],
                neg_ignore_thr = opt.yolofmmdet['neg_ignore_thr'],
                area_scale = opt.yolofmmdet['area_scale'],
                object_scale = opt.yolofmmdet['mmdet_obj_scale'],
                noobject_scale = opt.yolofmmdet['mmdet_noobj_scale'],
                loss_iou_weight = opt.yolofmmdet['loss_iou_weight']
            )
        elif self.version == 'yolox-mmdet':
            self.head_loss = YOLOXHead(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yoloxmmdet['loss_mean'],
                box_loss_type = opt.yoloxmmdet['box_loss_type'],
                show_pos_bbox = opt.yoloxmmdet['show_pos_bbox'],
                loss_cls_weight = opt.yoloxmmdet['loss_cls_weight'],
                loss_conf_weight = opt.yoloxmmdet['loss_conf_weight'],
                loss_reduction = opt.yoloxmmdet['loss_reduction'],
                area_scale = opt.yoloxmmdet['area_scale'],
                object_scale = opt.yoloxmmdet['mmdet_obj_scale'],
                noobject_scale = opt.yoloxmmdet['mmdet_noobj_scale'],
                loss_iou_weight = opt.yoloxmmdet['loss_iou_weight'],
                simOTA = opt.yoloxmmdet['simOTA']
            )
        elif self.version == 'yolo-atss-mmdet':
            self.head_loss = ATSSHead(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yoloatss['loss_mean'],
                box_loss_type = opt.yoloatss['box_loss_type'],
                show_pos_bbox = opt.yoloatss['show_pos_bbox'],
                loss_cls_weight = opt.yoloatss['loss_cls_weight'],
                loss_conf_weight = opt.yoloatss['loss_conf_weight'],
                loss_xy_weight = opt.yoloatss['loss_xy_weight'],
                loss_wh_weight = opt.yoloatss['loss_wh_weight'],
                loss_reduction = opt.yoloatss['loss_reduction'],
                topk = opt.yoloatss['topk'],
                area_scale = opt.yoloatss['area_scale'],
                object_scale = opt.yoloatss['mmdet_obj_scale'],
                noobject_scale = opt.yoloatss['mmdet_noobj_scale'],
                loss_iou_weight = opt.yoloatss['loss_iou_weight']
            )
        elif self.version == 'yolo-atss-yzf-mmdet':
            self.head_loss = ATSSHead_yzf(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yoloatss['loss_mean'],
                box_loss_type = opt.yoloatss['box_loss_type'],
                show_pos_bbox = opt.yoloatss['show_pos_bbox'],
                loss_cls_weight = opt.yoloatss['loss_cls_weight'],
                loss_conf_weight = opt.yoloatss['loss_conf_weight'],
                loss_xy_weight = opt.yoloatss['loss_xy_weight'],
                loss_wh_weight = opt.yoloatss['loss_wh_weight'],
                loss_reduction = opt.yoloatss['loss_reduction'],
                topk = opt.yoloatss['topk'],
                area_scale = opt.yoloatss['area_scale'],
                object_scale = opt.yoloatss['mmdet_obj_scale'],
                noobject_scale = opt.yoloatss['mmdet_noobj_scale'],
                loss_iou_weight = opt.yoloatss['loss_iou_weight']
            ) 
        elif self.version == 'yolo-fcos-mmdet':
            self.head_loss = FCOSHead(
                num_classes = self.nc,
                anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                add = not opt.yolofcos['loss_mean'],
                box_loss_type = opt.yolofcos['box_loss_type'],
                show_pos_bbox = opt.yolofcos['show_pos_bbox'],
                loss_cls_weight = opt.yolofcos['loss_cls_weight'],
                loss_conf_weight = opt.yolofcos['loss_conf_weight'],
                loss_reduction = opt.yolofcos['loss_reduction'],
                area_scale = opt.yolofcos['area_scale'],
                object_scale = opt.yolofcos['mmdet_obj_scale'],
                noobject_scale = opt.yolofcos['mmdet_noobj_scale'],
                loss_iou_weight = opt.yolofcos['loss_iou_weight'],
                regress_ranges = opt.yolofcos['regress_ranges'],
                center_sampling = opt.yolofcos['center_sampling'],
                center_sample_radius = opt.yolofcos['center_sample_radius'],
                norm_on_bbox = opt.yolofcos['norm_on_bbox'],
                centerness_on_reg = opt.yolofcos['centerness_on_reg'],
            ) 
        elif self.version == 'yolo-center-mmdet':
            self.head_loss = CenterNetHead(
                num_classes = self.nc,
                featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                loss_weight_heatmap= opt.yolocenter['loss_weight_heatmap'],
                loss_wh_weight= opt.yolocenter['loss_weight_heatmap'],
                loss_offset_weight= opt.yolocenter['loss_weight_heatmap'],
                show_pos_bbox=opt.yolocenter['show_pos_bbox']
            )
        elif self.version == 'yolo-gfl-mmdet':
            if self.distiller:
                self.head_loss = LDHead(
                    num_classes = self.nc,
                    anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                    featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                    add = not opt.yologfl['loss_mean'],
                    box_loss_type=opt.yologfl['box_loss_type'],
                    show_pos_bbox=opt.yologfl['show_pos_bbox'],
                    loss_cls_weight=opt.yologfl['loss_cls_weight'],
                    loss_dfl_weight=opt.yologfl['loss_dfl_weight'],
                    loss_iou_weight=opt.yologfl['loss_iou_weight'],
                    loss_reduction=opt.yologfl['loss_reduction'],
                    topk=opt.yologfl['topk'],
                    reg_max=opt.yologfl['reg_max'],
                    reg_topk=opt.yologfl['reg_topk'],
                    add_mean=opt.yologfl['add_mean'],
                    area_scale=opt.yologfl['area_scale'],
                    object_scale=opt.yologfl['mmdet_obj_scale'],
                    noobject_scale=opt.yologfl['mmdet_noobj_scale'],
                    use_sigmoid=opt.yologfl['use_sigmoid'],
                    **opt.distiller['ld_param']
                )
            else:
                self.head_loss = GFLHead(
                    num_classes = self.nc,
                    anchor_generator = np.array(opt.train_cfg['anchors']).reshape(len(opt.train_cfg['anchors']),-1,2).tolist(),
                    featmap_strides = list(det.stride.numpy()), #注意多卡时的写法：model.module
                    add = not opt.yologfl['loss_mean'],
                    box_loss_type=opt.yologfl['box_loss_type'],
                    show_pos_bbox=opt.yologfl['show_pos_bbox'],
                    loss_cls_weight=opt.yologfl['loss_cls_weight'],
                    loss_dfl_weight=opt.yologfl['loss_dfl_weight'],
                    loss_iou_weight=opt.yologfl['loss_iou_weight'],
                    loss_reduction=opt.yologfl['loss_reduction'],
                    topk=opt.yologfl['topk'],
                    reg_max=opt.yologfl['reg_max'],
                    reg_topk=opt.yologfl['reg_topk'],
                    add_mean=opt.yologfl['add_mean'],
                    area_scale=opt.yologfl['area_scale'],
                    object_scale=opt.yologfl['mmdet_obj_scale'],
                    noobject_scale=opt.yologfl['mmdet_noobj_scale'],
                    use_sigmoid=opt.yologfl['use_sigmoid']
                )
        else:
            print('not sport')
            raise "error"


    def __call__(self, p, targets, img_metas=None, imgs=None, feature=None):  # predictions, targets, model

        if self.version == 'yolov5':
            return compute_loss_v5(self,p,targets)
        elif self.version == 'yolov3':
            return compute_loss_v3(self,p,targets)
        elif self.version == 'yolov4':
            return compute_loss_v4(self,p,targets)            
        elif self.version == 'yolov3-gaussian':
            return compute_gaussian_yolo_loss(self,p,targets,imgs)
        elif self.version == 'darknet':
            return compute_loss_darknet(self,p,targets)
        elif self.version == 'yolo-mmdet':
            return self.head_loss.compute_loss_mmdet(p,targets[0],targets[1],img_metas)#gt_bboxes,gt_labels,
        elif self.version == 'yolof-mmdet':
            return self.head_loss.compute_loss_yolof(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolox-mmdet':
            return self.head_loss.compute_loss_yolox(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolox-yzf-mmdet':
            return self.head_loss.compute_loss_yzf(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolo-atss-mmdet':
            return self.head_loss.compute_loss_atss(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolo-atss-yzf-mmdet':
            return self.head_loss.compute_loss_atss_yzf(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolo-fcos-mmdet':
            return self.head_loss.compute_loss_fcos(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolo-center-mmdet':
            return self.head_loss.compute_loss_center(p,targets[0],targets[1],img_metas)
        elif self.version == 'yolo-gfl-mmdet':
            if self.distiller:
                return self.head_loss.compute_loss_gfl(p,targets[0],targets[1],img_metas,feature)
            else:
                return self.head_loss.compute_loss_gfl(p,targets[0],targets[1],img_metas)
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