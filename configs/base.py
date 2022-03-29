import os
import torch
gpu_num = torch.cuda.device_count()
train_cfg = {
    "data"          :"data_demo",
    "version"       :"mmdet",
    "cfg"           :"yolov5s.yaml",#
    "weights"       :"",
    "anchors"       :[[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]],
    "epochs"        :300,
    "batch_size"    :8 * gpu_num,
    "hight"         :576,
    "width"         :960,
    "step"          :[8, 16, 32],#, 64
    "rect"          :False,
    "resume"        :False,
    "nosave"        :False,
    "notest"        :False,
    "project"       :"runs/train",
    "name"          :"exp",
    "workers"       :1,
    "noautoanchor"  :True,
    "evolve"        :False,
    "bucket"        :'',
    "cache_images"  :False,
    "image_weights"  :False,
    "device"        :'',
    "multi_scale"   :False,
    "single_cls"    :False,
    "adam"          :False,
    "sync_bn"       :False,
    "log_imgs"      :16,
    "log_artifacts" :False,
    "exist_ok"      :False,
    "max_norm"      :0,
    "max_one_norm"  :0,

    #Debug模式
    "debug"         :0,
    "show_pos_img"  :1,   
}

hyp = {
    'lr0': 0.001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    'lrf': 0.2,  # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': 0.9,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # optimizer weight decay 5e-4
    'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
    'warmup_momentum': 0.9,  # warmup initial momentum
    'warmup_bias_lr': 0.0,  # warmup initial bias lr v5为0.1 其他为0
    'box': 0.01,  # box loss gain
    'cls': 0.5,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 1.0,  # obj loss gain (scale with pixels)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.20,  # IoU training threshold
    'anchor_t': 4.0,  # anchor-multiple threshold
    # 'anchors': 3,  # anchors per output layer (0 to ignore)
    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
    'hsv_h': 0.1,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
    'gaussianblur': 0.2,   # gaussianblur
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.0,  # image translation (+/- fraction)
    'scale': 0.0,  # image scale (+/- gain)
    'shear': 0.0,  # image shear (+/- deg)
    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,  # image flip up-down (probability)
    'fliplr': 0.0,  # image flip left-right (probability)
    'mosaic': 0.0,  # image mosaic (probability)
    'mixup': 0.0,  # image mixup (probability)
}