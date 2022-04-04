# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from yolodet.loss.distiller_loss_utils import LDHead


class KnowledgeDistillationDetector(nn.Module):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self, model_cfg, distill_params):
                #  backbone,
                #  neck,
                #  bbox_head,
                #  teacher_config,
                #  teacher_ckpt=None,
                #  eval_teacher=True,
                #  train_cfg=None,
                #  test_cfg=None,
                #  pretrained=None):
        super().__init__()
        # self.eval_teacher = eval_teacher
        # # Build teacher model
        # if isinstance(teacher_config, str):
        #     teacher_config = mmcv.Config.fromfile(teacher_config)
        # self.teacher_model = build_detector(teacher_config['model'])
        # if teacher_ckpt is not None:
        #     load_checkpoint(
        #         self.teacher_model, teacher_ckpt, map_location='cpu')

        self.teacher = model_cfg["teacher"][0]
        #load_state_dict(self.teacher, model_cfg["teacher"][1]['model'].float().state_dict())
        self.teacher.eval()
        #self._freeze_bn(teacher=True)
        self.student = model_cfg["student"][0]
        self.teacher.model.detect.config['distiller'] = True #教师提取特征的标志位
        self.student.model.detect.config['distiller'] = False #学生提取特征的标志位
        self.distill_losses = nn.ModuleDict()
        self.student_init = False

        if self.student_init:
            t_checkpoint = _load_checkpoint('/home/yu/workspace/yoloall/yoloall/run_det/distiller/teacher.pth')
            all_name = []
            for name, v in t_checkpoint['model'].state_dict().items():
                if 'backbone' in name:
                    continue
                else:
                    all_name.append((name, v))
                    #self.stud_layer_freeze.append(name) #保留层中的bn进行冻结

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
            self._freeze_bn(student=True)
            # self.student.load_state_dict(state_dict, strict=False)

    def _freeze_bn(self, teacher=False, student=False):
        if teacher:
            for k, v in self.teacher.named_parameters():
                if 'bn' in k:
                    v.requires_grad = False
                else:
                    v.requires_grad = True
        if student:
            for k, v in self.student.named_parameters():
                if 'bn' in k:
                    v.requires_grad = False
                else:
                    v.requires_grad = True


    def forward_train(self,
                      img,
                      img_metas,
                      target,
                      compute_loss_fun,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        self.student.train()
        self.student.model.detect.config['distiller'] = False
        x = self.student.forward(img) #学生特征
        with torch.no_grad():
            self.teacher.eval()
            self.teacher.model.detect.config['distiller'] = True
            teacher_feature = self.teacher.forward(img) 
            #teacher_feature = teacher_x.split() ##仅仅拿位置信息作为软标签
            #out_teacher = self.teacher_model.bbox_head(teacher_x)# 输出len = 2, 1、各个下采样层的分类特征图  2、各个下采样的定位特征图
        losses = compute_loss_fun(x, target, img_metas, feature = teacher_feature)
        return losses


    def forward(self, epoch, img, img_metas=None, target=None, compute_loss_fun=None, **kwargs):
        if compute_loss_fun != None:
            return self.forward_train(img, img_metas, target, compute_loss_fun, **kwargs)
        else:
            return self.forward_test(epoch, img, **kwargs)


    def forward_test(self, epoch, img, **kwargs):
        teacher_feat = ()
        student_feat = ()
        self.teacher.model.detect.config['distiller'] = False #教师提取特征的标志位
        self.student.model.detect.config['distiller'] = False #学生提取特征的标志位
        #已开始蒸馏和未开始蒸馏
        #if epoch > self.distill_start:
        self.teacher.eval()
        self.student.eval()
        teach_feat = self.teacher.forward(img)
        student_feat = self.student.forward(img)
        return teach_feat, student_feat

    # def train(self, mode=True):
    #     """Set the same train mode for teacher and student model."""
    #     if self.eval_teacher:
    #         self.teacher_model.train(False)
    #     else:
    #         self.teacher_model.train(mode)
    #     super().train(mode)
