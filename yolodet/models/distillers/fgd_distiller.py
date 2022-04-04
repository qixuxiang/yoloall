import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from yolodet.loss.distiller_loss_utils import FeatureLoss
from collections import OrderedDict
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict

logger = logging.getLogger(__name__)

class DetectionDistiller(nn.Module):
    def __init__(self, model_cfg, distill_params):  # model, input channels, number of classes
        super(DetectionDistiller, self).__init__()
        self.stud_layer_freeze =[]
        self.teacher = model_cfg["teacher"][0]
        self.teacher.eval()
        #self._freeze_bn(teacher=True)
        self.student = model_cfg["student"][0]
        self.distill_losses = nn.ModuleDict()
        self.distill_start = distill_params['distill_start']
        self.distill_layer = []
        self.student_init = True
        self.distill_step = distill_params['distill_step']
        self.teacher_param = {}
        self.student_param = {}

        self.teacher_dict = {}
        for k, v in self.teacher.named_parameters():
            self.teacher_dict[k] = v.shape
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        
        distill_cfg = []
        for k, v in self.student.named_parameters():
            _dic = {}
            distill_param = {**distill_params['distill_cfg']}
            if 'FPN' in k:
                self.stud_layer_freeze.append(k)
                if 'bn' not in k: #and 'FPN0' not in k and 'FPN1' not in k:
                    if k[6:10] not in self.distill_layer:
                        self.distill_layer.append(k[6:10]) #要蒸馏的layer
                    distill_param['name'] = k[:-7].replace('.','_')+'_loss'
                    distill_param['student_channels'] = v.size()[0]
                    distill_param['teacher_channels'] =  self.teacher_dict[k][0]
                    _dic['methods'] = [distill_param]
                    _dic['output_hook'] = True
                    #print('layer_name:{}, student_size:{}, teacher_size:{}'.format(k, v.size(), self.teacher_dict[k]))
                    _dic['student_module'] = k[:-7]
                    _dic['teacher_module'] = k[:-7]
                    distill_cfg.append(_dic)
        self.distill_cfg = distill_cfg
        
        if self.student_init:
            t_checkpoint = _load_checkpoint(distill_params['teacher_weights'])
            all_name = []
            for name, v in t_checkpoint['model'].state_dict().items():
                if 'backbone' in name:
                    continue
                else:
                    all_name.append((name, v))
                    #self.stud_layer_freeze.append(name) #保留层中的bn进行冻结

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
            #self._freeze_bn(student=True)
            # self.student.load_state_dict(state_dict, strict=False)


        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module, output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer(student_module, output)
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc['student_module'].replace('.','_')
            teacher_module = 'teacher_' + item_loc['teacher_module'].replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module)
            teacher_modules[item_loc['teacher_module']].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc['student_module']].register_forward_hook(hook_student_forward)

            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                self.distill_losses[loss_name] = FeatureLoss(**item_loss)
    
    def _freeze_bn(self, teacher=False, student=False):
        if teacher:
            for k, v in self.teacher.named_parameters():
                if 'bn' in k:
                    v.requires_grad = False
                    #这边与学生网络的参数做比较
                    if self.stud_layer_freeze != None and k in self.stud_layer_freeze:
                        self.teacher_param[k] = v
                
        elif student:
            for k, v in self.student.named_parameters():
                if 'bn' in k and k in self.stud_layer_freeze: #如果学生网络蒸馏层中存在BN层就进行冻结[仅仅冻结要蒸馏的层]
                    v.requires_grad = False
                    #这边与教师网络做比较
                    self.student_param[k] = v
                else:
                    v.requires_grad = True
        else:
            pass

    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])

    def forward(self, epoch, img, img_metas=None, target=None, compute_loss_fun=None, **kwargs):
        if compute_loss_fun != None:
            return self.forward_train(epoch, img, img_metas, target, compute_loss_fun, **kwargs)
        else:
            return self.forward_test(epoch, img, **kwargs)

    def forward_train(self, epoch, img, img_metas, target, compute_loss_fun, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        teach_loss,stud_loss,distill_loss = 0,0,0
        teach_loss_items,stud_loss_items = (),()
        distill_student_loss = {}
        distill_loss_items = {}
        loss_items = {}
        for layer_name in self.distill_layer:
            distill_loss_items[layer_name] = 0

        #teach_loss, teach_loss_items = compute_loss_fun(teach_pred, target, img_metas)
        #设置开始蒸馏的epoch
        #if epoch > self.distill_start:
        #如果学生网络在对比特征层中有bn层，就把学生的bn层给冻结
        #self._freeze_bn(student=True)
        stud_pred = self.student.forward(img)
        stud_loss, stud_loss_items = compute_loss_fun(stud_pred, target, img_metas)
        # for key in self.stud_layer_freeze:
        #     if 'bn' in key:
        #         print(self.teacher_param[key].equal(self.student_param[key]))

        if epoch > self.distill_start:
            with torch.no_grad():
                self.teacher.eval()
                #self._freeze_bn(teacher=True)
                teach_pred = self.teacher.forward(img)

            buffer_dict = dict(self.named_buffers()) #named_buffers是将学生网络中的所有层的参数进行了保存
            for item_loc in self.distill_cfg[::-1]:
                
                student_module = 'student_' + item_loc['student_module'].replace('.','_')
                teacher_module = 'teacher_' + item_loc['teacher_module'].replace('.','_')
                
                student_feat = buffer_dict[student_module]
                teacher_feat = buffer_dict[teacher_module]

                for item_loss in item_loc['methods']:
                    loss_name = item_loss['name']
                    # print('loss_layer:{}, student_featue_size:{}, teacher_feature_size:{}'.format(loss_name,student_feat.shape,teacher_feat.shape))
                    # print(self.distill_losses[loss_name].align)
                    distill_student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat, target[0], img_metas)
            #按照FPN结构进行loss重组
            for key, value in distill_student_loss.items():
                distill_loss += value
                distill_loss_items[key[6:10]] += value.detach()
                #按阶段优化
                # if epoch >= self.distill_step[-1]:#大于30，则计算全部FPN的loss
                #         distill_loss += value
                #         distill_loss_items[key[6:10]] += value.detach()
                # elif epoch < self.distill_step[-1] and epoch >= self.distill_step[-2]:#大于等于20小于30的则计算前三个FPN
                #     if 'FPN3' in key or 'FPN2' in key or 'FPN1' in key:
                #         distill_loss += value
                #         distill_loss_items[key[6:10]] += value.detach()
                # elif epoch < self.distill_step[-2] and epoch >= self.distill_step[-3]:
                #     if 'FPN3' in key or 'FPN2' in key:#小于10的则计算前二个FPN
                #         distill_loss += value
                #         distill_loss_items[key[6:10]] += value.detach()
                # else:
                #     if 'FPN3' in key:
                #         distill_loss += value
                #         distill_loss_items[key[6:10]] += value.detach()

                # for _item in self.distill_layer:
                #     if _item in key:
                #         distill_loss_items[_item] += value.detach()

        loss = stud_loss + distill_loss #teach_loss
        # distill_loss_items['total'] = loss.detach()
        loss_items = {'teacher': teach_loss_items, 'student': stud_loss_items, 'distiller': distill_loss_items}
        return loss, loss_items

    def forward_test(self, epoch, img, **kwargs):
        teacher_feat = ()
        student_feat = ()
        #已开始蒸馏和未开始蒸馏
        #if epoch > self.distill_start:
        student_feat = self.student.forward(img)
        teach_feat = self.teacher.forward(img)

        return teach_feat, student_feat
