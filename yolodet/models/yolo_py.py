import os
import sys
import math
import logging
import argparse
from pathlib import Path
from copy import deepcopy


from yolodet.models.experimental import MixConv2d, CrossConv
from yolodet.utils.general import make_divisible, check_file, set_logging
from yolodet.utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

from yolodet.models.common_py import *

#from yolodet.models.two_stage_utils import YOLOAnchorGenerator
from yolodet.loss.yolox_loss_utils import YOLOXHead
from yolodet.loss.yzf_loss_utils import YOLOXHead_yzf
from yolodet.loss.atss_loss_utils import ATSSHead
from yolodet.loss.fcos_loss_utils import FCOSHead
from yolodet.loss.centernet_loss_utils import CenterNetHead
from yolodet.loss.gfocal_loss_utills import GFLHead

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

logger = logging.getLogger(__name__)

import pdb


class Model(nn.Module):
    def __init__(self, cfg, ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        
        version         = cfg.train_cfg['version']
        anchors         = cfg.train_cfg['anchors']
        model_name      = cfg.train_cfg['model']
        # if cfg.get(version, None) != None:
        self.loss_param = eval('cfg.'+ cfg.version_info[version]) #cfg[version]
        # two_stage  = cfg.two_stage['two_stage_enabel']

        try:
            exec(f'from yolodet.models.py_model.torch_{model_name} import YOLO')
        except:
            exec(f'from yolodet.models.py_model.{model_name} import YOLO')

        self.names = [str(i) for i in range(nc)]  # default names
        self.model = eval('YOLO')(nc, anchors, ch)
        

        # Build strides, anchors
        m = self.model.detect  # Detect()
        m.version = version
        if version in ['yolo-fcos-mmdet']:
            m.config = self.loss_param
            m.decoupling_init_fcos_layers()
        elif version in ['yolo-center-mmdet']:
            m.decoupling_init_center_layers()
        elif version in ['yolo-gfl-mmdet']:
            m.config = self.loss_param
            m.decoupling_init_gfl_layers()
            if not self.loss_param.get('use_sigmoid', True): #Gfocal_v2配置
                m.box_coder = GFLHead(num_classes=nc, anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(), **self.loss_param)
        else:
            m.coupling_init_detect_layers()
        #获取stride
        if isinstance(m, Detect) or m.version in ['yolov3-gaussian']:
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            #self._initialize_biases()  # only run once 仅仅初始化检测头的偏置
            logger.info('Model Strides: %s' % m.stride.tolist())
        #获取解码、loss_type的参数
        if version == 'yolox-mmdet':
            m.box_coder = YOLOXHead(num_classes=nc,anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist())
        elif version == 'yolox-yzf-mmdet':
            m.box_coder = YOLOXHead_yzf(num_classes=nc,anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist())
        elif version == 'yolo-atss-mmdet':
            m.box_coder = ATSSHead(num_classes=nc,anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist())
        elif version == 'yolo-fcos-mmdet':
            m.box_coder = FCOSHead(num_classes=nc,anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist())
        elif version == 'yolo-center-mmdet':
            m.box_coder = CenterNetHead(num_classes=nc, featmap_strides=self.stride.tolist())
        elif version == 'yolo-gfl-mmdet':
            m.box_coder = GFLHead(num_classes=nc, anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist(),**self.loss_param)
        else:
            pass
        #m.prior_generator = YOLOAnchorGenerator(strides=self.stride.numpy().tolist(), base_sizes = np.array(anchors).reshape(len(anchors),-1,2).tolist())
        # # 更新detect函数中的参数
        # m = self.model.detect
        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        t = time_synchronized()
        x = self.model(x)
        dt = (time_synchronized() - t) * 100

        # y, dt = [], []  # outputs

        # for m in self.model:
        #     if m.f != -1:  # if not from previous layer
        #         x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        #     if profile:
        #         o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                
        #         for _ in range(10):
        #             _ = m(x)
        #         dt.append((time_synchronized() - t) * 100)
        #         print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

        #     x = m(x)  # run
        #     y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.c = fuse_conv_and_bn(m.c, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
