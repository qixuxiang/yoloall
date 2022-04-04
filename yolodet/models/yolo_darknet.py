# from utils.google_utils import *
import torch
import math
import torch.nn as nn
from yolodet.models.model_utils.layers import *
from yolodet.models.model_utils.parse_config import *
from yolodet.utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

ONNX_EXPORT = False


def create_modules(module_defs, nc, anchors):
    # Constructs module list of layer blocks from module configuration in module_defs

    #img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    output_filters = [module_defs[0]['channels']]  # input channels
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    detect_layer = []
    detech_ch = []
    
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'emb':
                modules.add_module('activation', F.normalize())
            elif mdef['activation'] == 'logistic':
                modules.add_module('activation', nn.Sigmoid())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())

        elif mdef['type'] == 'deformableconvolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('DeformConv2d', DeformConv2d(output_filters[-1],
                                                       filters,
                                                       kernel_size=k,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       stride=stride,
                                                       bias=not bn,
                                                       modulation=True))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())
                
        elif mdef['type'] == 'dropout':
            p = mdef['probability']
            modules = nn.Dropout(p)

        elif mdef['type'] == 'avgpool':
            modules = GAP()

        elif mdef['type'] == 'silence':
            filters = output_filters[-1]
            modules = Silence()

        elif mdef['type'] == 'scale_channels':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleChannel(layers=layers)

        elif mdef['type'] == 'sam':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleSpatial(layers=layers)

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'local_avgpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            avgpool = nn.AvgPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('AvgPool2d', avgpool)
            else:
                modules = avgpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'route2':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat2(layers=layers)

        elif mdef['type'] == 'route3':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat3(layers=layers)

        elif mdef['type'] == 'route_lhalf':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])//2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'reorg':  # yolov3-spp-pan-scale
            filters = 4 * output_filters[-1]
            modules.add_module('Reorg', Reorg())

        elif mdef['type'] == 'yolo':
            detect_layer.append(i)
            detech_ch.append(output_filters[-1])
            routs.append(i)
            
            # yolo_index += 1
            # stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides
            # if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):  # P5, P4, P3 strides
            #     stride = [32, 16, 8]
            # layers = mdef['from'] if 'from' in mdef else []
            # modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
            #                     nc=mdef['classes'],  # number of classes
            #                     img_size=img_size,  # (416, 416)
            #                     yolo_index=yolo_index,  # 0, 1, 2...
            #                     layers=layers,  # output layers
            #                     stride=stride[yolo_index])

            # # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # try:
            #     j = layers[yolo_index] if 'from' in mdef else -1
            #     bias_ = module_list[j][0].bias  # shape(255,)
            #     bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
            #     #bias[:, 4] += -4.5  # obj
            #     bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
            #     bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
            #     module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
                
            #     #j = [-2, -5, -8]
            #     #for sj in j:
            #     #    bias_ = module_list[sj][0].bias
            #     #    bias = bias_[:modules.no * 1].view(1, -1)
            #     #    bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)
            #     #    bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
            #     #    module_list[sj][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            # except:
            #     print('WARNING: smart bias initialization failure.')

        # elif mdef['type'] == 'jde':
        #     yolo_index += 1
        #     stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides
        #     if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):  # P5, P4, P3 strides
        #         stride = [32, 16, 8]
        #     layers = mdef['from'] if 'from' in mdef else []
        #     modules = JDELayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
        #                         nc=mdef['classes'],  # number of classes
        #                         img_size=img_size,  # (416, 416)
        #                         yolo_index=yolo_index,  # 0, 1, 2...
        #                         layers=layers,  # output layers
        #                         stride=stride[yolo_index])

        #     # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
        #     try:
        #         j = layers[yolo_index] if 'from' in mdef else -1
        #         bias_ = module_list[j][0].bias  # shape(255,)
        #         bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
        #         #bias[:, 4] += -4.5  # obj
        #         bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
        #         bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
        #         module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
        #     except:
        #         print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    #creat Detect layer
    module_list.append(Detect(nc = nc, anchors=anchors, ch=detech_ch))
    module_list[-1].f = detect_layer

    routs_binary = [False] * (i + 2)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), dim=5, nnx_enable=True):  # detection layer
        super(Detect, self).__init__()
        self.ch = ch
        self.nc = nc  # number of classes
        self.no = nc + dim  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.export = False  # onnx export
        self.stride = None  # strides computed during build

        self.version = 'yolov3'

        self.two_stage = False
        self.nnx_enable = nnx_enable
        # self.prior_generator = None #YOLOAnchorGenerator(strides=self.stride, base_sizes = np.array(anchors).reshape(len(anchors),-1,2).tolist())
        # self.box_coder = YOLOV5BBoxCoder()#YOLOIouBBoxCoder()

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output

        if self.two_stage:
            return self.forward_two_stage(x)

        if self.export:
            if self.nnx_enable:
                for i in range(self.nl):
                    x[i] = self.m[i](x[i])  # conv
                return x
            else:
                return x

        self.training |= self.export

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if 'mmdet' in self.version and self.training:
                pass

            else:
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                # from einops import rearrange
                # x[i] = rearrange(x[i], 'b (na no) h w -> b na h w no', na=self.na, no=self.no).contiguous()

                if not self.training:  # inference
                    if self.version == 'yolov5':
                        y = x[i].sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    elif self.version in ['yolov3','yolov4','yolo-mmdet','yolof-mmdet','vid-yolo-mmdet']:# 'mmdet',
                        y = x[i]
                        y[..., 0:2] =  (y[..., 0:2].sigmoid() + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = y[..., 2:4].exp() * self.anchor_grid[i]  # wh
                        y[..., 4:] = y[..., 4:].sigmoid()
                    
                    elif self.version in ['yolox-mmdet','yolox-yzf-mmdet']:
                        y = self.box_coder._get_box_single(pred_map=x[i],num_imgs=bs,level_idx=i)

                    if self.version in ['yolox-mmdet','yolox-yzf-mmdet']:
                        z.append(y)
                    else:
                        z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def forward_two_stage(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output

        x_new = []

        if self.export:
            for i in range(self.nl):
                x_new.append(self.m[i](x[i]))  # conv
            return None, x_new, x

        self.training |= self.export

        for i in range(self.nl):
            x_new.append(self.m[i](x[i]))  # conv
            bs, _, ny, nx = x_new[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            if self.version in ['mmdet', 'vid-yolo-mmdet'] and self.training:

                if self.two_stage:
                    x_new_i = x_new[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    
                    if self.grid[i].shape[2:4] != x_new_i.shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x_new_i.device)
                    if self.version == 'yolov5':
                        y = x_new_i.sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x_new_i.device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    elif self.version in ['yolov3','yolov4', 'mmdet', 'vid-yolo-mmdet']:
                        y = x_new_i
                        y[..., 0:2] =  (y[..., 0:2].sigmoid() + self.grid[i].to(x_new_i.device)) * self.stride[i]  # xy
                        y[..., 2:4] = y[..., 2:4].exp() * self.anchor_grid[i]  # wh
                        y[..., 4:] = y[..., 4:].sigmoid()
                    
                    z.append(y.view(bs, -1, self.no))

            else:
                x_new[i] = x_new[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x_new[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x_new[i].device)
                    if self.version == 'yolov5':
                        y = x_new[i].sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x_new[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    elif self.version in ['yolov3','yolov4', 'mmdet', 'vid-yolo-mmdet']:
                        y = x_new[i]
                        y[..., 0:2] =  (y[..., 0:2].sigmoid() + self.grid[i].to(x_new[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = y[..., 2:4].exp() * self.anchor_grid[i]  # wh
                        y[..., 4:] = y[..., 4:].sigmoid()
                    
                    z.append(y.view(bs, -1, self.no))

        if self.two_stage: #self.training:  
            return (z, x_new, x)
        else:
            return x if self.training else (torch.cat(z, 1), x)

        # return x if (self.training and not self.two_stage) else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg, ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()

        version    = cfg['version']
        anchors    = cfg['anchors']
        model_name = os.path.join('/home/yu/workspace/yoloall/yoloall/yolodet/models/cfg_model', cfg['model'])
        # two_stage  = cfg.two_stage['two_stage_enabel']

        self.module_def = parse_model_cfg(model_name, out_ch=len(anchors[0])//2*(5+nc))
        self.module_def[0]['channels'] = ch
        self.model, self.routs = create_modules(self.module_def, nc, anchors)
        
        self.names = [str(i) for i in range(nc)]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        m.version = version
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            logger.info('Model Strides: %s' % m.stride.tolist())
        
        if version == 'yolox-mmdet':
            m.box_coder = YOLOXHead(num_classes=nc,anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist())
        elif version == 'yolox-yzf-mmdet':
            m.box_coder = YZFHead(num_classes=nc,anchor_generator=np.array(anchors).reshape(len(anchors),-1,2).tolist(),featmap_strides=self.stride.tolist())
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
        out = []
        detect_index = len(self.model) - 1
        detect_input_index = self.model[-1].f

        for i, module in enumerate(self.model):
            name = module.__class__.__name__
            #print(name)
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l', 'ScaleChannel', 'ScaleSpatial']:  # sum, concat
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif i == detect_index:
                detect_input = [out[ind] for ind in detect_input_index]
                x = module(detect_input)
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
            out.append(x if self.routs[i] else [])

        dt = (time_synchronized() - t) * 100
        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
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


# class YOLOLayer(nn.Module):
#     def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
#         super(YOLOLayer, self).__init__()
#         self.anchors = torch.Tensor(anchors)
#         self.index = yolo_index  # index of this layer in layers
#         self.layers = layers  # model output layer indices
#         self.stride = stride  # layer stride
#         self.nl = len(layers)  # number of output layers (3)
#         self.na = len(anchors)  # number of anchors (3)
#         self.nc = nc  # number of classes (80)
#         self.no = nc + 5  # number of outputs (85)
#         self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
#         self.anchor_vec = self.anchors / self.stride
#         self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

#         if ONNX_EXPORT:
#             self.training = False
#             self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

#     def create_grids(self, ng=(13, 13), device='cpu'):
#         self.nx, self.ny = ng  # x and y grid size
#         self.ng = torch.tensor(ng, dtype=torch.float)

#         # build xy offsets
#         if not self.training:
#             yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
#             self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

#         if self.anchor_vec.device != device:
#             self.anchor_vec = self.anchor_vec.to(device)
#             self.anchor_wh = self.anchor_wh.to(device)

#     def forward(self, p, out):
#         ASFF = False  # https://arxiv.org/abs/1911.09516
#         if ASFF:
#             i, n = self.index, self.nl  # index in layers, number of layers
#             p = out[self.layers[i]]
#             bs, _, ny, nx = p.shape  # bs, 255, 13, 13
#             if (self.nx, self.ny) != (nx, ny):
#                 self.create_grids((nx, ny), p.device)

#             # outputs and weights
#             # w = F.softmax(p[:, -n:], 1)  # normalized weights
#             w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
#             # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

#             # weighted ASFF sum
#             p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
#             for j in range(n):
#                 if j != i:
#                     p += w[:, j:j + 1] * \
#                          F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

#         elif ONNX_EXPORT:
#             bs = 1  # batch size
#         else:
#             bs, _, ny, nx = p.shape  # bs, 255, 13, 13
#             if (self.nx, self.ny) != (nx, ny):
#                 self.create_grids((nx, ny), p.device)

#         # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
#         p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

#         if self.training:
#             return p

#         elif ONNX_EXPORT:
#             # Avoid broadcasting for ANE operations
#             m = self.na * self.nx * self.ny
#             ng = 1. / self.ng.repeat(m, 1)
#             grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
#             anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

#             p = p.view(m, self.no)
#             xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
#             wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
#             p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
#                 torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
#             return p_cls, xy * ng, wh

#         else:  # inference
#             io = p.sigmoid()
#             io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
#             io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
#             io[..., :4] *= self.stride
#             #io = p.clone()  # inference output
#             #io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
#             #io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
#             #io[..., :4] *= self.stride
#             #torch.sigmoid_(io[..., 4:])
#             return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


# class JDELayer(nn.Module):
#     def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
#         super(JDELayer, self).__init__()
#         self.anchors = torch.Tensor(anchors)
#         self.index = yolo_index  # index of this layer in layers
#         self.layers = layers  # model output layer indices
#         self.stride = stride  # layer stride
#         self.nl = len(layers)  # number of output layers (3)
#         self.na = len(anchors)  # number of anchors (3)
#         self.nc = nc  # number of classes (80)
#         self.no = nc + 5  # number of outputs (85)
#         self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
#         self.anchor_vec = self.anchors / self.stride
#         self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

#         if ONNX_EXPORT:
#             self.training = False
#             self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

#     def create_grids(self, ng=(13, 13), device='cpu'):
#         self.nx, self.ny = ng  # x and y grid size
#         self.ng = torch.tensor(ng, dtype=torch.float)

#         # build xy offsets
#         if not self.training:
#             yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
#             self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

#         if self.anchor_vec.device != device:
#             self.anchor_vec = self.anchor_vec.to(device)
#             self.anchor_wh = self.anchor_wh.to(device)

#     def forward(self, p, out):
#         ASFF = False  # https://arxiv.org/abs/1911.09516
#         if ASFF:
#             i, n = self.index, self.nl  # index in layers, number of layers
#             p = out[self.layers[i]]
#             bs, _, ny, nx = p.shape  # bs, 255, 13, 13
#             if (self.nx, self.ny) != (nx, ny):
#                 self.create_grids((nx, ny), p.device)

#             # outputs and weights
#             # w = F.softmax(p[:, -n:], 1)  # normalized weights
#             w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
#             # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

#             # weighted ASFF sum
#             p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
#             for j in range(n):
#                 if j != i:
#                     p += w[:, j:j + 1] * \
#                          F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

#         elif ONNX_EXPORT:
#             bs = 1  # batch size
#         else:
#             bs, _, ny, nx = p.shape  # bs, 255, 13, 13
#             if (self.nx, self.ny) != (nx, ny):
#                 self.create_grids((nx, ny), p.device)

#         # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
#         p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

#         if self.training:
#             return p

#         elif ONNX_EXPORT:
#             # Avoid broadcasting for ANE operations
#             m = self.na * self.nx * self.ny
#             ng = 1. / self.ng.repeat(m, 1)
#             grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
#             anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

#             p = p.view(m, self.no)
#             xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
#             wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
#             p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
#                 torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
#             return p_cls, xy * ng, wh

#         else:  # inference
#             #io = p.sigmoid()
#             #io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
#             #io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
#             #io[..., :4] *= self.stride
#             io = p.clone()  # inference output
#             io[..., :2] = torch.sigmoid(io[..., :2]) * 2. - 0.5 + self.grid  # xy
#             io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * self.anchor_wh  # wh yolo method
#             io[..., :4] *= self.stride
#             io[..., 4:] = F.softmax(io[..., 4:])
#             return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

# class Darknet(nn.Module):
#     # YOLOv3 object detection model

#     def __init__(self, cfg, img_size=(416, 416), verbose=False):
#         super(Darknet, self).__init__()

#         self.module_defs = parse_model_cfg(cfg)
#         self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
#         self.yolo_layers = get_yolo_layers(self)
#         # torch_utils.initialize_weights(self)

#         # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
#         self.info_mat = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
#         self.version = 'darknet'
#         self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
#         self.info(verbose) if not ONNX_EXPORT else None  # print model description

#     def forward(self, x, augment=False, verbose=False):

#         if not augment:
#             return self.forward_once(x)
#         else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
#             img_size = x.shape[-2:]  # height, width
#             s = [0.83, 0.67]  # scales
#             y = []
#             for i, xi in enumerate((x,
#                                     torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
#                                     torch_utils.scale_img(x, s[1], same_shape=False),  # scale
#                                     )):
#                 # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
#                 y.append(self.forward_once(xi)[0])

#             y[1][..., :4] /= s[0]  # scale
#             y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
#             y[2][..., :4] /= s[1]  # scale

#             # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
#             #     area = yi[..., 2:4].prod(2)[:, :, None]
#             #     if i == 1:
#             #         yi *= (area < 96. ** 2).float()
#             #     elif i == 2:
#             #         yi *= (area > 32. ** 2).float()
#             #     y[i] = yi

#             y = torch.cat(y, 1)
#             return y, None

#     def forward_once(self, x, augment=False, verbose=False):
#         img_size = x.shape[-2:]  # height, width
#         yolo_out, out = [], []
#         if verbose:
#             print('0', x.shape)
#             str = ''

#         # Augment images (inference and test only)
#         if augment:  # https://github.com/ultralytics/yolov3/issues/931
#             nb = x.shape[0]  # batch size
#             s = [0.83, 0.67]  # scales
#             x = torch.cat((x,
#                            torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
#                            torch_utils.scale_img(x, s[1]),  # scale
#                            ), 0)

#         for i, module in enumerate(self.module_list):
#             name = module.__class__.__name__
#             #print(name)
#             if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l', 'ScaleChannel', 'ScaleSpatial']:  # sum, concat
#                 if verbose:
#                     l = [i - 1] + module.layers  # layers
#                     sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
#                     str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
#                 x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
#             elif name == 'YOLOLayer':
#                 yolo_out.append(module(x, out))
#             elif name == 'JDELayer':
#                 yolo_out.append(module(x, out))
#             else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
#                 #print(module)
#                 #print(x.shape)
#                 x = module(x)

#             out.append(x if self.routs[i] else [])
#             if verbose:
#                 print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
#                 str = ''

#         if self.training:  # train
#             return yolo_out
#         elif ONNX_EXPORT:  # export
#             x = [torch.cat(x, 0) for x in zip(*yolo_out)]
#             return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
#         else:  # inference or test
#             x, p = zip(*yolo_out)  # inference output, training output
#             x = torch.cat(x, 1)  # cat yolo outputs
#             if augment:  # de-augment results
#                 x = torch.split(x, nb, dim=0)
#                 x[1][..., :4] /= s[0]  # scale
#                 x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
#                 x[2][..., :4] /= s[1]  # scale
#                 x = torch.cat(x, 1)
#             return x, p

#     def fuse(self):
#         # Fuse Conv2d + BatchNorm2d layers throughout model
#         print('Fusing layers...')
#         fused_list = nn.ModuleList()
#         for a in list(self.children())[0]:
#             if isinstance(a, nn.Sequential):
#                 for i, b in enumerate(a):
#                     if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
#                         # fuse this bn layer with the previous conv2d layer
#                         conv = a[i - 1]
#                         fused = torch_utils.fuse_conv_and_bn(conv, b)
#                         a = nn.Sequential(fused, *list(a.children())[i + 1:])
#                         break
#             fused_list.append(a)
#         self.module_list = fused_list
#         self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

#     def info(self, verbose=False):
#         torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ in ['YOLOLayer', 'JDELayer']]  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.info_mat = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.info_mat.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights', saveto='converted.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)
    ckpt = torch.load(weights)  # load checkpoint
    try:
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
        save_weights(model, path=saveto, cutoff=-1)
    except KeyError as e:
        print(e)

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {''}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
