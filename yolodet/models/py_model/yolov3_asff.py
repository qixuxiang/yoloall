import torch
import torch.nn as nn
import torch.nn.functional as F
# from .network_blocks import *
# from .yolov3_head import YOLOv3Head

from collections import defaultdict


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor) 
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4),dim=1)
        return out


class DropBlock(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)
    
    def reset(self, block_size, keep_prob):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)

    def calculate_gamma(self, x):
        return  (1-self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2) 

    def forward(self, x):
        if (not self.training or self.keep_prob==1): #set keep_prob=1 to turn off dropblock
            return x
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        if x.type() == 'torch.cuda.HalfTensor': #TODO: not fully support for FP16 now 
            FP16 = True
            x = x.float()
        else:
            FP16 = False
        p = torch.ones_like(x) * (self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        out =  mask * x * (mask.numel()/mask.sum())

        if FP16:
            out = out.half()
        return out


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch//2, 1, 1))
            resblock_one.append(add_conv(ch//2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


def build_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb):
    """
    Build yolov3 layer modules.
    Args:
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """
    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))           #0
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))          #1
    mlist.append(resblock(ch=64))                                           #2
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))         #3
    mlist.append(resblock(ch=128, nblocks=2))                               #4
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))        #5
    mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here     #6
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))        #7
    mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here     #8
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))       #9
    mlist.append(resblock(ch=1024, nblocks=4))                              #10

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=1, shortcut=False))              #11
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       #12
    #SPP Layer
    mlist.append(SPPLayer())                                                #13

    mlist.append(add_conv(in_ch=2048, out_ch=512, ksize=1, stride=1))       #14
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))       #15
    mlist.append(DropBlock(block_size=1, keep_prob=1))                    #16
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       #17

    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))        #18
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #19
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))        #20
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))        #21
    mlist.append(DropBlock(block_size=1, keep_prob=1))                    #22
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))               #23
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))        #24
    # 2nd yolo branch

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))        #25
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #26
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))        #27
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        #28
    mlist.append(DropBlock(block_size=1, keep_prob=1))                    #29
    mlist.append(resblock(ch=256, nblocks=1, shortcut=False))               #30
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))        #31
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        #32

    return mlist


class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, num_classes = 80, ignore_thre=0.7, label_smooth = False, rfb=False, vis=False, asff=False):
        """
        Initialization of YOLOv3 class.
        Args:
            ignore_thre (float): used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()
        self.module_list = build_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb)

        self.level_0_fusion = ASFF(level=0,rfb=rfb,vis=vis)

        # self.level_0_header = YOLOv3Head(anch_mask=[6, 7, 8], n_classes=num_classes, stride=32, in_ch=1024,
        #                       ignore_thre=ignore_thre,label_smooth = label_smooth, rfb=rfb)

        self.level_1_fusion = ASFF(level=1,rfb=rfb,vis=vis)

        # self.level_1_header = YOLOv3Head(anch_mask=[3, 4, 5], n_classes=num_classes, stride=16, in_ch=512,
        #                       ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb)

        self.level_2_fusion = ASFF(level=2,rfb=rfb,vis=vis)

        # self.level_2_header = YOLOv3Head(anch_mask=[0, 1, 2], n_classes=num_classes, stride=8, in_ch=256,
        #                       ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb)
        self.vis=vis

    def forward(self, x, targets=None, epoch=0):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """

        #train = targets is not None
        output = []
        # anchor_losses= []
        # iou_losses = []
        # l1_losses = []
        # conf_losses = []
        # cls_losses = []
        route_layers = []
        if self.vis:
            fuse_wegihts = []
            fuse_fs = []

        for i, module in enumerate(self.module_list):

            # yolo layers
            x = module(x)

            # route layers
            if i in [6, 8, 17, 24, 32]:
                route_layers.append(x)
            if i == 19:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 26:
                x = torch.cat((x, route_layers[0]), 1)
        
        for l in range(3):
            fusion = getattr(self, 'level_{}_fusion'.format(l))
            #header = getattr(self, 'level_{}_header'.format(l))

            if self.vis:
                fused, weight, fuse_f = fusion(route_layers[2],route_layers[3],route_layers[4])
                fuse_wegihts.append(weight)
                fuse_fs.append(fuse_f)
            else:
                fused = fusion(route_layers[2],route_layers[3],route_layers[4])
            output.append(fused)
        return output
        #     if train:
        #         x, anchor_loss, iou_loss, l1_loss, conf_loss, cls_loss = header(fused, targets)
        #         anchor_losses.append(anchor_loss)
        #         iou_losses.append(iou_loss)
        #         l1_losses.append(l1_loss)
        #         conf_losses.append(conf_loss)
        #         cls_losses.append(cls_loss)
        #     else:
        #         x = header(fused)

        #     output.append(x)

        # if train:
        #     losses = torch.stack(output, 0).unsqueeze(0).sum(1,keepdim=True)
        #     anchor_losses = torch.stack(anchor_losses, 0).unsqueeze(0).sum(1,keepdim=True)
        #     iou_losses = torch.stack(iou_losses, 0).unsqueeze(0).sum(1,keepdim=True)
        #     l1_losses = torch.stack(l1_losses, 0).unsqueeze(0).sum(1,keepdim=True)
        #     conf_losses = torch.stack(conf_losses, 0).unsqueeze(0).sum(1,keepdim=True)
        #     cls_losses = torch.stack(cls_losses, 0).unsqueeze(0).sum(1,keepdim=True)
        #     loss_dict = dict(
        #             losses = losses,
        #             anchor_losses = anchor_losses,
        #             iou_losses = iou_losses,
        #             l1_losses = l1_losses,
        #             conf_losses = conf_losses,
        #             cls_losses = cls_losses,
        #     )
        #     return loss_dict
        # else:
        #     if self.vis:
        #         return torch.cat(output, 1), fuse_wegihts, fuse_fs
        #     else:
        #         return torch.cat(output, 1)

if __name__ == "__main__":
    model = YOLOv3(num_classes = 80, ignore_thre=0.7, label_smooth = True, rfb=False, vis=False, asff=False)
    x = torch.rand([1,3,640,640])
    y = model(x)
    print(len(y))
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)

    