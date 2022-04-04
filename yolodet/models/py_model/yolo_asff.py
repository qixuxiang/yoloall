from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolodet.models.common_py import Conv, Focus, Catneck, Incept, Detect


def add_conv(in_ch, out_ch, ksize, stride):
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
    # if leaky:
    #     stage.add_module('leaky', nn.LeakyReLU(0.1))
    # else:
    #     stage.add_module('relu6', nn.ReLU6(inplace=True))

    stage.add_module('relu',nn.ReLU(inplace=True))
    return stage

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [64, 48, 48]
        self.inter_dim = self.dim[self.level]
        if level==0: #深度层
            self.stride_level_1 = add_conv(48, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(48, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 64, 3, 1)
        elif level==1: #中度等
            self.compress_level_0 = add_conv(64, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(48, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 48, 3, 1)
        elif level==2:#浅度层
            self.compress_level_0 = add_conv(64, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 48, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0: #
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


class YOLO(nn.Module):
    def __init__(self, nc, anchors, ch):
        super(YOLO, self).__init__()
        b1, b2, b3 = [], [], []
        b1.append(Focus(in_ch=ch, out_ch=32, pool_type='Max'))
        b1.append(Catneck(in_ch=64, cat_ch=16, d=2, nblocks=4)) 
        b1.append(Conv(in_ch=128, out_ch=64, ksize=1, stride=1)) 
        b1.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)) 
        b1.append(Catneck(in_ch=64, cat_ch=32, d=2, nblocks=4))  
        self.b1 = nn.Sequential(*b1)

        b2.append(Conv(in_ch=192, out_ch=128, ksize=1, stride=1))  
        b2.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))  
        b2.append(Catneck(in_ch=128, cat_ch=48, d=2, nblocks=5))  
        self.b2 = nn.Sequential(*b2)

        b3.append(Conv(in_ch=368, out_ch=128, ksize=1, stride=1))  
        b3.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))  
        b3.append(Catneck(in_ch=128, cat_ch=64, d=2, nblocks=2))  
        self.b3 = nn.Sequential(*b3)
        
        h1, h2, h3 = [], [], []
        h1.append(Incept(in_ch=192, out_ch=24, down_ch=32, shortcut=True))
        h1.append(Conv(in_ch=104, out_ch=48, ksize=1, stride=1))
        h1.append(Conv(in_ch=48, out_ch=48, ksize=3, stride=1))  
        h1.append(Conv(in_ch=48, out_ch=48, ksize=1, stride=1)) 
        h1.append(Conv(in_ch=48, out_ch=48, ksize=3, stride=1))  
        # h1.append(Conv(in_ch=48, out_ch=27, ksize=1, stride=1, out=True))  
        self.h1 = nn.Sequential(*h1)

        h2.append(Incept(in_ch=368, out_ch=32, down_ch=64, shortcut=True))  
        h2.append(Conv(in_ch=160, out_ch=96, ksize=1, stride=1))  
        h2.append(Conv(in_ch=96, out_ch=48, ksize=3, stride=1)) 
        h2.append(Conv(in_ch=48, out_ch=96, ksize=1, stride=1)) 
        h2.append(Conv(in_ch=96, out_ch=48, ksize=3, stride=1))
        # h2.append(Conv(in_ch=48, out_ch=27, ksize=1, stride=1, out=True)) 
        self.h2 = nn.Sequential(*h2)

        h3.append(Conv(in_ch=256, out_ch=128, ksize=1, stride=1)) 
        h3.append(Conv(in_ch=128, out_ch=64, ksize=3, stride=1)) 
        h3.append(Conv(in_ch=64, out_ch=128, ksize=1, stride=1))  
        h3.append(Conv(in_ch=128, out_ch=64, ksize=3, stride=1)) 
        # h3.append(Conv(in_ch=64, out_ch=27, ksize=1, stride=1, out=True))
        self.h3 = nn.Sequential(*h3)
        
        self.level_0_fusion = ASFF(level=0)
        self.level_1_fusion = ASFF(level=1)
        self.level_2_fusion = ASFF(level=2)

        self.detect = Detect(nc, anchors, [48, 48, 64])

    def forward(self, *x):
        out = []
        x, *other = x
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        route_layers = [self.h3(b3), self.h2(b2), self.h1(b1)] #[64(深度层), 48(中度层), 48(浅度层)]
        for i in range(len(route_layers)):
            fusion = getattr(self, 'level_{}_fusion'.format(i))
            fused = fusion(route_layers[0],route_layers[1],route_layers[2])
            out.append(fused)
        return self.detect(out[::-1])

if __name__ == '__main__':

    imgs = torch.rand([1, 3, 384, 576])
    model = YOLO(ch=3, anchors=[[6, 5, 12, 7, 10, 16, 23, 12], [17, 25, 36, 21, 34, 41, 66, 38], [60, 88, 110, 67, 156, 163, 319, 213]], nc=5)
    print(len(model(imgs)))
    