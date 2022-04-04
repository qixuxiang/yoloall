from __future__ import division
import torch
import torch.nn as nn
from yolodet.models.common_py import Conv, Focus, Catneck, Incept, Detect


class Bottleneck(nn.Module):
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation,
                 ):#norm_cfg=dict(type='BN', requires_grad=True)
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, 1)#, norm_cfg=norm_cfg
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            )#norm_cfg=norm_cfg
        self.conv3 = nn.Conv2d(
            mid_channels, in_channels, 1)#, norm_cfg=norm_cfg

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


class DilatedEncoder(nn.Module):
    """Dilated Encoder for YOLOF <https://arxiv.org/abs/2103.09460>`.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
    """

    def __init__(self, in_channels, out_channels, block_mid_channels,
                 num_residual_blocks):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [2, 4, 6, 8]
        self._init_layers()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.out_channels)
        self.fpn_conv = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.out_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    # def init_weights(self):
    #     caffe2_xavier_init(self.lateral_conv)
    #     caffe2_xavier_init(self.fpn_conv)
    #     for m in [self.lateral_norm, self.fpn_norm]:
    #         constant_init(m, 1)
    #     for m in self.dilated_encoder_blocks.modules():
    #         if isinstance(m, nn.Conv2d):
    #             normal_init(m, mean=0, std=0.01)
    #         if is_norm(m):
    #             constant_init(m, 1)

    def forward(self, feature):
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)



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
        
        h3 = []
        # h1, h2, h3 = [], [], []
        # h1.append(Incept(in_ch=192, out_ch=24, down_ch=32, shortcut=True))
        # h1.append(Conv(in_ch=104, out_ch=48, ksize=1, stride=1))
        # h1.append(Conv(in_ch=48, out_ch=48, ksize=3, stride=1))  
        # h1.append(Conv(in_ch=48, out_ch=48, ksize=1, stride=1)) 
        # h1.append(Conv(in_ch=48, out_ch=48, ksize=3, stride=1))  
        # # h1.append(Conv(in_ch=48, out_ch=27, ksize=1, stride=1, out=True))  
        # self.h1 = nn.Sequential(*h1)

        # h2.append(Incept(in_ch=368, out_ch=32, down_ch=64, shortcut=True))  
        # h2.append(Conv(in_ch=160, out_ch=96, ksize=1, stride=1))  
        # h2.append(Conv(in_ch=96, out_ch=48, ksize=3, stride=1)) 
        # h2.append(Conv(in_ch=48, out_ch=96, ksize=1, stride=1)) 
        # h2.append(Conv(in_ch=96, out_ch=48, ksize=3, stride=1))
        # # h2.append(Conv(in_ch=48, out_ch=27, ksize=1, stride=1, out=True)) 
        # self.h2 = nn.Sequential(*h2)

        h3.append(Conv(in_ch=128, out_ch=128, ksize=1, stride=1)) 
        h3.append(Conv(in_ch=128, out_ch=64, ksize=3, stride=1)) 
        h3.append(Conv(in_ch=64, out_ch=128, ksize=1, stride=1))  
        h3.append(Conv(in_ch=128, out_ch=64, ksize=3, stride=1)) 
        # h3.append(Conv(in_ch=64, out_ch=27, ksize=1, stride=1, out=True))
        self.h3 = nn.Sequential(*h3)
        
        self.Dilated = DilatedEncoder(in_channels = 256, 
                                      out_channels = 128, 
                                      block_mid_channels = 256,
                                      num_residual_blocks = 4)
        self.detect = Detect(nc, anchors, [64]) #48, 48, 
        

    def forward(self, *x):
        x, *other = x
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        out_put = self.Dilated(b3)
        return self.detect([self.h3(out_put)])#self.h1(b1), self.h2(b2), 



if __name__ == '__main__':

    imgs = torch.rand([64, 3, 384, 576]).cuda()
    model = YOLO(ch=3, anchors= [[6, 5,  23, 12,  36, 21,  66, 38,  110, 67, 319, 213]], nc=5).cuda()
    print(len(model(imgs)))
    print(model(imgs)[0].shape)
    