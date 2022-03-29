#!usr/bin/env python
#-*- encoding: utf-8 -*-
#Copyright (c) 2022
import torch
from torch import nn
from yolodet.models.common_py import Detect

class SiLU(nn.Module):
    
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type：{}".format(name))
    return module

class BaseConv(nn.Module):
    # def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", out=False):
    #     super().__init__()
    #     self.out = out
    #     self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=ksize,stride=stride,padding=(ksize - 1)//2,bias=out)
    #     if not out:
    #         self.bn = nn.BatchNorm2d(out_channels)
    #         self.act = nn.ReLU(inplace=True)

    # def forward(self,x):
    #     return self.conv(x) if self.out else self.act(self.bn(self.conv(x)))

    # def fuseforward(self,x):
    #     return self.conv(x) if self.out else self.act(self.conv(x))

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act
        )
        self.pconv = BaseConv(
            in_channels,out_channels,ksize=1,stride=1,groups=1,act=act
        )
    
    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

    
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class ResLayer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

class SPPBottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(5, 9, 13), activate="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activate)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_size
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_size) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activate)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2*hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise=depthwise, act=act)
                for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2),dim=1)
        return self.conv3(x)

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3),1)

        self.stem = nn.Sequential(
            BaseConv(3, base_channels // 2, ksize=3, stride=1, act=act),
            *self.make_group_layer(base_channels//2, num_blocks=1, stride=2, act=act),
        )

        #dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels*2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            )
        )

        #dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels*4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            )
        )

        #dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels*8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            )
        )


        #dark4
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels*16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activate=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            )
        )

    def make_group_layer(self, in_channels:int, num_blocks: int, stride: int = 1, act='lrelu'):
        return [
            BaseConv(in_channels, in_channels*2, ksize=3, stride=stride, act=act),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]
    
    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)       
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k,v in outputs.items() if k in self.out_features}


class SPPBottleneck_sum(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(5, 9, 13), activate="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_size
            ]
        )

    def forward(self, x):
        for m in self.m:
            x = x + m(x)
        return x

class CSPLayer_tsh(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise=depthwise, act=act)
                for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        #x = torch.cat((x_1, x_2),dim=1)
        x = x_1 + x_2
        return self.conv3(x)


class CSPLayer_FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise=depthwise, act=act)
                for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        #x = torch.cat((x_1, x_2),dim=1)
        x = x_1 + x_2
        return self.conv3(x)        

class CSPDarknet_tsh(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3),1)


        #stem
        #self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem = nn.Sequential(
            BaseConv(3, base_channels // 2, ksize=3, stride=1, act=act),
            *self.make_group_layer(base_channels//2, num_blocks=3, stride=2, act=act),
        )

        #dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels*2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            )
        )

        #dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels*4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            )
        )

        #dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels*8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            )
        )


        #dark4
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels*16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activate=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            )
        )

    def make_group_layer(self, in_channels:int, num_blocks: int, stride: int = 1, act='lrelu'):
        return [
            BaseConv(in_channels, in_channels*2, ksize=3, stride=stride, act=act),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)]
        ]
    
    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)       
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k,v in outputs.items() if k in self.out_features}


class YOLO(nn.Module):
    def __init__(self, nc, anchors, ch):
        super().__init__()
        depth = 1.25
        width = 1.0
        in_features=["dark3","dark4","dark5"]
        in_channels=[256, 512, 1024]
        depthwise=False
        act='relu'

        self.backbone = CSPDarknet_tsh(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.FPN0 = CSPLayer_FPN(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            round(1),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        
        self.FPN1 = CSPLayer_FPN(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            round(1),
            False,
            depthwise=depthwise,
            act=act,
        )

        #bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )

        self.FPN2 = CSPLayer_FPN(
            int(in_channels[0] * width),
            int(in_channels[1] * width),
            round(1),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )

        self.FPN3 = CSPLayer_FPN(
            int(in_channels[1] * width),
            int(in_channels[2] * width),
            round(1),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.detect = Detect(nc, anchors, [int(in_channels[0]* width), int(in_channels[1] * width), int(in_channels[2] * width)])

    def forward(self, input):
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2,x1,x0] = features

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = f_out0 + x1
        f_out0 = self.FPN0(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = f_out1 + x2
        pan_out2 = self.FPN1(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = p_out1 + fpn_out1
        pan_out1 = self.FPN2(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = p_out0 + fpn_out0
        pan_out0 = self.FPN3(p_out0)

        outputs = [pan_out2, pan_out1, pan_out0]
        return self.detect(outputs)

if __name__ == '__main__':
    x = torch.rand([1,3,576,960])
    anchors = [[17,7, 11,14, 29,11],[23,36, 61,16, 39,34],[101,109, 180,76, 481,141]]
    ch = 3
    model = YOLO(nc=80, anchors = anchors, ch=ch)
    model(x)
