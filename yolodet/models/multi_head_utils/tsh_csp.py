#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
from torch import nn
from yolodet.models.common_py import Detect


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
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
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
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
    "Residual layer with `in_channels` inputs."

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
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

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
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
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


# class Darknet(nn.Module):
#     # number of blocks from dark2 to dark5.
#     depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

#     def __init__(
#         self,
#         depth,
#         in_channels=3,
#         stem_out_channels=32,
#         out_features=("dark3", "dark4", "dark5"),
#     ):
#         """
#         Args:
#             depth (int): depth of darknet used in model, usually use [21, 53] for this param.
#             in_channels (int): number of input channels, for example, use 3 for RGB image.
#             stem_out_channels (int): number of output chanels of darknet stem.
#                 It decides channels of darknet layer2 to layer5.
#             out_features (Tuple[str]): desired output layer name.
#         """
#         super().__init__()
#         assert out_features, "please provide output features of Darknet"
#         self.out_features = out_features
#         self.stem = nn.Sequential(
#             BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
#             *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
#         )
#         in_channels = stem_out_channels * 2  # 64

#         num_blocks = Darknet.depth2blocks[depth]
#         # create darknet with `stem_out_channels` and `num_blocks` layers.
#         # to make model structure more clear, we don't use `for` statement in python.
#         self.dark2 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[0], stride=2)
#         )
#         in_channels *= 2  # 128
#         self.dark3 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[1], stride=2)
#         )
#         in_channels *= 2  # 256
#         self.dark4 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[2], stride=2)
#         )
#         in_channels *= 2  # 512

#         self.dark5 = nn.Sequential(
#             *self.make_group_layer(in_channels, num_blocks[3], stride=2),
#             *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
#         )

#     def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
#         "starts with conv layer then has `num_blocks` `ResLayer`"
#         return [
#             BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
#             *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
#         ]

#     def make_spp_block(self, filters_list, in_filters):
#         m = nn.Sequential(
#             *[
#                 BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
#                 BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
#                 SPPBottleneck(
#                     in_channels=filters_list[1],
#                     out_channels=filters_list[0],
#                     activation="lrelu",
#                 ),
#                 BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
#                 BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
#             ]
#         )
#         return m

#     def forward(self, x):
#         outputs = {}
#         x = self.stem(x)
#         outputs["stem"] = x
#         x = self.dark2(x)
#         outputs["dark2"] = x
#         x = self.dark3(x)
#         outputs["dark3"] = x
#         x = self.dark4(x)
#         outputs["dark4"] = x
#         x = self.dark5(x)
#         outputs["dark5"] = x
#         return {k: v for k, v in outputs.items() if k in self.out_features}

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

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem = nn.Sequential(
            BaseConv(3, base_channels//2, ksize=3, stride=1, act=act),
            *self.make_group_layer(base_channels//2, num_blocks=1, stride=2, act=act),
        )

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, act='lrelu'):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act=act),
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
        return {k: v for k, v in outputs.items() if k in self.out_features}

    
    
# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         in_features=("dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         super().__init__()

# class YOLO(nn.Module):
#     def __init__(self, nc, anchors, ch):
#         super().__init__()
#         """
#         YOLOv3 model. Darknet 53 is the default backbone of this model.
#         """
#         depth=1.0
#         width=0.75
#         in_features=("dark3", "dark4", "dark5")
#         in_channels=[256, 512, 1024]
#         depthwise=False
#         act="relu"
        
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         self.detect = Detect(nc, anchors, [int(in_channels[0] * width), int(in_channels[1] * width), int(in_channels[2] * width)])


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """

#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         [x2, x1, x0] = features

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

#         outputs = [pan_out2, pan_out1, pan_out0]
#         return self.detect(outputs)



class SPPBottleneck_sum(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
    def forward(self, x):
        for m in self.m:
            x = x + m(x)
        return x


class CSPLayer_tsh(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

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
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        # x = torch.cat((x_1, x_2), dim=1)
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

        # import pdb;pdb.set_trace()

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem = nn.Sequential(
            BaseConv(3, base_channels//2, ksize=3, stride=1, act=act),
            *self.make_group_layer(base_channels//2, num_blocks=3, stride=2, act=act), #这边改为了（3,3,9,9）
        )

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer_tsh(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer_tsh(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer_tsh(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck_sum(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer_tsh(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, act='lrelu'):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act=act),
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
        return {k: v for k, v in outputs.items() if k in self.out_features}

class CSPLayer_FPN(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

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
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        # x = torch.cat((x_1, x_2), dim=1)
        x = x_1 + x_2
        return self.conv3(x)

class YOLO(nn.Module):
    def __init__(self, nc, anchors, ch):
        super().__init__()
        """
        YOLOv3 model. Darknet 53 is the default backbone of this model.
        """
        depth=1.25
        width=1.0
        in_features=("dark3", "dark4", "dark5")
        in_channels=[256, 512, 1024]
        depthwise=False
        act="relu"
        
        self.backbone = CSPDarknet_tsh(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.FPN0 = CSPLayer_FPN(
            # int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            round(1),#3 * depth
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.FPN1 = CSPLayer_FPN(
            # int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            round(1),#3 * depth
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.FPN2 = CSPLayer_FPN(
            # int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            int(in_channels[1] * width),
            round(1),#3 * depth
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.FPN3 = CSPLayer_FPN(
            # int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            int(in_channels[2] * width),
            round(1),#3 * depth
            False,
            depthwise=depthwise,
            act=act,
        )

        self.detect = Detect(nc, anchors, [int(in_channels[0] * width), int(in_channels[1] * width), int(in_channels[2] * width)])


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = f_out0 + x1
        f_out0 = self.FPN0(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        f_out1 = f_out1 + x2
        pan_out2 = self.FPN1(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        p_out1 = p_out1 + fpn_out1
        pan_out1 = self.FPN2(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        p_out0 = p_out0 + fpn_out0
        pan_out0 = self.FPN3(p_out0)  # 1024->1024/32
        #print(pan_out0.shape)
        outputs = [pan_out2, pan_out1, pan_out0]
        return self.detect(outputs)

if __name__ == '__main__':
    x = torch.rand([1,3,576,960])
    nc = 80
    anchors =  [[17,7,  11,14,  29,11,  20,21,  38,19], [23,36,  61,16,  39,34,  64,28,  42,59],[101,109,  180,76,  249,98,  481,141,  338,233]]
    ch = 3
    model = YOLO(nc,anchors,ch)
    model(x)