import torch.nn as nn
import torch
import torch.nn.functional as F
#import config.yolov4_config as cfg
from yolodet.models.py_model.CSPDarknet53 import _BuildCSPDarknet53
from yolodet.models.py_model.mobilenetv2 import _BuildMobilenetV2
from yolodet.models.py_model.mobilenetv3 import _BuildMobilenetV3
from yolodet.models.py_model.mobilenetv2_CoordAttention import _BuildMobileNetV2_CoordAttention
from yolodet.models.py_model.global_context_block import ContextBlock2d


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
            Conv(feature_channels[-1] // 2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
        )

        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x] + features, dim=1)

        return features

    def __initialize_weights(self):
        print("**" * 10, "Initing head_conv weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1), nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)


class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.feature_transform3 = Conv(
            feature_channels[0], feature_channels[0] // 2, 1
        )
        self.feature_transform4 = Conv(
            feature_channels[1], feature_channels[1] // 2, 1
        )

        self.resample5_4 = Upsample(
            feature_channels[2] // 2, feature_channels[1] // 2
        )
        self.resample4_3 = Upsample(
            feature_channels[1] // 2, feature_channels[0] // 2
        )
        self.resample3_4 = Downsample(
            feature_channels[0] // 2, feature_channels[1] // 2
        )
        self.resample4_5 = Downsample(
            feature_channels[1] // 2, feature_channels[2] // 2
        )

        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2] * 2, feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
        )

        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.__initialize_weights()

    def forward(self, features):
        features = [
            self.feature_transform3(features[0]),
            self.feature_transform4(features[1]),
            features[2],
        ]

        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(
            torch.cat(
                [features[1], self.resample5_4(downstream_feature5)], dim=1
            )
        )
        downstream_feature3 = self.downstream_conv3(
            torch.cat(
                [features[0], self.resample4_3(downstream_feature4)], dim=1
            )
        )

        upstream_feature4 = self.upstream_conv4(
            torch.cat(
                [self.resample3_4(downstream_feature3), downstream_feature4],
                dim=1,
            )
        )
        upstream_feature5 = self.upstream_conv5(
            torch.cat(
                [self.resample4_5(upstream_feature4), downstream_feature5],
                dim=1,
            )
        )

        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def __initialize_weights(self):
        print("**" * 10, "Initing PANet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels):
        super(PredictNet, self).__init__()

        self.predict_conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(feature_channels[i] // 2, feature_channels[i], 3),
                    nn.Conv2d(feature_channels[i], target_channels, 1),
                )
                for i in range(len(feature_channels))
            ]
        )
        self.__initialize_weights()

    def forward(self, features):
        predicts = [
            predict_conv(feature)
            for predict_conv, feature in zip(self.predict_conv, features)
        ]

        return predicts

    def __initialize_weights(self):
        print("**" * 10, "Initing PredictNet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)

        p_de = self.__decode(p.clone())

        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = (
            grid_xy.unsqueeze(0)
            .unsqueeze(3)
            .repeat(batch_size, 1, 1, 3, 1)
            .float()
            .to(device)
        )

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return (
            pred_bbox.view(-1, 5 + self.__nC)
            if not self.training
            else pred_bbox
        )

class YOLOv4(nn.Module):
    def __init__(self, model_name, weight_path=None, out_channels=255, resume=False, showatt=False, feature_channels=0):
        super(YOLOv4, self).__init__()
        self.showatt = showatt
        if model_name == "YOLOv4":
            # CSPDarknet53 backbone
            self.backbone, feature_channels = _BuildCSPDarknet53(
                weight_path=weight_path, resume=resume
            )
        elif model_name == "Mobilenet-YOLOv4":
            # MobilenetV2 backbone
            self.backbone, feature_channels = _BuildMobilenetV2(
                weight_path=weight_path, resume=resume
            )
        elif model_name == "CoordAttention-YOLOv4":
            # MobilenetV2 backbone
            self.backbone, feature_channels = _BuildMobileNetV2_CoordAttention(
                weight_path=weight_path, resume=resume
            )
        elif model_name == "Mobilenetv3-YOLOv4":
            # MobilenetV3 backbone
            self.backbone, feature_channels = _BuildMobilenetV3(
                weight_path=weight_path, resume=resume
            )
        else:
            assert print("model type must be YOLOv4 or Mobilenet-YOLOv4")

        if self.showatt:
            self.attention = ContextBlock2d(feature_channels[-1], feature_channels[-1])
        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(feature_channels)

        # Path Aggregation Net
        self.panet = PANet(feature_channels)

        # predict
        self.predict_net = PredictNet(feature_channels, out_channels)

    def forward(self, x):
        atten = None
        features = self.backbone(x)
        if self.showatt:
            features[-1], atten = self.attention(features[-1])
        features[-1] = self.spp(features[-1])
        features = self.panet(features)
        predicts = self.predict_net(features)
        return predicts, atten

class Model(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, cfg, ch=3, nc=None, weight_path=None, resume=False, showatt=False):
        super(Model, self).__init__()
        self.__showatt = showatt
        
        a = cfg['anchors']
        self.stride = cfg['step']
        self.version = cfg['version']
        #per_scale = len(a[0])

        temp = []
        for i in range(len(a)):
            res = []
            for j in range(0,len(a[i]),2):
                res.append((round(a[i][j]/self.stride[i],3),round(a[i][j+1]/self.stride[i],3)))
            temp.append(res)
        per_scale = len(temp[0])
        self.__anchors = torch.FloatTensor(temp)
        self.__strides = torch.FloatTensor(self.stride)

        # if cfg.TRAIN["DATA_TYPE"] == "VOC":
        #     self.__nC = cfg.VOC_DATA["NUM"]
        # elif cfg.TRAIN["DATA_TYPE"] == "COCO":
        #     self.__nC = cfg.COCO_DATA["NUM"]
        # else:
        self.__nC = nc #cfg.Customer_DATA["NUM"]

        self.__out_channel = per_scale * (self.__nC + 5)

        self.__yolov4 = YOLOv4(
            model_name = cfg['cfg'],
            weight_path=weight_path,
            out_channels=self.__out_channel,
            resume=resume,
            showatt=showatt
        )
        # small
        self.__head_s = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0]
        )
        # medium
        self.__head_m = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1]
        )
        # large
        self.__head_l = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2]
        )

    def forward(self, x):
        out = []
        [x_s, x_m, x_l], atten = self.__yolov4(x)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            if self.__showatt:
                return p, torch.cat(p_d, 0), atten
            return p, torch.cat(p_d, 0)
