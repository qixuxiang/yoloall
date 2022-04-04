from __future__ import division
import torch
import torch.nn as nn
from yolodet.models.common_py import Conv, Focus, Catneck, Incept
from yolodet.loss.gaussian_loss_utils import y_pred_graph
import numpy as np


def correct_boxes_graph(y_pred_xy, y_pred_wh, input_shape, image_shape):
    """

    Args:
        y_pred_xy: (b, fh, fw, num_anchors_this_layer, 2)
        y_pred_wh: (b, fh, fw, num_anchors_this_layer, 2)
        input_shape: (b, 2), hw
        image_shape: (b, 2), hw

    Returns:
        boxes: (b, fh, fw, num_anchors_this_layer, 4), (y_min, x_min, y_max, x_max)

    """
    box_yx = y_pred_xy[..., ::-1]
    box_hw = y_pred_wh[..., ::-1]
    new_shape = torch.round(image_shape * torch.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = torch.stack([
        # y_min
        box_mins[..., 0:1],
        # x_min
        box_mins[..., 1:2],
        # y_max
        box_maxes[..., 0:1],
        # x_max
        box_maxes[..., 1:2]
    ],-1)
    boxes *= torch.stack([image_shape, image_shape],-1)
    return boxes


def correct_boxes_and_scores_graph(raw_y_pred, anchors, num_classes, input_shape, image_shape):
    """
    Args:
        raw_y_pred:
        anchors: (num_anchors_this_layer, 2)
        num_classes:
        input_shape: (2, ) hw
        image_shape: (batch_size, 2)

    Returns:
        boxes: (b, total_num_anchors_this_layer, 4), (y_min, x_min, y_max, x_max)
        boxes_scores: (b, total_num_anchors_this_layer, num_classes)

    """
    _, y_pred_box, _, _, y_pred_sigma, y_pred_confidence, y_pred_class_probs = y_pred_graph(raw_y_pred, anchors, input_shape)
    y_pred_xy = y_pred_box[..., :2]
    y_pred_wh = y_pred_box[..., 2:]
    # for batch predictions
    batch_size = image_shape
    input_shape = torch.unsqueeze(input_shape, axis=0)
    input_shape = input_shape.repeat([batch_size,1])
    boxes = correct_boxes_graph(y_pred_xy, y_pred_wh, input_shape, image_shape)
    box_scores = y_pred_confidence * y_pred_class_probs * torch.mean(y_pred_sigma, axis=-1, keepdims=True)
    boxes = boxes.reshape([batch_size, -1, 4])
    box_scores = box_scores.reshape([batch_size, -1, num_classes])

    return boxes, box_scores



class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), dim=5, nnx_enable=True):  # detection layer
        super(Detect, self).__init__()
        self.ch = ch
        self.nc = nc  # number of classes
        self.no = nc + dim + 4 # number of outputs per anchor (高斯yolo是输出8+1 = 9 + cls维度)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.export = False  # onnx export
        self.stride = None  # strides computed during build

        self.version = 'v3'

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
                z.append(x[i])

        if not self.training:
            print(z[0].shape)
            print(z[1].shape)
            print(z[2].shape)
            print(len(z))
            print(len(z))
            print(len(z))
            device = z[0].device
            yolo_outputs = [z[i].permute(0, 2, 3, 1, 4).contiguous() for i in range(len(z))][::-1]
            batch_image_shape = z[0].shape[0]
            input_shape = torch.tensor(yolo_outputs[0].shape[1:3]) * self.stride[-1]
            num_output_layers = len(yolo_outputs)
            anchors = (self.anchors.cpu() * torch.tensor(self.stride).reshape(len(self.stride),1,1)).reshape(-1,2).clone().detach()
            anchor_masks = np.array([i for i in range(anchors.shape[0])]).reshape(-1, anchors.shape[0] // len(self.stride)).tolist()[::-1]
            grid_shapes = [yolo_outputs[l].shape[1:3] for l in range(num_output_layers)]
            boxes_all_layers = []
            scores_all_layers = []
            for l in range(num_output_layers):
                yolo_output = yolo_outputs[l]
                grid_shape = grid_shapes[l]
                raw_y_pred = yolo_output.reshape([-1, grid_shape[0], grid_shape[1], self.na, self.nc + 9])

                boxes_this_layer, scores_this_layer = correct_boxes_and_scores_graph(raw_y_pred,
                                                                                    anchors[anchor_masks[l]],
                                                                                    self.nc,
                                                                                    input_shape,
                                                                                    batch_image_shape,
                                                                                    )
                boxes_all_layers.append(boxes_this_layer)
                scores_all_layers.append(scores_this_layer)

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
                    if self.version == 'v5':
                        y = x_new_i.sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x_new_i.device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    elif self.version in ['v3','v4', 'mmdet', 'vid-yolo-mmdet']:
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
                    if self.version == 'v5':
                        y = x_new[i].sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x_new[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    elif self.version in ['v3','v4', 'mmdet', 'vid-yolo-mmdet']:
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
        
        self.detect = Detect(nc, anchors, [48, 48, 64])

    def forward(self, *x):
        x, *other = x
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        return self.detect([self.h1(b1), self.h2(b2), self.h3(b3)])



if __name__ == '__main__':

    imgs = torch.rand([1, 3, 384, 576]).cuda()
    model = YOLO(ch=3, anchors=[[6, 5, 12, 7, 10, 16, 23, 12], [17, 25, 36, 21, 34, 41, 66, 38], [60, 88, 110, 67, 156, 163, 319, 213]], nc=14).cuda()
    y = model(imgs)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    