from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from yolodet.utils.general import non_max_suppression, tsh_batch_non_max_suppression # for test

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
        self.version = 'v3'

        self.config = {}
        self.export = False  # onnx export
        self.stride = None  # strides computed during build

        self.two_stage = False
        self.nnx_enable = nnx_enable
        # self.prior_generator = None #YOLOAnchorGenerator(strides=self.stride, base_sizes = np.array(anchors).reshape(len(anchors),-1,2).tolist())
        self.box_coder = None #YOLOV5BBoxCoder()#YOLOIouBBoxCoder()

    def coupling_init_detect_layers(self):
        """"Initialize layers of the coupling head. like yolov3/4/5"""
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in self.ch)  # output conv FCOS不用


    def decoupling_init_fcos_layers(self):
        """"Initialize layers of the decoupling head. like yolox fcos"""
        self.neck_reg = nn.ModuleList(Conv(x, 64, 3, 1) for x in self.ch)
        self.neck_cls = nn.ModuleList(Conv(x, 64, 3, 1) for x in self.ch)
        self.conv_reg = nn.ModuleList(nn.Conv2d(64, 4, 3, padding=1) for x in self.ch)
        self.conv_cls = nn.ModuleList(nn.Conv2d(64, self.nc, 3, padding=1) for x in self.ch)
        self.conv_centerness = nn.ModuleList(nn.Conv2d(64, 1, 3, padding=1) for x in self.ch) #
        self.scales = nn.ModuleList([Scale(1.0) for _ in [8, 16, 32]])

    def decoupling_init_center_layers(self):
        """"Initialize layers of the decoupling head. like yolox fcos"""
        self.neck_reg = nn.ModuleList(Conv(x, 64, 3, 1) for x in self.ch)
        self.neck_cls = nn.ModuleList(Conv(x, 64, 3, 1) for x in self.ch)
        self.conv_reg = nn.ModuleList(nn.Conv2d(64, 2, 3, padding=1) for x in self.ch)
        self.conv_cls = nn.ModuleList(nn.Conv2d(64, self.nc, 3, padding=1) for x in self.ch)
        self.conv_centerness = nn.ModuleList(nn.Conv2d(64, 2, 3, padding=1) for x in self.ch) #
    
    def decoupling_init_gfl_layers(self):
        """"Initialize layers of the decoupling head. """
        self.neck_reg = nn.ModuleList(Conv(x, 256, 3, 1) for x in self.ch)
        self.neck_cls = nn.ModuleList(Conv(x, 256, 3, 1) for x in self.ch)
        self.conv_reg = nn.ModuleList(nn.Conv2d(256, self.na * 4 * (self.config.get('reg_max', 0) + 1), 3, padding=1) for x in self.ch)
        self.conv_cls = nn.ModuleList(nn.Conv2d(256, self.na * self.nc, 3, padding=1) for x in self.ch) #这边类别+1为Gfcoals 该类别为背景类
        self.scales = nn.ModuleList(Scale(1.0) for _ in range(self.nl))
        if not self.config.get('use_sigmoid', True):# Gfocal_v2
            self.reg_conf = nn.ModuleList(nn.Sequential(nn.Conv2d(4 * (self.config.get('reg_topk', 0) + 1), 64, 1),          
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 1, 1), 
                                        nn.Sigmoid()) for x in self.ch)


    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output

        if self.two_stage:
            return self.forward_two_stage(x)

        if self.version in ['yolo-fcos-mmdet','yolo-center-mmdet','yolo-gfl-mmdet']:
            return self.forward_anchor_free(x)

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
                    if self.version == 'v5':
                        y = x[i].sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    elif self.version in ['v3','v4','yolo-mmdet','yolof-mmdet','vid-yolo-mmdet','yolo-atss-yzf-mmdet']:# 'mmdet',
                        y = x[i]
                        y[..., 0:2] =  (y[..., 0:2].sigmoid() + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        y[..., 2:4] = y[..., 2:4].exp() * self.anchor_grid[i]  # wh
                        y[..., 4:] = y[..., 4:].sigmoid()
                    
                    elif self.version in ['yolox-mmdet','yolox-yzf-mmdet','yolo-atss-mmdet']:
                        y = self.box_coder._get_box_single(pred_map=x[i],num_imgs=bs,level_idx=i)


                    if self.version in ['yolox-mmdet','yolox-yzf-mmdet','yolo-atss-mmdet']:
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

    def forward_anchor_free(self, x):
       # x = x.copy()  # for profiling
        z = []  # inference output
        bs, _, ny, nx = x[0].shape
        x_new = []
        out_puts = []
        distiller = self.config.get('distiller', False)
        if self.export:
            for i in range(len(self.ch)):
                x_new.append(self.m[i](x[i]))  # conv
            return None, x_new, x

        self.training |= self.export

        for i in range(len(self.ch)):
            cls_feat = self.neck_cls[i](x[i])
            reg_feat = self.neck_reg[i](x[i])
            # fcos输出框、类别、中心度3个向量
            if 'fcos' in self.version:
                #obj_feat = x[i] #Fcos没有框的置信度这一概念的它是中心度的概念及偏离中心点的距离： centerness 这个中心度的特征可以从分类特征中输出也可以从回归的特征中输出。
                #中心度 如果centerness_on_reg=False 表示从回归特征图上进行卷积;centerness_on_reg=True 表示从分类特征图上进行卷积
                if self.config.get('centerness_on_reg',False): #从回归特征图上进行卷积
                    centerness = self.conv_centerness[i](reg_feat)
                else:
                    centerness = self.conv_centerness[i](cls_feat)
                
                cls_score = self.conv_cls[i]((cls_feat))
                bbox_pred = self.scales[i](self.conv_reg[i](reg_feat)).float()
                if self.config.get('norm_on_bbox',False):
                    bbox_pred = F.relu(bbox_pred)
                    if not self.training:
                        bbox_pred *= self.stride[i]
                else:
                    bbox_pred = bbox_pred.exp()
                x[i] = torch.cat([bbox_pred,centerness,cls_score],1)
            # centernet 输出框、类别、中心度3个向量
            elif 'center' in self.version:
                cls_score = self.conv_cls[i]((cls_feat)).sigmoid() #centernet中这个是回归heat_map的（这里进行sigmoid()激活）
                bbox_pred = self.conv_reg[i]((reg_feat)) #centernet中这个是回归wh_map的
                centerness = self.conv_centerness[i](reg_feat) #centernet中这个是回归偏置的
                x[i] = torch.cat([bbox_pred,centerness,cls_score],1)
            #gfcoal 输出框、类别2个向量
            elif 'gfl' in self.version:
                #gfcoal_v1:
                #cls_score = self.conv_cls[i]((cls_feat)) #gfcoal_v1：开
                bbox_pred = self.scales[i](self.conv_reg[i]((reg_feat))).float() 
                #gfcoal_v2:
                # N, C, H, W = bbox_pred.size()
                # prob = F.softmax(bbox_pred.reshape(N, 4, self.na*(self.config.get('reg_max', 0)+1), H, W), dim=2)
                # prob_topk, _ = prob.topk(self.config.get('reg_topk', 0), dim=2) #self.na *
                # if self.config.get('add_mean', False):
                #     stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)], dim=2)
                # else:
                #     stat = prob_topk
                if not self.config.get('use_sigmoid', True):
                    stat = self.box_coder.gfcoal_v2(bbox_pred)
                    quality_score = self.reg_conf[i](stat)
                    cls_score = self.conv_cls[i]((cls_feat)).sigmoid() * quality_score
                else:
                    cls_score = self.conv_cls[i]((cls_feat)) #gfcoal_v1：开
                x[i] = torch.cat([bbox_pred,cls_score],1)

            if ('mmdet' in self.version and self.training) or distiller: #train的时候cat
                pass
            else:
                y = self.box_coder._get_box_single(pred_map=x[i],num_imgs=bs,level_idx=i)
                if isinstance(y,list):
                    z.extend(y)
                else:
                    z.append(y)
        # if not self.training:
        #     for batch_num in range(bs):
        #         ans = []
        #         for detect_num in range(self.nl):
        #             ans.append(z[detect_num][batch_num][:,1:])
        #         out_puts.append(torch.cat(ans,0))
        #(out_puts, None) #
        if self.version in ['yolo-center-mmdet']: #centernet不需要做nms
            return x if self.training else (y, None) #(torch.cat(z, 1), x)  #cls_score, bbox_pred, cls_feat, reg_feat #这边需要更改Centernet 不需要nms
        else:
            return x if self.training or distiller else (torch.cat(z, 1), x)
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, out=False):
        super().__init__()
        self.out = out
        self.c = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ksize, stride=stride, padding=(ksize - 1) // 2, bias=out) 
        if not out:              
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.c(x) if self.out else self.act(self.bn(self.c(x)))
    
    def fuseforward(self, x):
        return self.c(x) if self.out else self.act(self.c(x))


class Focus(nn.Module):
    def __init__(self, in_ch, out_ch, pool_type='Max'):
        super().__init__()
        self.c0 = Conv(in_ch=in_ch, out_ch=out_ch, ksize=3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) if pool_type == 'Max' else nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        self.c1 = Conv(in_ch=out_ch, out_ch=out_ch//2, ksize=1, stride=1)
        self.c2 = Conv(in_ch=out_ch//2, out_ch=out_ch, ksize=3, stride=2)
    def forward(self, x):
        x = self.c0(x)
        return torch.cat((self.pool(x), self.c2(self.c1(x))), dim=1)

class res(nn.Module):
    def __init__(self, in_ch, mid_ch=0, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.c1 = Conv(in_ch, mid_ch, 1, 1)
        self.c2 = Conv(mid_ch, in_ch, 3, 1)
    def forward(self, x):
        return x + self.c2(self.c1(x)) if self.shortcut else self.c2(self.c1(x))

class Resneck(nn.Module):
    def __init__(self, in_ch, mid_ch=0, nblocks=1, shortcut=True):
        super().__init__()
        self.n = nn.Sequential(*[res(in_ch, mid_ch, shortcut) for _ in range(nblocks)])
    def forward(self, x):
        return self.n(x)

class cat(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.c1 = Conv(in_ch, mid_ch, 1, 1)
        self.c2 = Conv(mid_ch, out_ch, 3, 1)
    def forward(self, x):
        return torch.cat((x, self.c2(self.c1(x))), dim=1)

class Catneck(nn.Module):
    def __init__(self, in_ch, cat_ch=16, d=4, nblocks=1):
        super().__init__()
        self.n = nn.Sequential(*[cat(in_ch+i*cat_ch, cat_ch*d, cat_ch) for i in range(nblocks)])
    def forward(self, x):
        return self.n(x)

class Incept(nn.Module):
    def __init__(self, in_ch, out_ch, down_ch=0, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.down = down_ch > 0
        if self.down:
            self.d = Conv(in_ch, down_ch, 1, 1)
            in_ch = down_ch
        # 1*1
        self.c1 = Conv(in_ch, out_ch, 1, 1)
        # 1*1 3*3
        self.c31 = Conv(in_ch, out_ch, 1, 1)
        self.c32 = Conv(out_ch, out_ch, 3, 1)
        # 1*1 3*3 3*3
        self.c51 = Conv(in_ch, out_ch, 1, 1)
        self.c52 = Conv(out_ch, out_ch, 3, 1)
        self.c53 = Conv(out_ch, out_ch, 3, 1)
    def forward(self, x):
        if self.down:
            x = self.d(x)
        y1 = self.c1(x)
        y2 = self.c32(self.c31(x))
        y3 = self.c53(self.c52(self.c51(x)))
        return torch.cat((x, y1, y2, y3), dim=1) if self.shortcut else torch.cat((y1, y2, y3), dim=1)



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

# CBAM注意力网络
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

# SE_Net注意力网络
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层 (降维)
            nn.ReLU(inplace=True),                                 # ReLU 非线性激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层 (升维)
            nn.Sigmoid()                                           # 非线性激活函数 + 数值范围约束 (0, 1)
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 即上文所述的 U
        y = self.fc(y).view(b, c, 1, 1)  # reshape 张量以便于进行通道重要性加权的乘法操作
 
        return x * y.expand_as(x)  # 按元素一一对应相乘

#skNet 注意力网络 版本1
# class SKConv(nn.Module):
#     def __init__(self, features, WH, M, G, r, stride=1, L=32):
#         super(SKConv, self).__init__()
#         d = max(int(features / r), L)
#         self.M = M
#         self.features = features
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             # 使用不同kernel size的卷积
#             self.convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(features,
#                               features,
#                               kernel_size=3 + i * 2,
#                               stride=stride,
#                               padding=1 + i,
#                               groups=G), nn.BatchNorm2d(features),
#                     nn.ReLU(inplace=False)))
            
#         self.fc = nn.Linear(features, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(nn.Linear(d, features))
#         self.softmax = nn.Softmax(dim=1)
 
#     def forward(self, x):
#         for i, conv in enumerate(self.convs):
#             fea = conv(x).unsqueeze_(dim=1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             print(i, fea_z.shape)
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             print(i, vector.shape)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector],
#                                               dim=1)
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feas * attention_vectors).sum(dim=1)
#         return fea_v

#skNet 注意力网络 版本2
class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r, L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 (3X3，dilation=2)来代替。且论文中建议分组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应 pool 到指定维度 - GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d, out_channels*M, 1, 1, bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定 dim=1 令两个 FCs 对应位置进行 softmax,保证 对应位置a+b+..=1
 
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x, y: x+y, output) # 逐元素相加生成 混合特征 U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z 降维
        a_b=self.fc2(z)  # Z->a, b 升维 - 论文用 conv 1x1 表示 FC。结果中前一半通道值为 a, 后一半为 b
        a_b=a_b.reshape(batch_size, self.M, self.out_channels, -1) # reshape 为两个 FCs 的值
        a_b=self.softmax(a_b)  # 令两个 FCs 对应位置进行 softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M, dim=1))  # split to a 和 b - chunk 将 tensor 按指定维度切分成几块
        a_b=list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))  # reshape 所有分块，即扩展两维
        V=list(map(lambda x, y: x*y, output, a_b)) # 权重与对应不同卷积核输出的 U 逐元素相乘
        V=reduce(lambda x, y: x+y, V)  # 两个加权后的特征 逐元素相加
        return V
 
 
class SKBlock(nn.Module):
    '''
    基于Res Block构造的SK Block
    ResNeXt有  1x1Conv（通道数：x） +  SKConv（通道数：x）  + 1x1Conv（通道数：2x） 构成
    '''
    expansion=2 #指 每个block中 通道数增大指定倍数
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(SKBlock,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(inplanes,planes,1,1,0,bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU(inplace=True))
        self.conv2=SKConv(planes,planes,stride)
        self.conv3=nn.Sequential(nn.Conv2d(planes,planes*self.expansion,1,1,0,bias=False),
                                 nn.BatchNorm2d(planes*self.expansion))
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
 
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)
 
 
class SKNet(nn.Module):
    '''
    参考 论文Table.1 进行构造
    '''
    def __init__(self,nums_class=1000,block=SKBlock,nums_block_list=[3, 4, 6, 3]):
        super(SKNet,self).__init__()
        self.inplanes=64
        # in_channel=3  out_channel=64  kernel=7x7 stride=2 padding=3
        self.conv=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.maxpool=nn.MaxPool2d(3,2,1) # kernel=3x3 stride=2 padding=1
        self.layer1=self._make_layer(block,128,nums_block_list[0],stride=1) # 构建表中 每个[] 的部分
        self.layer2=self._make_layer(block,256,nums_block_list[1],stride=2)
        self.layer3=self._make_layer(block,512,nums_block_list[2],stride=2)
        self.layer4=self._make_layer(block,1024,nums_block_list[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(1) # GAP全局平均池化
        self.fc=nn.Linear(1024*block.expansion,nums_class) # 通道 2048 -> 1000
        self.softmax=nn.Softmax(-1) # 对最后一维进行softmax
 
    def forward(self, input):
        output=self.conv(input)
        output=self.maxpool(output)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.avgpool(output)
        output=output.squeeze(-1).squeeze(-1)
        output=self.fc(output)
        output=self.softmax(output)
        return output
 
    def _make_layer(self,block,planes,nums_block,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(nn.Conv2d(self.inplanes,planes*block.expansion,1,stride,bias=False),
                                     nn.BatchNorm2d(planes*block.expansion))
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for _ in range(1,nums_block):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
 
 
def SKNet50(nums_class=1000):
    return SKNet(nums_class, SKBlock, [3, 4, 6, 3])  # 论文通过[3, 4, 6, 3]搭配出SKNet50
 
 
def SKNet101(nums_class=1000):
    return SKNet(nums_class, SKBlock, [3, 4, 23, 3])

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


if __name__ == '__main__':

    image = torch.ones([2,64,128,128])

    model = CBAM(64)

    model(image)