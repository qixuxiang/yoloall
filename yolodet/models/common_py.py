from __future__ import division
import torch
import torch.nn as nn

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), dim=5, nnx_enable=True):
        super(Detect, self).__init__()
        self.ch = ch
        self.nc = nc
        self.no = nc + dim
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl,-1,2)
        self.register_buffer('anchors',a)
        self.register_buffer('anchor_grid',a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no*self.na,1) for x in ch)

        self.export = False
        self.stride = None

        self.version = 'v3'
        self.two_stage = False
        self.nnx_enable = nnx_enable

    def forward(self,x):
        z = []
        if self.two_stage:
            return self.forward_two_stage(x)
        if self.export:
            if self.nnx_enable:
                for i in range(self.nl):
                    x[i] = self.m[i](x[i])
                return x
            else:
                return x
        self.training |= self.export
        
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape

            if self.version in ['mmdet'] and self.training:
                pass
            else:
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0,1,3,4,2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                
                if self.version == 'v5':
                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    y = x[i]
                    y[...,0:2] = (y[...,0:2].sigmoid() + self.grid[i].to(x[i].device)) * self.stride[i]
                    y[...,2:4] = y[...,2:4].exp() * self.anchor_grid[i]
                    y[...,4:] = y[...,4:].sigmoid()

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    def forward_two_stage(self,x):

        z = []
        x_new = []

        if self.export:
            for i in range(self.nl):
                x_new.append(self.m[i](x[i]))
            return None, x_new, x

        self.training |= self.export
        
        for i in range(self.nl):
            x_new[i] = self.m[i](x[i])
            bs, _, ny, nx = x_new[i].shape

            if self.version in ['mmdet'] and self.training:
                if self.two_stage:
                    x_new_i = x_new[i].view(bs, self.na, self.no, ny, nx).permute(0,1,3,4,2).contiguous()

                    if self.grid[i].shape[2:4] != x_new_i.shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x_new_i.device)
                    
                    if self.version == 'v5':
                        y = x_new_i.sigmoid()
                        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x_new_i.device)) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    else:
                        y = x_new_i
                        y[...,0:2] = (y[...,0:2].sigmoid() + self.grid[i].to(x_new_i.device)) * self.stride[i]
                        y[...,2:4] = y[...,2:4].exp() * self.anchor_grid[i]
                        y[...,4:] = y[...,4:].sigmoid()

                    z.append(y.view(bs, -1, self.no))
            else:
                x_new[i] = x_new[i].view(bs, self.na, self.no, ny, nx).permute(0,1,3,4,2).contiguous()

                if self.grid[i].shape[2:4] != x_new[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x_new[i].device)
                
                if self.version == 'v5':
                    y = x_new[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x_new[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    y = x_new[i]
                    y[...,0:2] = (y[...,0:2].sigmoid() + self.grid[i].to(x_new[i].device)) * self.stride[i]
                    y[...,2:4] = y[...,2:4].exp() * self.anchor_grid[i]
                    y[...,4:] = y[...,4:].sigmoid()

                z.append(y.view(bs, -1, self.no))

        if self.two_stage:
            return (z, x_new, x)
        else:
            return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, out=False):
        super().__init__()
        self.out = out
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=ksize,stride=stride,padding=(ksize - 1)//2,bias=out)
        if not out:
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.conv(x) if self.out else self.act(self.bn(self.conv(x)))

    def fuseforward(self,x):
        return self.conv(x) if self.out else self.act(self.conv(x))

class Focus(nn.Module):
    def __init__(self, in_ch, out_ch, pool_type='Max'):
        super().__init__()
        self.c0 = Conv(in_ch=in_ch, out_ch=out_ch, ksize=3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True) if pool_type=='Max' else nn.AvgPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.c1 = Conv(in_ch=out_ch, out_ch=out_ch//2, ksize=1, stride=1)
        self.c2 = Conv(in_ch=out_ch//2, out_ch=out_ch,ksize=3, stride=2)
    def forward(self,x):
        x = self.c0(x)
        return torch.cat((self.pool(x), self.c2(self.c1(x))), dim=1)

class res(nn.Module):
    def __init__(self,in_ch, mid_ch=0, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.c1 = Conv(in_ch, mid_ch, 1, 1)
        self.c2 = Conv(mid_ch, in_ch, 3, 1)
    def forward(self,x):
        return x + self.c2(self.c1(x)) if self.shortcut else self.c2(self.c1(x))

class Resneck(nn.Module):
    def __init__(self, in_ch, mid_ch=0, nblocks=1, shortcut=True):
        super().__init__()
        self.n = nn.Sequential(*[res(in_ch, mid_ch, shortcut) for _ in range(nblocks)])
    def forward(self,x):
        return self.n(x)

class cat(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.c1 = Conv(in_ch, mid_ch, 1, 1)
        self.c2 = Conv(mid_ch, out_ch, 3, 1)
    def forward(self, x):
        return torch.cat((x,self.c2(self.c1(x))),dim=1)

class Catneck(nn.Module):
    def __init__(self, in_ch, cat_ch=16, d=4, nblocks=1):
        super().__init__()
        self.n = nn.Sequential(*[cat(in_ch+i*cat_ch, cat_ch*d, cat_ch) for i in range(nblocks)])
    def forward(self,x):
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
        #1*1 3*3
        self.c31 = Conv(in_ch, out_ch, 1, 1)
        self.c32 = Conv(out_ch, out_ch, 3, 1)
        #1*1, 3*3, 3*3
        self.c51 = Conv(in_ch, out_ch, 1, 1)
        self.c52 = Conv(out_ch, out_ch, 3, 1)
        self.c53 = Conv(out_ch, out_ch, 3, 1)
    def forward(self, x):
        if self.down:
            x = self.d(x)
        y1 = self.c1(x)
        y2 = self.c32(self.c31(x))
        y3 = self.c53(self.c52(self.c51(x)))
        return torch.cat((x,y1,y2,y3), dim=1) if self.shortcut else torch.cat((y1,y2,y3),dim=1)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
            
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout,maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out






