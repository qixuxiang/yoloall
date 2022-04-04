from __future__ import division
import torch
import torch.nn as nn
from yolodet.models.common_py import Conv, Focus, Catneck, Incept, Detect


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

    imgs = torch.rand([64, 3, 384, 576]).cuda()
    model = YOLO(ch=3, anchors=[[6, 5, 12, 7, 10, 16, 23, 12], [17, 25, 36, 21, 34, 41, 66, 38], [60, 88, 110, 67, 156, 163, 319, 213]], nc=5).cuda()
    model(imgs)
    