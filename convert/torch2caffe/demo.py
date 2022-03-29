import os
import argparse
import torch
import torch.nn as nn

from convert.torch2caffe.pytoch_to_caffe import trans_net

#from torchvision.ops import RoIAlign, RoIPool

from mmcv.ops.roi_pool import RoIPool

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple_name", type=int, default=1)
    parser.add_argument("--pre", type=str, default='test')
    parser.add_argument("--draw", type=int, default=0)
    opt = parser.parse_args()

    return opt

class Model(nn.Modules):
    def __init__(self):
        super(Model, self).__init__()
        # self.pool1 = RoIPool(spatial_scale=0.25,
        #                     output_size = 7,
        #                     sampling_ratio=-1,
        #                     aligned=False)
        self.pool12 = RoIPool(spatial_scale=0.5,output_size=7)
        # self.rnn_cell = nn.GRUCell(100, 100)
    
    def forward(self, x):
        box = torch.tensor([[0,0,0,32,32]])
        # roi1 = self.pool1(x,box)
        # kk = torch.ones([3,3,3])
        
        roi2 = self.pool12(x, box)
        return roi2
    
if __name__ == "__main__":
    opt = get_args()

    model = Model()
    input_size = [1,3,800,1333]
    trans_net(model,
              input_size,
              ckpt='detect.path',
              merage_bn=True,
              simple_name=opt.simple_name,
              out_blob_pre=opt.pre,
              draw=opt.draw)


