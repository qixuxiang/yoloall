from torchvision.ops import roi_pool,roi_align,RoIPool, RoIAlign

import torch
import numpy as np

import torch.nn as nn
from torch.nn.modules.utils import _pair

from convert.torch2caffe.caffe import caffe_pb2 as pb
from convert.torch2caffe.utils import trans_log
from convert.torch2caffe.caffe_layer import Layer


try:
    from mmcv.ops.roi_pool import RoIPool as mmcv_RoIPool
    from mmcv.ops.roi_align import RoIAlign as mmcv_RoIAlign
except:
    pass

def _roi_pool_forward(self, input, rois):
    print('mmcv roi pool')
    print('\troi_pool bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [input(dim) for dim in input.shape]))
    
    if input.device.type == 'cpu':
        input_cu = input.to('cuda')
        rois_cu = rois.to('cuda')

        x = Raw.roi_pool_forward(self, input_cu, rois_cu)
        x = x.cpu()
    else:
        x = Raw.roi_pool_forward(self, input, rois)
    
    #rois input layer
    #creat roi in forward,use torch.ones or torch.zeros, and roi will be registed
    # roi_name = trans_log.add_layer(name='data')
    # roi_top_blobs = trans_log.add_blobs([rois],name='data')
    # input_layer = Layer(name=roi_name, type='Input', top=roi_top_blobs)
    # input_layer.input_param(rois.shape)
    # trans_log.cnet.add_layer(input_layer,after=trans_log.cnet.last_input_name)
    # trans_log.cnet.last_input_name = roi_name

    #rois pool layer
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]
    name = trans_log.add_layer(name='roi_pool')
    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name=name,
                  type='ROIPooling',
                  bottom=bottom_blobs,
                  top=top_blobs)
    if isinstance(self.output_size, int):
        self.output_size = [self.output_size,self.output_size]
    layer.roi_pooling_param(self.out_size[0], self.output_size[1], self.spatial_scale)
    
    trans_log.cnet.add_layer(layer)
    return x

def _roi_align_forward(self, input, rois):
    print('mmcv roi align')
    print('\troi_align bottom blob: {},dim:{}'.format(
        trans_log.blobs(input),[int(dim) for dim in input.shape]))
    
    if input.device.type == 'cpu':
        input_cu = input.to('cuda')
        rois_cu = rois.to('cuda')

        x = Raw.roi_pool_forward(self, input_cu, rois_cu)
        x = x.cpu()
    else:
        x = Raw.roi_pool_forward(self, input, rois)
    
    #rois input layer
    #creat roi in forward,use torch.ones or torch.zeros, and roi will be registed
    # roi_name = trans_log.add_layer(name='data')
    # roi_top_blobs = trans_log.add_blobs([rois],name='data')
    # input_layer = Layer(name=roi_name, type='Input', top=roi_top_blobs)
    # input_layer.input_param(rois.shape)
    # trans_log.cnet.add_layer(input_layer,after=trans_log.cnet.last_input_name)
    # trans_log.cnet.last_input_name = roi_name

    #rois pool layer
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]
    name = trans_log.add_layer(name='roi_align')
    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name=name,
                  type='ROIAlign',
                  bottom=bottom_blobs,
                  top=top_blobs)

    if isinstance(self.output_size, int):
        self.output_size = [self.output_size,self.output_size]
    layer.roi_align_param(self.out_size[0], self.output_size[1], self.spatial_scale)
    
    trans_log.cnet.add_layer(layer)
    return x

#针对 const_data = torch_tensor 的操作
try:
    from videodet.models.vid_common import TSHSub
except:
    pass
def _tsh_sub_forward(self, input):
    print('tsh sub')
    print('\ttsh_sub bottom blob:{}, dim:{}'.format(
        trans_log.blobs(input),[int(dim) for dim in input.shape]
    ))
    x = Raw.tsh_sub(self, input)
    top_blobs = trans_log.add_blobs([x],name='sub_blob')
    layer_name = trans_log.add_layer(name='sub_scale')
    layer = Layer(name=layer_name,
                  type='Scale',
                  bottom=[trans_log.blobs(input)],
                  top=top_blobs)
    
    layer.param.scale_param.bias_term = True
    weight = np.ones(x.shape) * -1
    bias = np.zeros(x.shape) + self.const
    layer.add_data(weight,bias)

    trans_log.cnet.add_layer(layer)

    return x

class Raw(object):
    def __init__(self):
        self.type = 'torch.Tensor'

def replace():
    pass

    try:
        setattr(Raw,'roi_pool_forward', mmcv_RoIPool.forward)
        mmcv_RoIPool.forward = _roi_pool_forward

        setattr(Raw, 'roi_align_forward', mmcv_RoIAlign.forward)
        mmcv_RoIAlign.forward = _roi_align_forward
    except:
        pass

    try:
        setattr(Raw,'tsh_sub',TSHSub.forward)
        TSHSub.forward = _tsh_sub_forward
    except:
        pass
def reset():
    try:
        mmcv_RoIPool.forward = Raw.roi_pool_forward
        mmcv_RoIAlign.forward = Raw.roi_align_forward
    except:
        pass

    try:
        TSHSub.forward = Raw.tsh_sub
    except:
        pass