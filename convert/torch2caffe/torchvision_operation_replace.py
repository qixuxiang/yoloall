import torch
import torchvision.ops as ops
from convert.torch2caffe.utils import trans_log, Rp
from convert.torch2caffe.caffe_layer import Layer


###################-------for torchvision operation-------------############

def _ror_pool(raw, input, rois, spatial_scale, out_size_h, out_size_w):
    print("\troi_pool bottom blob: {} ,dim:{}".format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))

    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]
    x,_ = raw(input, rois, spatial_scale, out_size_h, out_size_w)
    name = trans_log.add_layer(name='roi_pool')

    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name = name,
                  type = 'ROIPooling',
                  bottom=bottom_blobs,
                  top=top_blobs)

    layer.roi_pooling_param(out_size_h, out_size_w, spatial_scale)

    trans_log.cnet.add_layer(layer)
    return x, _


def _roi_align(raw,
               input,
               rois,
               spatial_scale,
               out_size_h,
               out_size_w,
               sampling_ratio=-1,
               aligned=False):
    
    print("\troi_align bottom blob: {}, dim:{}".format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))
    
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]

    x = raw(input, rois, spatial_scale, out_size_h, out_size_w, sampling_ratio, aligned)
    name = trans_log.add_layer(name='roi_align')
    top_blobs = trans_log.add_blobs([x],name='roi_blob')

    layer = Layer(name=name,
                  type='ROIAlign',
                  botto=bottom_blobs,
                  top=top_blobs)
    
    layer.roi_align_param(out_size_h, out_size_w, spatial_scale)

    trans_log.cnet.add_layer(layer)
    return x


from torch import _VF

def replace():

    torch.ops.torchvision.roi_pool = Rp(torch.ops.torchvision.roi_pool,
                                        _roi_align)
    torch.ops.torchvision.roi_align = Rp(torch.ops.torchvision.roi_align,
                                         _roi_align)

def reset():

    torch.ops.torchvision.roi_pool = torch.ops.torchvision.roi_pool.raw
    torch.ops.torchvision.roi_align = torch.ops.torchvision.roi_align.raw

