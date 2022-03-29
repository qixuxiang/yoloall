import torch
import numpy as np

from convert.torch2caffe.caffe import caffe_pb2 as pb
from convert.torch2caffe.utils import trans_log
from convert.torch2caffe.caffe_layer import Layer

def _view(input, *args):
    x = Raw.raw_view(input, *args)
    if not trans_log.NET_INITTED:
        return x
    layer_name = trans_log.add_layer(name='view')
    top_blobs = trans_log.add_blobs([x],name='view_blob')
    layer = Layer(name=layer_name,
                  type='Reshape',
                  bottom=[trans_log.blobs(input)],
                  top=top_blobs)
    #TODO:
    dims = list(args)
    dims[0] = 0
    layer.param.reshape_param.shape.CopyFrom(pb.BlobShape(dim=dims))
    trans_log.cnet.add_layer(layer)
    return x

def _mean(input, *args, **kwargs):
    x = Raw.raw_mean(input, *args, **kwargs)
    if not trans_log.NET_INITTED:
        return x
    layer_name = trans_log.add_layer(name='mean')
    top_blobs = trans_log.add_blobs([x],name='mean_blob')
    layer = Layer(name=layer_name,
                  type='Reduction',
                  bottom=[trans_log.blobs(input)],
                  top=top_blobs)
    if len(args) == 1:
        dim = args[0]
    elif 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        raise NotImplementedError('mean operation must specify a dim')
    layer.param.reduction_param.operation = 4
    layer.param.reduction_param.axis = dim
    trans_log.cnet.add_layer(layer)
    return x

def _add(input, *args):
    x = Raw.raw__add__(input, *args)
    if not trans_log.NET_INITTED:
        return x
    layer_name = trans_log.add_layer(name='add')
    top_blobs = trans_log.add_blobs([x],name='add_blob')
    if trans_log.blobs(args[0]) == None:
        trans_log.add_blobs([args[0]], name='extra_blob')
    else:
        layer = Layer(name=layer_name,
                    type='Eltwise',
                    bottom=[trans_log.blobs(input),
                            trans_log.blobs(args[0])],
                    top=top_blobs)
    layer.param.eltwise_param.operation = 1 #sum is 1
    trans_log.cnet.add_layer(layer)
    return x

def _iadd(input, *args):
    x = Raw.raw__iadd__(input, *args)
    if not trans_log.NET_INITTED:
        return x
    x = x.clone()
    layer_name = trans_log.add_layer(name='add')
    top_blobs = trans_log.add_blobs([x],name='add_blob')
    layer = Layer(name=layer_name,
                  type='Eltwise',
                  bottom=[trans_log.blobs(input),
                          trans_log.blobs(args[0])],
                  top=top_blobs)
    layer.param.eltwise_param.operation = 1 #sum is 1
    trans_log.cnet.add_layer(layer)
    return x

def _sub(input, *args):
    x = Raw.raw__sub__(input, *args)
    if not trans_log.NET_INITTED:
        return x
    layer_name = trans_log.add_layer(name='sub')
    top_blobs = trans_log.add_blobs([x],name='sub_blob')
    layer = Layer(name=layer_name,
                  type='Eltwise',
                  bottom=[trans_log.blobs(input),
                          trans_log.blobs(args[0])],
                  top=top_blobs)
    layer.param.eltwise_param.operation = 1 #sum is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    trans_log.cnet.add_layer(layer)
    return x

def _isub(input, *args):
    x = Raw.raw__isub__(input, *args)
    if not trans_log.NET_INITTED:
        return x
    x = x.clone()
    layer_name = trans_log.add_layer(name='sub')
    top_blobs = trans_log.add_blobs([x],name='sub_blob')
    layer = Layer(name=layer_name,
                  type='Eltwise',
                  bottom=[trans_log.blobs(input),
                          trans_log.blobs(args[0])],
                  top=top_blobs)
    layer.param.eltwise_param.operation = 1 #sum is 1
    trans_log.cnet.add_layer(layer)
    return x

def _mul(input, *args):
    x = Raw.raw__mul__(input, *args)
    if not trans_log.NET_INITTED:
        return x
    layer_name = trans_log.add_layer(name='mul')
    top_blobs = trans_log.add_blobs([x],name='mul_blob')
    layer = Layer(name=layer_name,
                  type='Eltwise',
                  bottom=[trans_log.blobs(input),
                          trans_log.blobs(args[0])],
                  top=top_blobs)
    layer.param.eltwise_param.operation = 0 #product is 1
    trans_log.cnet.add_layer(layer)
    return x

def _imul(input, *args):
    x = Raw.raw__imul__(input, *args)
    if not trans_log.NET_INITTED:
        return x
    x = x.clone()
    layer_name = trans_log.add_layer(name='mul')
    top_blobs = trans_log.add_blobs([x],name='mul_blob')
    layer = Layer(name=layer_name,
                  type='Eltwise',
                  bottom=[trans_log.blobs(input),
                          trans_log.blobs(args[0])],
                  top=top_blobs)
    layer.param.eltwise_param.operation = 0 #suproduct is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    trans_log.cnet.add_layer(layer)
    return x

def _permute(input, *args):
    x = Raw.raw__permute__(input, *args)
    name = trans_log.add_layer(name='permute')
    trans_log.add_blobs([x], name='permute_blob')
    layer = Layer(name=name,
                type='Permute',
                bottom=[trans_log.blobs(input)],
                top=[trans_log.blobs(x)])
    order1 = args[0]
    order2 = args[1]
    order3 = args[2]
    order4 = args[3]

    layer.permute_param(order1, order2, order3, order4)
    trans_log.cnet.add_layer(layer)
    return x

#contiguous
def _contiguous(input, *args):
    x = Raw.raw__contiguous__(input, *args)
    name = trans_log.add_layer(name='contiguous')
    trans_log.add_blobs([x],name='contiguous_blob')
    layer = Layer(name=name,
                  type='NeedRemove',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])
    trans_log.cnet.add_layer(layer)
    return x

#pow
def _pow(input, *args):
    x = Raw.raw__pow__(input, *args)
    trans_log.add_blobs([x],name='pow_blob')
    return x

#sum
def _sum(input, *args):
    x = Raw.raw__sum__(input, *args)
    trans_log.add_blobs([x],name='sum_blob')
    return x

#sqrt
def _sqrt(input, *args):
    x = Raw.raw__sqrt__(input, *args)
    trans_log.add_blobs([x],name='sqrt_blob')
    return x

#unsqueeze
def _unsqueeze(input, *args):
    x = Raw.raw__unsqueeze__(input, *args)
    trans_log.add_blobs([x],name='unsqueeze_blob')
    return x

def _expand_as(input, *args):
    #only support expand A(1,1,H,w) to B(1,c,h,w)

    x = Raw.raw__expand_as__(input, *args)

    layer_name = trans_log.add_layer(name='expand_as', with_num=True)
    trans_log.add_blobs([x],name='expand_as_blob')
    layer = Layer(name=layer_name,
                  type='Convolution',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])
    
    def constant_weight(shape):
        weights = np.ones(shape,dtype='float32')
        return weights
    
    channels = args[0].size(1)
    weight = constant_weight([channels, 1, 1, 1])
    layer.conv_param(channels,
                     kernel_size=1,
                     bias_term=False,
                     weight_filler_type='xavier')
    layer.add_data(weight)
    trans_log.cnet.add_layer(layer)
    return x

#flatten
def _flatten(input, *args):
    x = Raw.raw__flatten__(input, *args)

    if len(args) == 0:
        layer_name = trans_log.add_layer(name='reshape')
        top_blobs = trans_log.add_blobs([x],name='reshape_blob')
        layer = Layer(name=layer_name,
                    type='Reshape',
                    bottom=[trans_log.blobs(input)],
                    top=top_blobs)
        shape = list(x.shape)
        if x.shape[0] == input.shape[0]:
            shape[0] = 0
        else:
            shape[0] = -1
        layer.reshape_param(shape)
    else:
        layer_name = trans_log.add_layer(name='flatten')
        top_blobs = trans_log.add_blobs([x],name='flatten_blob') 
        layer = Layer(name=layer_name,
                     type='Flatten',
                     bottom=[trans_log.blobs(input)],
                     top=top_blobs)
        axis = args[0]
        end_axis = args[1] if len(args) > 1 else -1
        layer.flatten_param(axis, end_axis)
    trans_log.cnet.add_layer(layer)
    return x

#TODO other types of the view function
class Raw(object):
    def __init__(self):
        self.type = 'torch.Tensor'

def replace():

    setattr(Raw, 'raw_view', torch.Tensor.view)
    torch.Tensor.view = _view

    setattr(Raw, 'raw_mean', torch.Tensor.mean)
    torch.Tensor.mean = _mean

    setattr(Raw, 'raw__add__', torch.Tensor.__add__)
    torch.Tensor.__add__ = _add

    setattr(Raw, 'raw__iadd__', torch.Tensor.__iadd__)
    torch.Tensor.__iadd__ = _iadd

    setattr(Raw, 'raw__sub__', torch.Tensor.__sub__)
    torch.Tensor.__sub__ = _sub

    setattr(Raw, 'raw__isub__', torch.Tensor.__isub__)
    torch.Tensor.__isub__ = _isub

    setattr(Raw, 'raw__mul__', torch.Tensor.__mul__)
    torch.Tensor.__mul__ = _mul

    setattr(Raw, 'raw__imul__', torch.Tensor.__imul__)
    torch.Tensor.__imul__ = _imul

    setattr(Raw, 'raw__permute__', torch.Tensor.permute)
    torch.Tensor.permute = _permute

    setattr(Raw, 'raw__contiguous__', torch.Tensor.contiguous)
    torch.Tensor.contiguous = _contiguous



    setattr(Raw, 'raw__pow__', torch.Tensor.pow)
    torch.Tensor.pow = _pow

    setattr(Raw, 'raw__sum__', torch.Tensor.sum)
    torch.Tensor.sum = _sum

    setattr(Raw, 'raw__sqrt__', torch.Tensor.sqrt)
    torch.Tensor.sqrt = _sqrt

    setattr(Raw, 'raw__unsqueeze__', torch.Tensor.unsqueeze)
    torch.Tensor.unsqueeze = _unsqueeze

    setattr(Raw, 'raw__expand_as__', torch.Tensor.expand_as)
    torch.Tensor.expand_as = _expand_as

    setattr(Raw, 'raw__flatten__', torch.Tensor.flatten)
    torch.Tensor.flatten = _flatten

def reset():

    torch.Tensor.view = Raw.raw_view
    torch.Tensor.mean = Raw.raw_mean
    torch.Tensor.__add__ = Raw.raw__add__
    torch.Tensor.__iadd__ = Raw.raw__iadd__
    torch.Tensor.__sub__ = Raw.raw__sub__
    torch.Tensor.__isub__ = Raw.raw__isub__
    torch.Tensor.__mul__ = Raw.raw__mul__
    torch.Tensor.__imul__ = Raw.raw__imul__
    torch.Tensor.permute = Raw.raw__permute__
    torch.Tensor.contiguous = Raw.raw__contiguous__
    torch.Tensor.pow = Raw.raw__pow__
    torch.Tensor.sum = Raw.raw__sum__
    torch.Tensor.sqrt = Raw.raw__sqrt__
    torch.Tensor.unsqueeze = Raw.raw__unsqueeze__
    torch.Tensor.expand_as = Raw.raw__expand_as__
    torch.Tensor.flatten = Raw.raw__flatten__
