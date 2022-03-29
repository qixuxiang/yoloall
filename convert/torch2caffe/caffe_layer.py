from convert.torch2caffe.caffe import caffe_pb2 as pb
from convert.torch2caffe.utils import trans_log, pair_process, pair_reduce


class Layer(object):
    def __init__(self, name='', type='', top=(), bottom=()):
        self.param = pb.LayerParameter()
        self.name = self.param.name = name
        self.type = self.param.type = type

        self.bottom = self.param.bottom
        if bottom:
            self.bottom.extend(bottom)
            trans_log.cnet.bottom_blobs.extend(bottom)
            trans_log.cnet.blobs.extend(bottom)

        self.top = self.param.top
        if top:
            self.top.extend(top)
            trans_log.cnet.blobs.extend(top)
    
    def add_data(self, *args):
        """Args are data numpy blob
        """
        del self.param.blobs[:]
        for data in args:
            new_blob = self.param.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))
    
    def input_param(self, shape):

        if self.type != 'Input':
            raise TypeError(
                'the layer type must be Input if you want set input param')
        
        input_param = pb.InputParameter()
        new_shape = input_param.shape.add()
        new_shape.dim.extend(shape)

        self.param.input_param.CopyFrom(input_param)

    def fc_param(self,
                 num_output,
                 weight_filler='xavier',
                 bias_filler='constant',
                 has_bias=True):
        if self.type != 'InnerProduct':
            raise TypeError(
                'the layer type must be InnerProduct if you want set fc param')
        fc_param = pb.InnerProductParameter()
        fc_param.num_output = num_output
        fc_param.weight_filler.type = weight_filler
        fc_param.bias_term = has_bias
        if has_bias:
            fc_param.bias_filler.type = bias_filler
        self.param.inner_product_param.CopyFrom(fc_param)

    def conv_param(self,
                   num_output,
                   kernel_size,
                   stride=(1),
                   pad=(0,),
                   weight_filler_type='xavier',
                   bias_filler_type='constant',
                   bias_term=True,
                   dilation=None,
                   groups=None):
        if self.type not in ['Convolution','Deconvolution']:
            raise TypeError(
                'The layer type must be Convolution or Deconvolution if you want sent conv param')
        conv_param = pb.ConvolutionParameter()
        conv_param.num_output = num_output
        conv_param.bias_term = bias_term
        conv_param.weight_filler.type = weight_filler_type

        kernel_size = pair_reduce(kernel_size)
        if len(kernel_size) == 1:
            conv_param.kernel_size.extend(kernel_size)
        else:
            conv_param.kernel_h = kernel_size[0]
            conv_param.kernel_w = kernel_size[1]
        
        stride = pair_reduce(stride)
        if len(stride) == 1:
            conv_param.stride.extend(stride)
        else:
            conv_param.stride_h = stride[0]
            conv_param.stride_w = stride[1]
        
        pad = pair_reduce(pad)
        if len(pad) == 1:
            conv_param.pad.extend(pad)
        else:
            conv_param.pad_h = pad[0]
            conv_param.pad_w = pad[1]
        
        if bias_term:
            conv_param.bias_filler.type = bias_filler_type

        if dilation:
            conv_param.dilation.extend(pair_reduce(dilation))
        
        if groups:
            conv_param.group = groups
            if groups != 1:
                conv_param.engine = 1
        
        self.param.convolution_param.CopyFrom(conv_param)
    
    def norm_param(self, eps):
        l2norm_param = pb.NormalizeParameter()
        l2norm_param.across_spatial = False
        l2norm_param.channel_shared = False
        l2norm_param.eps = eps
        self.param.norm_param.CopyFrom(l2norm_param)
    
    def permute_param(self, order1, order2, order3, order4):
        permute_param = pb.PermuteParameter()
        permute_param.order.extend([order1, order2, order3, order4])

        self.param.permute_param.CopyFrom(permute_param)
    
    def pool_param(self,
                   type='MAX',
                   kernel_size=2,
                   stride=2,
                   pad=None,
                   ceil_mode=True):
        if self.type not in ['Pooling']:
            raise TypeError(
                'The layer type must be Pooling if you want set pool param')
        
        pool_param = pb.PoolingParameter()
        pool_param.pool = pool_param.PoolMethod.Value(type)

        kernel_size = pair_process(kernel_size)
        if kernel_size[0] == kernel_size[1]:
            pool_param.kernel_size = kernel_size[0]
        else:
            pool_param.kernel_h = kernel_size[0]
            pool_param.stride_w = kernel_size[1]

        stride = pair_process(stride)
        if stride[0] == stride[1]:
            pool_param.stride = stride[0]
        else:
            pool_param.stride_h = stride[0]
            pool_param.stride_w = stride[1]

        if pad:
            pad = pair_process(pad)
            if pad[0] == pad[1]:
                pool_param.pad = pad[0]
            else:
                pool_param.pad_h = pad[0]
                pool_param.pad_w = pad[1]

        self.param.pooling_param.CopyFrom(pool_param)

    def batch_morm_param(self,
                         use_global_stats=0,
                         moving_average_fraction=None,
                         eps=None):   
        bn_param = pb.BatchNormParameter()
        bn_param.use_global_stats = use_global_stats
        if moving_average_fraction:
            bn_param.moving_average_fraction = moving_average_fraction
        if eps:
            bn_param.eps = eps
        self.param.batch_morm_param.CopyFrom(bn_param)

    def upsample_param(self, size=None, scale_factor=None):
        upsample_param = pb.UpsampleParameter()
        if scale_factor:
            if isinstance(scale_factor, int):
                upsample_param.scale = scale_factor
            else:
                upsample_param.stride_h = scale_factor[0]
                upsample_param.stride_w = scale_factor[1]
        if size:
            if isinstance(size, int):
                upsample_param.upsample_h = size
            else:
                upsample_param.upsample_h = size[0] * scale_factor
                upsample_param.upsample_w = size[1] * scale_factor
        self.param.upsample_param.CopyFrom(upsample_param)
    
    def roi_pooling_param(self,
                          output_size_h=0,
                          output_size_w=0,
                          spatial_scale=1):
        if self.type != 'ROIPooling':
            raise TypeError(
                'the layer type must be ROIPooling if you want set roi pooling param')
        roi_pooling_param = pb.ROIPoolingParameter()

        roi_pooling_param.pooled_h = int(output_size_h)
        roi_pooling_param.pooled_w = int(output_size_w)
        roi_pooling_param.spatial_scale = spatial_scale
        self.param.roi_pooling_param.CopyFrom(roi_pooling_param)
    
    def roi_align_param(self,
                         output_size_h=0,
                         output_size_w=0,
                         spatial_scale=1):
        if self.type != 'ROIAlign':
            raise TypeError(
                'the layer type must be ROIPooling if you want want set roi ROIAlign param')
        roi_align_param = pb.RoIAlignParameter()

        roi_align_param.pooled_h = int(output_size_h)
        roi_align_param.pooled_w = int(output_size_w)

        roi_align_param.spatial_scale.CopyFrom(roi_align_param)

    def reshape_param(self, shape):
        if self.type != 'Reshape':
            raise TypeError(
                'the layer type must be Reshape if you want set roi reshape param'
            )
        
        reshape_param = pb.ReshapeParameter()
        new_shape = reshape_param.shape.add()
        new_shape.dim.extend(shape)

        self.param.reshape_param.CopyFrom(reshape_param)
    
    def flatten_param(self, axis=1, end_axis=-1):
        if self.type != 'Flatten':
            raise TypeError(
                'the layer type must be Flatten if you want set roi flatten param'
            )
        flatten_param = pb.FlattenParameter()
        flatten_param.axis = axis
        flatten_param.end_axis = end_axis

        self.param.flatten_param.CopyFrom(flatten_param)

    def slice_param(self, axis, slice_point):
        if self.type != 'Slice':
            raise TypeError(
                'the layer type must be slice if you want set roi slice param'
            )
        slice_param = pb.SliceParameter(axis=axis, slice_point=slice_point)
        self.param.slice_param.CopyFrom(slice_param)

    def continuation_indicator_param(self, time_step=0, batch_size=0):
        continuation_indicator_param = pb.ContinuationIndicatorParameter()
        continuation_indicator_param.time_step = int(time_step)
        continuation_indicator_param.batch_size = int(batch_size)
        self.param.continuation_indicator_param.CopyFrom(continuation_indicator_param)
    
    def recurrent_param(self, num_output, weight_filler, bias_filler, expose_hidden):
        recurrent_param = pb.RecurrentParameter()
        recurrent_param.num_output = int(num_output)
        recurrent_param.weight_filler = int(weight_filler)
        recurrent_param.bias_filler = int(bias_filler)
        recurrent_param.expose_hidden = int(expose_hidden)
        self.param.recurrent_param.CopyFrom(recurrent_param)

    def set_params_by_dict(self, dic):
        pass

    def copy_from(self, layer_param):
        pass

if __name__ == '__main__':

    import torch
    trans_log.init([torch.ones([1,1,384,576])])

    layer = Layer(name='test',type='Pooling')
    layer.pool_param(type='MAX',
                     kernel_size=(2,2),
                     stride=(2,2),
                     pad=None,
                     ceil_mode=False)
    layer = Layer(name='data', type='Input', top=['data'])
    layer.input_param(shape=[1,3,384,576])