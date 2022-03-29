import os 
import torch
from convert.torch2caffe.utils import trans_log
from convert.torch2caffe.torch_function_replace import replace as replace_torch_function
from convert.torch2caffe.torch_operation_replace import replace as replace_torch_operation
from convert.torch2caffe.torch_tensor_operation_replace import replace as replace_torch_tensor_operation
from convert.torch2caffe.torchvision_operation_replace import replace as replace_torchvision_operation
from convert.torch2caffe.torch_class_replace import replace as replace_torch_class

from convert.torch2caffe.torch_function_replace import reset as reset_torch_function
from convert.torch2caffe.torch_operation_replace import reset as reset_torch_operation
from convert.torch2caffe.torch_tensor_operation_replace import reset as reset_torch_tensor_operation
from convert.torch2caffe.torchvision_operation_replace import reset as reset_torchvision_operation
from convert.torch2caffe.torch_class_replace import reset as reset_torch_class

from convert.torch2caffe.draw_net import draw_caffe_net

""""
How to support a new layer type:
layer_name = trans_log.add_layer(layer_type_name)
top_blobs = trans_log.add_blobs(<out_put of that layer>)
layer = Layer(xxx)
<set layer parameter>
[<layer.add_data(*datas)>]
trans_log.cnet.add_layer(layer)
"""

def trans_net(net,
              input_size,
              ckpt='detect.pth',
              merage_bn=False,
              simple_name=True,
              out_blob_pre='',
              draw=False,
              mmdet=False):
    print('Starting Transform, This will take a while')
    net.eval()

    trans_log.simple_name = simple_name
    trans_log.out_blob_pre = out_blob_pre
    trans_log.ckpt = ckpt
    trans_log.merage_bn = merage_bn

    trans_log.init() #only init cnet

    trans_log.cnet.net.name = ckpt

    trans_log.NET_INITTED = True

    for name, layer in net.named_modules():
        trans_log.layer_names[layer] = name
    
    replace_torch_function()
    replace_torch_operation()
    replace_torch_tensor_operation()
    replace_torchvision_operation()
    replace_torch_class()

    if not mmdet:
        input_var = torch.ones(input_size)
        out = net.forward(input_var)
    else:
        input_var = torch.ones(input_size)
        out = net.forward_convert_caffe(input_var)
    print('Transform Completed\n')

    prototxt, caffemodel, output_blobs = trans_log.cnet.save()

    if draw:
        output_image_file = os.path.splitext(prototxt)[0] + '.jpg'
        draw_caffe_net(input_net_proto_file = prototxt,
                       output_image_file=output_image_file,
                       rankdir='LR',
                       phase=None,
                       display_lrm=None)
    reset_torch_function()
    reset_torch_operation()
    reset_torch_tensor_operation()
    reset_torchvision_operation()
    reset_torch_class()

    return prototxt, caffemodel, output_blobs

