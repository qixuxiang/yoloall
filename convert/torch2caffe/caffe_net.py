import os
import six
import numpy as np
from copy import deepcopy
import google.protobuf.text_format as text_format

from convert.torch2caffe.caffe import caffe_pb2 as pb

class Net(object):
    def __init__(self, ckpt='detect.pth', output_blob_pre='',caffemodel=None):
        self.net = pb.NetParameter()
        if caffemodel is not None and os.path.exists(caffemodel):
            with open(caffemodel,'rb') as f:
                self.net.ParseFromString(f.read())
        self.ckpt = os.path.join('caffemodel', os.path.basename(ckpt))
        if caffemodel:
            self.ckpt = os.path.join('caffemodel',os.path.basename(caffemodel))
        
        self.needChange = {}

        self.blobs = []
        self.bottom_blobs = []
        self.output_blobs = {}
        self.output_pre = output_blob_pre

        self.last_input_name = ''

    def save(self):

        self.remove_layer_by_type("NeedRemove")
        self.set_output_blob_name()

        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
        caffemodel = os.path.splitext(self.ckpt)[0] + '.caffemodel'
        prototxt = os.path.splitext(self.ckpt)[0] + '.prototxt'
        self.save_prototxt(prototxt)
        self.save_caffemodel(caffemodel)

        return prototxt,caffemodel, self.output_blobs

    def save_caffemodel(self, path):
        print(colorstr('blue','bold', '>>Save caffemodel:') + f"  {os.path.abspath(path)}")
        with open(path, 'wb') as f:
            f.write(self.net.SerializeToString())

    def save_prototxt(self, path):
        print(colorstr('blue','bold', '>>Save prototxt:') + f"  {os.path.abspath(path)}")

        net = deepcopy(self.net)

        name = net.name
        inputs = net.input
        inputs_num = len(inputs)
        input_dims = net.input_dim
        input_dims_num = len(input_dims)

        del_num = 0
        info = ''

        if inputs_num + input_dims_num > 0:
            info = f'name: "{name}"\n'
            info += 'layer {\n'
            info += '   name: "data"\n'

            info += '   type: "Input"\n'
            info += '   top: "data"\n'
            info += '   input_param: {\n'
            info += '       shape {\n'
            for dim in input_dims:
                info += f'              dim:{int(dim)}\n'
            info += '       }\n'
            info += '   }\n'
            info += '}\n'

            del_num = 1 + inputs_num + input_dims_num
        for layer in net.layer:
            del layer.blobs[:]
        
        net_text = text_format.MessageToString(net)
        net_text = net_text.split('\n')[del_num:]
        net_text = '\n'.join(net_text)

        for origin_name in self.needChange.keys():
            net_text = net_text.replace(origin_name,
                                        self.needChange[origin_name])
        
        if self.output_pre:
            print('change output blob name:')
            for output_blob in self.output_blobs.keys():
                print('\t' + output_blob + ' -> ' + 
                        colorstr('red', 'bold', self.output_blobs[output_blob]))
                
                net_text = net_text.replace(output_blob, 
                                            self.output_blobs[output_blob])
        else:
            print('output blob name:')
            for output_blob in self.output_blobs.keys():
                print('\t' + colorstr('red', 'bold', output_blob))
        
        with open(path, 'w') as f:
            f.write(info)
            f.write(net_text)
    
    def add_layer(self, layer, before='', after=''):
        index = -1
        if after != '':
            index = self.layer_index(before) + 1
        if before != '':
            index = self.layer_index(before)
        new_layer = pb.LayerParameter()
        new_layer.CopyFrom(layer.param)

        if index != -1:
            self.net.layer.add()
            for i in range(len(self.net.layer) -1, index, -1):
                self.net.layer[i].CopyFrom(self.net.layer[i - 1])
            self.net.layer[index].CopyFrom(new_layer)
        else:
            self.net.layer.extend([new_layer])
    
    def layer_index(self, layer_name):
        for i ,layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                return i
    
    def remove_layer_by_name(self, layer_name):
        layer_index = self.layer_index(layer_name)
        del self.net.layer[layer_index]
    
    def remove_layer_by_type(self, type_name):
        for i,layer in enumerate(self.net.layer):
            if layer.type == type_name:
                s1 = "\"" + layer.top[0] + "\""
                s2 = "\"" + layer.bottom[0] + "\""
                self.needChange[s1] = s2
                del self.net.layer[i]
        return
    
    def set_output_blob_name(self):
        self.bottom_blobs = set(self.bottom_blobs)

        output_num = 0
        for blob in self.blobs:
            if blob not in self.bottom_blobs and not blob.startswith('data'):

                if blob in self.needChange.keys():
                    blob = self.needChange[blob]
                self.output_blobs[blob] = f'{self.output_pre}{output_num}'
                output_num += 1

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

if __name__ == '__main__':
    caffemodel = ''
    model = Net(caffemodel=caffemodel)

    for layer in model.net.layer:
        print(layer.name)
        print(layer.type)
        print('blobs:',len(layer.blobs),
              [blob.shape.dim for blob in layer.blobs])
        print('bottoms:',len(layer.bottom),
              [bottom for bottom in layer.bottom])
        print('tops:', len(layer.top), [top for top in layer.top])
        print(
            '------------------------------------------------------------')
    model.save()



