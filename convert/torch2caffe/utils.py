import traceback
from convert.torch2caffe.caffe_net import Net

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

def pair_process(item, strict_one=False):
    if hasattr(item, '__iter__'):
        for i in item:
            if i != item[0]:
                if strict_one:
                    raise ValueError(
                        "number in item {} must be the same".format(item))
                else:
                    return item
        return item
    return [item,item]

def pair_reduce(item):
    if hasattr(item, '__iter__'):
        for i in item:
            if i != item[0]:
                return item
        return [item[0]]
    return [item]

class Blob_LOG():
    def __init__(self):
        self.data = {}
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class TransLog(object):
    def __init__(self):
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.debug = True

        self.NET_INITTED = False
        self.layer_names = {}
        
        self.pytorch_layer_name = None
        self.simple_name = True

        self.ckpt = 'detect.pth'
        self.output_blob_pre = ''
        self.cnet = None

        self.merge_bn = False
        self.layer_type_order = []
        self.layer_name_order = []

    def init(self):
        self.cnet = Net(ckpt=self.ckpt, output_blob_pre=self.output_blob_pre)
        # self.add_blobs(inputs, name='data', with_num = False)
    
    def add_layer(self, name='layer'):
        self.layer_type_order.append(name)
        if not self.simple_name:
            if self.pytorch_layer_name:
                name = self.pytorch_layer_name
        
        if name in self.layers and  name != 'data':
            self.layer_name_order.append(name)
            return self.layers[name]
        
        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0
        
        self.detail_layers[name] += 1
        if name == 'data' and self.detail_layers[name] == 1:
            name = '{}'.format(name)
        else:
            name = '{}{}'.format(name, self.detail_layers[name])
        
        self.layers[name] = name

        if self.debug:
            print("\t{} was add to layers".format(self.layers[name]))
        
        self.layer_name_order.append(name)
        return self.layers[name]

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst = []
        for blob_idx, blob in enumerate(blobs):
            self._blobs_data.append(blob)
            blob_id = int(id(blob))
            if name not in self.detail_blobs.keys():
                self.detail_blobs[name] = 0
            self.detail_blobs[name] += 1
            if with_num and self.detail_blobs[name] != 1:
                rst.append('{}{}'.format(name, self.detail_blobs[name]))
            else:
                rst.append('{}'.format(name))
            if self.detail_blobs:
                print(
                    "\ttop blob {}: {}:{}, dim:{}, was added to blobs".format(
                        blob_idx + 1, blob_id, rst[-1],
                        [int(dim) for dim in blob.shape]))
            
            self._blobs[blob_id] = rst[-1]
        return rst
    
    def blobs(self, var):
        var_id = id(var)
        try:
            return self._blobs[var_id]
        except:
            raise Exception("CANNOT FOUND blob {}, blob shape: {}".format(
                var_id, var.shape))

trans_log = TransLog()

class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        self.obj = replace
        self.raw = raw

    def __call__(self,*arg, **kwargs):
        if not trans_log.NET_INITTED:
            return self.raw(*arg,**kwargs)
        input_layer = True
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                if layer in trans_log.layer_names:
                    input_layer = False
                    trans_log.pytorch_layer_name = trans_log.layer_names[layer]
                    if trans_log.layer_names[layer]:
                        print(colorstr('blue','bold',trans_log.layer_names[layer]))
                    break
        if input_layer:
            print(colorstr('blue','bold', 'input'))
        out = self.obj(self.raw, *arg, **kwargs)
        return out


if __name__ == '__main__':

    def nothing():
        pass

    import torch
    import torchvision.ops as ops
    trans_log.NET_INITTED = True

    torch.ops.torchvision.roi_pool = Rp(torch.ops.torchvision.roi_pool,
                                        nothing)
    pool = ops.RoIPool(spatial_scale=0.5, output_size=7)

    input = torch.ones([1, 3, 256, 256])
    box = torch.Tensor([0, 0, 0, 32, 32])
    pool(input, box)
