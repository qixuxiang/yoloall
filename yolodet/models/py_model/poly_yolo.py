import torch
import torch.nn as nn
import torch.nn.functional as F
from yolodet.models.common_py import Detect #Conv, Focus, Catneck, Incept
cfgfile='/home/yu/workspace/yoloall/yoloall/yolodet/models/cfg_model/poly_yolov3.cfg'

def parse_cfg(cfgfile):
    """
    Parse cfg file, and retur a bloc dictionary.
    :param cfgfile: cfg file path
    :return: blocks
    """
    with open(cfgfile, 'r') as f:
        lines = f.readlines()

    blocks = []     # store info of all blocks
    block = {}      # store info of single block

    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        if line[0] == '[':
            if block:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class UpSample(nn.Module):

    def __init__(self, scale_factor=2, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        assert (x.dim() == 4)
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YoloLayer(nn.Module):

    def __init__(self, name):
        super(YoloLayer, self).__init__()
        self.name = name

    def forward(self, x):
        return {self.name: x}


class Conv2D_BN_Leaky(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super(Conv2D_BN_Leaky, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            padding = (self.kernel_size - 1) // 2
        elif isinstance(self.kernel_size, tuple):
            padding = (self.kernel_size[0] - 1) // 2
        self.Conv2D = nn.Conv2d(self.in_c, self.out_c, self.kernel_size, stride=1, padding=padding, bias=False)
        self.BN = nn.BatchNorm2d(self.out_c)
        self.Leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.Conv2D(x)
        x = self.BN(x)
        x = self.Leaky(x)
        return x


class yolo_body(nn.Module):

    def __init__(self, num_anchors=9, num_classes=13):
        super(yolo_body, self).__init__()
        """Create Poly-YOLo model CNN body in Pytorch."""
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, yolo_layers):
        tiny = yolo_layers['tiny']
        small = yolo_layers['small']
        medium = yolo_layers['medium']
        big = yolo_layers['big']
        base = 6
        tiny = Conv2D_BN_Leaky(tiny.shape[1], base*32, (1, 1))(tiny)
        small = Conv2D_BN_Leaky(small.shape[1], base*32, (1, 1))(small)
        medium = Conv2D_BN_Leaky(medium.shape[1], base*32, (1, 1))(medium)
        big = Conv2D_BN_Leaky(big.shape[1], base*32, (1, 1))(big)

        up = UpSample(scale_factor=2, mode='bilinear')
        all = medium + up(big)
        all = small + up(all)
        all = tiny + up(all)

        num_filters = base * 32
        all = Conv2D_BN_Leaky(all.shape[1], num_filters, (1, 1))(all)
        all = Conv2D_BN_Leaky(all.shape[1], num_filters * 2, (3, 3))(all)
        all = Conv2D_BN_Leaky(all.shape[1], num_filters, (1, 1))(all)
        all = Conv2D_BN_Leaky(all.shape[1], num_filters*2, (3, 3))(all)
        all = nn.Conv2d(all.shape[1], self.num_anchors * (self.num_classes + 5), (1, 1))(all)# + NUM_ANGLES3

        return all
        # print(tiny.shape, small.shape, medium.shape, big.shape)


class yolo_body(nn.Module):

    def __init__(self, in_dim, num_anchors=9, num_classes=1):
        super(yolo_body, self).__init__()
        """Create Poly-YOLo model CNN body in Pytorch."""
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.base = 6
        self.CBL_tiny = Conv2D_BN_Leaky(in_dim[0], self.base * 32, (1, 1))
        self.CBL_small = Conv2D_BN_Leaky(in_dim[1], self.base * 32, (1, 1))
        self.CBL_medium = Conv2D_BN_Leaky(in_dim[2], self.base * 32, (1, 1))
        self.CBL_big = Conv2D_BN_Leaky(in_dim[3], self.base * 32, (1, 1))
        num_filters = self.base * 32

        self.all1 = Conv2D_BN_Leaky(self.base * 32, num_filters, (1, 1))
        self.all2 = Conv2D_BN_Leaky(num_filters, num_filters * 2, (3, 3))
        self.all3 = Conv2D_BN_Leaky(num_filters * 2, num_filters, (1, 1))
        self.all4 = Conv2D_BN_Leaky(num_filters, num_filters, (3, 3))
        #self.all5 = nn.Conv2d(num_filters * 2, self.num_anchors * (self.num_classes + 5), (1, 1))# + NUM_ANGLES3

    def forward(self, yolo_layers):
        tiny = yolo_layers['tiny']
        small = yolo_layers['small']
        medium = yolo_layers['medium']
        big = yolo_layers['big']
        tiny = self.CBL_tiny(tiny)
        small = self.CBL_small(small)
        medium = self.CBL_medium(medium)
        big = self.CBL_big(big)

        up = UpSample(scale_factor=2, mode='bilinear')
        all = medium + up(big)
        all = small + up(all)
        all = tiny + up(all)

        all = self.all1(all)
        all = self.all2(all)
        all = self.all3(all)
        all = self.all4(all)
        #all = self.all5(all)

        return all
        # print(tiny.shape, small.shape, medium.shape, big.shape)


class YOLO(nn.Module):

    def __init__(self, nc, anchors, ch, cfgfile=cfgfile):
        super(YOLO, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        #print(len(self.blocks))
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = self.create_modules(self.blocks)
        self.num_classes = nc
        self.num_anchors = len(anchors[0]) // 2
        self.detect = Detect(nc, anchors, [192])
        # print(self.modules)

    def get_yolo_layers(self):
        yolo_layers = {}
        for module in self.models:
            if isinstance(module[0], YoloLayer):
                yolo_layers[module[0].name] = module[0]
        return yolo_layers


    def darknet_body(self, x):
        yolo_layers = {}
        output = {}
        for index, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional':
                x = self.models[index](x)
                output[index] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                from_layer = int(from_layer) if int(from_layer) > 0 else int(from_layer) + index
                x1 = output[from_layer]
                x2 = output[index - 1]
                x = x1 + x2
                output[index] = x
            elif block['type'] == 'yolo':
                output[index] = x
                name = block['name']
                yolo_layers[name] = x
            elif block['type'] == 'yolo_body':
                self.models[index] = yolo_body([yolo_layers['tiny'].shape[1], yolo_layers['small'].shape[1], yolo_layers['medium'].shape[1], yolo_layers['big'].shape[1]], num_anchors=self.num_anchors, num_classes=self.num_classes).to(x.device)
                x = self.models[index](yolo_layers)

            else:
                print("Unknown type {0}".format(block['type']))

        return x

    def forward(self, x):
        x = self.darknet_body(x)
        return self.detect([x])

    def create_modules(self, blocks):
        self.net_info = blocks[0]
        self.width = int(self.net_info['width'])
        self.height = int(self.net_info['height'])
        prev_filters = int(self.net_info['channels'])
        out_filters = []
        models = nn.ModuleList()

        # iterate all blocks
        for index, block in enumerate(blocks[1:]):
            module = nn.Sequential()

            if block["type"] == 'convolutional':
                activation_func = block['activation']
                kernel_size = int(block['size'])
                pad = int(block['pad'])
                filters = int(block['filters'])
                stride = int(block['stride'])

                try:
                    batch_normalize = int(block['batch_normalize'])
                    bias = False
                except KeyError:
                    # no BN
                    batch_normalize = 0
                    bias = True

                if pad:
                    padding = (kernel_size - 1) // 2
                else:
                    padding = 0

                conv_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    module.add_module('batch_norm_{}'.format(index), nn.BatchNorm2d(filters))

                if activation_func == 'leaky':
                    activation = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('leaky_{0}'.format(index), activation)

            elif block['type'] == 'shortcut':
                module.add_module('shortcut_{0}'.format(index), EmptyLayer())

            elif block['type'] == 'yolo':
                name = block['name']
                module.add_module('feature_{0}'.format(index), YoloLayer(name))
            elif block['type'] == 'yolo_body':
                module.add_module('yolo_body_{}'.format(index), EmptyLayer())

            models.append(module)
            prev_filters = filters
            out_filters.append(prev_filters)

        return models

if __name__ == "__main__":
    cfgfile='/home/yu/workspace/yoloall/yoloall/yolodet/models/cfg_model/poly_yolov3.cfg'
    x = torch.rand([1,3,960,576]).to('cuda')
    nc = 14
    anchors = [[11,11, 26,37, 74,36, 124,39, 198,102]]
    ch = 3
    model = YOLO(nc, anchors, ch, cfgfile=cfgfile).to('cuda')
    print(len(model(x)))
    print(model(x)[0].shape)