import os
import pdb
import math
import time
import yaml
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from threading import Thread

import torch
import torch.nn as nn
#from yolodet.models.common_py import Shortcut, Swish, Mish, DeformConv2d, FeatureConcat
from yolodet.models.common_py import Detect

def parse_model_cfg(path):
    #解析模型
    if not path.endswith('cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path
    
    with open(path, 'r')as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val
        
        if mdefs[-1]['type'] == 'yolo':
            mdf = mdefs[-2]
            assert mdf['type'] == 'convolutional', 'yolo input is not convolutional'
        
    return mdefs

def creat_modules(module_defs, nc, anchors, multi_data=False, multi_head=False):

    output_filters = [module_defs[0]['channels']]
    _ = module_defs.pop(0)
    module_list = nn.ModuleList()
    routs = []
    detect_layer = []
    detect_ch = []

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional' or mdef['type'] == 'out_conv':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                conv_name = 'conv'
                modules.add_module(conv_name, nn.Conv2d(in_channels=output_filters[-1],
                                   out_channels = filters,
                                   kernel_size=k,
                                   stride=stride,
                                   padding=k // 2 if mdef['pad'] else 0,
                                   groups=mdef['groups'] if 'groups' in mdef else 1,
                                   bias=not bn))
            else: #multiple-size conv
                # modules.add_module('MixConv2d',MixConv2d(in_ch=output_filters[-1],
                #                   out_ch=filters,
                #                   k=k,
                #                   stride=stride,
                #                   bias=not bn))
                raise

            if bn:
                modules.add_module('bn', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)
            
            if mdef['activation'] == 'relu':
                modules.add_module('act', nn.ReLU(inplace=True))
            if mdef['activation'] == 'leaky':
                modules.add_module('act', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('act', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('act', Mish())
            
        elif mdef['type'] == 'deformableconvolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module('DeformConv2d', DeformConv2d(output_filters[-1],
                                                    filters,
                                                    kernel_size=k,
                                                    padding=k // 2 if mdef['pad'] else 0,
                                                    stride=stride,
                                                    bias=not bn,
                                                    modulation=True))
            else:
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                            out_ch=filters,
                                                            k=k,
                                                            stride=stride,
                                                            bias=not bn))
            
            if bn:
                modules.add_module('bn', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routes.append(i)
            
            if mdef['activation'] == 'relu':
                modules.add_module('act', nn.ReLU(inplace=True))
            if mdef['activation'] == 'leaky':
                modules.add_module('act', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('act', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('act', Mish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])
            
        elif mdef['type'] == 'maxpool':
            k = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k -1) // 2, ceil_mode=True)
            if k == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0,1,0,1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
            
        elif mdef['type'] == 'upsample':
            if 'weight_filler' in mdef and mdef['weight_filler'] == 'bilinear':
                modules.add_module('deconv', nn.Upsample(scale_factor=mdef['stride'], mode='bilinear'))
            else:
                modules.add_module('upsample', nn.Upsample(scale_factor=mdef['stride'], mode='nearest'))

        elif mdef['type'] == 'route':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + 1 if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers, activation=mdef['activation'] if 'activation' in mdef else None)

        elif mdef['type'] == 'route2':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat2(layers=layers)
        
        elif mdef['type'] == 'route3':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat3(layers=layers)
        
        elif mdef['type'] == 'route_lhalf':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers]) // 2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mdef['type'] == 'shortcut':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = Shortcut(layers=layers)
        
        elif mdef['type'] == 'reorg3d':
            pass

        elif mdef['type'] == 'yolo':
            detect_layer.append(i)
            routs.append(i)
            output_filters[-1] = module_list[-1][0].in_channels
            module_list[-1] = nn.Identity()
            detect_ch.append(output_filters[-1])

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        module_list.append(modules)
        output_filters.append(filters)
        
    #create Detect layer
    module_list.append(Detect(nc, anchors, detect_ch))
    module_list[-1].f = detect_layer

    routs_binary = [False] * (i + 2)
    for i in routs:
        routs_binary[i] = True
    
    return module_list, routs_binary


def creat_new_torch_model(multi_data, multi_head):

    torch_model_list = [
        'from __future__ import division',
        'import torch',
        'import torch.nn as nn',
        'from yolodet.models.common_py import Conv, Focus, Catneck, Incept, Detect, Resneck',
        '',
        'class YOLO(nn.Module):'
        '   def __init__(self, nc, anchors, ch, heads=None):',
        '       super().__init__()',
        '',            
    ]
    torch_forward_list = [
        '',
        '   def forward(self, *x):',
        '       x, *other = x',
        '',
    ]

    if multi_head:
        detect_model_list = [
        '',
        '       self.heads = heads',
        '       input_channel = [{}]',
        '       build_detect(self, input_channel,nc anchors, heads)',
        ]

        detect_forward_list = [
        '',
        '       return forward_detect(self, out)',
        ]
    elif multi_head:

        detect_model_list = [
        '',
        '       self.heads = heads',
        '       input_channel = [{}]',
        '       build_detect(self, input_channel, nc, anchors, heads)',
        ]

        detect_forward_list = [
        '',
        '       return forward_detect(self, out)',
        ]

    else:
        detect_model_list = [
        '',
        '       self.detect = Detect(nc, anchors, [{}])'
        ]

        detect_forward_list = [
        '',
        '       return self.detect(out)',
        ]

    return torch_model_list, torch_forward_list, detect_model_list,detect_forward_list

dict_key = [
    'inchannels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups',
    'num_features', 'eps', 'momentum', 'affine', 'track_running_stats',
    'negative_slope', 'inplace',
    'size', 'scale_factor', 'mode', 'align_corners',
]

def replace_name(s):
    return s.replace('torch.nn.modules.upsampling.', 'nn.').replace('torch.nn.modules.conv.', 'nn.').replace('torch.nn.modules.batchnorm.', 'nn.')

def find_class_define(name_idx, name, module, in_blob_name, out_blob_name):

    model_list = []
    forward_list = []
    if name  == 'Sequential' and len(module) > 0:
        forward_str = f'{in_blob_name}'
        for m in module:
            class_name = replace_name(str(m.__class__).split("'")[1])
            attr_dict = m.__dict__
            attr_str = ', '.join([attr + '=' + str(attr_dict[attr]) for attr in attr_dict])
            attr_str = attr_str.replace('mode=bilinear',"mode='nearest")
            attr_str = attr_str.replace('mode=nearest',"mode='nearest'")

            if class_name == 'nn.Conv2d':
                if m.bias is None:
                    attr_str += ', bias=False'
                else:
                    attr_str += ', bias=True'

            self_name = 'self.m' + str(name_idx)
            name_idx += 1
            model_list.append(
            f'          {self_name} = {class_name}({attr_str})'
            )

            forward_str = f'{self_name}({forward_str})'
            in_blob_name = out_blob_name

        forward_str = f'        {out_blob_name} = ' + forward_str
        forward_list.append(forward_str)
        model_list.append('')
        forward_list.append('')

    else:
        class_name = replace_name(str(module.__class__).split("'")[1])
        attr_dict = module.__dict__
        attr_str = ', '.join([attr + '=' + str(attr_dict[attr]) for attr in dict_key if attr in attr_dict])

        self_name = 'self.' + name.lower() + str(name_idx)
        name_idx += 1
        model_list.append(
        f'          {self_name} = {class_name}({in_blob_name})'
        )

        model_list.append('')
        forward_list.append('')
    return name_idx, model_list, forward_list




