# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple, abc

import torch

if torch.__version__ != 'parrots':

    def load_ext(name, funcs):
        ext = importlib.import_module('mmcv.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
else:
    from parrots import extension
    from parrots.base import ParrotsException

    has_return_value_ops = [
        'nms',
        'softnms',
        'nms_match',
        'nms_rotated',
        'top_pool_forward',
        'top_pool_backward',
        'bottom_pool_forward',
        'bottom_pool_backward',
        'left_pool_forward',
        'left_pool_backward',
        'right_pool_forward',
        'right_pool_backward',
        'fused_bias_leakyrelu',
        'upfirdn2d',
        'ms_deform_attn_forward',
        'pixel_group',
        'contour_expand',
    ]

    def get_fake_func(name, e):

        def fake_func(*args, **kwargs):
            warnings.warn(f'{name} is not supported in parrots now')
            raise e

        return fake_func

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            try:
                ext_fun = extension.load(fun, name, lib_dir=lib_root)
            except ParrotsException as e:
                if 'No element registered' not in e.message:
                    warnings.warn(e.message)
                ext_fun = get_fake_func(fun, e)
                ext_list.append(ext_fun)
            else:
                if fun in has_return_value_ops:
                    ext_list.append(ext_fun.op)
                else:
                    ext_list.append(ext_fun.op_)
        return ExtModule(*ext_list)


def check_ops_exist():
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)
