import json
from functools import wraps
from util import registry, build_from_cfg
import torch.nn as nn


def singleton_arg(func):
    _instances = {}

    @wraps(func)
    def wrapper_call(*args, **kwargs):
        arg_str = '%s_%s' % (json.dumps(args), json.dumps(kwargs))
        if arg_str not in _instances:
            ret = func(*args, **kwargs)
            _instances[arg_str] = ret
        return _instances[arg_str]

    return wrapper_call


def build(cfg, module_name, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, module_name, default_args)
            for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, module_name, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, 'backbone')


def build_neck(cfg):
    """Build neck."""
    return build(cfg, 'neck')


def build_head(cfg):
    """Build heads."""
    return build(cfg, 'head')


@singleton_arg
def build_detector(cfg):
    return build_from_cfg(cfg, registry, 'detector')


def build_engine(cfg):
    return build_from_cfg(cfg, registry, 'engine')


@singleton_arg
def build_bbox_coder(cfg, **default_args):
    bbox_coder = build_from_cfg(cfg, registry, 'bbox_coder', default_args)
    return bbox_coder
