from uuid import uuid4 as uuid
import namegenerator
import logging
from collections.abc import Iterable
from collections import ChainMap
import torch
import numpy as np
from os import environ

# Logging
def get_logger(name, level=logging.NOTSET):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


# naming / unique id
experiment_name_creator = lambda:namegenerator.gen(lists=(namegenerator.LEFT, namegenerator.RIGHT))
trial_name_creator = lambda *args:namegenerator.gen()

uuid = lambda: uuid().hex


# def to_numpy(x):

#     if isinstance(x, dict):
#         return {key: to_numpy(value) for key, value in x.items()}
#     if isinstance(x, list):
#         return [to_numpy(value) for value in x]
#     if isinstance(x, tuple):
#         return tuple(to_numpy(value) for value in x)
    
#     if isinstance(x, torch.Tensor):
#         return x.detach().to('cpu').numpy()
    
#     # not a tensor, ignore??
#     return x
def to_device(x, device='cpu'):

    def to_device(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy()

        if isinstance(x, torch.Tensor):
            return x.to(device)

        return x


    return recurse_json_like(x, to_device)

def to_numpy(x):

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
             return x.detach().to('cpu').numpy()
        return x

    return recurse_json_like(x, to_numpy)

def recurse_json_like(x, func):

    if isinstance(x, dict):
        return {key: recurse_json_like(value, func) for key, value in x.items()}
    if isinstance(x, list):
        return [recurse_json_like(value, func) for value in x]
    if isinstance(x, tuple):
        return tuple([recurse_json_like(value, func) for value in x])
    
    return func(x)


def defaults_f(defaults, parent=environ):
    """
    Generate the defaults dictionary
    """

    return ChainMap(parent, defaults)