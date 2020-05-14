from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from importlib import import_module
from torch.nn import Module

class AbstractModel(Module):

    def __init__(self):

        super().__init__()

    @staticmethod
    def from_config(Model, **kwargs):

        if isinstance(Model, str):
            Model = import_module(Model)
        return Model(**kwargs)


    @contextmanager
    def train_model(self):

        self.train()

        try:
            yield
        finally:
            self.eval()


    @property
    def device(self):

        return next(self.parameters()).device

        