from abc import ABCMeta, abstractmethod
from importlib import import_module

class AbstractSolver(metaclass=ABCMeta):

    @abstractmethod
    def solve(self, model, X):
        return {}

    @abstractmethod
    def step(self):
        """
        Method called at the end of each epoch
        """
        pass

    @abstractmethod
    def state_dict(self):
        return {}

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def from_config(Solver, **kwargs):
        if isinstance(Solver, str):
            Solver = import_module(Solver)

        return Solver(**kwargs)

        