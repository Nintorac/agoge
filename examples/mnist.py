#%%
import torch

from pathlib import Path
from importlib import import_module
from itertools import starmap

from ray import tune
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from torch import nn
from torch.optim import Adadelta
from torch.nn import Module
from agoge import AbstractModel, AbstractSolver, TrainWorker, DEFAULTS
from agoge.utils import uuid, trial_name_creator, experiment_name_creator, get_logger

import logging

logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)


class MNISTDataset():

    def __init__(self, data_path='~/datasets', transform=None):
        """
        Simple dataset that wraps torchvision mnist dataset
        """
        
        transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

        if not isinstance(data_path, Path):
            data_path = Path(data_path).expanduser()

        train_dataset = datasets.MNIST(data_path.as_posix(), train=True, download=True,transform=transform)
        test_dataset = datasets.MNIST(data_path.as_posix(), train=False, download=True, transform=transform)

        self.dataset = ConcatDataset((train_dataset, test_dataset))

    def __getitem__(self, i):
        """
        returns a dictionary of {
            'x': torch.FloatTensor(batch_size, 1, 28, 28),
            'y': torch.LongTensor(batch_size)
        }
        """
        return dict(zip(['x', 'y'], self.dataset[i]))

    def __len__(self):

        return len(self.dataset)

class MNISTModel(AbstractModel):

    def __init__(self, 
            conv1=(1, 32, 3, 1),
            conv2=(32, 64, 3, 1),
            dropout1=(0.25, ),
            dropout2=(0.5, ),
            fc1=(9216, 128),
            fc2=(128, 10),
            **kwargs,
    ):

        super().__init__()
        self.conv1 = nn.Conv2d(*conv1)
        self.conv2 = nn.Conv2d(*conv2)
        self.dropout1 = nn.Dropout2d(*dropout1)
        self.dropout2 = nn.Dropout2d(*dropout2)
        self.fc1 = nn.Linear(*fc1)
        self.fc2 = nn.Linear(*fc2)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return {
            'y_hat': output
        }


class MNISTSolver(AbstractSolver):

    def __init__(self, model,
        Optim=Adadelta, optim_opts=dict(lr= 1.),
        Scheduler=StepLR, scheduler_opts=dict(step_size=1, gamma=0.7),
        **kwargs):

        if isinstance(Optim, str):
            Optim = import_module(Optim)
        if isinstance(Scheduler, str):
            Scheduler = import_module(Scheduler)
        self.optim = Optim(params=model.parameters(), **optim_opts)
        self.scheduler = Scheduler(optimizer=self.optim, **scheduler_opts)

        self.model = model

    def loss(self, y, y_hat, **kwargs):

        return F.cross_entropy(y_hat, y)

    def solve(self, X, **kwargs):
        
        Y = self.model(**X)
        loss = self.loss(**X, **Y)

        if self.model.training:

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        return {'loss': loss}

    
    def step(self):

        self.scheduler.step()


    def state_dict(self):
        
        state_dict = {
            'scheduler': self.scheduler.state_dict(),
            'optim': self.optim.state_dict()
        }

        return state_dict

    def load_state_dict(self, state_dict):
        
        load_component = lambda component, state: getattr(self, component).load_state_dict(state)
        list(starmap(load_component, state_dict.items()))


# points_per_epoch
# %%
