from pathlib import Path
from tqdm import tqdm
import mlflow
import torch
from ray.tune import Trainable
from agoge import AbstractModel as Model, Tracker, AbstractSolver as Solver, DataHandler
from agoge.utils import to_device
from agoge import DEFAULTS
from agoge.utils import get_logger

logger = get_logger(__name__)


class InferenceWorker():

    def __init__(self, path, with_data=False):

        self.path = Path(path).expanduser().as_posix()
        self.with_data = with_data
        self.setup_components() 

    def setup_components(self, **config):

        state_dict = torch.load(self.path, map_location=torch.device('cpu'))
        
        worker_config = state_dict['worker']

        self.model = Model.from_config(**worker_config['model'])
        if self.with_data:
            self.dataset = DataHandler.from_config(**worker_config['data_handler'])

        self.model.load_state_dict(state_dict['model'])
        self.model.eval()


