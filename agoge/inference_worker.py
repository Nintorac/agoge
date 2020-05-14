from pathlib import Path
from contextlib import suppress
from tqdm import tqdm
import mlflow
import torch
from ray.tune import Trainable
from agoge import AbstractModel as Model, Tracker, AbstractSolver as Solver, DataHandler
from agoge.utils import to_device
from agoge import DEFAULTS
from agoge.utils import get_logger, download_blob

logger = get_logger(__name__)

ARTIFACTS_ROOT = DEFAULTS['ARTIFACTS_ROOT']
BUCKET = DEFAULTS['BUCKET']


class InferenceWorker():

    def __init__(self, name, project, path=ARTIFACTS_ROOT, with_data=False):

            if not isinstance(path, Path):
                path = Path(path).expanduser()
            model_path = Path(project).joinpath(name).with_suffix('.box')
            full_path = path.joinpath(model_path)
            
            
            if not full_path.exists():
                print('downloading')
                with suppress(FileExistsError):
                    full_path.parent.mkdir()
                
                download_blob(BUCKET, model_path.as_posix(), full_path.as_posix())

            self.path = full_path.as_posix()
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


