from pathlib import Path
from contextlib import suppress
from urllib.request import urlretrieve
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
BASE_URL = DEFAULTS['BASE_URL']


class InferenceWorker():

    def __init__(self, name, project, path=ARTIFACTS_ROOT, with_data=False, base_url=None):

            if not isinstance(path, Path):
                path = Path(path).expanduser()
            model_path = Path(project).joinpath(name).with_suffix('.box')
            full_path = path.joinpath(model_path)
            
            
            if not full_path.exists():
                self.download_weights(full_path, model_path, base_url)

            self.path = full_path.as_posix()
            self.with_data = with_data
            self.setup_components()

    @staticmethod
    def download_weights(full_path, model_path, base_url):
        """
        If base_url is supplied then the model is downloaded from there,
        otherwise tries to download from the bucket set by environment variale
        `BUCKET`

        """

        if base_url is None:
            base_url = BASE_URL

        with suppress(FileExistsError):
            full_path.parent.mkdir()
        url = f'{base_url}/{model_path.as_posix()}'
        
        logger.info('Downloading weights from {url}...')
        urlretrieve(url, full_path.as_posix())

        

    def setup_components(self, **config):

        state_dict = torch.load(self.path, map_location=torch.device('cpu'))
        
        worker_config = state_dict['worker']

        self.model = Model.from_config(**worker_config['model'])
        if self.with_data:
            self.dataset = DataHandler.from_config(**worker_config['data_handler'])

        self.model.load_state_dict(state_dict['model'])
        self.model.eval()


