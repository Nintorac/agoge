import wandb
import torch
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm
from ray.tune import Trainable
from agoge import AbstractModel as Model, AbstractSolver as Solver, DataHandler
from agoge.utils import to_device
from agoge import DEFAULTS
from agoge.utils import get_logger

logger = get_logger(__name__)


class TrainWorker(Trainable):

    def _setup(self, config):

        self.setup_worker(config)
        self.setup_components(config)
        self.setup_tracking(config)

    @property
    def trial_name(self):
        return self._trial_info._trial_name

    def setup_worker(self, points_per_epoch=10, **kwargs):

        self.points_per_epoch = points_per_epoch

    def setup_tracking(self, config):

        wandb.init(
            project=config['experiment_name'],
            name=self.trial_name,
            resume=True
            )
        
        wandb.config.update({
            key.replace('param_', ''): value
                 for key, value in self.config.items() if 'param_' in key
        })

        wandb.watch(self.model)

    def setup_components(self, config):
        
        worker_config = config['config_generator'](**config)
        self.worker_config = worker_config

        self.model = Model.from_config(**worker_config['model'])
        self.solver = Solver.from_config(model=self.model, **worker_config['solver'])
        self.dataset = DataHandler.from_config(**worker_config['data_handler'])
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

    def epoch(self, loader, phase):
        
        for i, X in enumerate(tqdm(loader, disable=bool(DEFAULTS['TQDM_DISABLED']))):
            
            X = to_device(X, self.model.device)

            loss = self.solver.solve(X)
            wandb.log(loss)
            break
        
        return loss


    def _train(self):
        
        with self.model.train_model():
            self.epoch(self.dataset.loaders.train, 'train')
        with torch.no_grad():
            loss = self.epoch(self.dataset.loaders.evaluate, 'evaluate')

        return {'loss': loss}
        

    def _save(self, path):

        state_dict = {
            'model': self.model.state_dict(),
            'solver': self.solver.state_dict(),
            'worker': self.worker_config
        }

        path = Path(path).joinpath(f'{self.trial_name}.pt').as_posix()
        torch.save(state_dict, path)

        return path


    def _restore(self, path):

        state_dict = torch.load(path, map_location=torch.device('cpu'))

        self.model.load_state_dict(state_dict['model'])
        self.solver.load_state_dict(state_dict['solver'])

    def _stop(self):

        with TemporaryDirectory() as d:
            logger.critical(d)
            self._save(d)
            wandb.save(f'{d}/{self.trial_name}.pt')