from collections import defaultdict
from tempfile import TemporaryDirectory
from pathlib import Path
import wandb
import torch
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
        self.setup_tracking(**config)

    @property
    def trial_name(self):
        if self._trial_info is not None:
            return self._trial_info._trial_name
        return 'test'

    def setup_worker(self, points_per_epoch=10, **kwargs):

        self.points_per_epoch = points_per_epoch

    def setup_tracking(self, experiment_name, log_freq=50, **kwargs):

        self.log_freq = log_freq

        wandb.init(
            project=experiment_name,
            name=self.trial_name,
            resume=True
            )
        
        wandb.config.update({
            key.replace('param_', ''): value
                 for key, value in kwargs.items() if 'param_' in key
        })

        try:
            # hacky workaround to ensure not a jit script model
            wandb.watch(self.model)
        except:
            pass

    def setup_components(self, config):
        
        worker_config = config['config_generator'](**config)
        self.worker_config = worker_config

        self.model = Model.from_config(**worker_config['model'])
        self.solver = Solver.from_config(model=self.model, **worker_config['solver'])
        self.dataset = DataHandler.from_config(**worker_config['data_handler'])

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def epoch(self, loader, phase):
        """
        loader - pytorch data loader
        phase - training phase in {train, evaluate}
        """
        
        total_loss = defaultdict(int)

        # calculate steps so far
        steps = len(loader) * self.iteration

        for i, X in enumerate(tqdm(loader, disable=bool(DEFAULTS['TQDM_DISABLED']))):
            
            # pass data through solver
            X = to_device(X, self.model.device)
            loss = self.solver.solve(X)

            # accumulate total loss
            with torch.no_grad():
                for key, value in loss.items():
                    total_loss[key] += value

            # log instantaneous loss
            if not i % self.log_freq:
                wandb.log({
                    f'{phase}_step': steps+i,
                    **{f'{phase}_{key}': value for key, value in loss.items()}
                })
        
        # calculate epoch averages
        epoch_loss = {key: value/len(loader) for key, value in total_loss.items()}
        
        # log epoch loss
        wandb.log({
            'epoch': self.iteration,
            **{f'{phase}_epoch_{key}': value for key, value in epoch_loss.items()}
        })


        return epoch_loss


    def _train(self):
        
        with self.model.train_model():
            self.epoch(self.dataset.loaders.train, 'train')
        with torch.no_grad():
            loss = self.epoch(self.dataset.loaders.evaluate, 'evaluate')

        return {'loss': loss}
        

    def _save(self, path):
        self.model.cpu()
        state_dict = {
            'model': self.model.state_dict(),
            'solver': self.solver.state_dict(),
            'worker': self.worker_config
        }

        path = Path(path).joinpath(f'{self.trial_name}.pt').as_posix()
        torch.save(state_dict, path)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

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
