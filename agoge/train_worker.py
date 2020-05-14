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


class TrainWorker(Trainable):

    def _setup(self, config):

        config['trial_name'] = getattr(self, 'trial_name', '')

        self.setup_worker(**config)
        self.setup_components(**config)

    def setup_worker(self, points_per_epoch=100, **kwargs):

        self.points_per_epoch = points_per_epoch

    def setup_components(self, **config):
        
        worker_config = config['config_generator'](**config)
        self.worker_config = worker_config
        self.tracker = Tracker(**worker_config['tracker'])

        self.model = Model.from_config(**worker_config['model'])
        self.solver = Solver.from_config(model=self.model, **worker_config['solver'])
        self.dataset = DataHandler.from_config(**worker_config['data_handler'])
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()

    def epoch(self, loader, phase):
        
        plot_freq = len(loader)//self.points_per_epoch

        plotter = iter(self.tracker.metric_logger(phase, plot_freq))
        next(plotter)
        for i, X in enumerate(tqdm(loader, disable=DEFAULTS['TQDM_ENABLED'])):
            
            X = to_device(X, self.model.device)

            loss = self.solver.solve(X)
            plotter.send(loss)

        loss = plotter.send(None)
        
        return loss


    def _train(self):
        
        self.tracker.log_params(self.config)

        with self.model.train_model():
            self.epoch(self.dataset.loaders.train, 'train')
        with torch.no_grad():
            loss = self.epoch(self.dataset.loaders.evaluate, 'evaluate')

        return {'loss': loss}
        

    def _save(self, path):

        state_dict = {
            'model': self.model.state_dict(),
            'solver': self.solver.state_dict(),
            'tracker': self.tracker.state_dict(),
            'worker': self.worker_config
        }

        path = Path(path).joinpath('model.box').as_posix()
        torch.save(state_dict, path)

        return path


    def _restore(self, path):

        state_dict = torch.load(path, map_location=torch.device('cpu'))

        self.model.load_state_dict(state_dict['model'])
        self.solver.load_state_dict(state_dict['solver'])
        self.tracker.load_state_dict(state_dict['tracker'])

    def _stop(self):

        self.tracker.set_status('FINISHED')