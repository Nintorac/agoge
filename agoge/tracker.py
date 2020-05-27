
import mlflow
from math import ceil
from collections import defaultdict
from itertools import count
from contextlib import contextmanager
import numpy as np
from mlflow.tracking import MlflowClient

from agoge.utils import to_numpy, get_logger

logger = get_logger(__name__)

MAX_PARAMS_PER_LOG = 80

class Tracker():
    def __init__(self, experiment_name, trial_name, metrics=[], **kwargs):

        # initialise the tracking client
        self.trial_name = trial_name
        mlflow.set_experiment(experiment_name)
        self.run_id = None
        self.params_submitted = False
        self.step = defaultdict(int)
        self.metrics = metrics

    @contextmanager
    def run_context(self, exit_status='SCHEDULED'):
        """
        Set the run id to enable tracking to the correct trial
        """
        
        run = mlflow.start_run(run_name=self.trial_name, run_id=self.run_id)
        self.run_id = run.info.run_id

        try:
            yield
        finally:
            mlflow.end_run(exit_status)
    

    def set_status(self, exit_status='FINISHED'):
        """
        Change the status of this trial to the one supplied
        """

        if self.run_id is None:
            return

        with self.run_context(exit_status):
            pass
       
    def log_params(self, params):
        """
        Push the hyperparameters for the current run.. only does things on first  call
        """
        if self.params_submitted:
            # if params already submitted skip resubmission
            return

        def chunk(to_chunk, chunk_size):
            n_chunks = ceil(len(to_chunk)/chunk_size)
            idxs = lambda x: x * chunk_size
            yield from (to_chunk[idxs(i):idxs(i + 1)] for i in range(n_chunks))

        params = {key: value for key, value in params.items() if 'param_' in key}
        with self.run_context():
            for keys in chunk(list(params.keys()), MAX_PARAMS_PER_LOG):
                
                param_keys = map(lambda key: key.replace('param_', ''), keys)
                param_values = map(params.get, keys)

                mlflow.log_params(dict(zip(param_keys, param_values)))

        self.params_submitted = True
        


    def metric_logger(self, phase, plot_freq):
        """
        A coroutine to log metrics during training

        """

        metrics_history = {key: [] for key in self.metrics}

        def get_metric_dict(n=0, prefix_keys=False):
            
            plot_prefix = ''
            if prefix_keys:
                plot_prefix = f'{phase}_'

            metric_dict = {
                f'{plot_prefix}{key}': np.mean(value[-n:]) 
                        for key, value in metrics_history.items()
            }

            return metric_dict

        with self.run_context():

            for i in count(self.step[phase]):
                
                metrics = yield

                if metrics is None:
                    break

                assert set(metrics_history.keys()).issubset(set(metrics.keys())), 'Metrics missing'

                metrics = {key: to_numpy(value) for key, value in metrics.items()}
                metrics_history = {key: [*metrics_history[key], metrics[key]]
                                        for key in self.metrics}

                if not (i+1) % plot_freq:
                    try:
                        mlflow.log_metrics(get_metric_dict(plot_freq, True), step=i)
                    except:
                        pass

        # update steps
        self.step[phase] = i

        yield get_metric_dict()


    def load_state_dict(self, state_dict):

        logging.info('Loading tracker...')
        self.step = state_dict['step']
        self.run_id = state_dict['run_id']
        self.params_submitted = state_dict['params_submitted']

    def state_dict(self):

        print('saving')
        print(dict(self.step))

        return {
            'step': dict(self.step),
            'run_id': self.run_id,
            'params_submitted': self.params_submitted
            }
