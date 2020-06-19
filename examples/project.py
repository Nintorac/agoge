from ray import tune
from agoge import TrainWorker, DEFAULTS
from mnist import MNISTDataset, MNISTModel, MNISTSolver
from agoge.utils import uuid, trial_name_creator, experiment_name_creator, get_logger


def config(Model, Solver, experiment_name, batch_size=128, **kwargs):

    data_handler = {
        'Dataset': MNISTDataset,
        'dataset_opts': {'data_path': DEFAULTS['ARTIFACTS_ROOT']},
        'loader_opts': {
            'batch_size': batch_size,
            'shuffle': True
        },
    }

    model = {
        'Model': Model
    }

    solver = {
        'Solver': Solver
    }
    return {
        'data_handler': data_handler,
        'model': model,
        'solver': solver,
    }

if __name__=='__main__':

    # client = MlflowClient(tracking_uri='localhost:5000')
    experiment_name = 'mnist-'#+experiment_name_creator()
    # experiment_id = client.create_experiment(experiment_name)

    stopping_criterion = lambda trial_id, results: results['loss']['loss']<1e-3
    
    tune.run(TrainWorker, config={
        'config_generator': config,
        'experiment_name': experiment_name,
        'Model': MNISTModel,
        'Solver': MNISTSolver
    },
        trial_name_creator=trial_name_creator,
        stop=stopping_criterion,
        resources_per_trial={
            'cpu': 4
        }
    )