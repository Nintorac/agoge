from .utils import defaults_f
DEFAULTS = defaults_f({
    'ARTIFACTS_ROOT': '~/agoge/artifacts',
    'TQDM_DISABLED': False,
    'TRIAL_ROOT': 'Worker',
    'BUCKET': 'nintorac_model_serving',
    'BASE_URL': 'https://github.com/Nintorac/NeuralDX7-weights/raw/master'
})

from .data_handler import DataHandler

from .model import AbstractModel
from .solver import AbstractSolver
from .train_worker import TrainWorker
from .inference_worker import InferenceWorker
