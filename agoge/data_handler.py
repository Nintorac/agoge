#%%
from importlib import import_module
from collections import namedtuple
from torch.utils.data import DataLoader, random_split
from functools import partial


TrainTestEvaluate = namedtuple('Loaders', ('train', 'evaluate', 'test'))

class DataHandler():


    def __init__(self, dataset, split_sizes=(0.7, 0.2, 0.1), loader_opts={}, **kwargs):

        assert len(split_sizes)==3, 'expects train, evaluate and test split ratio'

        subset_size = lambda ratio: int(ratio*len(dataset))
        *dataset_sizes, _ = map(subset_size, split_sizes)
        dataset_sizes = (*dataset_sizes, len(dataset)-sum(dataset_sizes))
        
        ArgLoader = partial(DataLoader, **loader_opts)

        self.datasets = TrainTestEvaluate(*random_split(dataset, dataset_sizes))
        self.loaders = TrainTestEvaluate(*map(lambda data: ArgLoader(data), self.datasets))


    @staticmethod
    def from_config(Dataset, dataset_opts, **kwargs):
        if isinstance(Dataset, str):
            Dataset = import_module(Dataset)

        dataset = Dataset(**dataset_opts)

        return DataHandler(dataset, **kwargs)

if __name__=='__main__':
    from torchvision import datasets, transforms
    from pathlib import Path
    from  torch.utils.data import ConcatDataset

    batch_size = 32

    class MNISTDataset():

        def __init__(self, data_path='~/datasets', transform=None):

            if not isinstance(data_path, Path):
                data_path = Path(data_path).expanduser()

            train_dataset = datasets.MNIST(data_path.as_posix(), train=True, download=True,transform=transform)
            test_dataset = datasets.MNIST(data_path.as_posix(), train=True, download=True, transform=transform)
            print(type(train_dataset))
            
            self.dataset = ConcatDataset((train_dataset, test_dataset))

        def __getitem__(self, i):
            
            return dict(zip(['x', 'y'], self.dataset[i]))

        def __len__(self):

            return len(self.dataset)


    data_handler = {
            'Dataset': MNISTDataset,
            'dataset_opts': {'data_path': '~/audio/artifacts/'},
            'loader_opts': {
                'batch_size': batch_size,
            },
        }

    DataHandler.from_config(**data_handler)

        