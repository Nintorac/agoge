# Agoge

An opinionated ML development environment

Train pytorch models, hyperparameter tune them with single loc change.



## Libraries

### Pytorch

[Pytorch](https://pytorch.org/) is a Python first machine learning library 

### Ray

[Ray](https://ray-project.github.io/) Provides easy experiment scaling + hyper parameter optimisation

### MLFlow

Provides tracking of model metrics and hyperparameters. Also allows for intelligent storing of training artifacts

## Static Components

These components should not need to be customised for model specific use cases

### Train Worker

Setups all the required components to train a model

### Inference Worker

Setups all the required components for inference. Also attempts to download model weights if they are not found locally.

### Tracker

Slim interface into MLFlow, to set the MLFlow server address set the `MLFLOW_TRACKING_URI` environment variable either from the CLI or before importing `agoge`.

Examples

CLI
```
MLFLOW_TRACKING_URI="http://localhost:5001"
```

Python
```
from os import environ
environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'
import agoge
```

### Data Handler

Loads the dataset and handles the dataset split

## User Provided Components

These components need to be inherited by project specific classes

### Model

Provides some convenience functions around loading models. This class will hold all model specific code and is used by the train worker and inference workers

### Solver

Override the `solve` method with the code required to train your model

### Dataset

Any dataset that is compatiable with the [Pytorch map style dataset model](https://pytorch.org/docs/stable/data.html#map-style-datasets)


# Disclaimer

This code is subject to change. I will try not to break anything but can't promise. File an issue if an update breaks your code
