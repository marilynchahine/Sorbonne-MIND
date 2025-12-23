import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click


from sklearn.datasets import fetch_openml

# Changer le DATA_PATH
DATA_PATH = "/tmp/mnist"


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
TEST_RATIO = 0.2
def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


#  TODO:  Implémenter
