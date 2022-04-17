# basic random seed
import tensorflow as tf
import torch
import os
import random
import numpy as np

DEFAULT_RANDOM_SEED = 2021


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# tensorflow random seed


def seedTF(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)


# torch random seed


def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# basic + tensorflow + torch


def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)
