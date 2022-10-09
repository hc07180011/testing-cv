import logging
import random
import torch
import numpy as np

# Save and Load Functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(save_path, model, optimizer, loss, f1, val_f1, val_loss):

    if save_path == None:
        return

    state_dict = {
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'f1': f1,
        'val_f1': val_f1,
        'valid_loss': val_loss
    }

    if isinstance(model, tuple):
        state_dict['feature_extractor'] = model[0].state_dict()
        state_dict['model'] = model[1].state_dict()
    else:
        state_dict['model_state_dict'] = model.state_dict()

    torch.save(state_dict, save_path)
    logging.info(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    logging.info(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['loss'], state_dict['f1'], state_dict['valid_loss'], state_dict['valid_f1']


def save_metrics(save_path, loss_callback, f1_callback, val_loss_callback, val_f1_callback):

    if save_path == None:
        return

    state_dict = {'loss_callback': loss_callback,
                  'f1_callback': f1_callback,
                  'val_loss_callback': val_loss_callback,
                  'val_f1_callback': val_f1_callback, }

    torch.save(state_dict, save_path)
    logging.info(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    logging.info(f'Model loaded from <== {load_path}')

    return torch.Tensor(state_dict['loss_callback']).numpy(),\
        torch.Tensor(state_dict['f1_callback']).numpy(),\
        torch.Tensor(state_dict['val_loss_callback']).numpy(),\
        torch.Tensor(state_dict['val_f1_callback']).numpy()


def torch_seeding(seed=12345):
    """
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 torch_training.py --train
    """
    # Application-side randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Benchmarking randomness
    torch.backends.cudnn.benchmark = False

    # CUDA algorithmic randomness
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
