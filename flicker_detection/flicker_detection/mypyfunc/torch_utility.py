import logging
import re
import random
from tkinter import Y
import torch
import numpy as np
import pandas as pd
from io import StringIO

# Save and Load Functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(save_path, model, optimizer, loss, f1, val_f1, val_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,
                  'f1': f1,
                  'val_f1': val_f1,
                  'valid_loss': val_loss}

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


def torch_seeding():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def report_to_df(report):  # FIX ME
    report = re.sub(r" +", " ", report).replace("avg / total",
                                                "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report),
                            sep=' ', index_col=0, on_bad_lines='skip')
    report_df.to_csv("report.csv")
    return report_df


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc
