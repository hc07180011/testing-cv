import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torchmetrics import F1Score
from torch.utils.data import DataLoader
from mypyfunc.torch_models import LSTMModel
from mypyfunc.torch_data_loader import MYDS

from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save and Load Functions


def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    logging.info(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    logging.info(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    logging.info(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    logging.info(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def torch_training(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
    epochs: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_list_train = np.array(np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_0"])
    chunked_list = np.array_split(embedding_list_train, indices_or_sections=40)

    input_dim = 9216

    model = LSTMModel(input_dim, hidden_dim=256, layer_dim=1, output_dim=1)
    logging.info("{}".format(model.train()))
    model.to(device)

    criterion = nn.BCELoss()
    f1_torch = F1Score()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    ds = MYDS(embedding_list_train, label_path, mapping_path, data_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=16)

    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            model.eval()
            f1_score = f1_torch((y_pred.cpu() > 0.5).int(), y.cpu().int())
            model.train()

        logging.info(
            "Epoch: {}/{}, Loss: {}, f1: {}".format(epoch, epochs, loss, f1_score))
    return model


def torch_eval(model, test_loader, version='title', threshold=0.5):
    y_pred, y_true = (), ()

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            y_pred += ((output.cpu() > threshold).int().tolist(),)
            y_true += (y.cpu().int().tolist(),)

    logging.info('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])


if __name__ == "__main__":
    pass
