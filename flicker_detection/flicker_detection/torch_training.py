import logging
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torchmetrics import F1Score
from torch.utils.data import DataLoader
from mypyfunc.logger import init_logger
from mypyfunc.torch_models import LSTMModel
from mypyfunc.torch_data_loader import MYDS, Streamer

from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save and Load Functions


def save_checkpoint(save_path, model, optimizer, loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss}

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


def save_metrics(save_path, loss, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': loss,
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
    epochs: int = 10,
    criterion=nn.BCELoss(),
    f1_torch=F1Score(),
):

    embedding_list_train = np.array(np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_0"])

    model = LSTMModel(input_dim=18432, hidden_dim=256,
                      layer_dim=1, output_dim=1)
    logging.info("{}".format(model.train()))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ds = Streamer(embedding_list_train, label_path, mapping_path, data_dir)

    f1_callback, loss_callback = (), ()
    model.train()
    for epoch in range(epochs):
        minibatch_loss = 0
        for batch_n, (x, y) in enumerate(ds):
            x = x.to(device)
            y = y.to(device)
            # logging.debug("{}".format(x.shape))
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            minibatch_loss += loss.item()

        model.eval()
        loss = (minibatch_loss/batch_n,)
        f1_score = (f1_torch((y_pred.cpu() > 0.5).int(), y.cpu().int()),)
        loss_callback += loss
        f1_callback += f1_score
        logging.info(
            "Epoch: {}/{}, Loss: {}, f1: {}".format(epoch, epochs, loss, f1_score))

        if not bool(epoch % 10):
            save_checkpoint('model.pth', model,
                            optimizer, loss)
            save_metrics('metrics.pth', loss_callback, f1_callback)

        model.train()
        random.shuffle(embedding_list_train)
        ds.new_embeddings(embedding_list_train)

    return model


def torch_eval(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
    threshold: float = 0.5,
) -> None:
    embedding_list_test = np.array(np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_1"])
    ds = Streamer(embedding_list_test, label_path, mapping_path, data_dir)

    model = LSTMModel(9216, hidden_dim=256, layer_dim=1, output_dim=1)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    y_pred, y_true = (), ()
    with torch.no_grad():
        mini_batch_loss = 0
        for batch_step, (x, y) in enumerate(ds):
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            y_pred += ((output.cpu() > threshold).int(),)
            y_true += (y.cpu().int(),)

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
    init_logger()
    logging.info("Starting Training")
    torch_training(
        label_path="data/label.json",
        mapping_path="data/mapping_aug_data.json",
        data_dir="data/vgg16_emb",
        cache_path=".cache/train_test",
    )
    logging.info("Done Training")
    pass
