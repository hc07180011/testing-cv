import os
import logging
from sklearn.decomposition import IncrementalPCA
import torch
import numpy as np
import pickle as pk
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from typing import Callable

from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_eval import F1Score, F1_Loss, Evaluation
from mypyfunc.torch_models import LSTM
from mypyfunc.torch_data_loader import Streamer
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding, report_to_df
from sklearn.metrics import classification_report


def torch_validation(
    ds_val: Streamer,
    criterion: Callable,
    objective: Callable = nn.Softmax(),
    f1_metric: Callable = F1Score(),
):
    with torch.no_grad():
        minibatch_loss_val, minibatch_f1_val = 0, 0
        for n_val, (x, y) in enumerate(ds_val):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_f1 = f1_metric(
                torch.topk(objective(y_pred), k=1, dim=1).indices.flatten(), y)
            minibatch_loss_val += loss.item()
            minibatch_f1_val += val_f1.item()
    ds_val._shuffle()
    return ((minibatch_loss_val/(n_val + 1)) if n_val else minibatch_loss_val,),\
        ((minibatch_f1_val/(n_val + 1)) if n_val else minibatch_f1_val,)


def torch_training(
    ds_train: Streamer,
    ds_val: Streamer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    f1_metric: Callable = F1Score('macro'),
    epochs: int = 1000,
    criterion: Callable = nn.BCELoss(),
    objective: Callable = nn.Softmax()
) -> nn.Module:

    val_max_f1 = 0
    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        if loss_callback and epoch > 10 and loss_callback[-1] < 0.01:
            break

        minibatch_loss_train, minibatch_f1 = 0, 0
        for n_train, (x, y) in enumerate(ds_train):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            f1 = f1_metric(torch.topk(objective(y_pred),
                           k=1, dim=1).indices.flatten(), y)

            minibatch_loss_train += loss.item()
            minibatch_f1 += f1.item()

        model.eval()
        loss_callback += ((minibatch_loss_train / (n_train + 1))
                          if n_train else minibatch_loss_train,)
        f1_callback += ((minibatch_f1/(n_train + 1))
                        if n_train else minibatch_f1,)
        # scheduler.step(loss_callback[-1])
        val_loss, val_f1 = torch_validation(
            ds_val, criterion=criterion, objective=objective)
        val_loss_callback += val_loss
        val_f1_callback += val_f1

        logging.info(
            "Epoch: {}/{} Loss - {:.3f},f1 - {:.3f} val_loss - {:.3f}, val_f1 - {:.3f}".format(
                epoch + 1, epochs,
                loss_callback[-1],
                f1_callback[-1],
                val_loss_callback[-1],
                val_f1_callback[-1]
            ))

        if epoch > 10 and val_f1_callback[-1] > val_max_f1:
            save_checkpoint('model.pth', model,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics('metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)
            val_max_f1 = val_f1_callback[-1]

        model.train()
        ds_train._shuffle()

    torch.cuda.empty_cache()
    return model


def torch_testing(
    ds_test: Streamer,
    model: nn.Module,
    objective: Callable = nn.Softmax(),
    classes: int = 2
) -> None:
    metrics = Evaluation(plots_folder="plots/", classes=classes)

    model.eval()
    y_pred, y_true = None, None
    with torch.no_grad():
        for x, y in ds_test:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            y_pred = output if y_pred is None else\
                torch.cat((y_pred, output), dim=0)
            y_true = y if y_true is None else\
                torch.cat((y_true, y), dim=0)

    y_classes = torch.topk(objective(y_pred), k=1, dim=1).indices.flatten()
    metrics.cm(y_true.detach(), y_classes.detach())

    y_pred, y_true = objective(y_pred).cpu().numpy(), y_true.cpu().numpy()
    y_bin = np.zeros((y_true.shape[0], classes))
    idx = np.array([[i] for i in y_true])
    np.put_along_axis(y_bin, idx, 1, axis=1)

    metrics.roc_auc(y_bin, y_pred)
    metrics.pr_curve(y_bin, y_pred)

    # loss, f1, val_loss, val_f1 = load_metrics("metrics.pth")
    # metrics.plot_callback(loss, val_loss, "loss")
    # metrics.plot_callback(f1, val_f1, "f1")

    report = classification_report(
        y_true,
        y_classes.cpu().numpy(),
        digits=4)
    report_to_df(report)


if __name__ == "__main__":
    init_logger()

    label_path = "data/new_label.json"
    mapping_path = "data/mapping_aug_data.json"
    data_dir = "data/vgg16_emb/"
    cache_path = ".cache/train_test"
    model_path = "model.pth"

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    chunk_size = 30
    batch_size = 768

    ipca = pk.load(open("ipca.pk1", "rb")) if os.path.exists(
        "ipca.pk1") else IncrementalPCA(n_components=2)

    sm = SMOTE(random_state=42, n_jobs=-1)  # , k_neighbors=2)
    nm = NearMiss(version=3, n_jobs=-1)  # , n_neighbors=1)
    ds_train = Streamer(embedding_list_train, label_path,
                        mapping_path, data_dir, mem_split=3, chunk_size=chunk_size, batch_size=batch_size, sampler=sm)  # , ipca=ipca, ipca_fitted=True)
    ds_val = Streamer(embedding_list_val, label_path,
                      mapping_path, data_dir, mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)
    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_dir, mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)

    model = LSTM(input_dim=18432, output_dim=2, hidden_dim=256,
                 layer_dim=1, bidirectional=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=5, verbose=True)
    # lower gpu float precision for larger batch size
    torch_seeding()
    logging.info("{}".format(model.train()))

    parser = ArgumentParser()
    parser.add_argument(
        "-preprocess", "--preprocess", action="store_true",
        default=False, help="Whether to perform IPCA")
    parser.add_argument(
        "-train", "--train", action="store_true",
        default=False, help="Whether to do training")
    parser.add_argument(
        "-test", "--test", action="store_true",
        default=False, help="Whether to do testing")
    args = parser.parse_args()

    if args.preprocess:
        logging.info("Preprocessing start...")
        ev = ds_train._fit_ipca()
        logging.info("Explained variance: {}".format(ev))
        logging.info("Preprocessing Done.")

    if args.train:
        # model.load_state_dict(torch.load(model_path)['model_state_dict'])
        logging.info("Starting Training")
        model = torch_training(ds_train, ds_val, model,
                               optimizer, criterion=criterion)
        logging.info("Done Training")

    if args.test:
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        logging.info("Starting Evaluation")
        torch_testing(ds_test, model)
        logging.info("Done Evaluation")
