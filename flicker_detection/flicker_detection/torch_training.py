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
from mypyfunc.torch_eval import F1Score, Evaluation
from mypyfunc.torch_models import LSTM
from mypyfunc.torch_data_loader import Streamer
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding


def torch_validation(
    model: torch.nn.Module,
    ds_val: Streamer,
    criterion: Callable,
    objective: Callable = nn.Softmax(),
    f1_metric: Callable = F1Score(average='macro'),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    with torch.no_grad():
        minibatch_loss_val, minibatch_f1_val = 0, 0
        for n_val, (x, y) in enumerate(ds_val):
            x, y = x.to(device), y.to(device)
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
    train_encodings: Streamer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    f1_metric: Callable = F1Score(average='macro'),
    criterion: Callable = nn.BCELoss(),
    objective: Callable = nn.Softmax(),
    epochs: int = 1000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path: str = 'h5_models/',
) -> nn.Module:

    val_max_f1 = 0
    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        if loss_callback and epoch > 10 and loss_callback[-1] < 0.005:
            break

        minibatch_loss_train, minibatch_f1 = 0, 0
        for n_train, (x0, y0), (x1, _) in enumerate(zip(ds_train, train_encodings)):
            x0, x1, y0 = x0.to(device), x1.to(device), y0.to(device)
            y_pred0, y_pred1 = model(x0), model(x1)
            loss0, loss1 = criterion(y_pred0, y0), criterion(y_pred1, y0)
            loss_final = loss0 + loss1
            optimizer.zero_grad()
            loss_final.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            f1 = f1_metric(torch.topk(objective(y_pred0)+objective(y_pred1),  # TO DO - figure out best to add crossentropy loss
                           k=1, dim=1).indices.flatten(), y0)

            minibatch_loss_train += loss_final.item()
            minibatch_f1 += f1.item()

        model.eval()
        loss_callback += ((minibatch_loss_train / (n_train + 1))
                          if n_train else minibatch_loss_train,)
        f1_callback += ((minibatch_f1/(n_train + 1))
                        if n_train else minibatch_f1,)
        # scheduler.step(loss_callback[-1])
        val_loss, val_f1 = torch_validation(
            model, ds_val, criterion=criterion, objective=objective)
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
            save_checkpoint(f'{save_path}model.pth', model,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics(f'{save_path}metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)
            val_max_f1 = val_f1_callback[-1]

        model.train()
        ds_train._shuffle()

    torch.cuda.empty_cache()
    return model


def torch_testing(
    ds_test: Streamer,
    test_encodings: Streamer,
    model: nn.Module,
    objective: Callable = nn.Softmax(),
    classes: int = 6,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path: str = 'h5_models/',
) -> None:
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    metrics = Evaluation(plots_folder="plots/", classes=classes)

    model.eval()
    y_pred0, y_true0, y_pred1 = None, None, None
    with torch.no_grad():
        for (x0, y0), (x1, _) in zip(ds_test, test_encodings):
            x0, x1, y0 = x0.to(device), x1.to(device), y0.to(device)
            output0, output1 = model(x0), model(x1)
            y_pred0 = output0 if y_pred0 is None else\
                torch.cat((y_pred0, output0), dim=0)
            y_pred1 = output1 if y_pred1 is None else\
                torch.cat((y_pred1, output1), dim=0)
            y_true = y0 if y_true0 is None else\
                torch.cat((y_true0, y0), dim=0)
    y_classes = torch.topk(
        objective(y_pred0)+objective(y_pred1), k=1, dim=1).indices.flatten()
    metrics.cm(y_true.detach(), y_classes.detach())

    y_pred, y_true = (objective(y_pred0)+objective(y_pred1)
                      ).cpu().numpy(), y_true.cpu().numpy()
    y_bin = np.zeros((y_true.shape[0], classes))
    idx = np.array([[i] for i in y_true])
    np.put_along_axis(y_bin, idx, 1, axis=1)
    
    metrics.roc_auc(y_bin, y_pred)
    metrics.pr_curve(y_bin, y_pred)

    loss, f1, val_loss, val_f1 = load_metrics(f"{save_path}metrics.pth")
    metrics.plot_callback(loss, val_loss, "loss", num=43)
    metrics.plot_callback(f1, val_f1, "f1", num=42)
    metrics.report(y_true, y_classes.cpu().numpy())


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--label_path', type=str, default="data/new_label.json",
                        help='path of json that store the labeled frames')
    parser.add_argument('--mapping_path', type=str, default="data/mapping_test.json",
                        help='path of json that maps encrpypted video file name to simple naming')
    parser.add_argument('--data_dir', type=str, default="data/vgg16_emb/",
                        help='directory of extracted feature embeddings')
    parser.add_argument('--cache_path', type=str, default=".cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--model_path', type=str, default="h5_models/model.pth",
                        help='directory to store model weights and bias')
    parser.add_argument(
        "-preprocess", "--preprocess", action="store_true",
        default=False, help="Whether to perform IPCA")
    parser.add_argument(
        "-train", "--train", action="store_true",
        default=False, help="Whether to do training")
    parser.add_argument(
        "-test", "--test", action="store_true",
        default=False, help="Whether to do testing")
    return parser.parse_args()


def main() -> None:
    args = command_arg()
    label_path, mapping_path, data_dir, cache_path, model_path = args.label_path, args.mapping_path, args.data_dir, args.cache_path, args.model_path

    init_logger()
    torch_seeding()
    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    chunk_size = 5
    batch_size = 1024

    ipca = pk.load(open("ipca.pk1", "rb")) if os.path.exists(
        "ipca.pk1") else IncrementalPCA(n_components=2)
    # ,n_neighbors=1)
    nm = NearMiss(version=3, n_jobs=-1, sampling_strategy='majority')

    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=1)

    ds_train = Streamer(embedding_list_train, label_path,
                        mapping_path, data_dir, mem_split=20, chunk_size=chunk_size, batch_size=batch_size, sampler=sm)  # [('near_miss', nm), ('smote', sm)])
    ds_val = Streamer(embedding_list_val, label_path,
                      mapping_path, data_dir, mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)
    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_dir, mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)
    train_encodings = Streamer(embedding_list_train, label_path,
                               mapping_path, 'data/pts_encodings', mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=sm)
    test_encodings = Streamer(embedding_list_test, label_path,
                              mapping_path, 'data/pts_encodings', mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)

    model = LSTM(input_dim=18432, output_dim=6, hidden_dim=256,
                 layer_dim=1, bidirectional=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=5, verbose=True)
    # lower gpu float precision for larger batch size

    logging.info("{}".format(model.train()))
    if args.preprocess:
        logging.info("Preprocessing start...")
        ev = ds_train._fit_ipca(dest="samplers/ipca.pk1")
        logging.info("PCA Explained variance: {}".format(ev))
        logging.info("Preprocessing Done.")

    if args.train:
        # model.load_state_dict(torch.load(model_path)['model_state_dict'])
        logging.info("Starting Training")
        model = torch_training(ds_train, ds_val, train_encodings, model,
                               optimizer, criterion=criterion)
        logging.info("Done Training")

    if args.test:
        logging.debug(f"Loading from... -> {model_path}")
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        logging.info("Starting Evaluation")
        torch_testing(ds_test, test_encodings, model)
        logging.info("Done Evaluation")


if __name__ == "__main__":
    main()
