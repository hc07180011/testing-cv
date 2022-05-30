import os
import logging
import random
import re
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torchmetrics import F1Score
from io import StringIO
from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_models import LSTMModel, F1_Loss
from mypyfunc.torch_data_loader import Streamer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc, roc_auc_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save and Load Functions


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
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def torch_training(
    ds_train: Streamer,
    ds_val: Streamer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1000,
    criterion=nn.BCELoss(),
    f1_torch=F1Score()  # F1_Loss().cuda(),
) -> nn.Module:

    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        minibatch_loss_train, minibatch_f1 = 0, 0
        n_train = None
        for n_train, (x, y) in enumerate(ds_train):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            f1_score = f1_torch((y_pred.cpu() > 0.5).int(), y.cpu().int())
            loss.backward()
            optimizer.step()

            minibatch_loss_train += loss.item()
            minibatch_f1 += f1_score

        model.eval()
        loss_callback += (minibatch_loss_train/n_train,)
        f1_callback += (minibatch_f1/n_train,)

        with torch.no_grad():
            minibatch_loss_val, minibatch_f1_val = 0, 0
            for n_val, (x, y) in enumerate(ds_val):
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_f1 = f1_torch((y_pred.cpu() > 0.5).int(), y.cpu().int())
                minibatch_loss_val += loss.item()
                minibatch_f1_val += val_f1
            if n_val > 0:
                val_loss_callback += (minibatch_loss_val/n_val,)
                val_f1_callback += (minibatch_f1_val/n_val,)
            else:
                val_loss_callback += (minibatch_loss_val,)
                val_f1_callback += (minibatch_f1_val,)

        logging.info(
            "Epoch: {}/{}, Loss - {:.3f},f1 - {:.3f}, val_loss - {:3f}, val_f1 - {:3f}".format(
                epoch, epochs, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1]
            ))

        if not bool(epoch % 10):
            save_checkpoint('model.pth', model,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics('metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)

        model.train()

        random.shuffle(embedding_list_train)
        ds_train.new_embeddings(embedding_list_train)

    return model


def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total",
                                                "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report),
                            sep=' ', index_col=0, on_bad_lines='skip')
    report_df.to_csv("report.csv")
    return report_df


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plots_folder="plots/"
) -> None:

    # plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot([0, 1], [0, 1], linestyle="dashed")
    plt.plot(fpr, tpr, marker="o")
    plt.plot([0, 0, 1], [0, 1, 1], linestyle="dashed", c="red")
    plt.legend([
        "No Skill",
        "ROC curve (area = {:.2f})".format(auc(fpr, tpr)),
        "Perfect"
    ])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(os.path.join(plots_folder, "roc_curve.png"))
    plt.close()

    # plot PR Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot([0, 1], [0, 0], linestyle="dashed")
    plt.plot(recall, precision, marker="o")
    plt.legend([
        "No Skill",
        "Model"
    ])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall Curve")
    plt.savefig(os.path.join(plots_folder, "pc_curve.png"))
    plt.close()

    threshold_range = np.arange(0.1, 1.0, 0.001)
    f1_scores = tuple(f1_score(y_true, (y_pred > lambda_).astype(int))
                      for lambda_ in threshold_range)
    logging.info("Max f1: {:.4f}, at thres = {:.4f}".format(
        np.max(f1_scores), threshold_range[np.argmax(f1_scores)]
    ))

    # plot Confusion Matrix
    # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
    cm = confusion_matrix(
        y_true,
        (y_pred > threshold_range[np.argmax(f1_scores)]).astype(int),
        labels=[1, 0]
    )
    fig = plt.figure(num=-1)
    ax = fig.add_subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title("Max f1: {:.4f}, at thres = {:.4f}".format(
        np.max(f1_scores), threshold_range[np.argmax(f1_scores)]
    ))
    fig.savefig(os.path.join(plots_folder, "confusion_matrix.png"))


def plot_callback(train_metric: np.ndarray, val_metric: np.ndarray, name: str, num=0):
    plt.figure(num=num, figsize=(16, 4), dpi=200)
    plt.plot(val_metric)
    plt.plot(train_metric)
    plt.legend(["val_{}".format(name), "{}".format(name), ])
    plt.xlabel("# Epochs")
    plt.ylabel("{}".format(name))
    plt.title("{} LSTM, Chunked, Oversampling".format(name))
    plt.savefig("{}.png".format(
        os.path.join("plots/", name)))
    plt.close()


def torch_eval(
    ds_test: Streamer,
    model: nn.Module,
    threshold: float = 0.5,
) -> None:

    model.load_state_dict(torch.load('model.pth')['model_state_dict'])
    model.eval()

    y_pred, y_true = None, None
    with torch.no_grad():
        for batch_step, (x, y) in enumerate(ds_test):
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            y_pred = output if y_pred is None else torch.hstack(
                (y_pred, output))
            y_true = y if y_true is None else torch.hstack((y_true, y))

    loss, f1, val_loss, val_f1 = load_metrics("metrics.pth")
    plot_callback(loss, val_loss, "loss")
    plot_callback(f1, val_f1, "f1")

    y_pred, y_true = (y_pred.cpu().numpy() >
                      threshold).astype(np.uint), y_true.cpu().numpy().astype(np.uint)
    evaluate(y_true, y_pred)

    report = classification_report(y_true, y_pred, labels=[1, 0], digits=4)
    report_to_df(report)


if __name__ == "__main__":
    init_logger()

    label_path = "data/label.json"
    mapping_path = "data/mapping_aug_data.json"
    data_dir = "data/vgg16_emb"
    cache_path = ".cache/train_test"

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    ds_train = Streamer(embedding_list_train, label_path,
                        mapping_path, data_dir, mem_split=2, chunk_size=30, batch_size=1024)
    ds_val = Streamer(embedding_list_val, label_path,
                      mapping_path, data_dir, mem_split=2, chunk_size=30, batch_size=1024)
    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_dir, mem_split=2, chunk_size=30, batch_size=1024)

    model = LSTMModel(input_dim=18432, hidden_dim=256,
                      layer_dim=1)
    logging.info("{}".format(model.train()))

    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # FIX ME
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    parser = ArgumentParser()
    parser.add_argument(
        "-train", "--train", action="store_true",
        default=False,
        help="Whether to do training"
    )
    parser.add_argument(
        "-test", "--test", action="store_true",
        default=False,
        help="Whether to do testing"
    )
    args = parser.parse_args()

    torch_seeding()

    if args.train:
        logging.info("Starting Training")
        model = torch_training(ds_train, ds_val, model, optimizer)
        logging.info("Done Training")

    if args.test:
        logging.info("Starting Evaluation")
        torch_eval(ds_test, model)
        logging.info("Done Evaluation")
