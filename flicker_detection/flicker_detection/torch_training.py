import os
import logging
import random
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torchmetrics import F1Score
from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_models import LSTMModel
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

    return state_dict['loss_callback'], state_dict['f1_callback'], state_dict['val_loss_callback'], state_dict['valid_f1_callback']


def torch_training(
    ds_train: Streamer,
    ds_val: Streamer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1000,
    criterion=nn.BCELoss(),
    f1_torch=F1Score(),
) -> nn.Module:

    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        minibatch_loss_train = 0
        n_train = None
        for n_train, (x, y) in enumerate(ds_train):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            minibatch_loss_train += loss.item()

        model.eval()
        loss = minibatch_loss_train/n_train
        f1_score = f1_torch((y_pred.cpu() > 0.5).int(), y.cpu().int())
        loss_callback += (loss,)
        f1_callback += (f1_score,)

        with torch.no_grad():
            minibatch_loss_val = 0
            for n_val, (x, y) in enumerate(ds_val):
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                minibatch_loss_val += loss.item()
            val_loss = minibatch_loss_val/n_val
            val_f1 = f1_torch((y_pred.cpu() > 0.5).int(), y.cpu().int())
            val_loss_callback += (val_loss,)
            val_f1_callback += (val_f1,)

        logging.info(
            "Epoch: {}/{}, Loss - {:.3f},f1 - {:.3f}, val_loss - {:3f}, val_f1 - {:3f}".format(
                epoch, epochs, loss, f1_score, val_loss, val_f1
            ))

        if not bool(epoch % 10):
            save_checkpoint('model.pth', model,
                            optimizer, loss, val_loss, f1_score, val_f1)
            save_metrics('metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)

        model.train()

        random.shuffle(embedding_list_train)
        ds_train.new_embeddings(embedding_list_train)

    return model


def classification_report_csv(report: str) -> None:
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index=False)


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


def torch_eval(
    ds_test: Streamer,
    model: nn.Module,
    threshold: float = 0.5,
) -> None:

    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    y_pred, y_true = (), ()
    with torch.no_grad():
        mini_batch_loss = 0
        for batch_step, (x, y) in enumerate(ds_test):
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            y_pred += ((output.cpu() > threshold).int(),)
            y_true += (y.cpu().int(),)

    evaluate(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    classification_report_csv(report)


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
                        mapping_path, data_dir, batch_size=32)
    ds_val = Streamer(embedding_list_val, label_path,
                      mapping_path, data_dir, batch_size=32)
    ds_test = Streamer(embedding_list_train, label_path,
                       mapping_path, data_dir, batch_size=32)

    model = LSTMModel(input_dim=18432, hidden_dim=256,
                      layer_dim=1, output_dim=1)
    logging.info("{}".format(model.train()))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    if args.train:
        logging.info("Starting Training")
        model = torch_training(ds_train, ds_val, model, optimizer)
        logging.info("Done Training")

    if args.test:
        logging.info("Starting Evaluation")
        torch_eval(ds_test, model)
        logging.info("Done Evaluation")
