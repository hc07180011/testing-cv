import logging
import re
import os
import random
from tkinter import Y
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from mypyfunc.torch_eval import F1Score

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


def roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: int = 32,
    plots_folder="plots/"
):
    """
    plot ROC Curve
    """
    roc_auc, fpr, tpr = {}, {}, {}
    for i in range(y_pred.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot of a ROC curve for a specific class
    for i in range(classes+1):
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="dashed")
        plt.plot(fpr[i], tpr[i], marker="o")
        plt.plot([0, 0, 1], [0, 1, 1], linestyle="dashed", c="red")
        plt.legend([
            "No Skill",
            "ROC curve (area = {:.2f})".format(roc_auc[i]),
            "Perfect"
        ])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig(os.path.join(plots_folder, f"roc_curve_{i}.png"))
    plt.close()


def pr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plots_folder="plots/"
) -> None:
    """
    plot PR Curve
    """
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
    return thresholds


def cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    f1_metric: F1Score = F1Score(),
    plots_folder="plots/"
) -> None:
    pred_classes = torch.topk(y_pred, k=1, dim=1).indices.flatten()
    f1_score = f1_metric(pred_classes, y_true)

    logging.info("f1: {:.4f}".format(f1_score))

    # plot Confusion Matrix
    # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
    cm = confusion_matrix(
        y_true,
        pred_classes,
        labels=[1, 0]
    )
    fig = plt.figure(num=-1)
    ax = fig.add_subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title("f1: {:.4f}".format(f1_score))
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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc
