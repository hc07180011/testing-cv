import re
import os
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from typing import Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score, f1_score, classification_report
from torch.nn import functional as F
from torch import nn


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(
            predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(
                                    true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(
            f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(len(labels.unique())):
            f1, true_count = self.calc_f1_count_for_label(
                predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    #sklearn.metrics.f1_score
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


class Evaluation(object):
    """
    https://onlineconfusionmatrix.com/
    https://discuss.pytorch.org/t/bce-loss-vs-cross-entropy/97437/3
    """
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    def __init__(self,
                 plots_folder: str = "plots/",
                 classes: int = 2,
                 f1_metric: F1Score = F1Score(average='macro'),
                 ) -> None:
        self.plots_folder = plots_folder
        self.classes = classes
        self.f1_metric = f1_metric

    def roc_auc(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        plot ROC Curve
        https://stackoverflow.com/questions/45332410/roc-for-multiclass-classification
        """
        roc_auc, fpr, tpr = {}, {}, {}
        for i in range(self.classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot of a ROC curve for a specific class
        for i in range(self.classes):
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
            plt.title(f"Class-{i} ROC Curve")
            plt.savefig(os.path.join(self.plots_folder, f"roc_curve_{i}.png"))
        plt.close()

    def pr_curve(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        plot PR Curve
        """
        precision, recall = {}, {}
        for i in range(self.classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true[:, i], y_pred[:, i])
        # Plot of a ROC curve for a specific class
        for i in range(self.classes):
            plt.figure()
            plt.plot([0, 1], [0, 0], linestyle="dashed")
            plt.plot(recall[i], precision[i], marker="o")
            plt.legend([
                "No Skill",
                "Model"
            ])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Class-{i} Precision-recall Curve")
            plt.savefig(os.path.join(self.plots_folder, f"pc_curve_{i}.png"))
        plt.close()

    def cm(
        self,
        y_true: torch.tensor,
        y_pred: torch.tensor,
    ) -> None:
        f1_score = self.f1_metric(y_pred, y_true)
        logging.info("f1: {:.4f}".format(f1_score))

        # plot Confusion Matrix
        # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
        cm = confusion_matrix(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
        )
        fig = plt.figure(num=-1)
        ax = fig.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title("Multiclass F1 Harmonization: {:.4f}".format(f1_score))
        fig.savefig(os.path.join(self.plots_folder, "confusion_matrix.png"))

    @ staticmethod
    def plot_callback(
        train_metric: np.ndarray,
        val_metric: np.ndarray,
        name: str, num=0
    ) -> None:
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

    @ staticmethod
    def report_to_df(report) -> pd.DataFrame:  # FIX ME
        report = re.sub(r" +", " ", report).replace("avg / total",
                                                    "avg/total").replace("\n ", "\n")
        report_df = pd.read_csv(StringIO("Classes" + report),
                                sep=' ', index_col=0, on_bad_lines='skip')
        report_df.to_csv("plots/report.csv")
        return report_df

    def report(
        self,
        y_true: np.ndarray,
        y_classes: np.ndarray,
    ) -> pd.DataFrame:
        return self.report_to_df(
            classification_report(y_true, y_classes, digits=4)
        )

    @staticmethod
    def miss_classified(
        X_test: torch.Tensor,
        y_classes: torch.Tensor,
        y_true: torch.Tensor,
        data_src: str = 'data/vgg16_emb',
        missed_out: str = 'data/missed_labels.json',
        test_set: str = None,
    ) -> None:
        midx = (y_classes != y_true).nonzero().flatten()
        logging.debug(f"{X_test[midx]} - {len(midx)}")
        X_test = X_test[midx]
        # X_test = X_test[midx].reshape(
        # (X_test[midx].shape[0]*X_test[midx].shape[1], X_test[midx].shape[-1]))

        if len(midx) > 0 and os.path.isdir(data_src) and len(os.listdir(data_src)) != 0:
            missed_labels = {}
            for emb in test_set:
                embedding = torch.from_numpy(
                    np.load(f"{os.path.join(data_src,emb)}"))

                X_test = torch.cat((
                    X_test,
                    torch.ones(
                        (embedding.shape[0]//X_test.shape[1] - X_test.shape[0], *X_test.shape[1:]), dim=0)
                ))

                if embedding.shape[0]//X_test.shape[1] < X_test.shape[0]:
                    embedding = torch.cat((
                        embedding,
                        torch.ones((
                            (X_test.shape[1] - embedding.shape[0] % X_test.shape[1]) +
                            (X_test.shape[0] - embedding.shape[0]//X_test.shape[1] - 1) *
                            X_test.shape[1],
                            X_test.shape[-1]))
                    ), dim=0)

                embedding = embedding.reshape(
                    (embedding.shape[0]//X_test.shape[1], *X_test.shape[1:]))

                idx = torch.logical_not(
                    torch.sum((X_test - embedding), dim=1)).sum().item()
                logging.debug(f"{emb} - {idx} - {type(idx)}")
                missed_labels[emb] = idx//X_test.shape[1]
            json.dump(missed_labels, open(f"{missed_out}", "w"))
        return X_test[midx]


def test_sk() -> None:
    errors = 0
    for _ in range(10):
        labels = torch.randint(1, 10, (4096, 100)).flatten()
        predictions = torch.randint(1, 10, (4096, 100)).flatten()
        labels1 = labels.numpy()
        predictions1 = predictions.numpy()
        print(labels.cuda().unique(), predictions.cuda().unique())
        for av in ['micro', 'macro', 'weighted']:
            f1_metric = F1Score(av)
            my_pred = f1_metric(predictions.cuda(), labels.cuda())

            f1_pred = f1_score(labels1, predictions1, average=av)
            # print(my_pred, f1_pred)
            if not np.isclose(my_pred.item(), f1_pred.item()):
                print('!' * 50)
                print(f1_pred, my_pred, av)
                errors += 1

    if errors == 0:
        print('No errors!')


if __name__ == "__main__":
    test_sk()
