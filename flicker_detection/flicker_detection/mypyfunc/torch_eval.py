import torch
import numpy as np
from typing import Tuple
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch import nn
from torch_data_loader import Streamer
from torch_models import LSTM


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
        # print(labels)
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
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
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


def test() -> None:
    label_path = "../data/new_label.json"
    mapping_path = "../data/mapping_aug_data.json"
    data_dir = "../data/InceptionResNetV2_emb/"
    __cache__ = np.load("{}.npz".format(
        "../.cache/train_test"), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_dir, mem_split=1, batch_size=256, sampler=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f1_metric = F1Score()
    model = LSTM(input_dim=24576, output_dim=32, hidden_dim=256,
                 layer_dim=1, bidirectional=False)
    model.to(device)
    for x, y in ds_train:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_pred = torch.topk(y_pred, k=1, dim=1).indices.flatten()
        print(y_pred, '\n', y)
        f1 = f1_metric(y_pred, y)
        print(f"f1-score: {f1}")


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
    test()
