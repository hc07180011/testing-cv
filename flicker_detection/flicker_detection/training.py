import os
import logging
from sklearn.decomposition import IncrementalPCA
import torch
import numpy as np
import pickle as pk
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from decord import VideoLoader, VideoReader, cpu, gpu
from typing import Callable, Tuple

from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_eval import F1Score, Evaluation
from mypyfunc.torch_models import LSTM, SentimentRNN
from mypyfunc.torch_data_loader import Streamer
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding


def train(
    video_loader: VideoLoader,
    cache_shape:tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    f1_metric: Callable = F1Score(average='macro'),
    criterion: Callable = nn.CrossEntropyLoss(),
    objective: Callable = nn.Softmax(),
    epochs: int = 1000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path: str = 'model0',
) -> nn.Module:
    val_max_f1 = 0
    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    cache = torch.zeros(cache_shape,dtype=torch.uint8)
    for epoch in range(epochs):
        if loss_callback and epoch > 10 and loss_callback[-1] < 0.005:
            break
        minibatch_loss_train, minibatch_f1 = 0, 0
        for n_train, (x0, y0) in enumerate(video_loader):
            pass
    pass


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--label_path', type=str, default="data/new_label.json",
                        help='path of json that store the labeled frames')
    parser.add_argument('--mapping_path', type=str, default="data/mapping.json",
                        help='path of json that maps encrpypted video file name to simple naming')
    parser.add_argument('--data_dir', type=str, default="data/vgg16_emb/",
                        help='directory of extracted feature embeddings')
    parser.add_argument('--cache_path', type=str, default=".cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--model_path', type=str, default="model0",
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


if __name__ == "__main__":
    args = command_arg()
    label_path, mapping_path, data_dir, cache_path, model_path = args.label_path, args.mapping_path, args.data_dir, args.cache_path, args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_logger()
    torch_seeding(seed=12345)

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)
    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=7)

    vl = VideoLoader(
        embedding_list_train,
        ctx=[cpu(0)],
        shape=(11*1024, 3040//2, 1440//2, 3),
        interval=1,
        skip=5,
        shuffle=1
    )
    pass
