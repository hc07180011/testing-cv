import os
import json
import logging
import torch
import numpy as np
import torchvision
import torch.nn as nn
from imblearn.over_sampling import SMOTE

from typing import Callable, Tuple

from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_eval import F1Score, Evaluation
from mypyfunc.torch_models import CNN_LSTM
from mypyfunc.torch_data_loader import VideoLoader
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding


def training(
    train_loader: VideoLoader,
    val_loader: VideoLoader,
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
    for epoch in range(epochs):

        if loss_callback and epoch > 10 and loss_callback[-1] < 0.005:
            break
        model.train()

        minibatch_loss_train, minibatch_f1 = 0, 0
        for n_train, (input, labels) in enumerate(train_loader):
            input, labels = input.to(device), labels(device)
            output = model(input)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            f1 = f1_metric(torch.topk(objective(output),
                                      k=1, dim=1).indices.flatten(), labels)
            minibatch_loss_train += loss.item()
            minibatch_f1 += f1.item()

        model.eval()
        loss_callback += ((minibatch_loss_train / (n_train + 1))
                          if n_train else minibatch_loss_train,)
        f1_callback += ((minibatch_f1/(n_train + 1))
                        if n_train else minibatch_f1,)

        with torch.no_grad():
            minibatch_loss_val, minibatch_f1_val = 0, 0
            for n_val, (input, label) in enumerate(val_loader):
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = criterion(output, label)
                val_f1 = f1_metric(
                    torch.topk(objective(output), k=1, dim=1).indices.flatten(), label)
                minibatch_loss_val += loss.item()
                minibatch_f1_val += val_f1.item()

        val_loss_callback += ((minibatch_loss_val/(n_val + 1))
                              if n_val else minibatch_loss_val,)
        val_f1_callback += ((minibatch_f1_val/(n_val + 1))
                            if n_val else minibatch_f1_val,)

        logging.info(
            "Epoch: {}/{} Loss - {:.3f},f1 - {:.3f} val_loss - {:.3f}, val_f1 - {:.3f}".format(
                epoch + 1, epochs,
                loss_callback[-1],
                f1_callback[-1],
                val_loss_callback[-1],
                val_f1_callback[-1]
            ))

        if epoch > 10 and val_f1_callback[-1] > val_max_f1:
            save_checkpoint(f'{save_path}/model.pth', model,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics(f'{save_path}/metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)
            val_max_f1 = val_f1_callback[-1]

        train_loader.shuffle()
        val_loader.shuffle()

    torch.cuda.empty_cache()
    return model


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


def testing(
    test_loader: VideoLoader,
    model: nn.Module,
    objective: Callable = nn.Softmax(),
    classes: int = 6,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path: str = 'model0',
) -> None:
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    metrics = Evaluation(plots_folder="plots/", classes=classes)

    model.eval()
    y_pred, y_true = (), ()
    with torch.no_grad():
        for (input, label) in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            y_pred += (objective(output),)
            y_true += (label,)

    y_pred, y_true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
    y_pred, y_true = y_pred.detach().cpu().numpy(
    ), y_true.detach().cpu().numpy()  # changed here

    y_classes = torch.topk(y_pred, k=1, dim=1).indices.flatten()
    y_classes = y_classes.detach().cpu().numpy()

    # TO DO re implement frame tracking
    # metrics.miss_classified(X_test, y_classes, y_true,
    #                         test_set=ds_test.embedding_list_train)

    metrics.cm(y_true, y_classes)  # change here
    y_bin = np.zeros((y_true.shape[0], classes))
    idx = np.array([[i] for i in y_true])
    np.put_along_axis(y_bin, idx, 1, axis=1)

    metrics.roc_auc(y_bin, y_pred)
    metrics.pr_curve(y_bin, y_pred)

    loss, f1, val_loss, val_f1 = load_metrics(f"{save_path}/metrics.pth")
    metrics.plot_callback(loss, val_loss, "loss", num=43)
    metrics.plot_callback(f1, val_f1, "f1", num=42)
    metrics.report(y_true, y_classes)


def main() -> None:
    args = command_arg()
    label_path, mapping_path, data_dir, cache_path, model_path = args.label_path, args.mapping_path, args.data_dir, args.cache_path, args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_logger()
    torch_seeding(seed=12345)

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    train_list, val_list, test_list = tuple(
        __cache__[lst] for lst in __cache__)

    chunk_size = 11
    batch_size = 256
    input_dim = 18432
    output_dim = 2
    hidden_dim = 64
    layer_dim = 1
    vocab_size = 0
    bidirectional = True
    normalize = False

    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=7)

    train = VideoLoader(
        vid_list=train_list,
        label_path=label_path,
        data_dir=data_dir,
        chunk_size=chunk_size,
        batch_size=batch_size,
        shape=(15000, 380, 360, 3),
        sampler=sm,
        mov=False,
        norm=False
    )
    val = VideoLoader(
        vid_list=val_list,
        label_path=label_path,
        data_dir=data_dir,
        chunk_size=chunk_size,
        batch_size=batch_size,
        shape=(15000, 380, 360, 3),
        sampler=sm,
        mov=False,
        norm=False
    )
    test = VideoLoader(
        vid_list=test_list,
        label_path=label_path,
        data_dir=data_dir,
        chunk_size=chunk_size,
        batch_size=batch_size,
        shape=(15000, 380, 360, 3),
        sampler=sm,
        mov=False,
        norm=False
    )

    model = CNN_LSTM(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        bidirectional=bidirectional,
        normalize=normalize,
        vocab_size=vocab_size,
        model=torchvision.models.vgg16
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    training(train, val, optimizer, device=device)
    model.load_state_dict(torch.load(os.path.join(
        model_path, 'model.pth'))['model_state_dict'])

    logging.info("Starting Evaluation")
    testing(test, model, device=device, classes=2, save_path=model_path)
    logging.info("Done Evaluation")


if __name__ == "__main__":
    main()
