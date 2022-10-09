import os
import json
import gc
import logging
import tqdm
import torch
import numpy as np
import torchvision
import torch.nn as nn
from typing import Callable, Tuple

from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_eval import F1Score, Evaluation
from mypyfunc.torch_models import LSTM, ExtractorCNN
from mypyfunc.torch_data_loader import Loader, VideoLoader
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding


def training(
    train_loader: VideoLoader,
    val_loader: VideoLoader,
    models: Tuple[nn.Module, nn.Module],
    optimizer: torch.optim.Optimizer,
    f1_metric: Callable,
    criterion: Callable,
    objective: Callable,
    epochs: int,
    device: torch.device,
    save_path: str = 'model0',
) -> nn.Module:
    val_max_f1 = 0
    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        if loss_callback and epoch > 10 and loss_callback[-1] < 0.005:
            break

        models[0].train()
        models[-1].train()

        minibatch_loss_train, minibatch_f1 = 0, 0
        for n_train, (inputs, labels) in enumerate(tqdm.tqdm(train_loader)):
            inputs = torch.from_numpy(inputs).permute(
                0, 1, 4, 2, 3).float().to(device)
            labels = torch.from_numpy(labels).long().to(device)
            batch_size, chunk_size = inputs.shape[:2]

            features = models[0](inputs.flatten(
                end_dim=1)).flatten(start_dim=1)
            features = features.reshape(
                (batch_size, chunk_size, features.shape[-1]))
            outputs = models[-1](features)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(models[-1].parameters(), 1.0)
            optimizer.step()

            f1 = f1_metric(torch.topk(objective(outputs),
                                      k=1, dim=1).indices.flatten(), labels)
            minibatch_loss_train += loss.item()
            minibatch_f1 += f1.item()

        models[0].eval()
        models[-1].eval()
        loss_callback += ((minibatch_loss_train / (n_train + 1))
                          if n_train else minibatch_loss_train,)
        f1_callback += ((minibatch_f1/(n_train + 1))
                        if n_train else minibatch_f1,)

        with torch.no_grad():
            minibatch_loss_val, minibatch_f1_val = 0, 0
            for n_val, (inputs, labels) in enumerate(tqdm.tqdm(val_loader)):
                inputs = torch.from_numpy(inputs).permute(
                    0, 1, 4, 2, 3).float().to(device)
                labels = torch.from_numpy(labels).long().to(device)
                batch_size, chunk_size = inputs.shape[:2]

                features = models[0](inputs.flatten(
                    end_dim=1)).flatten(start_dim=1)
                features = features.reshape(
                    (batch_size, chunk_size, features.shape[-1]))
                outputs = models[-1](features)

                loss = criterion(outputs, labels)
                val_f1 = f1_metric(
                    torch.topk(objective(outputs), k=1, dim=1).indices.flatten(), labels)
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
            save_checkpoint(f'{save_path}/model.pth', models,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics(f'{save_path}/metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)
            val_max_f1 = val_f1_callback[-1]

        train_loader._shuffle()
        val_loader._shuffle()

    torch.cuda.empty_cache()
    return models


def testing(
    test_loader: VideoLoader,
    models: Tuple[nn.Module, nn.Module],
    objective: Callable,
    device: torch.device,
    classes: int = 2,
    save_path: str = 'model0',
) -> None:
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    metrics = Evaluation(plots_folder="plots/", classes=classes)

    models[0].eval()
    models[-1].eval()
    y_pred, y_true = (), ()
    with torch.no_grad():
        for (inputs, labels) in tqdm.tqdm(test_loader):
            inputs = torch.from_numpy(inputs).permute(
                0, 1, 4, 2, 3).float().to(device)
            labels = torch.from_numpy(labels).long().to(device)
            batch_size, chunk_size = inputs.shape[:2]

            features = models[0](inputs.flatten(
                end_dim=1)).flatten(start_dim=1)
            features = features.reshape(
                (batch_size, chunk_size, features.shape[-1]))
            outputs = models[-1](features)

            y_pred += (objective(outputs),)
            y_true += (labels,)

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


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--label_path', type=str, default="data/new_label.json",
                        help='path of json that store the labeled frames')
    parser.add_argument('--flicker_dir', type=str, default="data/flicker-chunks",
                        help='directory of processed flicker video chunks')
    parser.add_argument('--non_flicker_dir', type=str, default="data/meta-data",
                        help='directory of processed non flicker video chunks')
    parser.add_argument('--cache_path', type=str, default=".cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--model_path', type=str, default="model0",
                        help='directory to store model weights and bias')
    parser.add_argument(
        "-train", "--train", action="store_true",
        default=False, help="Whether to do training")
    parser.add_argument(
        "-test", "--test", action="store_true",
        default=False, help="Whether to do testing")
    return parser.parse_args()


def main() -> None:
    args = command_arg()
    label_path, flicker_path, non_flicker_path, cache_path, model_path = args.label_path, args.flicker_dir, args.non_flicker_dir, args.cache_path, args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_logger()
    torch_seeding(seed=12345)

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    flicker_train, non_flicker_train, fp_test, flicker_test, non_flicker_test = tuple(
        __cache__[lst] for lst in __cache__)
    labels = json.load(open(label_path, 'rb'))

    input_dim = 61952
    output_dim = 2
    hidden_dim = 64
    layer_dim = 1
    bidirectional = False
    in_mem_batches = 1000
    batch_size = 6
    # shape = (6, 10, 3, 360, 360)

    feature_extractor = ExtractorCNN(
        cnn=torchvision.models.vgg16(pretrained=True)
    )
    model = LSTM(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        bidirectional=bidirectional,
    )

    feature_extractor = torch.nn.DataParallel(feature_extractor)
    model = torch.nn.DataParallel(model)
    feature_extractor.to(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    metric = F1Score(average='macro')
    criterion = nn.CrossEntropyLoss()
    objective = nn.Softmax()
    epochs = 1000

    if args.train:
        logging.info("Loading training set..")
        ds_train = Loader(
            non_flicker_lst=non_flicker_train,
            flicker_lst=flicker_train,
            non_flicker_dir=non_flicker_path,
            flicker_dir=flicker_path,
            labels=labels,
            batch_size=batch_size,
            in_mem_batches=in_mem_batches
        )
        logging.info("Done loading training set")

        logging.info("Loading validtaion set..")
        ds_val = Loader(
            non_flicker_lst=np.concatenate(
                (non_flicker_test, fp_test), axis=0),
            flicker_lst=flicker_test,
            non_flicker_dir=non_flicker_path,
            flicker_dir=flicker_path,
            labels=labels,
            batch_size=batch_size,
            in_mem_batches=in_mem_batches
        )
        logging.info("Done loading validation set")

        logging.info(f"{feature_extractor.train()}")
        logging.info(f"{model.train()}")
        logging.info("Starting Training Video Model")
        training(
            train_loader=ds_train,
            val_loader=ds_val,
            models=(feature_extractor, model),
            optimizer=optimizer,
            f1_metric=metric,
            criterion=criterion,
            objective=objective,
            epochs=epochs,
            device=device
        )
        logging.info("Done Training Video Model...")

    if args.test:
        logging.info("Loading testing set..")
        ds_test = Loader(
            non_flicker_lst=np.concatenate(
                (non_flicker_test, fp_test), axis=0),
            flicker_lst=flicker_test,
            non_flicker_dir=non_flicker_path,
            flicker_dir=flicker_path,
            labels=labels,
            batch_size=batch_size,
            in_mem_batches=in_mem_batches
        )
        logging.info("Done loading testing set")

        logging.info("Starting Evaluation")
        feature_extractor.load_state_dict(torch.load(os.path.join(
            model_path, 'model.pth'))['feature_extractor'])
        model.load_state_dict(torch.load(os.path.join(
            model_path, 'model.pth'))['model'])
        testing(
            ds_test,
            (feature_extractor, model),
            objective=objective,
            device=device,
            classes=2,
            save_path=model_path
        )
        logging.info("Done Evaluation")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    """
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 training.py --train
    """
    main()
