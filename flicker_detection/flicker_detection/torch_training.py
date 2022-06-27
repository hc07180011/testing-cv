import logging

import torch
import numpy as np
import torch.nn as nn
from torchmetrics import F1Score
from argparse import ArgumentParser
from mypyfunc.logger import init_logger
from mypyfunc.torch_models import LSTM, F1_Loss
from mypyfunc.torch_data_loader import Streamer
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding, evaluate, plot_callback, report_to_df, device
from sklearn.metrics import classification_report, f1_score


def torch_training(
    ds_train: Streamer,
    ds_val: Streamer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epochs: int = 1000,
    criterion=nn.BCELoss(),
    f1_torch=f1_score,  # F1_Loss().cuda(),
) -> nn.Module:

    val_max_f1 = 0
    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        if loss_callback and epoch > 10 and loss_callback[-1] < 0.005:
            break

        minibatch_loss_train, minibatch_f1 = 0, 0

        for n_train, (data, targets) in enumerate(ds_train):
            x = data.to(device)
            y = targets.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            f1_score = f1_torch(
                (y_pred.cpu() > 0.5).int(), y.cpu().int())
            # f1_score = f1_torch(y, y_pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            minibatch_loss_train += loss.item()
            minibatch_f1 += f1_score.item()

        model.eval()
        loss_callback += (minibatch_loss_train / (n_train + 1)
                          if n_train else minibatch_loss_train,)
        f1_callback += ((minibatch_f1/(n_train + 1)
                        if n_train else minibatch_f1),)
        # scheduler.step(loss_callback[-1])

        with torch.no_grad():
            minibatch_loss_val, minibatch_f1_val = 0, 0
            for n_val, (x, y) in enumerate(ds_val):
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_f1 = f1_torch(
                    (y_pred.cpu() > 0.5).int(), y.cpu().int())
                # val_f1 = f1_torch(y, y_pred)
                minibatch_loss_val += loss.item()
                minibatch_f1_val += val_f1.item()

        ds_val.shuffle()
        val_loss_callback += (minibatch_loss_val /
                              (n_val + 1) if n_val else minibatch_loss_val,)
        val_f1_callback += ((minibatch_f1_val/(n_val + 1)
                            if n_val else minibatch_f1),)

        if epoch > 10 and val_f1_callback[-1] > val_max_f1:
            save_checkpoint('model.pth', model,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics('metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)
            val_max_f1 = val_f1_callback[-1]

        logging.info(
            "Epoch: {}/{} Loss - {:.3f},f1 - {:.3f} val_loss - {:.3f}, val_f1 - {:.3f}".format(
                epoch + 1, epochs,
                loss_callback[-1],
                f1_callback[-1],
                val_loss_callback[-1],
                val_f1_callback[-1]
            ))

        model.train()
        ds_train.shuffle()

    torch.cuda.empty_cache()
    return model


def torch_eval(
    ds_test: Streamer,
    model: nn.Module,
) -> None:

    model.eval()
    y_pred, y_true = None, None
    with torch.no_grad():
        for x, y in ds_test:
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            y_pred = output if y_pred is None else\
                torch.hstack((y_pred, output))
            y_true = y if y_true is None else\
                torch.hstack((y_true, y))

    loss, f1, val_loss, val_f1 = load_metrics("metrics.pth")
    plot_callback(loss, val_loss, "loss")
    plot_callback(f1, val_f1, "f1")

    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    best = evaluate(y_true, y_pred)

    report = classification_report(
        y_true.astype(np.uint),
        (y_pred > best).astype(np.uint),
        labels=[1, 0], digits=4)
    report_to_df(report)


if __name__ == "__main__":
    init_logger()

    label_path = "data/new_label.json"
    mapping_path = "data/mapping_aug_data.json"
    data_dir = "data/vgg16_emb"
    cache_path = ".cache/train_test"
    model_path = "model.pth"

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    ds_train = Streamer(embedding_list_train, label_path,
                        mapping_path, data_dir, mem_split=20, chunk_size=32, batch_size=1024, oversample=True)
    ds_val = Streamer(embedding_list_val, label_path,
                      mapping_path, data_dir, mem_split=1, chunk_size=32, batch_size=1024, oversample=False)
    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_dir, mem_split=1, chunk_size=32, batch_size=1024, oversample=True)

    model = LSTM(input_dim=18432, hidden_dim=256,
                 layer_dim=1)
    logging.info("{}".format(model.train()))

    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, verbose=True)
    # lower gpu float precision for larger batch size

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
        # model.load_state_dict(torch.load(model_path)['model_state_dict'])
        logging.info("Starting Training")
        model = torch_training(ds_train, ds_val, model,
                               optimizer, scheduler)
        logging.info("Done Training")

    if args.test:
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        logging.info("Starting Evaluation")
        torch_eval(ds_test, model)
        logging.info("Done Evaluation")
