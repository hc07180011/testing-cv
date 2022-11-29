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
from mypyfunc.torch_models import CNN_LSTM,CNN_Transformers,OHEMLoss
from mypyfunc.torch_utility import save_checkpoint, save_metrics, load_checkpoint, load_metrics, torch_seeding
from mypyfunc.streamer import MultiStreamer, VideoDataSet


def training(
    train_loader: MultiStreamer,
    val_loader: MultiStreamer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim,
    f1_metric: Callable,
    criterion: Callable,
    objective: Callable,
    epochs: int,
    device: torch.device,
    save_path: str,
) -> nn.Module:
    val_max_f1 = 0
    f1_callback, loss_callback, val_f1_callback, val_loss_callback = (), (), (), ()
    for epoch in range(epochs):
        if loss_callback and epoch > 11 and loss_callback[-1] < 0.05:
            break

        model.train()
        minibatch_loss_train, minibatch_f1 = 0, 0
        for n_train, (inputs, labels) in enumerate(tqdm.tqdm(train_loader)):
            inputs = inputs.permute(
                0, 1, 4, 2, 3).float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels,epoch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            f1 = f1_metric(torch.topk(objective(outputs),
                                      k=1, dim=1).indices.flatten(), labels)
            minibatch_loss_train += loss.item()
            minibatch_f1 += f1.item()

        model.eval()
        loss_callback += ((minibatch_loss_train / (n_train + 1))
                          if n_train else minibatch_loss_train,)
        f1_callback += ((minibatch_f1/(n_train + 1))
                        if n_train else minibatch_f1,)
        scheduler.step(loss_callback[-1])
        
        with torch.no_grad():
            minibatch_loss_val, minibatch_f1_val = 0, 0
            for n_val, (inputs, labels) in enumerate(tqdm.tqdm(val_loader)):
                inputs = inputs.permute(
                    0, 1, 4, 2, 3).float().to(device)
                labels = labels.long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels,epoch)
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
            save_checkpoint(f'{save_path}/model.pth', model,
                            optimizer, loss_callback[-1], f1_callback[-1], val_loss_callback[-1], val_f1_callback[-1])
            save_metrics(f'{save_path}/metrics.pth', loss_callback, f1_callback,
                         val_loss_callback, val_f1_callback)
            val_max_f1 = val_f1_callback[-1]

    torch.cuda.empty_cache()
    return model



def testing(
    test_loader: MultiStreamer,
    model: nn.Module,
    objective: Callable,
    device: torch.device,
    classes: int,
    save_path: str,
) -> None:
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    metrics = Evaluation(plots_folder="plots/", classes=classes)
    model.load_state_dict(torch.load(os.path.join(
        save_path, 'model.pth'))['model_state_dict'])
    model.eval()
    y_pred, y_true = (), () 
    with torch.no_grad():
        for (inputs, labels) in tqdm.tqdm(test_loader):
            inputs = inputs.permute(
                0, 1, 4, 2, 3).float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
              
            y_pred += (objective(outputs),)
            y_true += (labels,)

    y_pred, y_true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
    y_pred, y_true = y_pred.detach().cpu(), y_true.detach().cpu()
    y_classes = torch.topk(y_pred, k=1, dim=1).indices.flatten()
    y_classes = y_classes.detach().cpu()

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
    parser.add_argument('--flicker1', type=str, default="data/flicker1",
                        help='directory of flicker1')
    parser.add_argument('--flicker2', type=str, default="data/flicker2",
                        help='directory of flicker2')
    parser.add_argument('--flicker3', type=str, default="data/flicker3",
                        help='directory of flicker3')
    parser.add_argument('--flicker4', type=str, default="data/flicker4",
                        help='directory of flicker4')
    parser.add_argument('--non_flicker_dir', type=str, default="data/no_flicker",
                        help='directory of processed non flicker video chunks')
    parser.add_argument('--label_path', type=str, default="data/multi_label.json",
                        help='directory of labels json')
    parser.add_argument('--cache_path', type=str, default=".cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--model_path', type=str, default="cnn_lstm_model",
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
    label_path, flicker1_path, flicker2_path, flicker3_path, flicker4_path, non_flicker_path, cache_path, model_path =\
        args.label_path, args.flicker1, args.flicker2, args.flicker3, args.flicker4, args.non_flicker_dir, args.cache_path, args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_logger()
    torch_seeding(seed=12345)

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    flicker_train, non_flicker_train, flicker_test, non_flicker_test = tuple(
        __cache__[lst] for lst in __cache__)

    labels = json.load(open(label_path, 'r'))
    
    input_dim = 25088# 61952
    output_dim = 2
    hidden_dim = 64
    layer_dim = 1
    bidirectional = True
    batch_size = 4
    class_size = batch_size//output_dim
    max_workers = 1
    
    model = CNN_LSTM(
        cnn=torchvision.models.vgg19(pretrained=True),
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        bidirectional=bidirectional,
    )

    # model = CNN_Transformers(
    #     image_size=360,          # image size
    #     frames=10,               # number of frames
    #     image_patch_size=36,     # image patch size
    #     frame_patch_size=10,      # frame patch size
    #     num_classes=output_dim,
    #     dim=512,
    #     depth=6,
    #     heads=8,
    #     mlp_dim=512,
    #     cnn=torchvision.models.vgg19(pretrained=True),
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     pool='cls' 
    # )  # 16784 of 19456 gpu mb 0.6094
    
    model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.SGD(# try SGD
        model.parameters(),lr=1e-3, weight_decay=1e-4,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=5,verbose=True) 
    metric = F1Score(average='macro')
    criterion = OHEMLoss(batch_size=batch_size//2,init_epoch=40,criterion=nn.CrossEntropyLoss())  
    objective = nn.Softmax()
    epochs = 1000

    if args.train:
        logging.info("Loading training set..")
        non_flicker_train = [os.path.join(non_flicker_path, f)
                             for f in non_flicker_train]
        flicker1_train = [os.path.join(flicker1_path, f)
                          for f in flicker_train if f in os.listdir(flicker1_path)]
        flicker2_train = [os.path.join(flicker2_path, f)
                          for f in flicker_train if f in os.listdir(flicker2_path)]
        flicker3_train = [os.path.join(flicker3_path, f)
                          for f in flicker_train if f in os.listdir(flicker3_path)]
        flicker4_train = [os.path.join(flicker4_path, f)
                          for f in flicker_train if f in os.listdir(flicker4_path)]
        non_flicker_train = VideoDataSet.split_datasets(
            non_flicker_train, labels=labels, class_size=class_size, max_workers=max_workers, undersample=1000)
        flicker1_train = VideoDataSet.split_datasets(
            flicker1_train+flicker2_train+flicker3_train+flicker4_train, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)  # +flicker2_train+flicker3_train+flicker4_train
        # flicker2_train = VideoDataSet.split_datasets(
        #     flicker2_train, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)
        # flicker3_train = VideoDataSet.split_datasets(
        #     flicker3_train, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)
        # flicker4_train = VideoDataSet.split_datasets(
        #     flicker4_train, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)

        ds_train = MultiStreamer(
            non_flicker_train,
            flicker1_train,
            # flicker2_train,
            # flicker3_train,
            # flicker4_train,
            batch_size=batch_size,
        )
        logging.info("Done loading training set")

        logging.info("Loading validtaion set..")
        non_flicker_val = [os.path.join(non_flicker_path, f)
                           for f in non_flicker_test]
        flicker1_val = [os.path.join(flicker1_path, f)
                        for f in flicker_test if f in os.listdir(flicker1_path)]
        flicker2_val = [os.path.join(flicker2_path, f)
                        for f in flicker_test if f in os.listdir(flicker2_path)]
        flicker3_val = [os.path.join(flicker3_path, f)
                        for f in flicker_test if f in os.listdir(flicker3_path)]
        flicker4_val = [os.path.join(flicker4_path, f)
                        for f in flicker_test if f in os.listdir(flicker4_path)]
        non_flicker_val = VideoDataSet.split_datasets(
            non_flicker_val, labels=labels, class_size=class_size, max_workers=max_workers, undersample=300)
        flicker1_val = VideoDataSet.split_datasets(
            flicker1_val+flicker2_val+flicker3_val+flicker4_val, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)  # +flicker2_val+flicker3_val+flicker4_val
        # flicker2_val = VideoDataSet.split_datasets(
        #     flicker2_val, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)
        # flicker3_val = VideoDataSet.split_datasets(
        #     flicker3_val, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)
        # flicker4_val = VideoDataSet.split_datasets(
        #     flicker4_val, labels=labels, class_size=class_size, max_workers=max_workers, oversample=True)

        ds_val = MultiStreamer(
            non_flicker_val,
            flicker1_val,
            # flicker2_val,
            # flicker3_val,
            # flicker4_val,
            batch_size=batch_size,
        )
        logging.info("Done loading validation set")

        logging.info(f"{model.train()}")
        logging.info("Starting Training Video Model")
        training(
            train_loader=ds_train,
            val_loader=ds_val,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            f1_metric=metric,
            criterion=criterion,
            objective=objective,
            epochs=epochs,
            device=device,
            save_path=model_path,
        )
        logging.info("Done Training Video Model...")

    if args.test:
        logging.info("Loading testing set..")
        non_flicker_test = [os.path.join(non_flicker_path, f)
                            for f in non_flicker_test]
        flicker1_test = [os.path.join(flicker1_path, f)
                         for f in flicker_test if f in os.listdir(flicker1_path)]
        flicker2_test = [os.path.join(flicker2_path, f)
                         for f in flicker_test if f in os.listdir(flicker2_path)]
        flicker3_test = [os.path.join(flicker3_path, f)
                         for f in flicker_test if f in os.listdir(flicker3_path)]
        flicker4_test = [os.path.join(flicker4_path, f)
                         for f in flicker_test if f in os.listdir(flicker4_path)]
        non_flicker_test = VideoDataSet.split_datasets(
            non_flicker_test+flicker1_test+flicker2_test+flicker3_test+flicker4_test, labels=labels, class_size=1, max_workers=max_workers, undersample=0)#+flicker4_test


        ds_test = MultiStreamer(
            non_flicker_test,
            batch_size=batch_size,
            binary=output_dim < 3
        )
        logging.info("Done loading testing set")

        logging.info("Starting Evaluation")
        testing(
            ds_test,
            model,
            objective=objective,
            device=device,
            classes=output_dim,
            save_path=model_path
        )
        logging.info("Done Evaluation")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    """
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 training.py --train
    https://stackoverflow.com/questions/2763006/make-the-current-git-branch-a-master-branch
    """
    main()
