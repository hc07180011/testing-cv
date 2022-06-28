from cProfile import label
import os
import json
import cv2
from cv2 import threshold
from sklearn.metrics import f1_score
import tqdm
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from imblearn.over_sampling import SMOTE
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121, mobilenet, vgg16, InceptionResNetV2, InceptionV3
from tensorflow.keras import Model
from tensorflow_addons.metrics import F1Score
from mypyfunc.logger import init_logger
from typing import Tuple
from preprocessing.embedding.backbone import BaseCNN, Serializer
from mypyfunc.keras_models import Model, InferenceModel
from mypyfunc.keras_eval import Metrics
from mypyfunc.torch_data_loader import Streamer


data_base_dir = "data"
os.makedirs(data_base_dir, exist_ok=True)


def serialize_embed(
    video_data_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor, serializer = BaseCNN(), Serializer()
    # just change extractor to try different
    feature_extractor.extractor(mobilenet.MobileNet)

    for path in tqdm.tqdm(os.listdir(video_data_dir)):
        if os.path.exists(os.path.join(output_dir, "{}.tfrecords".format(path))):
            continue
        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        success, image = vidcap.read()
        embedding = ()
        while success:
            embedding += (feature_extractor.get_embedding(
                image, batched=True).flatten(),)
            success, image = vidcap.read()

        serializer.parse_batch(tf.convert_to_tensor(embedding),
                               filename=path)
        serializer.write_to_tfr()
        serializer.done_writing()
        serializer.get_schema()
        logging.info("Done extracting - {}".format(path))


def np_embed(
    video_data_dir: str,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor = BaseCNN()
    feature_extractor.extractor(vgg16.VGG16)  # mobilenet.MobileNet
    for path in tqdm.tqdm(os.listdir(video_data_dir)):
        if os.path.exists(os.path.join(output_dir, "{}.npy".format(path))):
            continue

        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        success, image = vidcap.read()

        embeddings = ()
        while success:
            embeddings += (
                feature_extractor.get_embed_cpu(
                    cv2.resize(image, (200, 200)), batched=False
                ).flatten(),)
            success, image = vidcap.read()

        embeddings = np.array(embeddings)

        np.save(os.path.join(output_dir, path), embeddings)


def preprocessing(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
) -> Tuple[np.ndarray, np.ndarray]:

    if os.path.exists("/{}.npz".format(cache_path)):
        __cache__ = np.load("/{}.npz".format(cache_path), allow_pickle=True)
        return tuple(__cache__[k] for k in __cache__)

    pass_videos = (
        "0126.mp4", "0127.mp4",
        "0178.mp4", "0180.mp4"
    )
    pass_videos += tuple(vid[:4]+f"_{i}.mp4.npy"for i in range(10)
                         for vid in pass_videos)
    raw_labels = json.load(open(label_path, "r"))
    encoding_filename_mapping = json.load(open(mapping_path, "r"))

    embedding_path_list = sorted([
        x for x in os.listdir(data_dir)
        if x.split(".tfrecords")[0] not in pass_videos
        and encoding_filename_mapping[x.replace(".npy", "")] in raw_labels
    ])

    # embedding_list_train, embedding_list_test, _, _ = train_test_split(
    #     tuple(file for file in embedding_path_list if "_" not in file),
    #     # dummy buffer just to split embedding_path_list
    #     tuple(
    #         range(len(tuple(file for file in embedding_path_list if "_" not in file)))),
    #     test_size=0.1,
    #     random_state=42
    # )
    embedding_list_test = (
        "0002.mp4.npy", "0003.mp4.npy", "0006.mp4.npy",
        "0016.mp4.npy", "0044.mp4.npy", "0055.mp4.npy",
        "0070.mp4.npy", "0108.mp4.npy", "0121.mp4.npy",
        "0169.mp4.npy", "0145.mp4.npy", "0179.mp4.npy",
        "0098.mp4.npy", "0147.mp4.npy", "0125.mp4.npy"
    )

    embedding_list_train = tuple(
        set(embedding_path_list) - set(embedding_list_test)
    )
    embedding_list_val = embedding_list_test

    length = max([len(embedding_list_test), len(
        embedding_list_val), len(embedding_list_train)])
    pd.DataFrame({
        "train": tuple(embedding_list_train) + ("",) * (length - len(embedding_list_train)),
        "val": tuple(embedding_list_val) + ("",) * (length - len(embedding_list_val)),
        "test": tuple(embedding_list_test) + ("",) * (length - len(embedding_list_test))
    }).to_csv("{}.csv".format(cache_path))

    np.savez(cache_path, embedding_list_train,
             embedding_list_val, embedding_list_test)


def decode_fn(record_bytes, key) -> tf.Tensor:
    string = tf.io.parse_single_example(
        record_bytes,
        {key: tf.io.FixedLenFeature([], dtype=tf.string), }
    )
    return tf.io.parse_tensor(string[key], out_type=tf.float32)


def training(
    ds_train: Streamer,
    ds_val: Streamer,
    epochs: int = 1,
) -> None:

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    metrics = Metrics()
    # metrics = F1Score(num_classes=2,threshold=0.5)

    buf = Model()
    model = buf.LSTM((30, 18432))
    buf.compile(model, loss_fn, optimizer, (metrics.f1,))

    buf.batch_train(
        epochs=epochs,
        train_loader=ds_train,
        val_loader=ds_val,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics)

    buf.save_callback()


def testing(
    ds_test: Streamer,
    model_path: str,
) -> None:
    X_test, y_test = np.array([]), np.array([])
    for x, y in ds_test:
        X_test = np.concatenate((X_test, x))
        y_test = np.concatenate((y_test, y))

    metrics = Metrics()
    model = InferenceModel(
        model_path,
        custom_objects={
            'f1': metrics.f1,
            # "PositionalEmbedding": PositionalEmbedding,
            # "TransformerEncoder": TransformerEncoder
        })
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)
    model.plot_callback()


def main():
    """
    can give minor classes higher weight

    GPU tensorflow memory managment
    https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
     You can do this by creating a new `tf.data.Options()` object then setting `options.experimenta
    l_distribute.auto_shard_policy = AutoShardPolicy.DATA` before applying the options object to the dataset via `dataset.with_options(options)`.
    """

    videos_path = "data/augmented"
    label_path = "data/new_label.json"
    mapping_path = "data/mapping_aug_data.json"
    data_path = "data/vgg16_emb"
    cache_path = ".cache/train_test"

    # tf.keras.utils.set_random_seed(12345)
    # tf.config.experimental.enable_op_determinism()
    # config = ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # tf.compat.v1.keras.backend.set_session(session)

    init_logger()

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

    logging.info("[Embedding] Start ...")
    np_embed(
        videos_path,
        data_path
    )
    logging.info("[Embedding] done.")

    logging.info("[Preprocessing] Start ...")
    preprocessing(
        label_path,
        mapping_path,
        data_path,
        cache_path,
    )
    logging.info("[Preprocessing] done.")

    __cache__ = np.load("{}.npz".format(cache_path), allow_pickle=True)

    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    ds_train = Streamer(embedding_list_train, label_path,
                        mapping_path, data_path, mem_split=20, batch_size=256, oversample=True, keras=True)
    ds_val = Streamer(embedding_list_test, label_path,
                      mapping_path, data_path, mem_split=1, batch_size=256, oversample=True, keras=True)
    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_path, mem_split=1, batch_size=256, oversample=True, keras=True)

    if args.train:
        logging.info("[Training] Start ...")
        training(
            ds_train,
            ds_val,
        )
        logging.info("[Training] done.")
    if args.test:
        logging.info("[Testing] start ...")
        testing(
            ds_test,
            "h5_models/test.h5"
        )
        logging.info("[Testing] done.")


if __name__ == "__main__":
    main()
