import os
import json
import cv2
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
from mypyfunc.logger import init_logger
from typing import Tuple
from preprocessing.embedding.backbone import BaseCNN, Serializer
from mypyfunc.keras_models import Model, InferenceModel
from mypyfunc.keras_eval import f1, recall, precision, specificity


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
        # if any(map(path.__contains__, ("0002_", "0003_", "0006_",
        #                                "0016_", "0044_", "0055_",
        #                                "0070_", "0108_", "0121_", "0169_"))):
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
        "0096.mp4", "0097.mp4", "0098.mp4",
        "0125.mp4", "0126.mp4", "0127.mp4",
        "0145.mp4", "0146.mp4", "0147.mp4",
        "0178.mp4", "0179.mp4", "0180.mp4"
    )
    pass_videos += tuple(vid[:4]+f"_{i}.mp4.npy"for i in range(10)
                         for vid in pass_videos)
    # logging.info("{}".format(pass_videos))
    raw_labels = json.load(open(label_path, "r"))
    encoding_filename_mapping = json.load(open(mapping_path, "r"))

    embedding_path_list = sorted([
        x for x in os.listdir(data_dir)
        if x.split(".tfrecords")[0] not in pass_videos
        and encoding_filename_mapping[x.replace(".npy", "")] in raw_labels
    ])

    embedding_list_test = (
        "0002.mp4.npy", "0003.mp4.npy", "0006.mp4.npy",
        "0016.mp4.npy", "0044.mp4.npy", "0055.mp4.npy",
        "0070.mp4.npy", "0108.mp4.npy", "0121.mp4.npy",
        "0169.mp4.npy"
    )
    embedding_list_val = tuple(
        file for file in embedding_path_list
        if any(map(file.__contains__, ("0002_", "0003_", "0006_",
                                       "0016_", "0044_", "0055_",
                                       "0070_", "0108_", "0121_", "0169_"))))
    embedding_list_train = tuple(
        file for file in embedding_path_list
        if file not in embedding_list_val and file not in embedding_list_test
    )
    pd.DataFrame({
        "train": embedding_list_train,
        "val": embedding_list_val,
        "test": embedding_list_test}).to_csv("{}.csv".format(cache_path))

    np.savez(cache_path, embedding_list_train,
             embedding_list_val, embedding_list_test)


def decode_fn(record_bytes, key) -> tf.Tensor:
    string = tf.io.parse_single_example(
        record_bytes,
        {key: tf.io.FixedLenFeature([], dtype=tf.string), }
    )
    return tf.io.parse_tensor(string[key], out_type=tf.float32)


def _get_chunk_array(input_arr: np.array, chunk_size: int) -> Tuple:
    chunks = np.array_split(
        input_arr,
        list(range(
            chunk_size,
            input_arr.shape[0] + 1,
            chunk_size
        ))
    )
    i_pad = np.zeros(chunks[0].shape)
    i_pad[:len(chunks[-1])] = chunks[-1]
    chunks[-1] = i_pad
    return tuple(map(tuple, chunks))


def load_embeddings(
    embedding_list_train: list,
    label_path: str,
    mapping_path: str,
    data_dir: str,
    batch_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    encoding_filename_mapping = json.load(open(mapping_path, "r"))
    raw_labels = json.load(open(label_path, "r"))
    X_train, y_train = (), ()
    for key in embedding_list_train:
        real_filename = encoding_filename_mapping[key.replace(
            ".tfrecords", "")]
        loaded = np.load("{}.npy".format(os.path.join(data_dir, key.replace(
            ".tfrecords", ""))))
        X_train += (*_get_chunk_array(loaded, batch_size),)
        # get flicker frame indexes
        flicker_idxs = np.array(raw_labels[real_filename]) - 1
        # buffer zeros array frame video embedding
        buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
        # set indexes in zeros array based on flicker frame indexes
        buf_label[flicker_idxs] = 1
        y_train += tuple(
            1 if sum(x) else 0
            for x in _get_chunk_array(buf_label, batch_size)
        )  # consider using tf reduce sum for multiclass
        # logging.info("Video - {}".format(key))
    return np.array(X_train), np.array(y_train)


def _oversampling(
    X_train: np.array,
    y_train: np.array,
) -> Tuple[np.array]:
    """
    batched alternative:
    https://imbalanced-learn.org/stable/references/generated/imblearn.keras.BalancedBatchGenerator.html
    """
    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=1)
    original_X_shape = X_train.shape
    X_train, y_train = sm.fit_resample(
        np.reshape(X_train, (-1, np.prod(original_X_shape[1:]))),
        y_train
    )
    X_train = np.reshape(X_train, (-1,) + original_X_shape[1:])
    return (X_train, y_train)


def training(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
    epochs: int = 1,
) -> Model:
    embedding_list_train = np.array(np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_0"])
    chunked_list = np.array_split(embedding_list_train, indices_or_sections=40)
    buf = Model(chunked_list, label_path, mapping_path, data_dir)
    buf.batch_train(epochs=epochs, metrics=(f1,), _oversampling=_oversampling,
                    load_embeddings=load_embeddings)
    buf.plot_history()
    return buf


def testing(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
    model_path: str,
) -> None:
    embedding_list_test = np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_1"]
    X_test, y_test = load_embeddings(
        embedding_list_test, label_path, mapping_path, data_dir)

    model = InferenceModel(
        model_path,
        custom_objects={
            'f1': f1,
            # "PositionalEmbedding": PositionalEmbedding,
            # "TransformerEncoder": TransformerEncoder
        })
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)


def main():
    """
    can give minor classes higher weight

    GPU tensorflow memory managment
    https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
     You can do this by creating a new `tf.data.Options()` object then setting `options.experimenta
    l_distribute.auto_shard_policy = AutoShardPolicy.DATA` before applying the options object to the dataset via `dataset.with_options(options)`.
    """
    data_base_dir = "data"
    os.makedirs(data_base_dir, exist_ok=True)
    cache_base_dir = ".cache"
    os.makedirs(cache_base_dir, exist_ok=True)

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
        os.path.join(data_base_dir, "augmented"),
        os.path.join(data_base_dir, "vgg16_emb")
    )
    logging.info("[Embedding] done.")

    logging.info("[Preprocessing] Start ...")
    preprocessing(
        os.path.join(data_base_dir, "label.json"),
        os.path.join(data_base_dir, "mapping_aug_data.json"),
        os.path.join(data_base_dir, "vgg16_emb"),  # or embedding
        os.path.join(cache_base_dir, "train_test")
    )
    logging.info("[Preprocessing] done.")
    if args.train:
        logging.info("[Training] Start ...")
        training(
            os.path.join(data_base_dir, "label.json"),
            os.path.join(data_base_dir, "mapping_aug_data.json"),
            os.path.join(data_base_dir, "vgg16_emb"),
            os.path.join(cache_base_dir, "train_test")
        )
        logging.info("[Training] done.")
    if args.test:
        logging.info("[Testing] start ...")
        testing(
            os.path.join(data_base_dir, "label.json"),
            os.path.join(data_base_dir, "mapping_aug_data.json"),
            os.path.join(data_base_dir, "vgg16_emb"),
            os.path.join(cache_base_dir, "train_test"),
            "h5_models/test.h5"
        )
        logging.info("[Testing] done.")


if __name__ == "__main__":
    main()
