import os
import json
import cv2
import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121, mobilenet, vgg16, InceptionResNetV2, InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Dropout, GlobalMaxPooling1D
from mypyfunc.logger import init_logger
from typing import Tuple
from preprocessing.embedding.backbone import BaseCNN, Serializer
from mypyfunc.custom_models import Model, InferenceModel
from mypyfunc.custom_eval import f1, recall, precision, specificity


data_base_dir = "data"
os.makedirs(data_base_dir, exist_ok=True)


def _embed(
    video_data_dir: str,
    output_dir: str,
    batch_size: int = 32,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor, serializer = BaseCNN(), Serializer()
    # just change extractor to try different
    feature_extractor.extractor(mobilenet.MobileNet)

    for path in tqdm.tqdm(os.listdir(video_data_dir)):
        if os.path.exists(os.path.join(output_dir, "{}.tfrecords".format(path))):
            continue
        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        h, w = int(vidcap.get(4)), int(vidcap.get(3))
        b_frames = np.zeros((batch_size, h, w, 3))
        embedding, success, frame_count = (), True, 0
        while success:
            if frame_count == batch_size:
                embedding += (feature_extractor.get_embedding(
                    b_frames, batched=True).flatten(),)
                frame_count = 0
                b_frames = np.zeros((batch_size, h, w, 3))

            success, b_frames[frame_count] = vidcap.read()
            frame_count += int(success)

        embedding += (feature_extractor.get_embedding(
            b_frames, batched=True).flatten(),)

        logging.info("{} {}".format(
            path, tf.convert_to_tensor(embedding).shape))
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
            embeddings += (feature_extractor.get_embed_cpu(cv2.resize(
                image, (200, 200)), batched=False).flatten(),)
            success, image = vidcap.read()

        embeddings = np.array(embeddings)

        np.save(os.path.join(output_dir, path), embeddings)


def preprocessing(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    manual testing
    0000.mp4
    0002.mp4
    0003.mp4
    0005.mp4
    """
    if os.path.exists("/{}.npz".format(cache_path)):
        __cache__ = np.load("/{}.npz".format(cache_path), allow_pickle=True)
        return tuple(__cache__[k] for k in __cache__)

    pass_videos = (
        "0096.mp4", "0097.mp4", "0098.mp4",
        "0125.mp4", "0126.mp4", "0127.mp4",
        "0145.mp4", "0146.mp4", "0147.mp4",
        "0178.mp4", "0179.mp4", "0180.mp4"
    )
    raw_labels = json.load(open(label_path, "r"))
    encoding_filename_mapping = json.load(open(mapping_path, "r"))

    embedding_path_list = sorted([
        x for x in os.listdir(data_dir)
        if x.split(".tfrecords")[0] not in pass_videos
        and encoding_filename_mapping[x.replace(".tfrecords", "")] in raw_labels
    ])

    embedding_list_train, embedding_list_test, _, _ = train_test_split(
        embedding_path_list,
        tuple(range(len(embedding_path_list))),
        test_size=0.1,
        random_state=42
    )

    np.savez(cache_path, embedding_list_train, embedding_list_test)


def decode_fn(record_bytes, key) -> tf.Tensor:
    string = tf.io.parse_single_example(
        record_bytes,
        {key: tf.io.FixedLenFeature([], dtype=tf.string), }
    )
    return tf.io.parse_tensor(string[key], out_type=tf.float32)


def _get_chunk_array(input_arr: np.array, chunk_size: int) -> Tuple:
    usable_vec = input_arr[:(
        np.floor(len(input_arr)/chunk_size)*chunk_size).astype(int)]
    i_pad = np.concatenate((usable_vec, np.array(
        [np.zeros(input_arr[-1].shape)]*(chunk_size-len(usable_vec) % chunk_size))))
    asymmetric_chunks = np.split(
        i_pad,
        list(range(
            chunk_size,
            input_arr.shape[0] + 1,
            chunk_size
        ))
    )
    return tuple(map(tuple, asymmetric_chunks))


def load_embeddings(
    embedding_list_train: list,
    label_path: str,
    mapping_path: str,
    data_dir: str,
    batch_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    encoding_filename_mapping = json.load(open(mapping_path, "r"))
    raw_labels = json.load(open(label_path, "r"))

    X_train, y_train = (), ()
    for key in embedding_list_train:
        real_filename = encoding_filename_mapping[key.replace(
            ".tfrecords", "")]

        loaded = np.load("{}.npy".format(os.path.join(data_dir, key.replace(
            ".tfrecords", ""))))
        logging.info("Video - {}".format(key))
        logging.info("Embedding Shape - {}".format(loaded.shape))
        X_train += (*_get_chunk_array(loaded, batch_size),)

        flicker_idxs = np.array(raw_labels[real_filename]) - 1
        buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
        buf_label[flicker_idxs] = 1
        y_train += tuple(
            1 if sum(x) else 0
            for x in _get_chunk_array(buf_label, batch_size)
        )
    return np.array(X_train), np.array(y_train)


def _oversampling(
    X_train: np.array,
    y_train: np.array,
) -> Tuple[np.array]:
    """
    batched alternative:
    https://imbalanced-learn.org/stable/references/generated/imblearn.keras.BalancedBatchGenerator.html
    """
    sm = SMOTE(random_state=42)
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
    epochs: int = 1000,
    model: tf.keras.Model = None,
) -> Model:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    embedding_list_train = np.array(np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_0"])
    chunked_list = np.array_split(embedding_list_train, indices_or_sections=2)

    for vid_chunk in chunked_list:
        X_train, y_train = _oversampling(*load_embeddings(
            embedding_list_train, label_path, mapping_path, data_dir))
        with mirrored_strategy.scope():
            if model is None:
                logging.info("Input shape {}".format(X_train.shape[1:]))
                model = Model()
                model.compile(
                    model=model.LSTM(X_train.shape[1:]),
                    loss="binary_crossentropy",
                    optimizer=Adam(learning_rate=1e-5),
                    metrics=(f1),  # , recall, precision, specificity),
                )

            model.train(X_train, y_train, epochs=epochs,
                        validation_split=0.1, batch_size=4096, model_path="online.h5")
            for k in ("loss", "f1"):
                model.plot_history(
                    k, title="{} - LSTM, Chunk, Oversampling".format(k))

    return model


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

    model = InferenceModel(model_path, custom_objects={'f1': f1})
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)


def main():
    """
    can give minor classes higher weight
    """
    data_base_dir = "data"
    os.makedirs(data_base_dir, exist_ok=True)
    cache_base_dir = ".cache"
    os.makedirs(cache_base_dir, exist_ok=True)

    tf.keras.utils.set_random_seed(12345)
    tf.config.experimental.enable_op_determinism()

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
    _embed(
        os.path.join(data_base_dir, "flicker-detection"),
        os.path.join(data_base_dir, "TFRecords")
    )
    logging.info("[Embedding] done.")

    logging.info("[Preprocessing] Start ...")
    preprocessing(
        os.path.join(data_base_dir, "label.json"),
        os.path.join(data_base_dir, "mapping_aug_data.json"),
        os.path.join(data_base_dir, "TFRecords"),  # or embedding
        os.path.join(cache_base_dir, "train_test")
    )
    logging.info("[Preprocessing] done.")
    if args.train:
        logging.info("[Training] Start ...")
        training(
            os.path.join(data_base_dir, "label.json"),
            os.path.join(data_base_dir, "mapping_aug_data.json"),
            os.path.join(data_base_dir, "embedding_original"),
            os.path.join(cache_base_dir, "train_test")
        )
        logging.info("[Training] done.")
    if args.test:
        logging.info("[Testing] start ...")
        testing(
            os.path.join(data_base_dir, "label.json"),
            os.path.join(data_base_dir, "mapping_aug_data.json"),
            os.path.join(data_base_dir, "embedding_original"),
            os.path.join(cache_base_dir, "train_test"),
            "online.h5"
        )
        logging.info("[Testing] done.")


if __name__ == "__main__":
    main()
