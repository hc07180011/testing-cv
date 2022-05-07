import os
import json
import cv2
import tqdm
import logging
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121, mobilenet, vgg16, InceptionResNetV2, InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from mypyfunc.logger import init_logger
from typing import Tuple
from preprocessing.embedding.backbone import BaseCNN, Serializer


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
        if os.path.exists(os.path.join(output_dir, "/{}.tfrecords".format(path))):
            continue

        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        h, w, total_frames, frame_count, n_batch = int(vidcap.get(
            4)), int(vidcap.get(3)), int(vidcap.get(7)), 0, 0
        b_frames = np.zeros((batch_size, h, w, 3))
        embedding, success = (), True
        while success:
            if frame_count == batch_size:
                embedding += (feature_extractor.get_embedding(
                    b_frames, batched=True).flatten(),)
                frame_count = 0
                n_batch += 1
                b_frames = np.zeros((batch_size, h, w, 3))

            success, b_frames[frame_count] = vidcap.read()
            frame_count += int(success)

        embedding += (feature_extractor.get_embedding(
            b_frames, batched=True).flatten(),)

        serializer.parse_batch(embedding,
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
        if x.split(".npy")[0] not in pass_videos
        and encoding_filename_mapping[x.replace(".npy", "")] in raw_labels
    ])

    embedding_list_train, embedding_list_test, _, _ = train_test_split(
        embedding_path_list,
        tuple(range(len(embedding_path_list))),
        test_size=0.1,
        random_state=42
    )

    np.savez(cache_path, embedding_list_train, embedding_list_test)

    return np.asarray(embedding_list_train), np.asarray(embedding_list_test)


def decode_fn(record_bytes, key):
    string = tf.io.parse_single_example(
        record_bytes,
        {key: tf.io.FixedLenFeature([], dtype=tf.string), }
    )
    return tf.io.parse_tensor(string[key], out_type=tf.float32)

# give minor classes higher weight


def extract_testing(
    data_dir: str,
    cache_path: str,
):
    embedding_list_test = np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_1"]

    X_test = ()
    for key in embedding_list_test:
        ds = tf.data.TFRecordDataset(
            os.path.join(data_dir, "TFRecords/{}".format(key))
        ).map(
            lambda byte: decode_fn(byte, key)
        )
        tensor = tf.io.parse_example(ds.get_single_element()[
                                     key], out_type=tf.float32)
        X_test += (tensor,)
    return np.array(X_test)


def training(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str,
    batch_size: int,
    model: Model,
) -> Model:

    embedding_list_train = np.load("{}.npz".format(
        cache_path), allow_pickle=True)["arr_0"]
    encoding_filename_mapping = json.load(open(mapping_path, "r"))
    raw_labels = json.load(open(label_path, "r"))

    epochs = 1000
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))

        for key in random.shuffle(embedding_list_train):
            real_filename = encoding_filename_mapping[key.replace(".npy", "")]
            X_train = tf.data.TFRecordDataset(data_dir)\
                .map(lambda byte: decode_fn(byte, key))

            flicker_idxs = np.array(raw_labels[real_filename]) - 1
            buf_label = np.zeros(X_train.shape[0], dtype=np.uint8)
            buf_label[flicker_idxs] = 1
            y_train = tf.data.Dataset.from_tensor_slices(buf_label)
            for bX_train, by_train in zip(X_train.padded_batch(batch_size), y_train.padded_batch(batch_size)):
                model.train_on_batch(bX_train, by_train)
    return model


def main():
    data_base_dir = "data"
    os.makedirs(data_base_dir, exist_ok=True)
    cache_base_dir = ".cache"
    os.makedirs(cache_base_dir, exist_ok=True)

    tf.keras.utils.set_random_seed(12345)
    tf.config.experimental.enable_op_determinism()

    configproto = tf.compat.v1.ConfigProto()
    configproto.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configproto)
    tf.compat.v1.keras.backend.set_session(sess)

    init_logger()

    logging.info("[Embedding] Start ...")
    _embed(
        os.path.join(data_base_dir, "flicker-detection"),
        os.path.join(data_base_dir, "TFRecords")
    )
    logging.info("[Embedding] done.")

    logging.info("[Preprocessing] Start ...")
    emb_train, emb_test = preprocessing(
        os.path.join(data_base_dir, "label.json"),
        os.path.join(data_base_dir, "mapping_aug_data.json"),
        os.path.join(data_base_dir, "embedding"),  # or embedding
        os.path.join(cache_base_dir, "train_test")
    )
    logging.info("[Preprocessing] done.")

    extract_testing(
        os.path.join(data_base_dir, "TFRecords"),
        os.path.join(cache_base_dir, "train_test")
    )

    training(
        os.path.join(data_base_dir, "label.json"),
        os.path.join(data_base_dir, "mapping_aug_data.json"),
        os.path.join(data_base_dir, "TFRecords"),
        os.path.join(cache_base_dir, "train_test")
    )


if __name__ == "__main__":
    main()
