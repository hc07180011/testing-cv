import os
import json
import cv2
import tqdm
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import resnet, mobilenet, vgg16, InceptionResNetV2, InceptionV3

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
    feature_extractor.adaptive_extractor(mobilenet.MobileNet, frequency=10)

    for path in tqdm.tqdm(os.listdir(video_data_dir)):
        if os.path.exists(os.path.join(output_dir, "{}.tfrecords".format(path))):
            continue

        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        success, image = vidcap.read()

        batched_image = np.zeros((batch_size+1,) + image.shape)
        batched_image[0] = image
        frame_count, n_batch = 1, 0
        while success:
            if frame_count == batch_size:
                serializer.parse_batch(feature_extractor.get_embedding(
                    batched_image, batched=True).flatten(), n_batch=n_batch, filename=os.path.join(output_dir, path))
                frame_count, n_batch = 0, n_batch+1
                batched_image = np.zeros((batch_size+1,) + image.shape)

            success, batched_image[frame_count+1] = vidcap.read()
            frame_count += int(success)

        serializer.write_to_tfr()
        serializer.done_writing()
        logging.info("Done extracting - {}".format(path))


def preprocessing(
    label_path: str,
    mapping_path: str,
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pass_videos = (
        "0096.mp4", "0097.mp4", "0098.mp4",
        "0125.mp4", "0126.mp4", "0127.mp4",
        "0145.mp4", "0146.mp4", "0147.mp4",
        "0178.mp4", "0179.mp4", "0180.mp4"
    )
    raw_labels = json.load(open(label_path, "r"))
    encoding_filename_mapping = json.load(open(mapping_path, "r"))
    # logging.info("{}".format(encoding_filename_mapping))
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
    # logging.info("{}".format(embedding_list_train))
    # logging.info("{}".format(embedding_list_test))
    # logging.info("{}".format(raw_labels))
    return np.asarray(embedding_list_train), np.asarray(embedding_list_test)


def main():
    data_base_dir = "data"
    os.makedirs(data_base_dir, exist_ok=True)
    cache_base_dir = ".cache"
    os.makedirs(cache_base_dir, exist_ok=True)

    tf.keras.utils.set_random_seed(12345)
    tf.config.experimental.enable_op_determinism()

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
        os.path.join(data_base_dir, "TFRecords"),  # or embedding

    )
    logging.info("[Preprocessing] done.")

    np.savez(os.path.join(cache_base_dir, "train_test"), emb_train, emb_test)


if __name__ == "__main__":
    main()
