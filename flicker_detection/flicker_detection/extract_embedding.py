import os
import cv2
import tqdm
import logging
import numpy as np
import tensorflow as tf
from preprocessing.embedding.backbone import BaseCNN, Serializer
from tensorflow.keras.applications import resnet, mobilenet, vgg16, InceptionResNetV2, InceptionV3
from mypyfunc.logger import init_logger

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
        if os.path.exists(os.path.join(output_dir, "/{}.npy".format(path))):
            continue

        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        success, image = vidcap.read()

        batched_image = np.zeros((batch_size+1,) + image.shape)
        batched_image[0] = image
        frame_count, n_batch = 1, 0
        while success:
            if frame_count == batch_size:
                serializer.parse_batch(feature_extractor.get_embedding(
                    batched_image, batched=True).flatten(), n_batch=n_batch, filename=path)
                frame_count, n_batch = 0, n_batch+1
                batched_image = np.zeros((batch_size+1,) + image.shape)

            success, batched_image[frame_count+1] = vidcap.read()
            frame_count += int(success)

        serializer.write_to_tfr()
        serializer.done_writing()
        logging.info("done extracting - {}".format(path))


def main():
    tf.keras.utils.set_random_seed(12345)
    tf.config.experimental.enable_op_determinism()
    init_logger()

    logging.info("[Embedding] Start ...")
    _embed(
        os.path.join(data_base_dir, "flicker-detection"),
        os.path.join(data_base_dir, "embedding")
    )
    logging.info("[Embedding] done.")


if __name__ == "__main__":
    main()
