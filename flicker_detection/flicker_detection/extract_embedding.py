import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from preprocessing.embedding.backbone import Backbone
from tensorflow.keras.applications import resnet, mobilenet, vgg16, InceptionResNetV2, InceptionV3
from mypyfunc.logger import init_logger

data_base_dir = "data"
os.makedirs(data_base_dir, exist_ok=True)


def _embed(
    video_data_dir: str,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor = Backbone()
    # just change extractor to try different
    feature_extractor.adaptive_extractor(mobilenet.MobileNet)

    for path in tqdm.tqdm(os.listdir(video_data_dir)):
        if os.path.exists(os.path.join(output_dir, "{}.npy".format(path))):
            continue

        vidcap = cv2.VideoCapture(os.path.join(video_data_dir, path))
        success, image = vidcap.read()

        embeddings = ()
        while success:
            embeddings = embeddings + tuple(feature_extractor.get_embedding(cv2.resize(
                image, (200, 200)), batched=False).flatten())
            success, image = vidcap.read()

        embeddings = np.array(embeddings)

        np.save(os.path.join(output_dir, path), embeddings)


def main():
    logging.info("[Embedding] Start ...")
    _embed(
        os.path.join(data_base_dir, "flicker-detection"),
        os.path.join(data_base_dir, "embedding")
    )
    logging.info("[Embedding] done.")


if __name__ == "__main__":
    main()
