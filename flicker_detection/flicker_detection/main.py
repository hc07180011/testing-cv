import os
import time
import logging
from argparse import ArgumentParser

import cv2
import numpy as np

from mypyfunc.logger import init_logger
from preprocessing.embedding.facenet import Facenet
from mypyfunc.keras import InferenceModel


def _main() -> None:

    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str,
        default="data/flicker-detection/0003.mp4",
        help="Path to the video"
    )
    parser.add_argument(
        "-i", "--img_dir", type=str,
        default=".img",
        help="directory of experiments .png"
    )
    parser.add_argument(
        "-dc", "--disable_cache", action="store_true",
        default=False,
        help="disable caching function"
    )
    parser.add_argument(
        "-c", "--cache_dir", type=str,
        default=".cache",
        help="directory of caches"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        default=False,
        help="print debugging logs"
    )
    args = parser.parse_args()

    init_logger()
    logging.info("Program starts")

    def _get_chunk_array(input_arr: np.array, chunk_size: int) -> list:
        asymmetric_chunks = np.split(
            input_arr,
            list(range(
                chunk_size,
                input_arr.shape[0] + 1,
                chunk_size
            ))
        )
        # TODO: should we take the last chunk?
        return np.array(asymmetric_chunks[:-1])

    logging.info("Reading video ...")
    vidcap = cv2.VideoCapture(args.data_path)
    success, image = vidcap.read()
    raw_images = list()
    while success:
        raw_images.append(cv2.resize(image, (200, 200)))
        success, image = vidcap.read()
    logging.info("done.")

    logging.info("Getting embedding ...")
    facenet = Facenet()
    embeddings = facenet.get_embedding(np.array(raw_images))
    logging.info("done.")

    logging.info("Getting chunks ...")
    chunk_size = 30
    X_test = _get_chunk_array(embeddings, chunk_size)
    logging.info("done.")

    logging.info("Model inference ...")
    model = InferenceModel("model.h5")
    y_pred = model.predict(X_test.reshape(-1, chunk_size, np.prod(X_test.shape[2:])))
    logging.info("done.")

    thres = 0.8
    for i in range(len(y_pred)):
        if y_pred[i] > 0.8:
            logging.warning("{:.0f}-{:.0f} FLICKER".format(i * chunk_size, (i + 1) * chunk_size - 1))
        else:
            logging.debug("{:.0f}-{:.0f} ok".format(i * chunk_size, (i + 1) * chunk_size - 1))


    logging.info("Program finished.")


if __name__ == "__main__":
    _main()
