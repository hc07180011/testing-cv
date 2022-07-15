import os
import gc
import time
import logging
from argparse import ArgumentParser

import cv2
import numpy as np

from mypyfunc.logger import init_logger
from preprocessing.embedding.facenet import Facenet
from mypyfunc.keras_models import InferenceModel


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
        "-c", "--cache_dir", type=str,
        default=".cache",
        help="directory of caches"
    )
    parser.add_argument(
        "-o", "--output", action="store_true",
        default=False,
        help="output video chunks"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        default=False,
        help="print debugging logs"
    )
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)

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
        raw_images.append(image)
        success, image = vidcap.read()
    logging.info("done.")

    logging.info("Getting embedding ...")
    facenet = Facenet()
    embeddings = list()
    for image in raw_images:
        embeddings.append(
            facenet.get_embedding(np.array(
                cv2.resize(image, (200, 200))
            ), batched=False)[0]
        )
    if not args.output:
        del raw_images
        gc.collect()
    embeddings = np.array(embeddings)
    logging.info("done.")

    logging.info("Getting chunks ...")
    chunk_size = 30
    X_test = _get_chunk_array(embeddings, chunk_size)
    logging.info("done.")

    logging.info("Model inference ...")
    model = InferenceModel("model.h5")
    y_pred = model.predict(X_test.reshape(-1, chunk_size, np.prod(X_test.shape[2:])))
    logging.info("done.")

    thres = 0.95
    logging.info("Prediction with threshold =\t{}".format(thres))
    for i in range(len(y_pred)):
        left = i * chunk_size
        right = (i + 1) * chunk_size

        if y_pred[i] > thres:
            logging.warning("{:.0f}-{:.0f}\tFLICKER\t(score={:.2f})".format(left, right - 1, y_pred[i]))
        else:
            logging.debug("{:.0f}-{:.0f}\tok\t(score={:.2f})".format(left, right - 1, y_pred[i]))

        if args.output:
            output_shape = raw_images[0].shape[:-1]
            video_writer = cv2.VideoWriter(
                os.path.join(
                    args.cache_dir,
                    args.data_path.replace("/", "_").replace(".mp4", "") + "_{:04d}({:.2f}).mp4".format(i, y_pred[i])
                ),
                cv2.VideoWriter_fourcc(*"mp4v"),
                50.0,
                output_shape
            )
            for j in range(left, right):
                video_writer.write(cv2.resize(raw_images[j], output_shape))
            video_writer.release()

    logging.info("Program finished.")


if __name__ == "__main__":
    _main()
