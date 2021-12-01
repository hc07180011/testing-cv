import os
import cv2
import time
from argparse import ArgumentParser

import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.engine.sequential import Sequential

from preprocessing.embedding.facenet import Facenet


def get_chunks(embedding_arr: np.array, chunk_size: int = 30) -> np.array:
    embedding_chunks = list()
    for j in range(embedding_arr.shape[0] // chunk_size):
        embedding_chunks.append(embedding_arr[chunk_size * j: chunk_size * (j + 1)])

    return np.array(embedding_chunks)


def load_inference_model(model_path: str = os.path.join("core", "models", "model.h5")) -> Sequential:
    def _f1_m(y_true, y_pred):
        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    model = load_model(model_path, custom_objects={"f1_m": _f1_m})
    return model


def inference(embedding_chunk: np.array, model: Sequential, chunk_size: int = 30) -> np.array:
    X_test = embedding_chunk
    y_test = np.reshape(model.predict(X_test), (X_test.shape[0], ))
    return dict({
        "{}-{}".format(s * chunk_size, e * chunk_size): y_test[i]
        for i, (s, e) in enumerate(zip(range(0, y_test.shape[0]), range(1, y_test.shape[0] + 1)))
    })


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str,
        default=os.path.join("data", "flicker_detection", "0147.mp4"),
        help="the relative path to video"
    )
    parser.add_argument(
        "-t", "--threshold", type=float,
        default=0.5,
        help="the threshold of positive labels (for dev ONLY, should be tuned with the precision-recall curve)"
    )
    args = parser.parse_args()

    facenet = Facenet()
    inference_model = load_inference_model()

    if not os.path.exists(args.data_path):
        print("\"{}\" not exists!".format(args.data_path))
        exit(1)

    print("Testing: {}".format(args.data_path))

    start_ = time.perf_counter()

    # First, we use Facenet to do the embedding.
    embedding_arr = list()
    vidcap = cv2.VideoCapture(args.data_path)
    success, image = vidcap.read()
    embedding_arr = list()
    while success:
        embedding_arr.append(facenet.get_embedding(image, batched=False).flatten())
        success, image = vidcap.read()

    # Then, we divide the embeddings of all frames into chunks.
    embedding_chunk = get_chunks(np.array(embedding_arr))

    # Last, we are able to inference the flicker score (probability) of every chunk.
    results = inference(embedding_chunk, inference_model)

    print("===== Results =====")
    for key in results:
        print("During frame {}, score ~= {:.8f}".format(key, results[key]))
        print("with thres = {}, {}".format(args.threshold, "ok" if results[key] < args.threshold else "FLICKER!!!"))
        print("------------------------")

    print("takes: {} s".format(time.perf_counter() - start_))
