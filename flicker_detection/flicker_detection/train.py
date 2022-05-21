import json
import os
import logging
import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple
from argparse import ArgumentParser
from mypyfunc.keras_models import Model, InferenceModel
from mypyfunc.logger import init_logger
from mypyfunc.keras_eval import f1, precision, recall, specificity, fbeta, negative_predictive_value, matthews_correlation_coefficient, equal_error_rate

data_base_dir = "data"
os.makedirs(data_base_dir, exist_ok=True)
cache_base_dir = ".cache"
os.makedirs(cache_base_dir, exist_ok=True)


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


def _preprocess(
    label_path: str,
    mapping_path: str,
    data_dir: str,
    cache_path: str
) -> Tuple[np.array]:
    """
    can consider reducing precision of np.float32 to np.float16 to reduce memory consumption

    abstract:
    https://towardsdatascience.com/overcoming-data-preprocessing-bottlenecks-with-tensorflow-data-service-nvidia-dali-and-other-d6321917f851
    cuda solution:
    https://stackoverflow.com/questions/60996756/how-do-i-assign-a-numpy-int64-to-a-torch-cuda-floattensor
    static memory allocation solution:
    https://pytorch.org/docs/stable/generated/torch.zeros.html
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
        if x.split(".npy")[0] not in pass_videos
        and encoding_filename_mapping[x.replace(".npy", "")] in raw_labels
    ])

    embedding_list_train, embedding_list_test, _, _ = train_test_split(
        embedding_path_list,
        tuple(range(len(embedding_path_list))),
        test_size=0.1,
        random_state=42
    )
    with open("used_videos.txt", "w") as f:
        for video in [*embedding_list_train, *embedding_list_test]:
            f.write("{}\n".format(video))

    chunk_size = 32  # batch sizes must be even number

    video_embeddings_list_train = ()
    video_labels_list_train = ()
    logging.debug(
        "taking training chunks, length = {}".format(len(embedding_list_train))
    )
    for path in tqdm.tqdm(embedding_list_train):
        real_filename = encoding_filename_mapping[path.replace(".npy", "")]

        buf_embedding = np.load(os.path.join(data_dir, path))

        batch = _get_chunk_array(buf_embedding, chunk_size)
        video_embeddings_list_train = video_embeddings_list_train + (*batch,)

        flicker_idxs = np.array(raw_labels[real_filename]) - 1  # this line
        buf_label = np.zeros(buf_embedding.shape[0]).astype(np.uint8)
        buf_label[flicker_idxs] = 1
        video_labels_list_train = video_labels_list_train + tuple(
            1 if sum(x) else 0
            for x in _get_chunk_array(buf_label, chunk_size)
        )

    video_embeddings_list_test = ()
    video_labels_list_test = ()
    logging.debug(
        "taking testing chunks, length = {}".format(len(embedding_list_test))
    )
    for path in tqdm.tqdm(embedding_list_test):
        real_filename = encoding_filename_mapping[path.replace(".npy", "")]

        buf_embedding = np.load(os.path.join(data_dir, path))

        video_embeddings_list_test = video_embeddings_list_test +\
            (*_get_chunk_array(buf_embedding, chunk_size),)

        flicker_idxs = np.array(raw_labels[real_filename]) - 1
        buf_label = np.zeros(buf_embedding.shape[0]).astype(np.uint8)
        buf_label[flicker_idxs] = 1
        video_labels_list_test = video_labels_list_test + tuple(
            1 if sum(x) else 0
            for x in _get_chunk_array(buf_label, chunk_size)
        )
    X_train = np.array(video_embeddings_list_train)
    X_test = np.array(video_embeddings_list_test)
    y_train = np.array(video_labels_list_train)
    y_test = np.array(video_labels_list_test)

    logging.debug("ok. got training: {}/{}, testing: {}/{}".format(
        X_train.shape, y_train.shape,
        X_test.shape, y_test.shape
    ))

    np.savez(cache_path, X_train, X_test, y_train, y_test)

    return (X_train, X_test, y_train, y_test)


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


def _train(X_train: np.array, y_train: np.array) -> Model:
    """
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    """
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Model(chunked_list=None, label_path=None,
                      mapping_path=None, data_dir=None)
        model.compile(
            model=model.LSTM(X_train.shape[1:]),
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            metrics=(
                # precision,
                # recall,
                f1,
                # tf.metrics.AUC(),
                # fbeta,
                # specificity,
                # negative_predictive_value,
                # matthews_correlation_coefficient,
                # equal_error_rate
            )
        )
        model.train(X_train, y_train, 1000, 0.1, 256)  # 1024 , 8192
    # for k in ("loss", "precision",
    #           "recall", "f1", "fbeta", "specificity",
    #           "negative_predictive_value",
    #           "matthews_correlation_coefficient", "equal_error_rate"):
    # model.plot_history()  # FIX ME

    return model


def _test(model_path: str, X_test: np.array, y_test: np.array) -> None:
    custom_objects = {
        # "precision": precision,
        # "recall": recall,
        "f1": f1,
        # "auroc": tf.metrics.AUC(),
        # "fbeta": fbeta,
        # "specificity": specificity,
        # "negative_predictive_value": negative_predictive_value,
        # "matthews_correlation_coefficient": matthews_correlation_coefficient,
        # "equal_error_rate": equal_error_rate
    }
    model = InferenceModel(model_path, custom_objects=custom_objects)
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)


def _main() -> None:
    tf.keras.utils.set_random_seed(12345)
    tf.config.experimental.enable_op_determinism()

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

    init_logger()

    logging.info("[Preprocessing] Start ...")
    X_train, X_test, y_train, y_test = _preprocess(
        os.path.join(data_base_dir, "label.json"),
        os.path.join(data_base_dir, "mapping_aug_data.json"),
        os.path.join(data_base_dir, "vgg16_emb"),
        os.path.join(cache_base_dir, "train_test")
    )
    logging.info("[Preprocessing] done.")

    if args.train:
        # logging.info("[Oversampling] Start ...")
        # X_train, y_train = _oversampling(
        #     X_train,
        #     y_train
        # )
        # logging.info("[Oversampling] done.")

        logging.info("[Training] Start ...")
        _ = _train(
            X_train,
            y_train
        )
        logging.info("[Training] done.")

    if args.test:
        logging.info("[Testing] Start ...")
        _test("model0.h5", X_test, y_test)
        logging.info("[Testing] done.")


if __name__ == "__main__":
    _main()
    # vimdiff ~/googlecv/train.py /home/henrychao/googlecv/train.py
    # https://github.com/3b1b/manim/issues/1213 opencv issue
