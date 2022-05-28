import os
import json
import random
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Callable
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from mypyfunc.transformers import TransformerEncoder, PositionalEmbedding

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)


class Model:
    """
    callbacks:
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    """

    def __init__(
        self,
        chunked_list, label_path, mapping_path, data_dir,
        summary: bool = True,
        overview: str = 'preprocessing/embedding/models/flicker_detection.txt'
    ) -> None:
        self.chunked_list = chunked_list
        self.label_path = label_path
        self.mapping_path = mapping_path
        self.data_dir = data_dir
        self.model_path = None
        self.summary = summary
        self.overview = overview
        self.history = None
        self.figures = None
        self.model = None
        self.metrics = None

    def compile(
        self,
        model: tf.keras.models.Sequential,
        loss: str,
        optimizer: tf.keras.optimizers,
        metrics: tuple,
    ) -> None:
        self.model = model
        self.metrics = [metric.__name__ if not isinstance(
            metric, tf.keras.metrics.AUC) else "auc" for metric in metrics]
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )
        if self.summary:
            with open(self.overview, 'w') as fh:
                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def LSTM(self, input_shape: Tuple = None) -> tf.keras.models.Sequential:
        buf = Sequential()
        buf.add(LSTM(units=256, input_shape=(input_shape)))
        buf.add(Dense(units=128, activation="relu"))
        buf.add(Flatten())
        buf.add(Dense(units=1, activation="sigmoid"))
        return buf

    def BiLSTM(self, input_shape: Tuple) -> tf.keras.models.Sequential:
        buf = Sequential()
        buf.add(Bidirectional(LSTM(units=256),
                              input_shape=(input_shape)))
        buf.add(Dense(units=128, activation="relu"))
        buf.add(Flatten())
        buf.add(Dense(units=1, activation="sigmoid"))
        return buf

    def transformers(self,
                     input_shape: Tuple,
                     sequence_length: int = 20,
                     num_heads: int = 1
                     ) -> tf.keras.Model:

        embed_dim = input_shape[-1]  # 9216
        dense_dim = input_shape[-2]  # 4
        inputs = tf.keras.Input(shape=input_shape)
        x = PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = TransformerEncoder(embed_dim, dense_dim, num_heads,
                               name="transformer_layer")(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def train(
        self,
        X_train: np.array,
        y_train: np.array,
        epochs: int,
        validation_split: float,
        batch_size: int,
        model_path: str = "model0.h5",
        monitor: str = "val_f1",
        mode: str = "max"
    ) -> None:
        if self.model_path is None:
            self.model_path = model_path
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    model_path,
                    save_best_only=True,
                    monitor=monitor,
                    mode=mode
                )
            ]
        )
        self.model.save(model_path)

    def save_callback(self) -> None:  # FIX ME
        with open('history.json', 'w') as file_pi:
            json.dump(self.history if isinstance(self.history, dict)
                      else self.history.history, file_pi)

    def batch_train(self, epochs: int, metrics: tuple,
                    _oversampling: Callable, load_embeddings: Callable,
                    model_path: str = "h5_models/test.h5",
                    ) -> None:

        mirrored_strategy = tf.distribute.MirroredStrategy()

        for epoch in range(epochs):

            random.shuffle(self.chunked_list)
            for vid_chunk in self.chunked_list:

                X_train, y_train = _oversampling(*load_embeddings(
                    vid_chunk, self.label_path, self.mapping_path, self.data_dir))  # FIX ME x val y val

                with mirrored_strategy.scope():

                    if self.model is None and os.path.exists("{}".format(model_path)):
                        logging.info("{}".format(metrics))
                        self.model = tf.keras.models.load_model(
                            model_path,
                            custom_objects=dict(
                                map(lambda metric: (metric.__name__, metric), metrics))
                        )
                    elif self.model is None:
                        self.compile(
                            model=self.LSTM(X_train.shape[1:]),
                            loss="binary_crossentropy",
                            optimizer=Adam(learning_rate=1e-5),
                            # , recall, precision, specificity),
                            metrics=metrics,
                        )

                    train_metrics = self.model.train_on_batch(
                        X_train, y_train)
                    y_pred = self.model.predict(X_train)
                    logging.debug("YPRED SHAPE - {}".format(y_pred.shape))
                    val_metrics = self.model.evaluate(
                        X_train, y_train)  # FIX ME x val y val

                logging.info(
                    "EPOCH {}: loss - {:.3f}, f1 - {:.3f}, val_loss - {:.3f}, val_f1 - {:.3f}".format(epoch, *train_metrics, *val_metrics))

                if self.model.metrics and self.history is None:
                    self.metrics = self.model.metrics_names
                    self.history = {}
                    for metric in self.model.metrics_names:
                        self.history[metric] = []
                        self.history["val_{}".format(metric)] = []

                    self.figures = tuple(map(lambda i: plt.figure(
                        num=i, figsize=(16, 4), dpi=200), range(len(self.model.metrics_names))))

            for idx, metric in enumerate(self.model.metrics_names):
                self.history[metric].append(train_metrics[idx])
                self.history["val_{}".format(metric)].append(
                    val_metrics[idx])

            if not (epoch % 1):
                self.model.save("{}".format(model_path))
                del self.model
                self.model = None
                tf.compat.v1.reset_default_graph()
                tf.keras.backend.clear_session()


class InferenceModel:

    def __init__(
        self,
        model_path: str,
        custom_objects: dict,
        plots_folder: str = "plots/",
    ) -> None:
        self.plots_folder = plots_folder
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        self.metrics = self.model.metrics_names
        os.makedirs(self.plots_folder, exist_ok=True)

    def predict(self, X_test: np.array) -> np.array:
        y_pred = self.model.predict(X_test)
        return y_pred.flatten()

    def plot_callback(self) -> None:
        with open('history.json', 'r') as file_pi:
            history = json.load(file_pi)

        for idx, metric in enumerate(self.metrics):
            plt.figure(num=idx, figsize=(16, 4), dpi=200)
            plt.plot(history["{}".format(metric)])
            plt.plot(history["val_{}".format(metric)])
            plt.legend(["{}".format(metric), "val_{}".format(metric)])
            plt.xlabel("# Epochs")
            plt.ylabel("{}".format(metric))
            plt.title("{} LSTM, Chunked, Oversampling".format(metric))
            plt.savefig("{}.png".format(
                os.path.join(self.plots_folder, metric)))
            plt.close()

    def evaluate(self, y_true: np.array, y_pred: np.array, plots_folder="plots/") -> None:
        threshold_range = np.arange(0.1, 1.0, 0.001)

        f1_scores = list()
        for lambda_ in threshold_range:
            f1_scores.append(
                f1_score(y_true, (y_pred > lambda_).astype(int)))

        # plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot([0, 1], [0, 1], linestyle="dashed")
        plt.plot(fpr, tpr, marker="o")
        plt.plot([0, 0, 1], [0, 1, 1], linestyle="dashed", c="red")
        plt.legend([
            "No Skill",
            "ROC curve (area = {:.2f})".format(auc(fpr, tpr)),
            "Perfect"
        ])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig(os.path.join(plots_folder, "roc_curve.png"))
        plt.close()

        # plot PR Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        plt.plot([0, 1], [0, 0], linestyle="dashed")
        plt.plot(recall, precision, marker="o")
        plt.legend([
            "No Skill",
            "Model"
        ])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-recall Curve")
        plt.savefig(os.path.join(plots_folder, "pc_curve.png"))
        plt.close()

        logging.info("Max f1: {:.4f}, at thres = {:.4f}".format(
            np.max(f1_scores), threshold_range[np.argmax(f1_scores)]
        ))

        # plot Confusion Matrix
        # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
        cm = confusion_matrix(
            y_true,
            (y_pred > threshold_range[np.argmax(f1_scores)]).astype(int),
            labels=[1, 0]
        )
        fig = plt.figure(num=-1)
        ax = fig.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title("Max f1: {:.4f}, at thres = {:.4f}".format(
            np.max(f1_scores), threshold_range[np.argmax(f1_scores)]
        ))
        fig.savefig(os.path.join(plots_folder, "confusion_matrix.png"))
