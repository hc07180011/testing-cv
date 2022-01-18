import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MyMetrics:

    def __init__(self) -> None:
        pass

    def precision(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(
            tf.keras.backend.round(
                tf.keras.backend.clip(y_true * y_pred, 0, 1)
            )
        )
        predicted_positives = tf.keras.backend.sum(
            tf.keras.backend.round(
                tf.keras.backend.clip(y_pred, 0, 1)
            )
        )
        precision = true_positives / \
                    (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def recall(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(
            tf.keras.backend.round(
                tf.keras.backend.clip(y_true * y_pred, 0, 1)
            )
        )
        possible_positives = tf.keras.backend.sum(
            tf.keras.backend.round(
                tf.keras.backend.clip(y_true, 0, 1)
            )
        )
        recall = true_positives / \
                    (possible_positives + tf.keras.backend.epsilon())
        return recall

    def f1(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * ((precision * recall) / \
                    (precision + recall + tf.keras.backend.epsilon()))


_my_metrics = MyMetrics()


class Model:

    def __init__(
        self,
        model: tf.keras.models.Sequential,
        loss: str,
        optimizer: tf.keras.optimizers,
        metrics: list = list((
            "accuracy",
            _my_metrics.f1,
            tf.keras.metrics.AUC()
        )),
        summary=True
    ) -> None:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.model = model
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )
        if summary:
            print(self.model.summary())

    def train(
        self,
        X_train: np.array,
        y_train: np.array,
        epochs: int,
        validation_split: float,
        batch_size: int,
        model_path: str = "model.h5",
        monitor: str = "val_f1",
        mode: str = "max"
    ) -> None:
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

    def plot_history(self, key: str) -> None:
        plt.plot(history.history["{}".format(key)])
        plt.plot(history.history["val_{}".format(key)])
        plt.legend(["{}".format(key), "val_{}".format(key)])
        plt.savefig("{}.png".format(key))
        plt.close()