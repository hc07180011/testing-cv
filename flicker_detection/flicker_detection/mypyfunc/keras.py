import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc


logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)


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
        return 2 * ((precision * recall) /
                    (precision + recall + tf.keras.backend.epsilon()))

    def auc(self, y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        tf.keras.backend.get_session().run(tf.local_variables_initializer())
        return auc

    def specificity(self, y_true, y_pred):
        """
        param:
        y_pred - Predicted labels
        y_true - True labels 
        Returns:
        Specificity score
        """
        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred
        fp = tf.keras.backend.sum(neg_y_true * y_pred)
        tn = tf.keras.backend.sum(neg_y_true * neg_y_pred)
        specificity = tn / (tn + fp + tf.keras.backend.epsilon())
        return specificity


_my_metrics = MyMetrics()


class Model:

    def __init__(
        self,
        model: tf.keras.models.Sequential,
        loss: str,
        optimizer: tf.keras.optimizers,
        metrics: list = list((
            _my_metrics.f1,
            _my_metrics.auc,
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            _my_metrics.specificity,
            tf.keras.metrics.SpecificityAtSensitivity(0.5),
            tf.keras.metrics.SensitivityAtSpecificity(0.5),
            'accuracy')),
        summary=True
    ) -> None:
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

    def plot_history(self, key: str, title=None) -> None:
        plt.figure(figsize=(16, 4), dpi=200)
        plt.plot(self.history.history["{}".format(key)])
        plt.plot(self.history.history["val_{}".format(key)])
        plt.legend(["{}".format(key), "val_{}".format(key)])
        plt.xlabel("# Epochs")
        plt.ylabel("{}".format(key))
        if title:
            plt.title("{}".format(title))
        plt.savefig("{}.png".format(key))
        plt.close()


class InferenceModel:

    def __init__(
        self,
        model_path: str,
        custom_objects: dict = dict({
            'f1': _my_metrics.f1,
            'auc': _my_metrics.auc,
            'precision': tf.keras.metrics.Precision(),
            'recall': tf.keras.metrics.Recall(),
            'specificity': _my_metrics.specificity,
            'spec_at_sen': tf.keras.metrics.SpecificityAtSensitivity(0.5),
            'sen_at_spec': tf.keras.metrics.SensitivityAtSpecificity(0.5), })
    ) -> None:
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )

    def predict(self, X_test: np.array) -> np.array:
        y_pred = self.model.predict(X_test)
        return y_pred.flatten()

    """
    TO DO if needed plot 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    """

    def evaluate(self, y_true: np.array, y_pred: np.array) -> None:
        threshold_range = np.arange(0.1, 1.0, 0.001)

        f1_scores = list()
        for lambda_ in threshold_range:
            f1_scores.append(f1_score(y_true, (y_pred > lambda_).astype(int)))

        logging.info("Max f1: {:.4f}, at thres = {:.4f}".format(
            np.max(f1_scores), threshold_range[np.argmax(f1_scores)]
        ))

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
        plt.savefig("roc_curve.png")
        plt.close()

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
        plt.savefig("pc_curve.png")

        print(confusion_matrix(
            y_true,
            (y_pred > threshold_range[np.argmax(f1_scores)]).astype(int)
        ))
