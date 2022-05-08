import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Dropout, GlobalMaxPooling1D
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
        summary: bool = True,
        plots_folder: str = "plots/"
    ) -> None:
        self.summary = summary
        self.plots_folder = plots_folder
        os.makedirs(self.plots_folder, exist_ok=True)

    def compile(
        self,
        model: tf.keras.models.Sequential,
        loss: str,
        optimizer: tf.keras.optimizers,
        metrics: tuple,
    ):
        """
        metrics=[
                # precision,
                # recall,
                f1,
                # auroc,
                # fbeta,
                # specificity,
                # negative_predictive_value,
                # matthews_correlation_coefficient,
                # equal_error_rate
            ]
        """
        self.model = model
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )
        if self.summary:
            with open('preprocessing/embedding/models/flicker_detection.txt', 'w') as fh:
                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def LSTM(self, input_shape: Tuple) -> tf.keras.models.Sequential:
        buf = Sequential()
        buf.add(LSTM(units=256, input_shape=(input_shape)))
        buf.add(Dense(units=128, activation="relu"))
        buf.add(Flatten())
        buf.add(Dense(units=1, activation="sigmoid"))
        return buf

    def BiLSTM(self, input_shape: Tuple) -> tf.keras.models.Sequential:
        buf = Sequential()
        buf.add(Bidirectional(LSTM(units=256, activation='relu'),
                              input_shape=(input_shape)))

        buf.add(Dense(units=128, activation="relu"))
        buf.add(Flatten())
        buf.add(Dense(units=1, activation="sigmoid"))
        return buf

    def transformers(self, input_shape: Tuple) -> tf.keras.Model:
        sequence_length = 20
        embed_dim = 9216
        dense_dim = 4
        num_heads = 1
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
        plt.savefig("{}.png".format(os.path.join(self.plots_folder, key)))
        plt.close()


class InferenceModel:

    def __init__(
        self,
        model_path: str,
        custom_objects: dict,
    ) -> None:
        """
        = {
                # "precision": precision,
                # "recall": recall,
                "f1": f1,
                # "auroc":auroc,
                # "fbeta": fbeta,
                # "specificity": specificity,
                # "negative_predictive_value": negative_predictive_value,
                # "matthews_correlation_coefficient": matthews_correlation_coefficient,
                # "equal_error_rate": equal_error_rate
            }
        """
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )

    def predict(self, X_test: np.array) -> np.array:
        y_pred = self.model.predict(X_test)
        return y_pred.flatten()

    def evaluate(self, y_true: np.array, y_pred: np.array, plots_folder="plots/") -> None:
        threshold_range = np.arange(0.1, 1.0, 0.001)

        f1_scores = list()
        for lambda_ in threshold_range:
            f1_scores.append(f1_score(y_true, (y_pred > lambda_).astype(int)))

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
        cm = confusion_matrix(
            y_true,
            (y_pred > threshold_range[np.argmax(f1_scores)]).astype(int)
        )
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title("Max f1: {:.4f}, at thres = {:.4f}".format(
            np.max(f1_scores), threshold_range[np.argmax(f1_scores)]
        ))
        fig.savefig(os.path.join(plots_folder, "confusion_matrix.png"))
