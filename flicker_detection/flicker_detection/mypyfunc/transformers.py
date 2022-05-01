import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu),
             layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class Transformers:
    def __init__(
            self,
            X_train: np.array,
            loss: str,
            optimizer: tf.keras.optimizers,
            strategy: tf.distribute.Strategy,
            metrics: tuple = (
            "accuracy", f1,
            # tf.keras.metrics.AUC()
            ),
    ) -> None:
        self.strategy = strategy
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        with self.strategy.scope():
            self.model = self.transformers(X_train)
            self.compile()

    def compile(self) -> Model:
        with self.strategy.scope():
            self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.metrics
            )
            return self.model

    def summary(self) -> None:
        print(self.model.summary())

    def transformers(self, X_train: np.array) -> Model:
        sequence_length = 20
        embed_dim = 9216
        dense_dim = 4
        num_heads = 1
        inputs = tf.keras.Input(shape=X_train.shape[1:])
        x = PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = TransformerEncoder(embed_dim, dense_dim, num_heads,
                               name="transformer_layer")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

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
