import os
import cv2
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import resnet


class Facenet:

    def __init__(self) -> None:

        super().__init__()

        self.__target_shape = (200, 200)

        base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=self.__target_shape + (3,), include_top=False
        )

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        self.__embedding = Model(base_cnn.input, output, name="Embedding")

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        anchor_input = layers.Input(
            name="anchor", shape=self.__target_shape + (3,))
        positive_input = layers.Input(
            name="positive", shape=self.__target_shape + (3,))
        negative_input = layers.Input(
            name="negative", shape=self.__target_shape + (3,))

        distances = DistanceLayer()(
            self.__embedding(resnet.preprocess_input(anchor_input)),
            self.__embedding(resnet.preprocess_input(positive_input)),
            self.__embedding(resnet.preprocess_input(negative_input)),
        )

        siamese_network = Model(
            inputs=[anchor_input, positive_input,
                    negative_input], outputs=distances
        )

        self.__siamese_model = SiameseModel(siamese_network)

        model_base_dir = os.path.join("preprocessing", "embedding", "models")

        model_settings = json.load(
            open(os.path.join(model_base_dir, "model.json"), "r"))
        model_path = os.path.join(model_base_dir, model_settings["name"])

        if os.path.exists(model_path):
            self.__siamese_model.load_weights(model_path)
        else:
            raise NotImplementedError
            self.__siamese_model.compile(optimizer=optimizers.Adam(0.0001))
            checkpoint = ModelCheckpoint(
                model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            # self.__siamese_model.fit(train_dataset, epochs=10,
            #                          validation_data=val_dataset, callbacks=callbacks_list)

    def get_embedding(self, images: np.ndarray) -> np.ndarray:
        assert len(
            images.shape) == 4, "images should be an array of image with shape (width, height, 3)"
        resized_images = np.array([cv2.resize(image, dsize=self.__target_shape,
                                              interpolation=cv2.INTER_CUBIC) for image in images])
        image_tensor = tf.convert_to_tensor(resized_images, np.float32)
        return self.__embedding(resnet.preprocess_input(image_tensor)).numpy()


class DistanceLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(
            loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]
