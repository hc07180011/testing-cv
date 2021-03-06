from email.mime import base
import os
import cv2
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet, mobilenet
from tensorflow_addons.layers import AdaptiveMaxPooling3D, AdaptiveMaxPooling2D


class Facenet:
    """
    adaptive pooling sample:
    https://ideone.com/cJoN3x
    """

    def __init__(self) -> None:

        super().__init__()

        self.__target_shape = (200, 200)

        np.random.seed(0)

        base_cnn = self.adaptive_mobilenet()

        adaptive_1 = AdaptiveMaxPooling3D(
            output_size=base_cnn.output.shape[1:])(base_cnn.output)

        output = layers.Dense(256)(adaptive_1)

        self.__embedding = Model(
            base_cnn.input, output, name="Embedding")

        with open('preprocessing/embedding/models/embedding_summary.txt', 'w') as fh:
            self.__embedding.summary(print_fn=lambda x: fh.write(x + '\n'))

        for layer in base_cnn.layers[:-23]:
            layer.trainable = False

        anchor_input = layers.Input(
            name="anchor", shape=self.__target_shape + (3,)
        )

        positive_input = layers.Input(
            name="positive", shape=self.__target_shape + (3,)
        )

        negative_input = layers.Input(
            name="negative", shape=self.__target_shape + (3,)
        )

        distances = DistanceLayer()(
            self.__embedding(resnet.preprocess_input(anchor_input)),
            self.__embedding(resnet.preprocess_input(positive_input)),
            self.__embedding(resnet.preprocess_input(negative_input)),
        )

        siamese_network = Model(
            inputs=[
                anchor_input,
                positive_input,
                negative_input,
            ],
            outputs=distances
        )

        with open('preprocessing/embedding/models/siamese_summary.txt', 'w') as fh:
            siamese_network.summary(print_fn=lambda x: fh.write(x + '\n'))

        adaptive_0 = AdaptiveMaxPooling2D(
            output_size=siamese_network.output[0].shape[1:])(siamese_network.output)

        adaptive_siamese_network = Model(siamese_network.input, adaptive_0)

        self.__siamese_model = SiameseModel(adaptive_siamese_network)
        self.__siamese_model.built = True

        with open('preprocessing/embedding/models/siamese_adaptive_summary.txt', 'w') as fh:
            self.__siamese_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        model_base_dir = os.path.join("preprocessing", "embedding", "models")

        model_settings = json.load(
            open(os.path.join(model_base_dir, "model.json"), "r")
        )
        model_path = os.path.join(model_base_dir, model_settings["name"])

        if os.path.exists(model_path):
            self.__siamese_model.load_weights(model_path)
        else:
            raise NotImplementedError

    def get_embedding(self, images: np.ndarray, batched=True) -> np.ndarray:
        assert (not batched) or len(
            images.shape) == 4, "images should be an array of image with shape (width, height, 3)"
        if not batched:
            images = np.array([images, ])
        resized_images = np.array([cv2.resize(image, dsize=self.__target_shape,
                                              interpolation=cv2.INTER_CUBIC) for image in images])
        image_tensor = tf.convert_to_tensor(resized_images, np.float32)
        return self.__embedding(resnet.preprocess_input(image_tensor)).numpy()

    def adaptive_mobilenet(self):
        base_cnn = mobilenet.MobileNet(
            weights="imagenet", input_shape=self.__target_shape + (3,), include_top=False)

        base_cnn.trainable = False

        new_model = None
        for idx, layer in enumerate(base_cnn.layers[1:]):
            layer.trainable = False
            if new_model is None:
                input = layer(base_cnn.layers[idx].output)
                new_model = Model(base_cnn.input, input)
                continue
            if idx % 10 == 0:
                input = AdaptiveMaxPooling3D(
                    output_size=new_model.output.shape[1:])(new_model.output)
                new_model = Model(new_model.input, input)

            input = layer(new_model.output)
            new_model = Model(new_model.input, input)

        with open('base_cnn.txt', 'w') as fh:
            new_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        return new_model


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
