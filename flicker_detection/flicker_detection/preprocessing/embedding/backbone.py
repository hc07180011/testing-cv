import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow_addons.layers import AdaptiveMaxPooling3D


class BaseCNN:
    """
    adaptive pooling sample:
    https://ideone.com/cJoN3x
    """

    def __init__(self) -> None:
        self.__target_shape = (200, 200)
        self.__embedding = None
        np.random.seed(0)
        tf.get_logger().setLevel('INFO')

    def get_embedding(self, images: np.ndarray, batched=True) -> np.ndarray:
        if not batched:
            images = np.expand_dims(images, axis=0)
        resized_images = tf.image.resize(
            images, self.__target_shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return self.__embedding.predict(resnet.preprocess_input(resized_images))

    def adaptive_extractor(self, extractor: Model, frequency: int) -> Model:
        """
        https://stackoverflow.com/questions/58660613/how-to-add-another-layer-on-a-pre-loaded-network
        """
        base_cnn = extractor(
            weights="imagenet",
            input_shape=self.__target_shape + (3,),
            include_top=False,
            pooling="Max"
        )

        new_model = None
        for idx, layer in enumerate(base_cnn.layers[1:]):
            layer.trainable = False
            if new_model is None:
                input = layer(base_cnn.layers[idx].output)
                new_model = Model(base_cnn.input, input)
                continue
            if idx % frequency == 0:
                input = AdaptiveMaxPooling3D(
                    output_size=new_model.output.shape[1:])(new_model.output)
                new_model = Model(new_model.input, input)

            input = layer(new_model.output)
            new_model = Model(new_model.input, input)

        output = layers.Dense(256)(new_model.output)

        self.__embedding = Model(
            new_model.input, output, name="Embedding")

        with open('preprocessing/embedding/models/feature_extractor.txt', 'w') as fh:
            tf.keras.utils.plot_model(
                self.__embedding, 'preprocessing/embedding/models/feature_extractor.png', show_shapes=True)
            self.__embedding.summary(print_fn=lambda x: fh.write(x + '\n'))
        return self.__embedding
