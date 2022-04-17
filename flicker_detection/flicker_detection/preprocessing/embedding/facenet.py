import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow_addons.layers import AdaptiveMaxPooling3D


class Backbone:
    """
    adaptive pooling sample:
    https://ideone.com/cJoN3x
    """

    def __init__(self) -> None:

        super().__init__()

        self.__target_shape = (200, 200)
        self.__embedding = None
        np.random.seed(0)

    def get_embedding(self, images: np.ndarray, batched=True) -> np.ndarray:
        assert (not batched) or len(
            images.shape) == 4, "images should be an array of image with shape (width, height, 3)"
        if not batched:
            images = np.array([images, ])
        resized_images = np.array([cv2.resize(image, dsize=self.__target_shape,
                                              interpolation=cv2.INTER_CUBIC) for image in images])
        image_tensor = tf.convert_to_tensor(resized_images, np.float32)
        return self.__embedding(resnet.preprocess_input(image_tensor)).numpy()

    def adaptive_extractor(self, extractor: Model) -> Model:
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
            if idx % 10 == 0:
                input = AdaptiveMaxPooling3D(
                    output_size=new_model.output.shape[1:])(new_model.output)
                new_model = Model(new_model.input, input)

            input = layer(new_model.output)
            new_model = Model(new_model.input, input)

        output = layers.Dense(256)(new_model.output)

        self.__embedding = Model(
            new_model.input, output, name="Embedding")

        with open('preprocessing/embedding/models/feature_extractor.txt', 'w') as fh:
            self.__embedding.summary(print_fn=lambda x: fh.write(x + '\n'))
        return self.__embedding
