import cv2
import json
import gc
from re import S
import numpy as np
import tensorflow as tf

# from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
# from tensorflow_addons.layers import AdaptiveMaxPooling3D


class BaseCNN:
    """
    adaptive pooling sample:
    https://ideone.com/cJoN3x
    """
    # tf.random.set_seed(12345)
    tf.keras.utils.set_random_seed(12345)
    tf.config.experimental.enable_op_determinism()

    def __init__(self) -> None:
        self.__target_shape = (200, 200)
        self.__embedding = None
        self.strategy = tf.distribute.MirroredStrategy()
        tf.get_logger().setLevel('INFO')

    def get_embedding(self, images: np.ndarray, batched=True) -> np.ndarray:
        if not batched:
            images = np.expand_dims(images, axis=0)
        with self.strategy.scope():
            resized_images = tf.image.resize(
                images, self.__target_shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # check resize differences
            return self.__embedding.predict(resnet.preprocess_input(resized_images))

    def get_embed_cpu(self, images: np.ndarray, batched=True) -> np.ndarray:
        if not batched:
            images = np.expand_dims(images, axis=0)
        resized_images = np.array([cv2.resize(image, dsize=self.__target_shape,
                                              interpolation=cv2.INTER_CUBIC) for image in images])
        with self.strategy.scope():
            image_tensor = tf.convert_to_tensor(resized_images, np.float32)
            return self.__embedding(resnet.preprocess_input(image_tensor)).numpy()

    def extractor(self, extractor: Model, weights: str = "imagenet", pooling: str = "Max") -> Model:
        with self.strategy.scope():
            self.__embedding = extractor(
                weights=weights,
                input_shape=self.__target_shape + (3,),
                include_top=False,
                pooling=pooling
            )
            return self.__embedding

    """
    TO DO - REWRITE MODEL RETRAINING METHOD
    
    def adaptive_extractor(self, extractor: Model, frequency: int, weights: str = "imagenet", pooling: str = "Max") -> Model:
        # https://stackoverflow.com/questions/58660613/how-to-add-another-layer-on-a-pre-loaded-network
        with self.strategy.scope():
            base_cnn = extractor(
                weights=weights,
                input_shape=self.__target_shape + (3,),
                include_top=False,
                pooling=pooling
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
    """


class Serializer:
    def __init__(self) -> None:
        self.data = None
        self.filename = None
        self.schema = {}

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value ist tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a floast_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_array(self, array):
        return tf.io.serialize_tensor(array)

    def parse_batch(self, batch: np.ndarray, filename: str = "images") -> None:
        # define the dictionary -- the structure -- of our single example
        if self.filename is None:
            self.data = {}
            self.filename = filename
            self.schema[self.filename] = None

        self.data["{}".format(self.filename)] = self._bytes_feature(
            self.serialize_array(batch))

    def write_to_tfr(self):
        self.writer = tf.io.TFRecordWriter(
            "data/TFRecords/{}.tfrecords".format(self.filename))
        # create an Example, wrapping the single features
        embedding = tf.train.Example(
            features=tf.train.Features(feature=self.data))
        # create a writer that'll store our data to disk
        self.writer.write(embedding.SerializeToString())

    def done_writing(self):
        self.data = None
        self.filename = None
        self.writer.close()
        self.writer = None
        gc.collect()

    def get_schema(self):
        with open('data/schema.json', 'w') as fh:
            json.dump(self.schema, fh)
