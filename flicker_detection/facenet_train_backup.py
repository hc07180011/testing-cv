from genericpath import exists
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array


target_shape = (200, 200)

cache_dir = Path(Path.home()) / ".keras"
anchor_images_path = cache_dir / "left"
positive_images_path = cache_dir / "right"


def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
)

positive_images = sorted(
    [str(positive_images_path / f) for f in os.listdir(positive_images_path)]
)

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip(
    (anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])


visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)


class SiameseModel(Model):

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


if __name__ == "__main__":

    local_cache_dir = "cache"

    dataset_dir = "177193533"

    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    current_cache_dir = os.path.join(local_cache_dir, dataset_dir)
    os.makedirs(current_cache_dir, exist_ok=True)

    images_path = np.sort(os.listdir(raw_dir))

    if not os.path.exists(os.path.join(current_cache_dir, "embeddings.npy")):

        siamese_model = SiameseModel(siamese_network)
        filepath = "model.h5"

        if os.path.exists(os.path.join("model", filepath)):
            siamese_model.load_weights(os.path.join("model", filepath))
        else:
            siamese_model.compile(optimizer=optimizers.Adam(0.0001))
            checkpoint = ModelCheckpoint(
                filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            siamese_model.fit(train_dataset, epochs=10,
                              validation_data=val_dataset, callbacks=callbacks_list)

        embeddings = []
        embeddings_mean = []
        for image_path, image_path_next in zip(images_path[:-1], images_path[1:]):

            print("Embedding: {}".format(image_path))

            image_array = img_to_array(
                load_img(os.path.join(raw_dir, image_path)))
            image_array = cv2.resize(image_array, dsize=target_shape,
                                     interpolation=cv2.INTER_CUBIC)
            image_tensor = tf.convert_to_tensor(
                np.reshape(image_array, (1, 200, 200, 3)), np.float32)
            embeddings.append(embedding(resnet.preprocess_input(image_tensor)))
            embeddings_mean.append(
                embedding(resnet.preprocess_input(image_tensor)))

            image_array_next = img_to_array(
                load_img(os.path.join(raw_dir, image_path_next)))
            image_array_next = cv2.resize(image_array_next, dsize=target_shape,
                                          interpolation=cv2.INTER_CUBIC)
            image_tensor_next = tf.convert_to_tensor(
                np.reshape((image_array / 2 + image_array_next / 2), (1, 200, 200, 3)), np.float32)
            embeddings_mean.append(
                embedding(resnet.preprocess_input(image_tensor_next)))

        np.save(os.path.join(current_cache_dir, "embeddings.npy"), embeddings)
        np.save(os.path.join(current_cache_dir,
                             "embeddings_mean.npy"), embeddings_mean)

    else:
        embeddings = np.load(os.path.join(current_cache_dir, "embeddings.npy"))
        embeddings_mean = np.load(os.path.join(
            current_cache_dir, "embeddings_mean.npy"))

    cosine_similarity = metrics.CosineSimilarity()

    results_dir = os.path.join(dataset_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    similarity_1 = []
    similarity_5 = []
    for i, (emb1, emb2) in enumerate(zip(embeddings[:-5], embeddings[5:])):
        print("Processing: {}".format(i))
        similarity_5.append(np.divide(np.inner(emb1, emb2), np.sqrt(np.multiply(
            np.sum(np.multiply(emb1, emb1), axis=1), np.inner(emb2, emb2))))[0][0])
        emb3 = embeddings[i+1]
        similarity_1.append(np.divide(np.inner(emb1, emb3), np.sqrt(np.multiply(
            np.sum(np.multiply(emb1, emb1), axis=1), np.inner(emb3, emb3))))[0][0])

    similarity_mean = []
    for i, (emb1, emb2) in enumerate(zip(embeddings_mean[0::2], embeddings_mean[1::2])):
        print("Processing: {}".format(i))
        similarity_mean.append(np.divide(np.inner(emb1, emb2), np.sqrt(np.multiply(
            np.sum(np.multiply(emb1, emb1), axis=1), np.inner(emb2, emb2))))[0][0])

    plt.figure(figsize=(16, 3), dpi=1000)
    plt.plot(similarity_1, label="Window Size = 2")
    plt.plot(similarity_5, label="Window Size = 6")
    plt.plot(similarity_mean, label="Mean", c="r", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "resnet.png"))
    plt.close()


