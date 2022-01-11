import cv2
import time

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.applications import mobilenet


model = mobilenet.MobileNet()

intermediate_layer_models = list([
    Model(
        inputs=model.input,
        outputs=model.get_layer(layer.name).output
    )
    for layer in model.layers
])

low = intermediate_layer_models[9]
high = intermediate_layer_models[91]

image = cv2.resize(cv2.imread("input.png"), (224, 224))

s = time.perf_counter()
_ = low.predict(np.reshape(image, (-1, 224, 224, 3)))
print(time.perf_counter() - s)

s = time.perf_counter()
_ = high.predict(np.reshape(image, (-1, 224, 224, 3)))
print(time.perf_counter() - s)
