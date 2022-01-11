from math import e
import os
import cv2
import numpy as np

from tqdm import tqdm

from tensorflow.keras import Model
from tensorflow.keras.applications import mobilenet
from preprocessing.embedding.facenet import Facenet


data_dir = os.path.join("data", "flicker_detection")
data_paths = os.listdir(data_dir)

output_dir = os.path.join("data", "embedding")
os.makedirs(output_dir, exist_ok=True)

model = mobilenet.MobileNet()
low = Model(
    inputs=model.input,
    outputs=model.get_layer(model.layers[9].name).output
)
high = Model(
    inputs=model.input,
    outputs=model.get_layer(model.layers[91].name).output
)

for path in tqdm(data_paths):
    vidcap = cv2.VideoCapture(os.path.join(data_dir, path))
    success, image = vidcap.read()

    raw_images = list()
    while success:
        raw_images.append(cv2.resize(image, (224, 224)))
        success, image = vidcap.read()

    embeddings = model.predict(np.array(raw_images))

    np.save(os.path.join(output_dir, path), embeddings)
