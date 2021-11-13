import os
import cv2
import numpy as np

from tqdm import tqdm

from preprocessing.embedding.facenet import Facenet


data_dir = os.path.join("data", "flicker_detection")
data_paths = os.listdir(data_dir)

output_dir = os.path.join("data", "embedding")
os.makedirs(output_dir, exist_ok=True)

facenet = Facenet()

for path in tqdm(data_paths):
    vidcap = cv2.VideoCapture(os.path.join(data_dir, path))
    success, image = vidcap.read()
    embedding = list()
    while success:
        embedding.append(facenet.get_embedding(image, batched=False).flatten())
        success, image = vidcap.read()

    np.save(os.path.join(output_dir, path), embedding)
