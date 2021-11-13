import os
import json
from cv2 import mean
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


pass_videos = list([
    "0096.mp4", "0097.mp4", "0098.mp4",
    "0125.mp4", "0126.mp4", "0127.mp4",
    "0145.mp4", "0146.mp4", "0147.mp4",
    "0178.mp4", "0179.mp4", "0180.mp4"
])

embedding_dir = os.path.join("data", "embedding")
embedding_path_list = sorted(os.listdir(embedding_dir))

embeddings = list()
video_lengths = dict()
for path in embedding_path_list:
    if path.split(".npy")[0] in pass_videos:
        continue

    buf_embedding = np.load(os.path.join(embedding_dir, path))
    video_lengths[path] = buf_embedding.shape[0]
    embeddings.extend(buf_embedding)

raw_labels = json.load(open(os.path.join("data", "label.json"), "r"))
encoding_filename_mapping = json.load(open(os.path.join("data", "mapping.json"), "r"))

labels = list()
for i, path in enumerate(embedding_path_list):
    if path.split(".npy")[0] in pass_videos:
        continue

    buf_label = np.zeros(video_lengths[path]).astype(int)
    if encoding_filename_mapping[path.replace(".npy", "")] in raw_labels:
        flicker_idxs = np.array(raw_labels[encoding_filename_mapping[path.replace(".npy", "")]]) - 1
        buf_label[flicker_idxs] = 1
    labels.extend(buf_label.tolist())

embedding_chunks = list()
label_chunks = list()

chunk_size = int(30)

prefix_length = 0
for path in embedding_path_list:
    if path.split(".npy")[0] in pass_videos:
        continue

    for j in range(video_lengths[path] // chunk_size):
        embedding_chunks.append(embeddings[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)])
        label_chunks.append(labels[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)])

    prefix_length += video_lengths[path]

X_train, X_test, y_train, y_test = train_test_split(
    np.array(embedding_chunks),
    np.array(label_chunks),
    test_size=0.1,
    random_state=42,
    shuffle=False
)

print(X_train.shape)

model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

model.fit(X_train, y_train, epochs=3, validation_split=0.1)

y_pred = model.predict(X_test)
# prob_baseline = np.mean(y_pred)
# y_pred = np.array([
#     1 if y > prob_baseline else 0
#     for y in y_pred
# ])

print(y_pred.shape)
print(y_test.shape)

# print(confusion_matrix(y_test, y_pred))