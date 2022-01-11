import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# pass the videos without label
pass_videos = list([
    "0096.mp4", "0097.mp4", "0098.mp4",
    "0125.mp4", "0126.mp4", "0127.mp4",
    "0145.mp4", "0146.mp4", "0147.mp4",
    "0178.mp4", "0179.mp4", "0180.mp4"
])

raw_labels = json.load(open(os.path.join("data", "label.json"), "r"))
encoding_filename_mapping = json.load(open(os.path.join("data", "mapping.json"), "r"))

embedding_dir = os.path.join("data", "embedding")
if not os.path.exists:
    input("No embeddings found!")
    exit(0)

embedding_path_list = sorted(os.listdir(embedding_dir))

for path in embedding_path_list:
    if encoding_filename_mapping[path.replace(".npy", "")] not in raw_labels:
        pass_videos.append(path.replace(".npy", ""))

embeddings = list()
video_lengths = dict()
for path in embedding_path_list:
    if path.split(".npy")[0] in pass_videos:
        continue

    buf_embedding = np.load(os.path.join(embedding_dir, path))
    video_lengths[path] = buf_embedding.shape[0]
    embeddings.extend(buf_embedding)

labels = list()
for i, path in enumerate(embedding_path_list):
    if path.split(".npy")[0] in pass_videos:
        continue

    buf_label = np.zeros(video_lengths[path]).astype(int)
    if encoding_filename_mapping[path.replace(".npy", "")] in raw_labels:
        flicker_idxs = np.array(raw_labels[encoding_filename_mapping[path.replace(".npy", "")]]) - 1
        buf_label[flicker_idxs] = 1
    labels.extend(buf_label.tolist())

chunk_size = int(30)

prefix_length = 0
chunk_prefix_length = 0

chunk_offsets = dict()

# To prevent chunks in the same video to be in the same set
embedding_chunks_train = list()
embedding_chunks_test = list()

label_chunks_train = list()
label_chunks_test = list()

cnt = 0
for path in embedding_path_list:
    if path.split(".npy")[0] in pass_videos:
        continue

    for j in range(video_lengths[path] // chunk_size):
        if cnt < 70:
            embedding_chunks_train.append(embeddings[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)])
        else:
            embedding_chunks_test.append(embeddings[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)])
        if cnt < 70:
            label_chunks_train.append(1 if sum(labels[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)]) else 0)
        else:
            label_chunks_test.append(1 if sum(labels[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)]) else 0)
    cnt += 1

    chunk_offsets[path] = (chunk_prefix_length, chunk_prefix_length + j)
    chunk_prefix_length += j

    prefix_length += video_lengths[path]

embedding_chunks_train = np.array(embedding_chunks_train)
embedding_chunks_test = np.array(embedding_chunks_test)
label_chunks_train = np.array(label_chunks_train)
label_chunks_test = np.array(label_chunks_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(
    np.reshape(embedding_chunks_train, (-1, 30 * 9216)),
    label_chunks_train
)
X_train = np.reshape(X_train, (-1, 30, 9216))
X_test, y_test = embedding_chunks_test, label_chunks_test

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def f1_m(y_true, y_pred):
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=3, input_shape=(X_train.shape[1:]), padding="same"))
model.add((LSTM(units=64, input_shape=(X_train.shape[1:]))))
model.add(Dense(units=16, activation="relu"))
model.add(Flatten())
model.add(Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=["accuracy", f1_m])
print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs=1000,
    validation_split=0.1, batch_size=512,
    callbacks=[ModelCheckpoint('model.h5', save_best_only=True, monitor="val_f1_m", mode="max")]
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "validation loss"])
plt.savefig("loss.png")
plt.close()

plt.plot(history.history["f1_m"])
plt.plot(history.history["val_f1_m"])
plt.legend(["f1", "validation f1"])
plt.savefig("f1.png")

np.save("history", history.history)

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test).reshape(X_test.shape[0])

# TODO (most -> least important): 
#
# 1. Be more rigorous about testing! Video-wise testing 
#    to make sure we're not just over-fitting!
#    
#    Sklearn shuffles videos because stratify is set to True; 
#    you can e.g. add a second [independent] sampler.
#
# 2. You should compare your results to your baseline method.
#
# 3. Add training curves plots.
# 
# 4. Add monitors, e.g. ReduceLR or saving (selecting)
#    best-performing checkpoint, early stopping, etc.
# 
# 5. Control batch size.
#
# 6. Try focal loss. 
# 
# 7. Use CNN instead of LSTM and compare which one is better.


# NOTE 
# 1. Don't train on imbalanced datasets! your negative 
#    samples do not contribute to the learning process
#    (just the opposite). 
# 
# 2. Don't use sigmoid as an activation function;
#    just use ReLU. You can Google for reasons.
# 
# 3. Give it more time to converge (not just a couple 
#    of epochs). Start with small LR because your model 
#    will always converge (eventually) if the manifold is 
#    well-behaved. Then you can increase your LR so that
#    the model converges faster. 
