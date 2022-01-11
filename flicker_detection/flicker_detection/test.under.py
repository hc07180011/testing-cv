import os
import time
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc


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

embedding_chunks = list()
label_chunks = list()
for path in embedding_path_list:
    if path.split(".npy")[0] in pass_videos:
        continue

    for j in range(video_lengths[path] // chunk_size):
        embedding_chunks.append(embeddings[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)])
        label_chunks.append(1 if sum(labels[prefix_length + chunk_size * j: prefix_length + chunk_size * (j + 1)]) else 0)

    chunk_offsets[path] = (chunk_prefix_length, chunk_prefix_length + j)
    chunk_prefix_length += j

    prefix_length += video_lengths[path]

label_chunks = np.array(label_chunks)
embedding_chunks = np.array(embedding_chunks)

video_count = len(video_lengths.values())
train_size = round(video_count * 0.9)

X_train, X_test, y_train, y_test = list(), list(), list(), list()

for k in list(chunk_offsets.keys())[-train_size:]:
    current_label_chunks = label_chunks[chunk_offsets[k][0]: chunk_offsets[k][1]]
    current_embedding_chunks = embedding_chunks[chunk_offsets[k][0]: chunk_offsets[k][1]]
    positive_idxs = np.where(current_label_chunks == 1)[0]
    negative_idxs = np.random.choice(np.where(current_label_chunks != 1)[0], int(np.sum(current_label_chunks)))
    taking_idxs = positive_idxs.tolist() + negative_idxs.tolist()
    for i in taking_idxs:
        X_train.append(current_embedding_chunks[i].tolist())
        y_train.append(current_label_chunks[i].tolist())
for k in list(chunk_offsets.keys())[:-train_size]:
    current_label_chunks = label_chunks[chunk_offsets[k][0]: chunk_offsets[k][1]]
    current_embedding_chunks = embedding_chunks[chunk_offsets[k][0]: chunk_offsets[k][1]]
    positive_idxs = np.where(current_label_chunks == 1)[0]
    negative_idxs = np.random.choice(np.where(current_label_chunks != 1)[0], int(np.sum(current_label_chunks)))
    taking_idxs = positive_idxs.tolist() + negative_idxs.tolist()
    for i in taking_idxs:
        X_test.append(current_embedding_chunks[i].tolist())
        y_test.append(current_label_chunks[i].tolist())

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


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

from keras.models import load_model
model = load_model("model.facenet.undersampling.h5", custom_objects={"f1_m": f1_m})

s = time.perf_counter()
y_pred = model.predict(X_test)
y_pred = np.reshape(y_pred, (y_pred.shape[0], ))
print("Prediction takes {}s".format(time.perf_counter() - s))

plt.hist(y_pred, bins=100)
plt.xlabel("probability of being a flicker")
plt.ylabel("count(s)")
plt.title("Counts of Probability being Flickers")
plt.savefig("test1.png")
plt.close()

model.evaluate(X_test, y_test)
print(max([f1_score(y_test, (y_pred > x).astype(int)) for x in np.arange(0.1, 1.0, 0.001)]))

np.save("y_test", y_test)
np.save("y_pred", y_pred)

threshold_range = np.arange(0.1, 1.0, 0.001)

f1_scores = list()
precisions = list()
recalls = list()
for lambda_ in threshold_range:
    f1_scores.append(f1_score(y_test, (y_pred > lambda_).astype(int)))
    precisions.append(precision_score(y_test, (y_pred > lambda_).astype(int)))
    recalls.append(recall_score(y_test, (y_pred > lambda_).astype(int)))

print(np.max(f1_scores), np.mean(f1_scores), np.std(f1_scores))

plt.plot(threshold_range, f1_scores)
plt.scatter(threshold_range[np.argmax(f1_scores)], np.max(f1_scores), c="r")
plt.xlabel("threshold")
plt.ylabel("f1 score")
plt.title("F1-score Under Certain Threshold")
plt.savefig("test2.png")
plt.close()

plt.plot(threshold_range, recalls)
plt.plot(threshold_range, precisions)
plt.legend(["Recall", "Precision"])
plt.xlabel("threshold")
plt.ylabel("score")
plt.title("Precision & Recall Under Certain Threshold")
plt.savefig("test3.png")
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot([0, 1], [0, 1], linestyle="dashed")
plt.plot(fpr, tpr, marker="o")
plt.plot([0, 0, 1], [0, 1, 1], linestyle="dashed", c="red")
plt.legend([
    "No Skill",
    "ROC curve (area = {:.2f})".format(auc(fpr, tpr)),
    "Perfect"
])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("test4.png")
plt.close()

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot([0, 1], [0, 0], linestyle="dashed")
plt.plot(recall, precision, marker="o")
plt.legend([
    "No Skill",
    "Model"
])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall Curve")
plt.savefig("test5.png")

print(confusion_matrix(y_test, (y_pred > 0.5).astype(int)))
print(confusion_matrix(y_test, (y_pred > 0.9).astype(int)))

history = np.load("history.npy", allow_pickle=True).tolist()

print(np.min(history["val_loss"]))
print(np.max(history["val_f1_m"]))

plt.figure(figsize=(16, 4))
plt.plot(history["loss"][:2000])
plt.plot(history["val_loss"][:2000])
plt.xlabel("#epochs")
plt.ylabel("loss value")
plt.title("Loss with LSTM - Oversample")
plt.savefig("loss.png")
plt.close()

plt.figure(figsize=(16, 4))
plt.plot(history["f1_m"][:2000])
plt.plot(history["val_f1_m"][:2000])
plt.xlabel("#epochs")
plt.ylabel("f1 score")
plt.title("F1 score with LSTM - Oversample")
plt.savefig("f1.png")
plt.close()

# positive = 288
# negative = 8986 - 288

# resample_positive = 288
# resample_negative = 576 - 288


# plt.bar(0, negative, bottom=positive, color="dodgerblue")
# plt.bar(0, positive, bottom=0, color="midnightblue")
# plt.bar(1, resample_negative, bottom=resample_positive, color="dodgerblue")
# plt.bar(1, resample_positive, bottom=0, color="midnightblue")

# plt.legend(["Negative", "Positive"])

# plt.xticks([0, 1], ["Original", "Undersample"])
# plt.ylabel("count(s)")

# plt.savefig("test.png")

y_true = np.load("y_test.npy")
y_scores = np.load("y_pred.npy")

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

print(precision)
print(recall)
print(thresholds)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("test.png")