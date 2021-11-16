import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve



# history = np.load("history.npy", allow_pickle=True).tolist()

# plt.figure(figsize=(24, 4))
# plt.plot(history["loss"])
# plt.plot(history["val_loss"])
# plt.xlabel("#epochs")
# plt.ylabel("loss value")
# plt.title("Loss with LSTM - Undersample")
# plt.savefig("loss.png")

# plt.figure(figsize=(24, 4))
# plt.plot(history["f1_m"])
# plt.plot(history["val_f1_m"])
# plt.xlabel("#epochs")
# plt.ylabel("f1 score")
# plt.title("F1 score with LSTM - Undersample")
# plt.savefig("f1.png")

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
