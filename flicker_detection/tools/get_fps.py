import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('data/flicker_detection/0001.mp4')
print(cap.get(cv2.CAP_PROP_FPS))
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while cap.isOpened():
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
    else:
        break

cap.release()

diff = [y - x for x, y in zip(timestamps[:-1], timestamps[1:])]
print(len(diff) / 37)

diff = [1000.0 / x if x else x for x in diff if x < 1000 and x > -1000]

plt.figure(figsize=(16, 4))
plt.plot(diff)
plt.plot([max(diff)] * len(diff), c="r", zorder=-10000)
plt.plot([60.0] * len(diff), c="green", zorder=10000)
plt.annotate("max ≈ {:.2f}".format(max(diff)), (25, 200), c="r")
plt.annotate("mean ≈ 60", (350, 75), c="green")
plt.title("FPS between frames")
plt.xlabel("Number of frame")
plt.ylabel("FPS")
plt.savefig("test.png")
