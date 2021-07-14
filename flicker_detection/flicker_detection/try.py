import cv2
import copy
import numpy as np

from preprocessing.embedding.facenet import Facenet
from util.utils import take_snapshots, euclidean_distance


processing_frames = take_snapshots("data/001.mp4")

f1 = processing_frames[0]
f2 = processing_frames[1]

# facenet = Facenet()

# init_score = euclidean_distance(facenet.get_embedding(np.array([f1, ])), facenet.get_embedding(np.array([f2, ])))

scale = (1, 1)

m1 = copy.deepcopy(f1)
m2 = copy.deepcopy(f2)

scores = []

for p in range(int(f1.shape[0] / scale[0])):
    for q in range(int(f1.shape[1] / scale[1])):

        print(p, q)

        # t1 = copy.deepcopy(f1)
        # t2 = copy.deepcopy(f2)

        # for i in range(p * scale[0], (p + 1) * scale[0]):
        #     for j in range(q * scale[1], (q + 1) * scale[1]):
        #         t1[i][j] = [0, 0, 0]
        #         t2[i][j] = [0, 0, 0]

        t1 = copy.deepcopy(
            f1[p * scale[0]: (p + 1) * scale[0], q * scale[1]: (q + 1) * scale[1]])
        t2 = copy.deepcopy(
            f2[p * scale[0]: (p + 1) * scale[0], q * scale[1]: (q + 1) * scale[1]])

        scores.append(np.linalg.norm(t1 - t2))

scores = np.array(scores)
np.save("scores.npy", scores)
init_score = np.mean(scores[scores != 0.0])

cnt = 0
for p in range(int(f1.shape[0] / scale[0])):
    for q in range(int(f1.shape[1] / scale[1])):

        score = scores[cnt]
        cnt += 1

        factor = (init_score + np.abs(init_score - score)) / init_score

        if score > init_score:
            for i in range(p * scale[0], (p + 1) * scale[0]):
                for j in range(q * scale[1], (q + 1) * scale[1]):
                    m1[i][j] = [0, 0, int(m1[i][j][2] * factor)]
                    m2[i][j] = [0, 0, int(m2[i][j][2] * factor)]

        if score < init_score:
            for i in range(p * scale[0], (p + 1) * scale[0]):
                for j in range(q * scale[1], (q + 1) * scale[1]):
                    m1[i][j] = [0, int(m1[i][j][1] * factor), 0]
                    m2[i][j] = [0, int(m2[i][j][1] * factor), 0]


cv2.imwrite("1.png", m1)
cv2.imwrite("2.png", m2)
