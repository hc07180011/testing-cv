import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

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

iter_layer_num = list([9, 16, 22, 29, 35, 42, 48, 54, 60, 66, 72, 79]) # 85, 91


# vidcap = cv2.VideoCapture(os.path.join("data", "flicker_detection", "0180.mp4"))

# raw_arr = list()
# success, image = vidcap.read()
# while success:
#     raw_arr.append(cv2.resize(image, (224, 224)))
#     success, image = vidcap.read()


white_black = list()
white_object1 = list()
object1_object2 = list()
shift = list()
rotation = list()

for layer_num in iter_layer_num:

    test_model = intermediate_layer_models[layer_num] 

    # embedding_arr = test_model.predict(np.array(raw_arr))
    # print(embedding_arr.shape)

    def cos_sim(vector1, vector2):
        distance = np.linalg.norm(vector1 - vector2)
        return float(distance)
        # vector1 = vector1.flatten()
        # vector2 = vector2.flatten()
        # return float(
        #     np.dot(vector1, vector2) /
        #     (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        # )

    # similarities = list()
    # for emb1, emb2 in zip(embedding_arr[:-1], embedding_arr[1:]):
    #     similarities.append(cos_sim(emb1, emb2))

    # plt.figure(figsize=(16, 4))
    # plt.plot(similarities)
    # plt.xlabel("# Frames")
    # plt.ylabel("Distance")
    # plt.title("Euclidean Distance with [{}]".format(model.layers[layer_num].name))
    # plt.savefig("intermediate_{}.png".format(layer_num))

    white = cv2.resize(cv2.imread("white.png"), (224, 224))
    black = cv2.resize(cv2.imread("black.png"), (224, 224))
    object1 = cv2.resize(cv2.imread("object1.png"), (224, 224))
    object2 = cv2.resize(cv2.imread("object2.png"), (224, 224))
    shift1 = cv2.resize(cv2.imread("shift1.png"), (224, 224))
    shift2 = cv2.resize(cv2.imread("shift2.png"), (224, 224))
    rotation1 = cv2.resize(cv2.imread("rotation1.png"), (224, 224))
    rotation2 = cv2.resize(cv2.imread("rotation2.png"), (224, 224))

    white_black.append(cos_sim(
        test_model.predict(np.reshape(white, (1, 224, 224, 3))),
        test_model.predict(np.reshape(black, (1, 224, 224, 3)))
    ))
    white_object1.append(cos_sim(
        test_model.predict(np.reshape(white, (1, 224, 224, 3))),
        test_model.predict(np.reshape(object1, (1, 224, 224, 3)))
    ))
    print(cos_sim(
        test_model.predict(np.reshape(white, (1, 224, 224, 3))),
        test_model.predict(np.reshape(object2, (1, 224, 224, 3)))
    ))
    object1_object2.append(cos_sim(
        test_model.predict(np.reshape(object1, (1, 224, 224, 3))),
        test_model.predict(np.reshape(object2, (1, 224, 224, 3)))
    ))
    shift.append(cos_sim(
        test_model.predict(np.reshape(shift1, (1, 224, 224, 3))),
        test_model.predict(np.reshape(shift2, (1, 224, 224, 3)))
    ))
    rotation.append(cos_sim(
        test_model.predict(np.reshape(rotation1, (1, 224, 224, 3))),
        test_model.predict(np.reshape(rotation2, (1, 224, 224, 3)))
    ))

plt.plot(white_black)
plt.xlabel("Depth")
plt.ylabel("Euclidean Distance")
plt.savefig("white_black.png")
plt.close()

plt.plot(white_object1)
plt.savefig("white_object1.png")
plt.close()

plt.plot(object1_object2)
plt.savefig("object1_object2.png")
plt.close()

plt.plot(shift)
plt.xlabel("Depth")
plt.ylabel("Euclidean Distance")
plt.savefig("shift.png")
plt.close()

plt.plot(rotation)
plt.xlabel("Depth")
plt.ylabel("Euclidean Distance")
plt.savefig("rotation.png")
plt.close()
