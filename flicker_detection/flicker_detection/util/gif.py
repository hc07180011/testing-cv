import os
import cv2
import imageio
import numpy as np


img_dir = "dump"
images = []
filenames = [x for x in os.listdir(img_dir) if "1-" in x]
filenames = sorted(filenames)

for filename in filenames:
    try:
        arr = np.array(imageio.imread(os.path.join(img_dir, filename)))
        images.append(cv2.resize(
            arr, (int(arr.shape[1] / 4), int(arr.shape[0] / 4))))
    except Exception as e:
        print(repr(e))
        pass
imageio.mimsave("test.gif", images, duration=(1 / 30.0))
