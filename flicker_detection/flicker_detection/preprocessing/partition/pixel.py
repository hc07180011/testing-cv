import sys
import cv2
import logging
import numpy as np

from typing import List


class Pixel:

    def __init__(self) -> None:
        logging.info("Initializing preprocessing.partion.pixel")
        pass

    def __warpMask(self, image: np.ndarray, mask: np.ndarray, p: int, q: int, scale: tuple) -> None:
        if mask > 0:
            index = 1  # red, large distance
            image[p * scale[0]: (p + 1) * scale[0], q * scale[1]: (q + 1) * scale[1], index] = \
                np.clip(image[p * scale[0]: (p + 1) * scale[0], q * scale[1]: (q + 1) * scale[1], index].astype(
                    float) * mask, 0.0, 255.0).astype("uint8")
        elif mask < 0:
            index = 2  # blue, small distance
            image[p * scale[0]: (p + 1) * scale[0], q * scale[1]: (q + 1) * scale[1], index] = \
                np.clip(image[p * scale[0]: (p + 1) * scale[0], q * scale[1]: (q + 1) * scale[1], index].astype(
                    float) * (-mask), 0.0, 255.0).astype("uint8")

    def get_heatmap(self, image1: np.ndarray, image2: np.ndarray, scale: tuple = (5, 5), output=False) -> List[np.ndarray]:

        assert np.all(image1.shape ==
                      image2.shape), "shape of two images should be same"

        logging.info("Start getting pixel-wise differences map.")

        cv2.imwrite("_1.png", image1)
        cv2.imwrite("_2.png", image2)

        scores = []
        for p in range(int(image1.shape[0] / scale[0])):
            for q in range(int(image1.shape[1] / scale[1])):

                t1 = image1[p * scale[0]: (p + 1) * scale[0], q *
                            scale[1]: (q + 1) * scale[1]]
                t2 = image2[p * scale[0]: (p + 1) * scale[0], q *
                            scale[1]: (q + 1) * scale[1]]

                scores.append(np.linalg.norm(t1 - t2))  # euclidian distance

        scores = np.array(scores)
        baseline = np.mean(scores[scores != 0.0])

        map_mask = np.zeros(
            (int(image1.shape[0] / scale[0]), int(image1.shape[1] / scale[1])))
        count = 0
        for p in range(int(image1.shape[0] / scale[0])):  # y
            for q in range(int(image1.shape[1] / scale[1])):  # x
                score = scores[count]
                count += 1
                factor = (score - baseline) / baseline
                if score == 0.0:
                    map_mask[p][q] = 0.0
                else:
                    map_mask[p][q] = factor

        for p in range(int(image1.shape[0] / scale[0])):  # y
            for q in range(int(image1.shape[1] / scale[1])):  # x
                self.__warpMask(image1, map_mask[p][q], p, q, scale)
                self.__warpMask(image2, map_mask[p][q], p, q, scale)

        if output:
            logging.info(
                "Choose to output: {}, {}. Exitting...".format("1.png", "2.png"))
            cv2.imwrite("1.png", image1)
            cv2.imwrite("2.png", image2)
            sys.exit(0)

        logging.info("ok")

        return image1, image2
