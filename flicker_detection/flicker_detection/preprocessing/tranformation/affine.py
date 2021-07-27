import cv2
import time
import logging
import itertools
import numpy as np
from typing import List, Union

from util.utils import euclidean_distance


class Affine:

    def __init__(self) -> None:
        logging.info("Initializing preprocessing.transformation.affine")
        pass

    def __scale(self, shape, x: float = 1.1, y: float = 1.0) -> np.array:
        return np.array([[x, 0.0, 0.0], [0.0, y, 0.0]])

    def __rotate(self, shape, angle: float = 5.0, scale: float = 1.0) -> np.array:
        return cv2.getRotationMatrix2D(tuple(np.array(shape[1::-1]) / 2), angle, scale)

    def __shear(self, shape, x: float = 0.1, y: float = 0.0) -> np.array:
        return np.array([[1.0, x, -0.5 * x * shape[0]], [y, 1.0, -0.5 * y * shape[1]]])

    def __transformation_trial(self, mask, anchor_embedding, transformation_func, image1, image2_border) -> List[Union[int, float]]:
        s = time.perf_counter()
        value1_range = np.arange(-5, 6)
        value2_range = np.arange(1, 2)
        iter_list = list(itertools.product(value1_range, value2_range))
        logging.debug("Testing: {}, {}".format(value1_range, value2_range))
        transforms = []
        for value1, value2 in iter_list:
            transforms.append(cv2.warpAffine(image1, transformation_func(
                image1.shape, value1, value2), image1.shape[1::-1], borderValue=image2_border))
        transforms = np.reshape(np.clip(
            transforms + mask, 0, 255).astype("uint8"), (-1, image1.shape[0] * image1.shape[1] * 3))
        distances = np.linalg.norm(transforms - anchor_embedding, axis=1)
        logging.debug("takes {} s".format(time.perf_counter() - s))
        return iter_list, distances

    def compare_transformation(self, image1: np.array, image2: np.array) -> dict:
        """
        Transform image1 to be as similar as possible with image2
        """
        assert len(image1.shape) == 3 and len(image2.shape) == 3

        logging.info("Comparing every transformation")

        anchor_embedding = np.reshape(
            image2, (image2.shape[0] * image2.shape[1] * 3))

        # get the background mask
        # this is neccessary otherwise transformation results might exceed border
        background_mask = (image2 == image2[0][0]).astype(
            int) * (255 if image2[0][0][0] else -255)
        # y, x, _ = np.where(image2 != image2[0][0])
        # index_mask = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])

        keys, values = self.__transformation_trial(
            background_mask, anchor_embedding, self.__rotate, image1, tuple(image2[0][0].tolist()))

        logging.info("ok")
        return dict({ "rotate": keys[np.argmin(values)] })
