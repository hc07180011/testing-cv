import cv2
import itertools
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch

from util.utils import euclidean_distance


class Affine:

    def __init__(self) -> None:
        pass

    def __scale(self, shape, x: float = 1.1, y: float = 1.0) -> np.array:
        return np.array([[x, 0.0, 0.0], [0.0, y, 0.0]])

    def __rotate(self, shape, angle: float = 5.0, scale: float = 1.0) -> np.array:
        return cv2.getRotationMatrix2D(tuple(np.array(shape[1::-1]) / 2), angle, scale)

    def __shear(self, shape, x: float = 0.1, y: float = 0.0) -> np.array:
        return np.array([[1.0, x, -0.5 * x * shape[0]], [y, 1.0, -0.5 * y * shape[1]]])

    def __transformation_trial(self, mask, embedding_func, anchor_embedding, transformation_func, image1, image2_border) -> dict:
        results = dict()
        for value1, value2 in list(itertools.product(range(-10, 11), range(1, 2))):
            transformed = cv2.warpAffine(image1, transformation_func(
                image1.shape, value1, value2), image1.shape[1::-1], borderValue=image2_border)
            transformed = np.clip(
                transformed + mask, 0, 255).astype("uint8")
            current_similarity = np.abs(euclidean_distance(embedding_func(
                transformed, batched=False), anchor_embedding))
            results[(value1, value2)] = current_similarity
        return results

    def compare_transformation(self, embedding_func, image1: np.array, image2: np.array) -> dict:
        """
        Transform image1 to be as similar as possible with image2
        """
        assert len(image1.shape) == 3 and len(image2.shape) == 3
        anchor_embedding = embedding_func(image2, batched=False)
        anchor_similarity = np.abs(euclidean_distance(
            embedding_func(image1, batched=False), anchor_embedding))

        # get the background mask
        background_mask = (image2 == image2[0][0]).astype(
            int) * (255 if image2[0][0][0] else -255)
        # y, x, _ = np.where(image2 != image2[0][0])
        # index_mask = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])

        res = self.__transformation_trial(
            background_mask, embedding_func, anchor_embedding, self.__rotate, image1, tuple(image2[0][0].tolist()))
        return res
