import cv2
import numpy as np


class Affine:

    def __init__(self) -> None:
        pass

    def __scale(self, x: float = 1.1, y: float = 1.0) -> np.array:
        return np.array([[x, 0.0, 0.0], [0.0, y, 0.0]])

    def __rotate(self, theta: float = 3.0) -> np.array:
        degree = theta / 180.0 * np.pi
        return np.array([[np.cos(degree), -np.sin(degree), 0.0], [np.sin(degree), np.cos(degree), 0.0]])

    def __shear(self, x: float = 0.1, y: float = 0.0) -> np.array:
        return np.array([[1.0, x, 0.0], [y, 1.0, 0.0]])

    def get_transformation(self, image: np.array) -> dict:
        assert len(image.shape) == 3
        rows, cols = image.shape[:2]
        dst1 = cv2.warpAffine(image, self.__scale(), (cols, rows))
        dst2 = cv2.warpAffine(image, self.__rotate(), (cols, rows))
        dst3 = cv2.warpAffine(image, self.__shear(), (cols, rows))
        cv2.imwrite("original.png", image)
        cv2.imwrite("scale.png", dst1)
        cv2.imwrite("rotate.png", dst2)
        cv2.imwrite("shear.png", dst3)
        return dict({
            "scale": dst1,
            "rotate": dst2,
            "shear": dst3
        })
