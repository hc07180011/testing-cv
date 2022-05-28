import os
import sys
import cv2
import time
import logging
import numpy as np

from typing import List
from numpy.linalg import norm


class Pixel:

    def __init__(self) -> None:
        logging.info("Initializing preprocessing.partion.pixel")
        pass

    def __warpMask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        p: int,
        q: int,
        scale: tuple
    ) -> None:
        if mask > 0:
            index = 1  # red, large distance
            image[
                p * scale[0]: (p + 1) * scale[0],
                q * scale[1]: (q + 1) * scale[1],
                index
            ] = np.clip(
                image[
                    p * scale[0]: (p + 1) * scale[0],
                    q * scale[1]: (q + 1) * scale[1],
                    index
                ].astype(float) * mask,
                0.0,
                255.0
            ).astype("uint8")
        elif mask < 0:
            index = 2  # blue, small distance
            image[
                p * scale[0]: (p + 1) * scale[0],
                q * scale[1]: (q + 1) * scale[1],
                index
            ] = np.clip(
                image[
                    p * scale[0]: (p + 1) * scale[0],
                    q * scale[1]: (q + 1) * scale[1],
                    index
                ].astype(float) * (-mask),
                0.0,
                255.0
            ).astype("uint8")

    def get_heatmap(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        scale: tuple = (5, 5),
        output: bool = False,
        exit_: bool = False,
        dump_dir: str = "."
    ) -> np.ndarray:

        assert np.all(image1.shape == image2.shape), \
            "shape of two images should be same"

        logging.info("Start getting pixel-wise differences map.")

        h, w, __ = image1.shape
        splitted_image1 = image1.reshape(
            h // scale[0],
            scale[0],
            -1,
            scale[1],
            3
        ).swapaxes(1, 2).reshape(
            (h // scale[0]) * (w // scale[1]),
            -1
        )
        splitted_image2 = image2.reshape(
            h // scale[0],
            scale[0],
            -1,
            scale[1],
            3
        ).swapaxes(1, 2).reshape(
            (h // scale[0]) * (w // scale[1]),
            -1
        )

        scores = norm(
            splitted_image1 - splitted_image2,
            axis=1
        ).reshape(h // scale[0], w // scale[1])
        baseline = np.mean(scores[scores != 0.0])

        map_mask = (scores - baseline) / baseline * scores.astype(bool)
        for p in range(int(image1.shape[0] / scale[0])):  # y
            for q in range(int(image1.shape[1] / scale[1])):  # x
                self.__warpMask(image1, map_mask[p][q], p, q, scale)

        if output:
            image_paths = (
                os.path.join(
                    dump_dir,
                    "1-{}.png".format(int(time.time() * 1000000))
                ),
                os.path.join(
                    dump_dir,
                    "2-{}.png".format(int(time.time() * 1000000))
                )
            )
            logging.info("Choose to output: {}, {}. Exitting...".format(
                image_paths[0],
                image_paths[1]
            ))
            cv2.imwrite(image_paths[0], image1)
            if exit_:
                sys.exit(0)

        return image1

