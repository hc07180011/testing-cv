import cv2
import logging
import numpy as np

from typing import List


class Brisk:

    def __init__(self) -> None:
        super().__init__()
        self.BRISK = cv2.BRISK_create()
        self.BFMatcher = cv2.BFMatcher(
            normType=cv2.NORM_HAMMING, crossCheck=True)
        self.__target_shape = (228, 228)

    def __dump_image(self, img1: np.ndarray, img2: np.ndarray, keypoints1: List[cv2.KeyPoint],
                     keypoints2: List[cv2.KeyPoint], matches: List[cv2.DMatch], dump_img_path: str) -> None:
        np.random.shuffle(matches)
        output = cv2.drawMatches(img1=img1,
                                 keypoints1=keypoints1,
                                 img2=img2,
                                 keypoints2=keypoints2,
                                 matches1to2=matches[:100],
                                 outImg=None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(dump_img_path, output)

    def calculate_movement(self, image1: np.ndarray, image2: np.ndarray, dump_img_path=None) -> tuple:

        image1 = cv2.resize(image1, self.__target_shape,
                            interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, self.__target_shape,
                            interpolation=cv2.INTER_AREA)

        try:
            keypoints1, descriptors1 = self.BRISK.detectAndCompute(
                image1, None)
            keypoints2, descriptors2 = self.BRISK.detectAndCompute(
                image2, None)

            matches = self.BFMatcher.match(queryDescriptors=descriptors1,
                                           trainDescriptors=descriptors2)

            if dump_img_path:
                self.__dump_image(image1, image2, keypoints1,
                                  keypoints2, matches, dump_img_path)

            delta_x = np.mean([keypoints2[p.trainIdx].pt[0] -
                               keypoints1[p.queryIdx].pt[0] for p in matches])
            delta_y = np.mean([keypoints2[p.trainIdx].pt[1] -
                               keypoints1[p.queryIdx].pt[1] for p in matches])

        except:
            # if shape error, value error, or else
            delta_x, delta_y = 0.0, 0.0

        if delta_x != delta_x or delta_y != delta_y:
            # empty frame
            delta_x, delta_y = 0.0, 0.0

        logging.debug("Movement between two images is: ({}, {})".format(
            delta_x, delta_y))

        return (delta_x, delta_y)
