import logging
import numpy as np

import matplotlib.pyplot as plt


class Flicker:

    def __init__(self, fps, similarities, suspects, horizontal_displacements, vertical_displacements) -> None:
        self.fps = fps
        self.similarities = similarities
        self.suspects = suspects
        self.horizontal_displacements = horizontal_displacements
        self.vertical_displacements = vertical_displacements

    def __moving_avg(self, data: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(data, np.ones(window), "valid") / window

    def __continuous_behaviour_detection(self, data, description, z_score_threshold, human_reaction_threshold, plotting=False) -> np.ndarray:
        moving_average = self.__moving_avg(data, human_reaction_threshold)
        z_scores = (moving_average - np.mean(moving_average)
                    ) / np.std(moving_average)  # z-value
        z_score_threshold = z_score_threshold

        if z_score_threshold > 0:
            ma_outliers = np.where(z_scores > z_score_threshold)[0]
        else:
            ma_outliers = np.where(z_scores < z_score_threshold)[0]

        anything_detected = False
        for seq in np.split(ma_outliers, np.where(np.diff(ma_outliers) != 1)[0] + 1):
            if len(seq) > (human_reaction_threshold * 2 - 1):  # original + ma
                logging.debug("Continuous [{}] at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                    description, seq[0], seq[-1], seq[0] / self.fps, seq[-1] / self.fps))
                anything_detected = True

        if not anything_detected:
            logging.debug("Cannot detect continuous [{}]".format(description))

        if plotting:  # dev only
            plt.figure(figsize=(16, 4), dpi=200)
            plt.plot(data)
            plt.plot(moving_average)
            plt.savefig("cont_behav_detect.png")

    def flicker_detection(self, human_reaction_threshold: int = 3):

        logging.info("Start core function: flicker detection")

        logging.debug("Length: {}".format(len(self.similarities[0])))

        self.__continuous_behaviour_detection(
            self.horizontal_displacements, "right movement", 1.0, human_reaction_threshold)
        self.__continuous_behaviour_detection(
            self.horizontal_displacements, "left movement", -1.0, human_reaction_threshold)
        self.__continuous_behaviour_detection(
            self.vertical_displacements, "up movement", -1.0, human_reaction_threshold)
        self.__continuous_behaviour_detection(
            self.vertical_displacements, "down movement", 1.0, human_reaction_threshold)

        logging.info("ok")
