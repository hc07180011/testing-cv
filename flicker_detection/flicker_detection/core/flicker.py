import sys
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
        results = []
        for seq in np.split(ma_outliers, np.where(np.diff(ma_outliers) != 1)[0] + 1):
            if len(seq) > (human_reaction_threshold * 2 - 1):  # original + ma
                logging.debug("Continuous [{}] at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                    description, seq[0], seq[-1], seq[0] / self.fps, seq[-1] / self.fps))
                anything_detected = True
                results.append([seq[0], seq[-1]])

        if not anything_detected:
            logging.debug("Cannot detect continuous [{}]".format(description))

        if plotting:  # dev only
            plt.figure(figsize=(16, 4), dpi=200)
            plt.plot(data)
            plt.plot(moving_average)
            for i in ma_outliers:
                plt.scatter(i, -3, s=3, c="r")
            for i in range(0, len(data), 10):
                plt.vlines(i, ymin=int(np.min(data)),
                           ymax=int(np.max(data)), colors="red", linestyles="dashed")
            plt.savefig("cont_behav_detect.png")
            # sys.exit(0)

        return np.array(results)

    def flicker_detection(self, human_reaction_threshold: int = 5):

        # need further tunning of reaction threshold

        logging.info("Start core function: flicker detection")

        logging.debug("Length: {}".format(len(self.similarities[0])))

        exemption_mask = []

        movement_base_threshold = 1.0
        for seq in self.__continuous_behaviour_detection(
                self.horizontal_displacements, "right movement", movement_base_threshold, human_reaction_threshold):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)
        for seq in self.__continuous_behaviour_detection(
                self.horizontal_displacements, "left movement", -movement_base_threshold, human_reaction_threshold):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)
        for seq in self.__continuous_behaviour_detection(
                self.vertical_displacements, "up movement", -movement_base_threshold, human_reaction_threshold):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)
        for seq in self.__continuous_behaviour_detection(
                self.vertical_displacements, "down movement", movement_base_threshold, human_reaction_threshold):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)

        similarity_base_threshold = 0.1
        continuous_results = self.__continuous_behaviour_detection(
            self.similarities[0], "similarity-2", -similarity_base_threshold, human_reaction_threshold, plotting=True)
        continuous_results = continuous_results + human_reaction_threshold - 1

        similarity_base_threshold = 0.5
        z_scores = (self.similarities[0] - np.mean(self.similarities[0])
                    ) / np.std(self.similarities[0])  # z-value
        outliers = np.where(z_scores < -similarity_base_threshold)[0]
        sequences = np.split(outliers, np.where(np.diff(outliers) != 1)[0] + 1)

        # print(continuous_results)
        # print(sequences)

        for seq1, seq2 in zip(sequences[:-1], sequences[1:]):
            # and seq2[-1] - seq1[0] > (human_reaction_threshold * 2 - 1):
            if seq2[0] - seq1[-1] <= human_reaction_threshold:
                for res in continuous_results:
                    if np.any(np.isin(range(seq1[0], seq2[-1] + 1), range(res[0], res[1] + 1))):
                        logging.debug("Suspect flickers at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                            seq1[0], seq2[-1], seq1[0] / self.fps, seq2[-1] / self.fps))
                        # any? all?
                        if np.any(np.isin(range(seq1[0], seq2[-1] + 1), exemption_mask)):
                            logging.debug("Pass")
                        else:
                            logging.warning("Flickers at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                                seq1[0], seq2[-1], seq1[0] / self.fps, seq2[-1] / self.fps))

        logging.info("ok")
