import os
import cv2
import json
import logging
import numpy as np

import matplotlib.pyplot as plt


class Flicker:

    def __init__(
        self,
        fps: float,
        similarities: np.ndarray,
        suspects: np.ndarray,
        horizontal_displacements: np.ndarray,
        vertical_displacements: np.ndarray,
        img_dir: str,
    ) -> None:
        self.fps = fps
        self.similarities = similarities
        self.suspects = suspects
        self.horizontal_displacements = horizontal_displacements
        self.vertical_displacements = vertical_displacements
        self.__img_dir = img_dir

    def __moving_avg(self, data: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(data, np.ones(window), "valid") / window

    def __continuous_behaviour_detection(
        self,
        data: np.ndarray,
        description: str,
        z_score_threshold: float,
        human_reaction_threshold: float,
        plotting: str = False
    ) -> np.ndarray:

        moving_average = self.__moving_avg(data, human_reaction_threshold)
        z_scores = (moving_average - np.mean(moving_average)) / \
            np.std(moving_average)  # z-value
        z_score_threshold = z_score_threshold

        if z_score_threshold > 0:
            ma_outliers = np.where(z_scores > z_score_threshold)[0]
        else:
            ma_outliers = np.where(z_scores < z_score_threshold)[0]

        anything_detected = False
        results = []
        for seq in np.split(
                ma_outliers,
                np.where(np.diff(ma_outliers) != 1)[0] + 1
        ):
            if len(seq) > (human_reaction_threshold * 2 - 1):  # original + ma
                logging.debug(
                    "Continuous [{}] at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                        description, seq[0], seq[-1],
                        seq[0] / self.fps, seq[-1] / self.fps
                    )
                )
                anything_detected = True
                results.append([seq[0], seq[-1]])

        if not anything_detected:
            logging.debug("Cannot detect continuous [{}]".format(description))

        if plotting:  # dev only
            plt.figure(figsize=(16, 4), dpi=200)
            plt.plot(data)
            # plt.plot(self.similarities[human_reaction_threshold])
            plt.plot(moving_average)
            plt.legend(list(["Similarity", "Moving Average"]))
            for i in ma_outliers:
                plt.scatter(i, 1.0, s=1, c="r")
            # for i in range(0, len(data), 10):
            #     plt.vlines(
            #         i, ymin=int(np.min(data)), ymax=int(np.max(data)),
            #         linewidth=0.1, colors="red", linestyles="dashed"
            #     )

            plt.xlabel("Number of Frames")
            plt.ylabel("Similarity")
            plt.savefig(os.path.join(self.__img_dir, "cont_behav_detect.png"))

        return np.array(results)

    def flicker_detection(
            self,
            output_path,
            output: bool = True,
            human_reaction_threshold: int = 5
    ) -> dict:

        # need further tunning of reaction threshold

        logging.info("Start core function: flicker detection")

        logging.debug("Length: {}".format(len(self.similarities[0])))

        flickers = list()
        flicker_types = [
            "flickering",
            "discontinuous",
            "retoation",
            "imbalance"
        ]

        exemption_mask = list()

        # discontinuous
        logging.debug("Start testing discontinuous")
        movement_base_threshold = 1.0
        for seq in self.__continuous_behaviour_detection(
                self.horizontal_displacements,
                "right movement",
                movement_base_threshold,
                human_reaction_threshold
        ):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)
        for seq in self.__continuous_behaviour_detection(
                self.horizontal_displacements,
                "left movement",
                -movement_base_threshold,
                human_reaction_threshold
        ):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)
        for seq in self.__continuous_behaviour_detection(
                self.vertical_displacements,
                "up movement",
                -movement_base_threshold,
                human_reaction_threshold
        ):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)
        for seq in self.__continuous_behaviour_detection(
                self.vertical_displacements,
                "down movement",
                movement_base_threshold,
                human_reaction_threshold
        ):
            for idx in range(seq[0], seq[1] + 1):
                exemption_mask.append(idx)

        similarity_base_threshold = 0.1
        continuous_results = self.__continuous_behaviour_detection(
            self.similarities[0],
            "similarity-2",
            -similarity_base_threshold,
            human_reaction_threshold,
            plotting=True
        )
        continuous_results = continuous_results + human_reaction_threshold - 1

        similarity_base_threshold = 0.5
        z_scores = (self.similarities[0] - np.mean(self.similarities[0])) / \
            np.std(self.similarities[0])  # z-value
        outliers = np.where(z_scores < -similarity_base_threshold)[0]
        sequences = np.split(outliers, np.where(np.diff(outliers) != 1)[0] + 1)

        for seq1, seq2 in zip(sequences[:-1], sequences[1:]):
            # and seq2[-1] - seq1[0] > (human_reaction_threshold * 2 - 1):
            if seq2[0] - seq1[-1] <= human_reaction_threshold:
                for res in continuous_results:
                    if np.any(
                            np.isin(
                                range(seq1[0], seq2[-1] + 1),
                                range(res[0], res[1] + 1)
                            )
                    ):
                        logging.debug(
                            "Suspect flickers at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                                seq1[0], seq2[-1], seq1[0] / self.fps,
                                seq2[-1] / self.fps
                            )
                        )
                        # any? all?
                        if np.any(
                                np.isin(
                                    range(seq1[0], seq2[-1] + 1),
                                    exemption_mask
                                )
                        ):
                            logging.debug("Pass")
                        else:
                            logging.warning(
                                "discontinuous at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                                    seq1[0], seq2[-1], seq1[0] / self.fps,
                                    seq2[-1] / self.fps
                                )
                            )
                            flickers.append(dict({
                                "type": "discontinuous",
                                "frame": [
                                    int(seq1[0]),
                                    int(seq2[-1])
                                ],
                                "time": [
                                    "{:.2f}".format(seq1[0] / self.fps),
                                    "{:.2f}".format(seq2[-1] / self.fps)
                                ]
                            }))

        # flickering
        logging.debug("Start testing flickering")
        lookbefore_window = human_reaction_threshold
        suspects_sequences = np.split(
            self.suspects, np.where(np.diff(self.suspects) != 1)[0] + 1)

        for seq in suspects_sequences:
            sus_seq_min = np.min(seq)
            sus_seq_max = np.max(seq)
            logging.debug("testing: {}-{}".format(sus_seq_min, sus_seq_max))
            if sus_seq_min >= lookbefore_window and \
                    sus_seq_max < len(self.similarities[0]):
                similarity_baseline = 0
                cnt = 0
                for suspect in self.suspects:
                    if suspect < len(self.similarities[lookbefore_window]):
                        similarity_baseline += \
                            self.similarities[lookbefore_window][suspect]
                        cnt += 1
                similarity_baseline /= cnt
                reverse_times = 0
                current_direction = 0  # 0 -> up, 1 -> down
                for idx in range(
                        sus_seq_min - lookbefore_window,
                        sus_seq_max + 1
                ):
                    if not current_direction and \
                            self.similarities[lookbefore_window][idx] < similarity_baseline:
                        reverse_times += 1
                        current_direction = 1
                    elif current_direction and \
                            self.similarities[lookbefore_window][idx] > similarity_baseline:
                        reverse_times += 1
                        current_direction = 0
                if reverse_times >= 3 and not \
                        np.any(
                            np.isin(
                                range(sus_seq_min, sus_seq_max + 1),
                                exemption_mask
                            )
                        ):
                    logging.debug(
                        "Suspect flickering, check if overlap with discontinuous"
                    )
                    overlapped = False
                    for flicker in flickers:
                        if flicker["type"] == "discontinuous":
                            if np.any(
                                    np.isin(
                                        range(sus_seq_min, sus_seq_max + 1),
                                        range(
                                            flicker["frame"][0],
                                            flicker["frame"][1] + 1
                                        )
                                    )
                            ):
                                logging.debug("overlapped")
                                overlapped = True
                                break
                    if not overlapped:
                        logging.warning(
                            "flickering at frame {}-{} ({:.2f}s-{:.2f}s)".format(
                                sus_seq_min, sus_seq_max, sus_seq_min / self.fps,
                                sus_seq_max / self.fps
                            )
                        )
                        flickers.append(dict({
                            "type": "flickering",
                            "frame": [
                                int(sus_seq_min),
                                int(sus_seq_max)
                            ],
                            "time": [
                                "{:.2f}".format(sus_seq_min / self.fps),
                                "{:.2f}".format(sus_seq_max / self.fps)
                            ]
                        }))

        logging.info("Final flickers: {}".format(dict({"label": flickers})))

        if output:
            with open(output_path, "w") as f:
                json.dump(dict({"label": flickers}), f, indent=4)

        logging.info("ok")

        return dict({
            "labels": flickers,
            "similarity": np.round(self.similarities[0], 2).tolist()
        })


def fullscreen_same_color(frame: np.ndarray, threshold: float = 0.9) -> bool:
    frame = cv2.resize(
        frame,
        tuple((np.array(frame.shape) / 10)[1::-1].astype(int).tolist())
    )
    value, counts = np.unique(frame.reshape(-1, 3), return_counts=True, axis=0)
    """
    Here we assume that background is always rgb=(0,0,0)
    """
    value = value[1:]
    counts = counts[1:]
    if not len(value) or not len(counts):
        return False

    # print("ratio = {}".format(np.max(counts) / np.sum(counts)))

    if np.max(counts) > np.sum(counts) * threshold:
        return True
    return False
