import time
import logging
import numpy as np

from typing import List

from util.feature_extraction import Features
from util.processing import consine_similarity


def feature_extraction(video_path: str, enable_cache: bool, cache_dir: str) -> List[np.ndarray]:

    start_time = time.perf_counter()

    logging.info("Start flicker detection ..")
    logging.info("Video path: {}, cache directory: {}".format(
        video_path, cache_dir))

    video_features = Features(video_path, enable_cache, cache_dir)
    embeddings, suspects, horizontal_displacements, vertical_displacements = video_features.extract()

    logging.info("Start testing similarity ...")

    similarities = []
    window_size_max = 10
    for window_size in range(2, window_size_max + 1):
        compare_with_next = window_size - 1
        similarity = []
        for emb1, emb2 in zip(embeddings[:-(1+window_size_max)], embeddings[compare_with_next:-(1+window_size_max-compare_with_next)]):
            similarity.append(consine_similarity(emb1, emb2))
        similarities.append(similarity)
    similarities = np.array(similarities)

    end_time = time.perf_counter()
    logging.info("Execution takes {} second(s).".format(end_time - start_time))

    return [similarities, suspects, horizontal_displacements, vertical_displacements]


def flicker_detection(similarities, suspects, horizontal_displacements, vertical_displacements, human_reaction_threshold=3):

    # mean without outlier (outside 1 (std_degree) standard error)
    std_degree = 1
    horizontal_baseline = np.mean(horizontal_displacements[abs(
        horizontal_displacements - np.mean(horizontal_displacements)) < std_degree * np.std(horizontal_displacements)])
    vertical_baseline = np.mean(vertical_displacements[abs(
        vertical_displacements - np.mean(vertical_displacements)) < std_degree * np.std(vertical_displacements)])

    """
    If continuously smaller than or greater than -> scroll
    """
    scroll_tolerant_error = human_reaction_threshold
    status_window = np.zeros(scroll_tolerant_error).astype(
        int).tolist()  # 0 -> nothing | 1 -> greater | -1 -> smaller
    current_count = 0
    current_status = 0
    for i, h in enumerate(horizontal_displacements):
        if i in suspects:
            if h - horizontal_baseline > 0.5 * np.std(vertical_displacements):
                status = 1
            elif horizontal_baseline - h > 0.5 * np.std(vertical_displacements):
                status = -1
            else:
                status = 0
        else:
            status = 0
        if status != current_status and current_status not in status_window:  # change status
            if current_status and current_count > scroll_tolerant_error * 2:
                print("End = {}, Length = {}, status = {}".format(
                    i - len(status_window) - 1, current_count - len(status_window), current_status))
            current_status = status
            current_count = 0
            for s in status_window[::-1]:
                if s == status:
                    current_count += 1
        else:
            current_count += 1
        status_window.append(status)
        status_window.pop(0)

    for i, h in enumerate(horizontal_displacements):
        if i in suspects:
            if h > horizontal_baseline:
                print(i, 1)
            else:
                print(i, -1)
