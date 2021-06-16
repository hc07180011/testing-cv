import time
import logging
import numpy as np

from typing import List

from util.feature_extraction import Features
from util.processing import consine_similarity


def flicker_detection(video_path: str, enable_cache: bool, cache_dir: str) -> List[np.ndarray]:

    start_time = time.perf_counter()

    logging.info("Start flicker detection ..")
    logging.info("Video path: {}, cache directory: {}".format(
        video_path, cache_dir))

    video_features = Features(video_path, enable_cache, cache_dir)
    embeddings, suspects, horizontal_displacements, vertical_displacements = video_features.extract()

    logging.info("Start testing similarity ...")

    similarities2 = []
    for emb1, emb2 in zip(embeddings[:-5], embeddings[1:-4]):
        similarities2.append(consine_similarity(emb1, emb2))
    similarities2 = np.array(similarities2)
    similarities6 = []
    for emb1, emb2 in zip(embeddings[:-5], embeddings[5:]):
        similarities6.append(consine_similarity(emb1, emb2))
    similarities6 = np.array(similarities6)

    # plt.figure(figsize=(16, 4), dpi=1000)
    # plt.scatter(0, 1, c="r", s=1, alpha=0.5, label="Horizontal")
    # plt.scatter(0, 1, c="b", s=1, alpha=0.5, label="Vertical")

    # for i in [5, 3, 1]:
    #     similarities = []
    #     for emb1, emb2 in zip(embeddings[:-i-10], embeddings[i:-10]):
    #         similarities.append(consine_similarity(emb1, emb2))
    #     plt.plot(similarities, label="window size = {}".format(
    #         i+1), linewidth=(0.5 if i != 1 else 1), alpha=(0.5 if i != 1 else 0.8))
    # similarities = np.array(similarities)

    # similarity_baseline = np.mean(similarities)
    # plt.plot(similarity_baseline * np.ones(len(similarities)),
    #          label="Threshold", linewidth=0.3, alpha=0.5)

    # __tmp_x = np.clip(np.array(horizontal_displacements) /
    #                   10.0 + 1.0, 0.9, 1.1)
    # __tmp_y = np.clip(np.array(vertical_displacements) / 10.0 + 1.0, 0.9, 1.1)
    # logging.debug("Threshold = {}".format(similarity_baseline))
    # for i, suspect in enumerate(suspects):
    # logging.debug("{}: {:.4f}, x={:.4f} y={:.4f}".format(
    #     suspect, similarities[suspect], horizontal_displacements[i], vertical_displacements[i]))
    # plt.scatter(suspect, __tmp_x[i], c="r", s=1, alpha=0.5)
    # plt.scatter(suspect, __tmp_y[i], c="b", s=1, alpha=0.5)

    # plt.legend()
    # plt.savefig("Result.png")

    end_time = time.perf_counter()
    logging.info("Execution takes {} second(s).".format(end_time - start_time))
    print("Execution takes {} second(s).".format(end_time - start_time))

    return [similarities2, similarities6, suspects, horizontal_displacements, vertical_displacements]
