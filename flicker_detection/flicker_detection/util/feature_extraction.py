import os
import gc
import hashlib
import logging
import numpy as np

from typing import List

from motion.brisk import Brisk
from similarity.facenet import Facenet
from util.processing import take_snapshots, consine_similarity


class Features:
    """Extracting frames from the given video

    Args:
        video_path (:obj:`str`): The path to the video file.
        limit (:obj:`int`, optional): Maximum frame to take. Defaults to np.inf.

    Returns:
        np.ndarray: A numpy array that includes all frames

    """

    def __init__(self, video_path: str, enable_cache: bool, cache_dir: str) -> None:
        self.__video_path = video_path
        self.__cache_dir = cache_dir
        self.__enable_cache = enable_cache

    def extract(self) -> List[np.ndarray]:

        md5 = hashlib.md5(open(self.__video_path, 'rb').read()).hexdigest()
        cache_data_path = os.path.join(self.__cache_dir, "{}.npz".format(md5))

        if self.__enable_cache and os.path.exists(cache_data_path):
            __cache = np.load(cache_data_path)
            embeddings, suspects, horizontal_displacements, vertical_displacements = [
                __cache[__cache.files[i]] for i in range(len(__cache.files))]
            logging.info("Cache exists, using cache")

        else:

            processing_frames = take_snapshots(self.__video_path)
            logging.info("Snapshots OK! Got {} frames with shape: {}".format(
                processing_frames.shape[0], processing_frames.shape[1:]))

            logging.info("Start embedding ...")
            facenet = Facenet()
            embeddings = facenet.get_embedding(processing_frames)
            logging.info("Embedding OK!")

            logging.info("Start BRISK algorithm ...")
            brisk = Brisk()

            similarities = []
            for emb1, emb2 in zip(embeddings[:-1], embeddings[1:]):
                similarities.append(consine_similarity(emb1, emb2))
            similarities = np.array(similarities)
            similarity_baseline = np.mean(similarities)

            suspects = []
            horizontal_displacements = []
            vertical_displacements = []
            for i, (frame1, frame2) in enumerate(zip(processing_frames[:-1], processing_frames[1:])):
                if similarities[i] < similarity_baseline:
                    suspects.append(i)
                delta_x, delta_y = brisk.calculate_movement(frame1, frame2)
                horizontal_displacements.append(delta_x)
                vertical_displacements.append(delta_y)
            suspects = np.array(suspects)
            horizontal_displacements = np.array(horizontal_displacements)
            vertical_displacements = np.array(vertical_displacements)
            logging.info("BRISK OK!")

            if self.__enable_cache:
                np.savez(cache_data_path, embeddings, suspects,
                         horizontal_displacements, vertical_displacements)
                logging.debug("Cache save at: {} with size = {} bytes".format(
                    cache_data_path, os.path.getsize(cache_data_path)))

            del processing_frames
            gc.collect()
            logging.info("Delete frames and free the memory")

        return [embeddings, suspects, horizontal_displacements, vertical_displacements]
