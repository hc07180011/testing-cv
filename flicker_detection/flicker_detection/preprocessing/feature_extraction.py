import os
import gc
import cv2
import time
import hashlib
import logging
import numpy as np

from typing import List

from preprocessing.movement.brisk import Brisk
from preprocessing.embedding.facenet import Facenet
from preprocessing.tranformation.affine import Affine
from preprocessing.partition.pixel import Pixel
from util.utils import parse_fps, take_snapshots, euclidean_distance
from core.flicker import fullscreen_same_color


class Features:
    """Extracting frames from the given video

    Args:
        video_path (:obj:`str`): The path to the video file.
        limit (:obj:`int`, optional): Max #frames to take. Defaults to np.inf.

    Returns:
        np.ndarray: A numpy array that includes all frames

    """

    def __init__(
        self,
        facenet: Facenet,
        video_path: str,
        img_dir: str,
        enable_cache: bool,
        cache_dir: str
    ) -> None:
        self.facenet = facenet
        self.__video_path = video_path
        self.fps = parse_fps(self.__video_path)
        self.__img_dir = img_dir
        self.__cache_dir = cache_dir
        self.__enable_cache = enable_cache

    def __get_affine_types(
        self,
        affine: Affine,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> dict():
        """ Affine transformation example, pass now and might be used later
            video 007: 309, 310
        """
        transformation_results = affine.compare_transformation(
            frame1,
            frame2
        )
        return transformation_results

    def __get_partition_result(
        self,
        pixel: Pixel,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """ Pixel-wise distance example, pass now, video 001: 0, 1, 105, 106
        """
        pixel_result = pixel.get_heatmap(
            frame1,
            frame2,
            dump_dir=self.__img_dir
        )
        return pixel_result

    def __extract(self) -> List[np.ndarray]:

        md5 = hashlib.md5(open(self.__video_path, 'rb').read()).hexdigest()
        cache_data_path = os.path.join(self.__cache_dir, "{}.npz".format(md5))

        if self.__enable_cache and os.path.exists(cache_data_path):
            __cache = np.load(cache_data_path)
            embeddings, suspects, \
                horizontal_displacements, vertical_displacements = [
                    __cache[__cache.files[i]]
                    for i in range(len(__cache.files))
                ]
            logging.info("Cache exists, using cache")

        else:

            brisk = Brisk()

            vidcap = cv2.VideoCapture(self.__video_path)
            success, image = vidcap.read()
            last_frame = image
            last_embedding = self.facenet.get_embedding(image, batched=False)

            embeddings = []
            similarities = []
            horizontal_displacements = []
            vertical_displacements = []

            count = 0
            while success:
                embeddings.append(last_embedding)

                success, image = vidcap.read()

                try:
                    embedding = self.facenet.get_embedding(
                        image, batched=False)
                except Exception as e:
                    logging.debug("{}".format(repr(e)))
                    break

                similarities.append(
                    euclidean_distance(last_embedding, embedding)
                )
                delta_x, delta_y = brisk.calculate_movement(last_frame, image)
                horizontal_displacements.append(delta_x)
                vertical_displacements.append(delta_y)

                fullscreen_same_color(image)

                last_frame = image
                last_embedding = embedding
                logging.debug('Parsing image: #{:04d}'.format(count))
                count += 1

            embeddings = np.array(embeddings)
            similarities = np.array(similarities)
            similarity_baseline = np.mean(similarities)

            suspects = []
            for i in range(similarities.shape[0]):
                if similarities[i] < similarity_baseline:
                    suspects.append(i)
            suspects = np.array(suspects)
            horizontal_displacements = np.array(horizontal_displacements)
            vertical_displacements = np.array(vertical_displacements)

            if self.__enable_cache:
                np.savez(cache_data_path, embeddings, suspects,
                         horizontal_displacements, vertical_displacements)
                logging.debug("Cache save at: {} with size = {} bytes".format(
                    cache_data_path,
                    os.path.getsize(cache_data_path)
                ))

            gc.collect()
            logging.info("Delete frames and free the memory")

        return list([
            embeddings,
            suspects,
            horizontal_displacements,
            vertical_displacements
        ])

    def feature_extraction(self) -> None:

        start_time = time.perf_counter()

        logging.info("Start flicker detection ..")
        logging.info("Video path: {}, cache directory: {}".format(
            self.__video_path, self.__cache_dir))

        embeddings, suspects, \
            horizontal_displacements, vertical_displacements = self.__extract()

        logging.info("Start testing similarity ...")

        similarities = []
        window_size_max = 10
        for window_size in range(2, window_size_max + 1):
            compare_with_next = window_size - 1
            similarity = []
            for emb1, emb2 in zip(
                    embeddings[:-(1+window_size_max)],
                    embeddings[
                        compare_with_next:
                        -(1+window_size_max-compare_with_next)
                    ]):
                similarity.append(euclidean_distance(emb1, emb2))
            similarities.append(similarity)
        similarities = np.array(similarities)

        end_time = time.perf_counter()
        logging.info("Execution takes {} second(s).".format(
            end_time - start_time
        ))

        self.embeddings = embeddings
        self.similarities = similarities
        self.suspects = suspects
        self.horizontal_displacements = horizontal_displacements
        self.vertical_displacements = vertical_displacements

