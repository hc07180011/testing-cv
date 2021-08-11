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
        self.fps = parse_fps(self.__video_path)
        self.__cache_dir = cache_dir
        self.__enable_cache = enable_cache

    def __extract(self) -> List[np.ndarray]:

        md5 = hashlib.md5(open(self.__video_path, 'rb').read()).hexdigest()
        cache_data_path = os.path.join(self.__cache_dir, "{}.npz".format(md5))

        if self.__enable_cache and os.path.exists(cache_data_path):
            __cache = np.load(cache_data_path)
            embeddings, suspects, horizontal_displacements, vertical_displacements = [
                __cache[__cache.files[i]] for i in range(len(__cache.files))]
            logging.info("Cache exists, using cache")

        else:

            facenet = Facenet()
            brisk = Brisk()

            vidcap = cv2.VideoCapture(self.__video_path)
            success, image = vidcap.read()
            last_frame = image
            last_embedding = facenet.get_embedding(image, batched=False)

            embeddings = []
            similarities = []
            horizontal_displacements = []
            vertical_displacements = []

            count = 0
            while success:
                embeddings.append(last_embedding)

                success, image = vidcap.read()

                try:
                    embedding = facenet.get_embedding(image, batched=False)
                except:
                    break

                similarities.append(euclidean_distance(
                    last_embedding, embedding))
                delta_x, delta_y = brisk.calculate_movement(last_frame, image)
                horizontal_displacements.append(delta_x)
                vertical_displacements.append(delta_y)

                last_frame = image
                last_embedding = embedding
                logging.debug('Parsing image: #{:04d}'.format(count))
                count += 1

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
                    cache_data_path, os.path.getsize(cache_data_path)))

            gc.collect()
            logging.info("Delete frames and free the memory")

            """ Affine transformation example, pass now and might be used later
                video 007: 309, 310
            """
            # affine = Affine()
            # with open("rotation.txt", "w") as f:
            #     for i in range(len(processing_frames) - 1):
            #         print(i, end=" ")
            #         transformation_results = affine.compare_transformation(
            #             processing_frames[i], processing_frames[i + 1])
            #         f.write("{:04d} {:04.2f} {}\n".format(
            #             i, i / self.fps, transformation_results["rotate"][0]))
            # exit()
            """ Pixel-wise distance example, pass now, video 001: 0, 1, 105, 106
            """
            # pixel = Pixel()
            # for i in range(len(processing_frames) - 1):
            #     pixel.get_heatmap(
            #         processing_frames[i], processing_frames[i+1], output=True, dump_dir="./dump")
            # from util.gif import gen_gif_temp
            # gen_gif_temp("dump")
            # exit()

            """ Embedding & no Embedding experiments
            """
            # import cv2
            # import matplotlib.pyplot as plt
            # facenet = Facenet()
            # row, col, _ = processing_frames[0].shape
            # times1 = []
            # times2 = []
            # for i in range(1, 60):
            #     a = cv2.resize(
            #         processing_frames[0], (int(row / i), int(col / i)))
            #     b = cv2.resize(
            #         processing_frames[1], (int(row / i), int(col / i)))
            #     s = time.perf_counter()
            #     b1 = facenet.get_embedding(a, batched=False)
            #     b2 = facenet.get_embedding(b, batched=False)
            #     for __ in range(100):
            #         ___ = euclidean_distance(b1, b2)
            #     times1.append((time.perf_counter() - s) / 100)
            #     s = time.perf_counter()
            #     for __ in range(100):
            #         ___ = euclidean_distance(a, b)
            #     times2.append((time.perf_counter() - s) / 100)
            # plt.plot(times1)
            # plt.plot(times2)
            # plt.legend(["Embedding", "No Embedding"])
            # plt.xlabel("Compressing Scale")
            # plt.ylabel("Time")
            # plt.savefig("embedding.png")
            # exit()

        return [embeddings, suspects, horizontal_displacements, vertical_displacements]

    def feature_extraction(self):

        start_time = time.perf_counter()

        logging.info("Start flicker detection ..")
        logging.info("Video path: {}, cache directory: {}".format(
            self.__video_path, self.__cache_dir))

        embeddings, suspects, horizontal_displacements, vertical_displacements = self.__extract()

        logging.info("Start testing similarity ...")

        similarities = []
        window_size_max = 10
        for window_size in range(2, window_size_max + 1):
            compare_with_next = window_size - 1
            similarity = []
            for emb1, emb2 in zip(embeddings[:-(1+window_size_max)], embeddings[compare_with_next:-(1+window_size_max-compare_with_next)]):
                similarity.append(euclidean_distance(emb1, emb2))
            similarities.append(similarity)
        similarities = np.array(similarities)

        end_time = time.perf_counter()
        logging.info("Execution takes {} second(s).".format(
            end_time - start_time))

        self.embeddings = embeddings
        self.similarities = similarities
        self.suspects = suspects
        self.horizontal_displacements = horizontal_displacements
        self.vertical_displacements = vertical_displacements
