import os
import json
import gc
import tqdm
import random
import psutil
import logging
import itertools
import cv2
import numpy as np
import skvideo.io
import torch
import torchvision
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import make_dataset

from typing import Callable, Tuple
from decord import VideoLoader
from decord import cpu, gpu


class Streamer(object):
    """
    https://jamesmccaffrey.wordpress.com/2021/03/08/working-with-huge-training-data-files-for-pytorch/
    """

    def __init__(self,
                 embedding_list_train: list,
                 label_path: str,
                 data_dir: str,
                 mem_split: int,
                 chunk_size: int,
                 batch_size: int,
                 sampler: Callable = None,
                 multiclass: bool = False,
                 overlap_chunking: bool = False,
                 ) -> None:
        self.multiclass = multiclass
        self.overlap_chunking = overlap_chunking

        self.embedding_list_train = embedding_list_train
        self.chunk_embedding_list = np.array_split(
            embedding_list_train, mem_split)
        self.data_dir = data_dir
        self.raw_labels = json.load(open(label_path, "r"))

        self.mem_split = mem_split
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.sampler = sampler
        self.sampling_params = None

        self.cur_chunk = 0
        self.X_buffer, self.y_buffer = (), ()

    def __len__(self) -> int:
        # FIX ME
        return len(self.embedding_list_train)*len(self.chunk_embedding_list)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.X_buffer or not self.y_buffer) and self.cur_chunk == len(self.chunk_embedding_list):
            gc.collect()
            raise StopIteration

        if (not self.X_buffer or not self.y_buffer):
            self._load_embeddings(
                self.chunk_embedding_list[self.cur_chunk])

            self.cur_chunk += 1
            X, y = self._re_sample()
            self.X_buffer, self.y_buffer = self._batch_sample(
                X, y, self.batch_size)
            gc.collect()

        X, y = self.X_buffer.pop(), self.y_buffer.pop()
        idx = np.arange(X.shape[0]) - 1
        random.shuffle(idx)
        return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx]).long()

    def _re_sample(self,) -> Tuple[np.ndarray, np.ndarray]:
        X, y = np.array(self.X_buffer), np.array(self.y_buffer)
        if self.sampler is None and self.ipca is None or not np.any(np.array(self.y_buffer)) == 1:
            return X, y

        if self.sampler:
            return self._sampling(X, y, self.sampler)

        return X, y

    def _load_embeddings(
        self,
        embedding_list_train: list,
        mov_dif: bool = False,
    ) -> None:
        for key in embedding_list_train:
            real_filename = key.replace("reduced_", "").replace(".npy", "")
            loaded = np.load(
                "{}".format(os.path.join(
                    self.data_dir, key))
            )

            flicker_idxs = np.array(
                self.raw_labels[real_filename], dtype=np.uint16) - 1
            if self.overlap_chunking:
                self.X_buffer += (*self._overlap_chunks(loaded,
                                  flicker_idxs, self.chunk_size),)
                self.y_buffer += (1,)*flicker_idxs.size
                loaded = np.delete(loaded, flicker_idxs, axis=0)
                flicker_idxs = np.array([])

            buf_label = np.zeros(loaded.shape[0])
            buf_label[flicker_idxs.tolist()] = 1
            self.X_buffer += (*self._get_chunk_array(loaded,
                                                     self.chunk_size),)
            self.y_buffer += tuple(
                sum(x) if self.multiclass else 1 if sum(x) else 0
                for x in self._get_chunk_array(buf_label, self.chunk_size)
            )
            gc.collect()

    def _shuffle(self) -> None:
        random.shuffle(self.embedding_list_train)
        self.chunk_embedding_list = np.array_split(
            self.embedding_list_train, self.mem_split)
        self.cur_chunk = 0
        self.X_buffer, self.y_buffer = (), ()
        gc.collect()

    @staticmethod
    def _mov_dif_chunks(
        input_arr: np.ndarray,
    ) -> np.ndarray:
        difference = np.diff(input_arr, axis=-1)
        return (255*(difference - np.min(difference))/np.ptp(difference)).astype(np.int8)

    @staticmethod
    def _overlap_chunks(
        input_arr: np.ndarray,
        labels: np.ndarray,
        chunk_size: int
    ) -> np.ndarray:
        vid_pad = np.zeros(
            (input_arr.shape[0]+chunk_size, *input_arr.shape[1:]))
        vid_pad[chunk_size//2:-chunk_size//2] = input_arr
        return np.array([
            vid_pad[idx:idx+chunk_size]
            for idx in labels
        ])

    @staticmethod
    def _get_chunk_array(input_arr: np.array, chunk_size: int) -> list:
        chunks = np.array_split(
            input_arr,
            list(range(
                chunk_size,
                input_arr.shape[0] + 1,
                chunk_size
            ))
        )
        i_pad = np.zeros(chunks[0].shape)
        i_pad[:len(chunks[-1])] = chunks[-1]
        chunks[-1] = i_pad
        return chunks

    @staticmethod
    def _sampling(
        X_train: np.array,
        y_train: np.array,
        sampler: Callable,
    ) -> Tuple[np.array, np.array]:
        """
        batched alternative:
        https://imbalanced-learn.org/stable/references/generated/imblearn.keras.BalancedBatchGenerator.html
        """
        if isinstance(sampler, list):
            sampler = Pipeline(sampler)
        original_X_shape = X_train.shape
        X_train, y_train = sampler.fit_resample(
            np.reshape(X_train, (-1, np.prod(original_X_shape[1:]))),
            y_train
        )
        X_train = np.reshape(X_train, (-1,) + original_X_shape[1:])
        return X_train, y_train

    @staticmethod
    def _batch_sample(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = [
            X[i:i+batch_size]
            for i in range(0, len(X), batch_size)
        ]
        y = [
            y[i:i+batch_size]
            for i in range(0, len(y), batch_size)
        ]
        return X, y


class Loader(object):
    def __init__(
        self,
        non_flicker_lst: str,
        flicker_lst: str,
        non_flicker_dir: str,
        flicker_dir: str,
        labels: dict,
        batch_size: int,
        shape: tuple,
        in_mem_batches: int,
    ) -> None:
        self.labels = labels
        self.batch_size = batch_size
        self.shape = shape
        self.batch_idx = self.cur_batch = 0
        self.in_mem_batches = in_mem_batches

        self.non_flicker_lst = [os.path.join(
            non_flicker_dir, f) for f in non_flicker_lst]
        self.flicker_lst = [os.path.join(flicker_dir, f) for f in flicker_lst]

        self.flicker_vids = self._load(self.flicker_lst, self.shape)
        self.out_x = None
        self.out_idxs = list(range(self.batch_size))

    def __len__(self) -> int:
        return len(self.non_flicker_lst) // ((self.batch_size//2)*self.in_mem_batches) + 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx > len(self.non_flicker_lst):
            gc.collect()
            raise StopIteration

        if not bool(self.cur_batch):
            non_flickers = [
                self.non_flicker_lst[i % len(self.non_flicker_lst)]
                for i in range(self.batch_idx, self.batch_idx+(self.batch_size//2)*self.in_mem_batches)
            ]
            random.shuffle(non_flickers)
            self.out_x = self._load(
                non_flickers, self.shape)
            self.cur_batch = self.in_mem_batches
            self.batch_idx = self.batch_idx + \
                (self.batch_size//2)*self.in_mem_batches
            gc.collect()

        start_idx = (self.in_mem_batches - self.cur_batch)*self.batch_size//2
        self.cur_batch -= 1
        cur_batch_idxs = list(range(start_idx, start_idx + self.batch_size//2))

        non_flicker_X = self.out_x[self._idx_mapping(
            len(self.out_x), cur_batch_idxs)]
        flicker_X = self.flicker_vids[self._idx_mapping(len(
            self.flicker_vids), cur_batch_idxs)]

        X = np.vstack((non_flicker_X, flicker_X))
        y = np.array([0]*(self.batch_size//2) + [1] *
                     (self.batch_size//2), dtype=np.uint8)
        random.shuffle(self.out_idxs)
        return X[self.out_idxs], y[self.out_idxs]

    def shuffle(self) -> None:
        random.shuffle(self.non_flicker_lst)
        np.random.shuffle(self.flicker_vids)
        gc.collect()

    @staticmethod
    def _idx_mapping(
        arr_length: int,
        cur_batch_idxs: list
    ) -> list:
        return list(map(lambda x: x % arr_length, cur_batch_idxs))

    @staticmethod
    def _load(
        vid_lst: list,
        shape: tuple
    ) -> np.ndarray:
        logging.info("LOADING from storage..")
        vl = VideoLoader(
            vid_lst,
            ctx=list(map(cpu, range(os.cpu_count()))),
            shape=shape,
            interval=0,
            skip=0,
            shuffle=1
        )
        return np.array([
            chunk[0].asnumpy()
            for chunk in vl  # tqdm.tqdm(vl)
        ], dtype=np.uint8)
        # print(loaded.shape)
        # return loaded.reshape((len(vid_lst)*shape[0], *shape[1:]))
        # return np.array([
        #     skvideo.io.vread(path)
        #     for path in tqdm.tqdm(vid_lst)
        # ], dtype=np.uint8)


def cpu_stats() -> None:
    # print(sys.version)
    print("CPU USAGE - ", psutil.cpu_percent())
    print("MEMORY USAGE - ", psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    # memory use in GB...I think
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('memory GB:', memoryUse)


def test_mem() -> None:
    input = ()
    for vid in tqdm.tqdm(os.listdir(non_flicker_dir)):
        loaded = skvideo.io.vread(os.path.join(non_flicker_dir, vid))
        input += (loaded,)
        cpu_stats()


if __name__ == "__main__":
    label_path = "../data/new_label.json"
    data_dir = "../data/augmented"
    cache_path = "../.cache/train_test"
    mapping_path = "../data/mapping.json"
    flicker_dir = "../data/flicker-chunks"
    non_flicker_dir = "../data/meta-data"

    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    flicker_train, non_flicker_train, fp_test, flicker_test, non_flicker_test = tuple(
        __cache__[lst] for lst in __cache__)
    labels = json.load(open(label_path, 'rb'))

    ds_train = Loader(
        non_flicker_lst=non_flicker_train,
        flicker_lst=flicker_train,
        non_flicker_dir=non_flicker_dir,
        flicker_dir=flicker_dir,
        labels=labels,
        batch_size=32,
        shape=(12, 360, 360, 3),
        in_mem_batches=10  # os.cpu_count()-4
    )
    for x, y in ds_train:
        print("OUTPUT SHAPE", x.shape, y.shape)
    test_mem()
