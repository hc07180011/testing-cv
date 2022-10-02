import os
import json
import gc
import random
import cv2
import torch
import numpy as np
import pickle as pk
import multiprocessing as mp
import seaborn as sns
import logging
import skvideo.io
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple


class MYDS(Dataset):
    def __init__(
        self,
        x_paths: list,
        label_path: str,
        mapping_path: str,
        data_dir: str,
        chunk_size: int = 30,
    ):
        self.label_path = label_path
        self.mapping_path = mapping_path
        self.data_dir = data_dir
        self.chunk_size = chunk_size

        self.xs, self.ys = self.load_embeddings(
            x_paths
        )

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx: int):
        return self.xs[idx].astype(np.float32), self.ys[idx].astype(np.float32)

    def update(self, new_x_paths: list):
        self.xs, self.ys = self.load_embeddings(
            new_x_paths
        )

    @staticmethod
    def _get_chunk_array(input_arr: np.array, chunk_size: int) -> Tuple:
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
        return tuple(map(tuple, chunks))

    def load_embeddings(
        self,
        embedding_list_train: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        encoding_filename_mapping = json.load(open(self.mapping_path, "r"))
        raw_labels = json.load(open(self.label_path, "r"))
        X_train, y_train = (), ()
        for key in embedding_list_train:
            real_filename = encoding_filename_mapping[key.replace(
                ".npy", "")]
            loaded = np.load("{}.npy".format(os.path.join(self.data_dir, key.replace(
                ".npy", ""))))
            X_train += (*self._get_chunk_array(loaded, self.chunk_size),)
            flicker_idxs = np.array(raw_labels[real_filename]) - 1
            buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
            buf_label[flicker_idxs] = 1
            y_train += tuple(
                1 if sum(x) else 0
                for x in self._get_chunk_array(buf_label, self.chunk_size)
            )
        return np.array(X_train), np.asarray(y_train)


class Streamer(object):
    """
    https://jamesmccaffrey.wordpress.com/2021/03/08/working-with-huge-training-data-files-for-pytorch/
    """

    def __init__(self,
                 embedding_list_train: list,
                 label_path: str,
                 mapping_path: str,
                 data_dir: str,
                 mem_split: int,
                 chunk_size: int,
                 batch_size: int,
                 sampler: Callable = None,
                 ipca: Callable = None,
                 ipca_fitted: bool = False,
                 text_based: bool = False,
                 multiclass: bool = False,
                 overlap_chunking: bool = False,
                 moving_difference: bool = False,
                 ) -> None:
        self.text_based = text_based
        self.multiclass = multiclass
        self.overlap_chunking = overlap_chunking
        self.moving_difference = moving_difference

        self.embedding_list_train = embedding_list_train
        self.chunk_embedding_list = np.array_split(
            embedding_list_train, mem_split)
        self.data_dir = data_dir
        self.encoding_filename_mapping = json.load(open(mapping_path, "r"))
        self.raw_labels = json.load(open(label_path, "r"))

        self.mem_split = mem_split
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.sampler = sampler
        self.sampling_params = None
        self.ipca = ipca
        self.ipca_fitted = ipca_fitted

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
            if self.moving_difference:
                # print(type(self.X_buffer[0]), type(self.y_buffer[0]))
                self._load_embeddings(
                    self.chunk_embedding_list[self.cur_chunk],
                    self.moving_difference
                )
            self.cur_chunk += 1
            X, y = self._re_sample()
            self.X_buffer, self.y_buffer = self._batch_sample(
                X, y, self.batch_size)
            gc.collect()

        X, y = self.X_buffer.pop(), self.y_buffer.pop()
        idx = np.arange(X.shape[0]) - 1
        random.shuffle(idx)
        return torch.from_numpy(X[idx]).long() if self.text_based else torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx]).long()

    def _re_sample(self,) -> Tuple[np.ndarray, np.ndarray]:
        X, y = np.array(self.X_buffer), np.array(self.y_buffer)
        if self.sampler is None and self.ipca is None or not np.any(np.array(self.y_buffer)) == 1:
            return X, y

        if self.sampler:
            return self._sampling(X, y, self.sampler)

        if self.ipca is not None and self.ipca_fitted:
            x_origin = X.shape
            X = self.ipca.inverse_transform(self.ipca.transform(
                np.reshape(X, (-1, np.prod(x_origin[1:])))
            ))
            X = np.reshape(X, (-1,) + x_origin[1:])
        return X, y

    def _fit_ipca(self, dest: str) -> str:
        if self.ipca is None:
            raise NotImplementedError
        for x, _ in self:
            x_origin = x.shape
            self.ipca.partial_fit(np.reshape(x, (-1, np.prod(x_origin[1:]))))
        self.ipca_fitted = True
        pk.dump(self.ipca, open(f"{dest}", "wb"))
        return "Explained variance: {}".format(self.ipca.explained_variance_)

    def _load_embeddings(
        self,
        embedding_list_train: list,
        mov_dif: bool = False,
    ) -> None:
        for key in embedding_list_train:
            # print(f"{key}")
            # real_filename = self.encoding_filename_mapping[key.replace(
            #     ".npy", "")]
            real_filename = key.replace("reduced_", "").replace(".mp4", "")
            loaded = np.load(
                "{}".format(os.path.join(
                    self.data_dir, key))
            )
            if mov_dif:
                loaded = self._mov_dif_chunks(loaded)

            flicker_idxs = np.array(
                self.raw_labels[real_filename], dtype=np.uint16) - 1
            print("LABELS ", loaded.shape, flicker_idxs)
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

    def plot_dist(self, dest: str) -> None:
        """
        FIX ME bugged
        """
        import matplotlib.pyplot as plt
        self._load_embeddings(
            self.chunk_embedding_list[self.cur_chunk])
        self.cur_chunk += 1
        X, y = self._re_sample()
        original_X_shape = X.shape
        sns.displot(np.reshape(X, (-1, np.prod(original_X_shape[1:]))))
        plt.savefig(f'{dest}')

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
        non_flicker_dir: str,
        flicker_dir: str,
        labels: dict,
        batch_size: int,
        in_mem_batches: int,
    ) -> None:
        self.labels = labels
        self.batch_size = batch_size
        self.batch_idx = self.cur_batch = 0
        self.in_mem_batches = in_mem_batches

        self.non_flicker_dir = non_flicker_dir
        self.flicker_dir = flicker_dir
        self.non_flicker_lst = os.listdir(non_flicker_dir)
        self.flicker_lst = os.listdir(flicker_dir)

        self.manager = mp.Manager()
        self.producer_q = self.manager.Queue()
        self.out_x = self.manager.Queue()
        self.out_y = self.manager.Queue()
        self.lock = self.manager.Lock()
        self.event = mp.Event()

    def __len__(self) -> int:
        return len(self.non_flicker_lst) // ((self.batch_size//2)*self.in_mem_batches) + 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        print("EPOCH STOP CONDITION ", self.batch_idx, len(self.non_flicker_lst))
        if self.batch_idx > len(self.non_flicker_lst):
            gc.collect()
            raise StopIteration

        if not bool(self.cur_batch):
            self.event.set()
            self.event = mp.Event()

            non_flickers = [
                os.path.join(self.non_flicker_dir,
                             self.non_flicker_lst[i % len(self.non_flicker_lst)])
                for i in range(self.batch_idx, self.batch_idx+(self.batch_size//2)*self.in_mem_batches)
            ]
            flickers = [
                os.path.join(self.flicker_dir,
                             self.flicker_lst[i % len(self.flicker_lst)])
                for i in range(self.batch_idx, self.batch_idx+(self.batch_size//2)*self.in_mem_batches)
            ]
            chunk_lst = non_flickers + flickers
            random.shuffle(chunk_lst)
            self._load(chunk_lst)
            self.cur_batch = self.in_mem_batches
            self.batch_idx = self.batch_idx + \
                (self.batch_size//2)*self.in_mem_batches
            gc.collect()

        self.cur_batch -= 1
        X, y = self.out_x.get(), self.out_y.get()
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.uint8))

    def _shuffle(self) -> None:
        random.shuffle(self.non_flicker_lst)
        random.shuffle(self.flicker_lst)
        gc.collect()

    def _load(
        self,
        chunk_lst: list,
    ) -> tuple:
        split = np.split(np.array(chunk_lst), self.in_mem_batches)
        producers = tuple(
            mp.Process(
                target=self._producers,
                args=(chunk, self.labels, self.producer_q, self.lock)
            )
            for _, chunk in zip(range(self.in_mem_batches), split)
        )
        consumers = tuple(mp.Process(
            target=self._consumers,
            args=(self.producer_q, self.out_x, self.out_y,
                  self.batch_size, self.lock, self.event))
            for _ in range(os.cpu_count()-self.in_mem_batches)
        )
        for c in consumers:
            c.daemon = True
            c.start()

        for p in producers:
            p.start()
        for p in producers:
            p.join()
        for p in producers:
            p.close()

        # for c in consumers:
        #     c.kill()
        print("LOADED")

    @staticmethod
    def _producers(
        cur_batch_lst: list,
        labels: dict,
        producer_q: mp.Queue,
        lock: mp.Lock,
    ) -> None:
        for path in cur_batch_lst:
            idx, vid_name = path.split(
                "/")[3].replace(".mp4", "").split("_", 1)
            producer_q.put(
                (int(idx in labels[vid_name]), skvideo.io.vread(path)))
            with lock:
                print(f"PRODUCER {vid_name} {os.getpid()}")

    @staticmethod
    def _consumers(
        q: mp.Queue,
        out_x: mp.Queue,
        out_y: mp.Queue,
        batch_size: int,
        lock: mp.Lock,
        event: mp.Event
    ) -> None:
        X = y = ()
        while not event.is_set():
            if len(X) == len(y) == batch_size:
                out_x.put(np.array(X))
                out_y.put(np.array(y))
                X = y = ()
                with lock:
                    print(f"CONSUMER NEW BATCH {os.getpid()}")
            label, input = q.get()
            X += (input,)
            y += (label,)


if __name__ == "__main__":
    label_path = "../data/new_label.json"
    data_dir = "../data/augmented"
    cache_path = "../.cache/train_test"
    mapping_path = "../data/mapping.json"
    flicker_dir = "../data/flicker-chunks"
    non_flicker_dir = "../data/meta-data"
    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)
    labels = json.load(open(label_path, "r"))
    ds_train = Loader(
        non_flicker_dir=non_flicker_dir,
        flicker_dir=flicker_dir,
        labels=labels,
        batch_size=256,
        in_mem_batches=os.cpu_count()-2
    )
    for x, y in ds_train:
        print("OUTPUT SHAPE", x.shape, y.shape)
