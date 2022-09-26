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
from sklearn.decomposition import IncrementalPCA
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
            real_filename = key.replace("reduced_", "").replace(".npy", "")
            loaded = np.load(
                "{}.npy".format(os.path.join(
                    self.data_dir, key.replace(".npy", "")))
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


class VideoLoader(object):
    def __init__(
        self,
        vid_list: list,
        label_path: str,
        data_dir: str,
        chunk_size: int,
        batch_size: int,
        shape: tuple,
        sampler: Callable = None,
        mov: bool = False,
        norm: bool = False
    ) -> None:
        self.raw_labels = json.load(open(label_path, "r"))
        self.vid_list = self.to_process = vid_list
        self.data_dir = data_dir

        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.sampler = sampler
        self.mov = mov
        self.norm = norm

        self.pidx = shape[0]
        self.start_idx = self.end_idx = 0

        self.cache_x = np.zeros(shape)
        self.cache_y = np.zeros(shape[0])
        self.out_sequence = self.out_x = self.out_y = ()

    def __len__(self) -> int:
        return len(self.out_sequence)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pidx > len(self.out_sequence) and len(self.to_process) == 1:
            gc.collect()
            raise StopIteration

        if self.pidx > len(self.out_sequence):
            # TO DO fix me need to refresh cache with empty tensor
            # self.out_x.fill(0)
            # self.out_y.fill(0)
            self._load(self.to_process.tolist())
            self.out_x += self._chunking(
                self.cache_x,
                self.out_sequence,
                self.chunk_size
            )
            self.out_sequence = list(self.out_sequence)
            self.out_x, self.out_y = np.array(
                self.out_x), self.cache_y[self.out_sequence]
            if self.sampler and np.any(self.out_y) == 1:
                self.out_x, self.out_y = self._sampling(
                    self.out_x,
                    self.out_y,
                    self.sampler
                )
            self.pidx = 0
            random.shuffle(self.out_sequence)
            gc.collect()

        X = self.out_x[self.pidx:(self.pidx+self.batch_size) if (
            self.pidx + self.batch_size) < len(self.out_sequence) else -1]
        y = self.out_y[self.pidx:(self.pidx+self.batch_size) if (
            self.pidx + self.batch_size) < len(self.out_sequence) else -1]
        self.pidx += self.batch_size
        return torch.from_numpy(X), torch.from_numpy(y)

    def _load(
        self,
        videos: list
    ) -> Tuple[int, int]:
        repeated = ()
        while videos:
            vid = videos.pop(0)
            print("WTF", vid)
            if vid in repeated:
                break
            loaded = np.load(
                os.path.join(self.data_dir, vid)
            ).astype(np.uint8)

            flicker_idxs = np.array(
                self.raw_labels[vid.replace(
                    "reduced_", "").replace(".npy", "")],
                dtype=np.uint16) - 1
            buf_label = np.zeros(loaded.shape[0], dtype=np.uint16)
            print("labels", buf_label.shape, flicker_idxs)
            buf_label[flicker_idxs.tolist()] = 1

            self.start_idx += self.end_idx + self.chunk_size//2
            self.end_idx = self.start_idx + loaded.shape[0]

            print("buffer", self.end_idx, self.cache_x.shape[0])
            if self.end_idx > self.cache_x.shape[0]:
                videos += [vid]
                repeated += (vid,)
                # print("repeated", repeated)
                continue

            self.cache_y[self.start_idx:self.end_idx] = buf_label
            self.cache_x[self.start_idx:self.end_idx] = loaded

            self.out_sequence += tuple(np.array([
                tuple(range(i-self.chunk_size//2, i+self.chunk_size//2))
                for i in range(self.start_idx, self.end_idx) if self.cache_y[i] == 1
            ]).flatten().tolist())

            self.out_sequence += tuple(
                i for i in range(self.start_idx, self.end_idx)
                if i not in self.out_sequence
            )

        self.start_idx = self.end_idx = 0
        self.to_process = np.array(repeated)
        gc.collect()
        return self.start_idx, self.end_idx

    def _shuffle(self) -> None:
        random.shuffle(self.vid_list)
        self.to_process = self.vid_list
        gc.collect()

    @staticmethod
    def _chunking(
        X: np.ndarray,
        idx: list,
        chunk_size: int
    ) -> tuple:
        return tuple(
            X[i-chunk_size//2:(i+1)+chunk_size//2]
            for i in idx
        )

    @staticmethod
    def _sampling(
        X_train: np.array,
        y_train: np.array,
        sampler: Callable,
    ) -> Tuple[np.array, np.array]:
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
    def _mov_div(arr: np.ndarray) -> np.ndarray:
        arr = cv2.resize(
            arr,
            dsize=(*np.array(arr.shape)//2,),
            interpolation=cv2.INTER_CUBIC
        )
        arr = np.diff(arr, axis=0).astype(np.uint8)
        return np.apply_along_axis(
            lambda frame: (frame*(255/frame.max())).astype(np.uint8),
            axis=0, arr=arr)

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        arr = cv2.resize(
            arr,
            dsize=(*np.array(arr.shape)//2,),
            interpolation=cv2.INTER_CUBIC
        )
        return np.apply_along_axis(
            lambda frame: (frame - frame.mean())/frame.std().astype(np.uint8),
            axis=0, arr=arr)


if __name__ == "__main__":
    label_path = "../data/new_label.json"
    data_dir = "../data/meta_data"
    cache_path = "../.cache/train_test"
    mapping_path = "../data/mapping.json"
    __cache__ = np.load(
        "{}.npz".format(cache_path), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=7)

    vl = VideoLoader(
        vid_list=embedding_list_train,
        label_path=label_path,
        data_dir=data_dir,
        chunk_size=11,
        batch_size=256,
        shape=(10000, 380, 360, 3),
        sampler=None,
        mov=False,
        norm=False
    )
    ds_train = Streamer(
        embedding_list_train,
        label_path,
        mapping_path,
        data_dir,
        mem_split=10,
        chunk_size=11,
        batch_size=256,
        multiclass=False,
        sampler=sm,
        overlap_chunking=True,
        moving_difference=False
    )  # [('near_miss', nm), ('smote', sm)])
    for x, y in ds_train:
        print(x.shape, y.shape)
