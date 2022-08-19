import os
import json
import gc
import random
import numpy as np
import pickle as pk
import torch
import tensorflow as tf
import multiprocessing as mp
import seaborn as sns
import logging
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
            # get flicker frame indexes
            flicker_idxs = np.array(raw_labels[real_filename]) - 1
            # buffer zeros array frame video embedding
            buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
            # set indexes in zeros array based on flicker frame indexes
            buf_label[flicker_idxs] = 1
            y_train += tuple(
                1 if sum(x) else 0
                for x in self._get_chunk_array(buf_label, self.chunk_size)
            )  # consider using tf reduce sum for multiclass
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
                 mem_split: int = 8,
                 chunk_size: int = 30,
                 batch_size: int = 32,
                 sampler: Callable = None,
                 ipca: Callable = None,
                 ipca_fitted: bool = False,
                 keras: bool = False,
                 multiclass: bool = False,
                 ) -> None:
        self.keras = keras
        self.multiclass = multiclass
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
            self.cur_chunk += 1
            X, y = self._re_sample()
            self.X_buffer, self.y_buffer = self._batch_sample(
                X, y, self.batch_size)
            gc.collect()

        X, y = self.X_buffer.pop(), self.y_buffer.pop()
        idx = np.arange(X.shape[0]) - 1
        random.shuffle(idx)
        input = torch.from_numpy(X[idx]).float()\
            if len(torch.from_numpy(X[idx]).float().shape) >= 3 else\
            torch.unsqueeze(torch.from_numpy(X[idx]).float(), -1)
        if self.keras:
            return tf.convert_to_tensor(X[idx], dtype=tf.float32), tf.convert_to_tensor(y[idx], dtype=tf.float32)
        return input, torch.from_numpy(y[idx]).long()

    def _re_sample(self,) -> None:
        if self.sampler is None and self.ipca is None:
            return np.array(self.X_buffer), np.array(self.y_buffer)

        if self.sampler is not None:
            X, y = self._sampling(np.array(self.X_buffer),
                                  np.array(
                self.y_buffer),
                self.sampler)
        if self.ipca is not None and self.ipca_fitted:
            X, y = np.array(self.X_buffer), np.array(self.y_buffer)
            x_origin = X.shape
            X = self.ipca.transform(np.reshape(X, (-1, np.prod(x_origin[1:]))))
            X = self.ipca.inverse_transform(X)
            X = np.reshape(X, (-1,) + x_origin[1:])
        return X, y

    def _fit_ipca(self, dest: str) -> None:
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
    ) -> None:
        for key in embedding_list_train:
            real_filename = self.encoding_filename_mapping[key.replace(
                ".npy", "")]
            loaded = np.load(
                "{}.npy".format(os.path.join(
                    self.data_dir, key.replace(".npy", "")))
            )
            self.X_buffer += (*self._get_chunk_array(loaded, self.chunk_size),)
            flicker_idxs = np.array(self.raw_labels[real_filename]) - 1
            buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
            buf_label[flicker_idxs] = 1
            self.y_buffer += tuple(
                # sum(x)  # FIX ME
                # for x in self._get_chunk_array(buf_label, self.chunk_size)
                sum(x) if self.multiclass else 1 if sum(x) else 0 for x in self._get_chunk_array(buf_label, self.chunk_size)
            )

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
    ) -> Tuple[list, list]:
        X = [
            X[i:i+batch_size]
            for i in range(0, len(X), batch_size)
        ]
        y = [
            y[i:i+batch_size]
            for i in range(0, len(y), batch_size)
        ]
        return X, y


def test_MYDS(
    embedding_list_train: list,
    embedding_list_val: list,
    embedding_list_test: list,
    label_path: str,
    mapping_path: str,
    data_dir: str,
) -> None:
    d_train = MYDS(embedding_list_train, label_path, mapping_path, data_dir)
    ds = MYDS(embedding_list_val, label_path, mapping_path, data_dir)
    ds = MYDS(embedding_list_test, label_path, mapping_path, data_dir)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=16)

    t_sample = 0
    for batch_idx, (x, y) in enumerate(dl):
        print(batch_idx, x.shape, y.shape)
        t_sample += y.shape[0]
    print(t_sample)


if __name__ == '__main__':
    label_path = "../data/new_label.json"
    mapping_path = "../data/mapping_test.json"
    data_dir = "../data/vgg16_emb/"
    __cache__ = np.load("{}.npz".format(
        "../.cache/train_test"), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    chunk_size = 5
    batch_size = 1024
    ipca = pk.load(open("../ipca.pk1", "rb"))\
        if os.path.exists("../ipca.pk1") else IncrementalPCA(n_components=2)
    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=1)
    nm = NearMiss(version=3, n_jobs=-1)  # , n_neighbors=1)
    ds_train = Streamer(embedding_list_train, label_path,
                        mapping_path, data_dir, mem_split=10, chunk_size=chunk_size, batch_size=batch_size, sampler=None)  # [('near_miss', nm), ('smote', sm)])  # ipca=ipca, ipca_fitted=True)
    ds_val = Streamer(embedding_list_val, label_path,
                      mapping_path, data_dir, batch_size=256)
    ds_test = Streamer(embedding_list_test, label_path,
                       mapping_path, data_dir, mem_split=1, batch_size=256, sampler=None)

    train_encodings = Streamer(embedding_list_train, label_path,
                               mapping_path, '../data/pts_encodings', mem_split=10, chunk_size=chunk_size, batch_size=batch_size, sampler=None)
    # ds_train.plot_dist(dest='../plots/X_train_dist.png')
    image, pts = 0, 0
    for (x, y), (x0, y0) in zip(ds_train, train_encodings):
        # print(x.shape, y.shape)
        print(y.shape, y0.shape)
        image += y.shape[0]
        pts += y0.shape[0]
    print(image, pts)
