import os
import json
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


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
                ".tfrecords", "")]
            loaded = np.load("{}.npy".format(os.path.join(self.data_dir, key.replace(
                ".tfrecords", ""))))
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


class Streamer():
    """
    https://jamesmccaffrey.wordpress.com/2021/03/08/working-with-huge-training-data-files-for-pytorch/
    """

    def __init__(self,
                 embedding_list_train: list,
                 label_path: str,
                 mapping_path: str,
                 data_dir: str,
                 chunk_size: int = 32,
                 batch_size: int = 32,
                 ) -> None:
        self.embedding_list_train = embedding_list_train
        self.label_path = label_path
        self.mapping_path = mapping_path
        self.data_dir = data_dir

        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.cur_chunk = 0
        self.X_buffer, self.y_buffer = [], []

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_chunk >= len(self.embedding_list_train):
            gc.collect()
            raise StopIteration

        if len(self.X_buffer) == 0 and len(self.y_buffer) == 0:
            gc.collect()
            self.load_embeddings([self.embedding_list_train[self.cur_chunk]])
            self.cur_chunk += 1

        X, y = np.array(self.X_buffer.pop()), np.array(self.y_buffer.pop())
        if X.shape == (0,):
            return self.__next__()
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

    def new_embeddings(self, new_emb_list: list):
        self.embedding_list_train = new_emb_list
        self.cur_chunk = 0
        self.X_buffer, self.y_buffer = [], []

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
        for key in embedding_list_train:
            real_filename = encoding_filename_mapping[key.replace(
                ".npy", "")]
            loaded = np.load("{}.npy".format(os.path.join(self.data_dir, key.replace(
                ".npy", ""))))
            self.X_buffer.extend(
                self._get_chunk_array(loaded, self.chunk_size))
            # get flicker frame indexes
            flicker_idxs = np.array(raw_labels[real_filename]) - 1
            # buffer zeros array frame video embedding
            buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
            # set indexes in zeros array based on flicker frame indexes
            buf_label[flicker_idxs] = 1
            self.y_buffer.extend(tuple(
                1 if sum(x) else 0
                for x in self._get_chunk_array(buf_label, self.chunk_size)
            ))  # consider using tf reduce sum for multiclass

        self.X_buffer = [
            self.X_buffer[i:self.batch_size]
            for i in range(0, len(self.X_buffer), self.batch_size)
        ]

        self.y_buffer = [
            self.y_buffer[i:self.batch_size]
            for i in range(0, len(self.y_buffer), self.batch_size)
        ]


if __name__ == '__main__':

    pass_videos = (
        "0096.mp4", "0097.mp4", "0098.mp4",
        "0125.mp4", "0126.mp4", "0127.mp4",
        "0145.mp4", "0146.mp4", "0147.mp4",
        "0178.mp4", "0179.mp4", "0180.mp4"
    )
    raw_labels = json.load(open("../data/label.json", "r"))
    encoding_filename_mapping = json.load(
        open("../data/mapping_aug_data.json", "r"))

    embedding_path_list = sorted([
        x.split(".npy")[0] for x in os.listdir("../data/embedding_original/")
        if x.split(".npy")[0] not in pass_videos
        and encoding_filename_mapping[x.replace(".npy", "")] in raw_labels
    ])
    label_path = "../data/label.json"
    mapping_path = "../data/mapping_aug_data.json"
    data_dir = "../data/embedding_original/"

    ds = MYDS(embedding_path_list, label_path, mapping_path, data_dir)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    for batch_idx, (x, y) in enumerate(dl):
        # loading batches only from x_paths[-1] and y_paths[-1] numpy files
        print(batch_idx, x.shape, y.shape)

    ds = Streamer(embedding_path_list, label_path, mapping_path, data_dir)
    for idx, (x, y) in enumerate(ds):
        print(idx, x.shape, y.shape)
