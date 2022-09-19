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
        if self.sampler is None and self.ipca is None:
            return np.array(self.X_buffer), np.array(self.y_buffer)

        if self.sampler is not None:
            # print(type(self.X_buffer[0]), type(self.y_buffer[0]))
            X, y = self._sampling(
                np.array(self.X_buffer),
                np.array(self.y_buffer),
                self.sampler
            )
        if self.ipca is not None and self.ipca_fitted:
            X, y = np.array(self.X_buffer), np.array(self.y_buffer)
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
            real_filename = self.encoding_filename_mapping[key.replace(
                ".npy", "")]
            loaded = np.load(
                "{}.npy".format(os.path.join(
                    self.data_dir, key.replace(".npy", "")))
            )
            if mov_dif:
                loaded = self._mov_dif_chunks(loaded)

            flicker_idxs = np.array(
                self.raw_labels[real_filename], dtype=np.int16) - 1

            if self.overlap_chunking:
                self.X_buffer += (*self._overlap_chunks(loaded,
                                  flicker_idxs, self.chunk_size),)
                self.y_buffer += (1,)*flicker_idxs.size
                loaded = np.delete(loaded, flicker_idxs, axis=0)
                flicker_idxs = np.array([])

            buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
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


class DecordLoader(object):
    def __init__(
        self,
        embedding_list_train: list,
        label_path: str,
        mapping_path: str,
        data_dir: str,
        chunk_size: int,
        batch_size: int,
        shape: tuple,
        sampler: Callable = None,
    ) -> None:
        self.encoding_filename_mapping = json.load(open(mapping_path, "r"))
        self.raw_labels = json.load(open(label_path, "r"))
        self.embedding_list_train = self.to_process = embedding_list_train
        self.data_dir = data_dir

        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.sampler = sampler

        self.chunk_idx = self.end_idx = 0
        self.start_idx = self.pidx = chunk_size//2

        self.next_chunk = False
        self.cache = np.zeros(shape)
        self.out_x = np.zeros((shape[0], shape[1]//2, *shape[2:]))
        self.out_y = np.zeros(shape[0])
        self.out_sequence = []

    def __len__(self) -> int:
        return len(self.out_sequence)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pidx > self.out_sequence and len(self.to_process) == 1:
            gc.collect()
            raise StopIteration

        if self.pidx > self.out_sequence and self.next_chunk:
            self._load(self.to_process)
            self.out_x[:, :, :self.out_x.shape[3] //
                       2, :] = self._mov_div(self.cache)
            self.out_x[:, :, self.out_x.shape[3] //
                       2:, :] = self._norm(self.cache)

            self.cur_chunk += 1
            self.next_chunk = False
            gc.collect()

        # Need to handle chunk padding in out sequence
        # How to handle transition between videos, chunks at the cut off between videos?
        idx = self.out_sequence[self.pidx:(self.pidx+self.batch_size) if (
            self.pidx + self.batch_size) < len(self.out_sequence) else -1]
        self.pidx += self.batch_size
        return self._batch_sample(self.out_x, self.out_y, idx, self.chunk_size)

    def _load(
        self,
        videos: list
    ) -> Tuple[int, int]:
        for i in range(videos):
            loaded = skvideo.io.vread(
                os.path.join(self.data_dir, videos[i])
            ).astype(np.uint8)

            self.start_idx += self.end_idx
            self.end_idx += loaded.shape[0]

            if self.end_idx > self.cache.shape[0]:
                self.next_chunk = True
                self.start_idx = self.end_idx = 0
                self.to_process = videos[i:]
                break

            flicker_idxs = np.array(
                self.raw_labels[videos[i]], dtype=np.int8) - 1
            buf_label = np.zeros(loaded.shape[0], dtype=np.uint8)
            buf_label[flicker_idxs.tolist()] = 1

            # TO DO over sample here
            self.out_y[self.start_idx:self.end_idx] = buf_label
            self.cache[self.start_idx:self.end_idx] = loaded

            self.out_sequence += np.array([
                tuple(range(i-self.chunk_size//2, i+self.chunk_size//2))
                for i in range(self.start_idx, self.end_idx) if self.out_y[i] == 1
            ]).flatten().tolist()

            self.out_sequence += [
                i for i in range(self.start_idx, self.end_idx)
                if i not in self.out_sequence
            ]

        return self.start_idx, self.end_idx

    def _shuffle(self) -> None:
        random.shuffle(self.embedding_list_train)
        random.shuffle(self.out_sequence)
        self.to_process = self.embedding_list_train
        self.cur_chunk = 0
        self.out_sequence = []
        gc.collect()

    @staticmethod
    def _batch_sample(
        X: np.ndarray,
        y: np.ndarray,
        idx: list,
        chunk_size: int
    ) -> Tuple[torch.Tensors, torch.Tensors]:
        X = torch.from_numpy([X[i-chunk_size//2:i+chunk_size//2]
                             for i in idx]).long()
        y = torch.from_numpy([y[i-chunk_size//2:i+chunk_size//2]
                             for i in idx]).long()
        return X, y

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
    """
    torch.Size([390, 30, 1, 64])                                                                                                                                                                 
    torch.Size([390, 30, 1, 64])                                                                                                                                                                 
    Traceback (most recent call last):                                                                                                                                                           
    File "/home/ntu-cv/testing-cv/flicker_detection/flicker_detection/torch_training.py", line 375, in <module>                                                                                
        main()                                                                                                                                                                                   
    File "/home/ntu-cv/testing-cv/flicker_detection/flicker_detection/torch_training.py", line 345, in main                                                                                    
        torch_training(train_encodings, val_encodings, model0,                                                                                                                                   
    File "/home/ntu-cv/testing-cv/flicker_detection/flicker_detection/torch_training.py", line 60, in torch_training                                                                           
        y_pred = model(x0)                                                                                                                                                                       
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl                                                                    
        return forward_call(*input, **kwargs)                                                                                                                                                    
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 168, in forward                                                                
        outputs = self.parallel_apply(replicas, inputs, kwargs)                                                                                                                                  
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 178, in parallel_apply                                                         
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])                                                                                                         
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/parallel/parallel_apply.py", line 86, in parallel_apply                                                         
        output.reraise()                                                                                                                                                                         
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/_utils.py", line 461, in reraise                                                                                   
        raise exception                                                                                                                                                                          
    RuntimeError: Caught RuntimeError in replica 0 on device 0.                                                                                                                                  
    Original Traceback (most recent call last):                                                                                                                                                  
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/parallel/parallel_apply.py", line 61, in _worker                                                                
        output = module(*input, **kwargs)                                                                                                                                                        
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl                                                                    
        return forward_call(*input, **kwargs)                                                                                                                                                    
    File "/home/ntu-cv/testing-cv/flicker_detection/flicker_detection/mypyfunc/torch_models.py", line 131, in forward                                                                          
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)                                                                                                                                   
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl                                                                    
        return forward_call(*input, **kwargs)                                                                                                                                                    
    File "/home/ntu-cv/anaconda3/envs/cv2/lib/python3.9/site-packages/torch/nn/modules/rnn.py", line 760, in forward                                                                           
        raise RuntimeError(msg)                                                                                                                                                                  
    RuntimeError: For unbatched 2-D input, hx and cx should also be 2-D but got (3-D, 3-D) tensors
    """
    label_path = "../data/new_label.json"
    mapping_path = "../data/mapping.json"
    data_dir = "../data/vgg16_emb/"
    __cache__ = np.load("{}.npz".format(
        "../.cache/train_test"), allow_pickle=True)
    embedding_list_train, embedding_list_val, embedding_list_test = tuple(
        __cache__[lst] for lst in __cache__)

    chunk_size = 30
    batch_size = 1024  # GPU memory
    ipca = pk.load(open("../ipca.pk1", "rb"))\
        if os.path.exists("../ipca.pk1") else IncrementalPCA(n_components=2)
    sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=7)
    nm = NearMiss(version=3, n_jobs=-1)  # , n_neighbors=1)
    ds_train = Streamer(embedding_list_train,
                        label_path,
                        mapping_path,
                        data_dir,
                        mem_split=4,
                        chunk_size=30,
                        batch_size=1024,
                        multiclass=False,
                        sampler=sm,
                        overlap_chunking=True,
                        moving_difference=False,
                        )  # [('near_miss', nm), ('smote', sm)])  # ipca=ipca, ipca_fitted=True)
    ds_val = Streamer(embedding_list_val,
                      label_path,
                      mapping_path,
                      data_dir,
                      mem_split=1,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      multiclass=False,
                      sampler=None,
                      overlap_chunking=True,
                      )
    ds_test = Streamer(
        embedding_list_test,
        label_path,
        mapping_path,
        data_dir,
        mem_split=1,
        chunk_size=chunk_size,
        batch_size=batch_size,
        sampler=None,
        overlap_chunking=True,
    )

    train_encodings = Streamer(
        embedding_list_train,
        label_path,
        mapping_path,
        '../data/pts_encodings',
        mem_split=1,
        chunk_size=30,
        batch_size=1024,
        sampler=sm,
        text_based=True,
        multiclass=False,
    )
    vl = DecordLoader(
        os.listdir('../data/flicker-detection'),
        ctx=[cpu(0)],
        shape=(11, 3040//2, 1440//2, 3),
        interval=1,
        skip=5,
        shuffle=0,
        labels=json.load(open(label_path, "r"))
    )
    image = 0  # (1024,30,flattened feature embeddding size) - (h,w,(rgb))
    for x in vl:
        print(x.shape)
