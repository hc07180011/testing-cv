import os
import random
import json
import itertools
import tqdm
import numpy as np
import skvideo.io
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from torchvision.datasets.folder import make_dataset

from typing import Tuple, Callable


class VideoDataSet(IterableDataset):
    def __init__(
        self,
        vid_lst: list,
        labels: dict,
        class_size: int,
        oversample: bool,
        undersample: int = 0,
    ) -> None:
        self.__vid_lst = vid_lst
        self.__labels = labels
        self.batch_size = class_size
        self.oversample = oversample
        self.undersample = undersample

    @property
    def shuffled_data_list(self):
        return random.sample(self.__vid_lst, len(self.__vid_lst))

    @staticmethod
    def _load(vid: str, label: int = -1) -> np.ndarray:
        video = skvideo.io.vread(vid)
        if label is not None and label < 0:
            yield video
            return
        pad = np.zeros((video.shape[0]+1, *video.shape[1:]))
        pad[-1].fill(label if label is not None else int(bool(label)))
        yield pad

    def _get_stream(self, vid_lst: list,) -> itertools.chain.from_iterable:
        if not self.oversample and not self.undersample:
            mapping = map(lambda f: self._load(
                f, self.__labels.get(f.split("/", 3)[-1].replace(".mp4", ""))), vid_lst)
            return itertools.chain.from_iterable(mapping)
        if self.oversample:
            vid_lst = itertools.cycle(vid_lst)
        elif self.undersample:
            vid_lst = random.sample(vid_lst, self.undersample)
        return itertools.chain.from_iterable(map(self._load, vid_lst))

    def _get_streams(self) -> zip:
        return zip(*[
            self._get_stream(self.shuffled_data_list)
            for _ in range(self.batch_size)
        ])

    @classmethod
    def split_datasets(
        cls: IterableDataset,
        vid_lst: list,
        labels: dict,
        class_size: int,
        max_workers: int,
        oversample: bool = False,
        undersample: int = 0,
    ) -> list:
        for n in range(max_workers, 0, -1):
            if class_size % n == 0:
                num_workers = n
                break
        split_lst = np.array_split(np.array(vid_lst), num_workers)
        return [cls(
            lst.tolist(),
            labels=labels,
            class_size=class_size,
            oversample=oversample,
            undersample=undersample)
            for lst in split_lst]

    def __iter__(self) -> itertools.chain.from_iterable:
        return self._get_streams()


class MultiStreamer(object):
    """
    https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    TODO merge multiclass binary method
    """

    def __init__(
        self,
        *args: Tuple[IterableDataset],
        batch_size: int,
    ) -> None:
        self.balanced = len(args) > 1
        self.batch_size = batch_size
        self.__batch_idx = list(range(batch_size))

        self.__datasets = args
        self.__streams = None

    @staticmethod
    def _get_stream_loaders(vid_lst: list):
        random.shuffle(vid_lst)
        return zip(*[
            DataLoader(ds, num_workers=1, batch_size=None, pin_memory=True)
            for ds in vid_lst
        ])

    @staticmethod
    def _balanced(
        streams: zip,
        idx: list,
        multiclass: bool = False,
    ):
        """
        USAGE: MULTICLASS

        batch_size = 5
        non_flickers = VideoDataSet.split_datasets(
            non_flicker_files[:8], class_size=1, max_workers=1)  # , undersample=len(flicker_files[:4]))
        flicker1 = VideoDataSet.split_datasets(
            flicker1_files[:8], class_size=1, max_workers=1)  # oversample=True)
        flicker2 = VideoDataSet.split_datasets(
            flicker2_files[:8], class_size=1, max_workers=1)
        flicker3 = VideoDataSet.split_datasets(
            flicker3_files[:8], class_size=1, max_workers=1)
        flicker4 = VideoDataSet.split_datasets(
            flicker4_files[:8], class_size=1, max_workers=1)

        loader = MultiStreamer(non_flickers,
                            flicker1, flicker2, flicker3, flicker4, batch_size=batch_size, multiclass=True)


        USAGE: BINARY

        batch_size = 4
        non_flicker_train = VideoDataSet.split_datasets(
            non_flicker_train, class_size=batch_size//2, max_workers=1, undersample=len(flicker_train))
        flicker_train = VideoDataSet.split_datasets(
            flicker_train, class_size=batch_size//2, max_workers=1)

        ds_train = MultiStreamer(non_flicker_train, flicker_train, batch_size)


        USAGE: IMBALANCED

        non_flickers = VideoDataSet.split_datasets(
            non_flicker_files[:12], labels=labels, class_size=1, max_workers=8, undersample=0)
        loader = MultiStreamer(
            non_flickers, batch_size=batch_size)  
        """
        for stream in streams:
            stream = list(itertools.chain(
                *tuple(map(lambda s: list(itertools.chain(*s)), stream))
            ))
            inputs = torch.stack(stream)
            if multiclass:
                labels = torch.arange(len(stream))
            else:
                labels = torch.zeros(inputs.shape[0])
                labels[labels.size(dim=0)//2:] = 1
            random.shuffle(idx)
            yield inputs[idx].float(), labels[idx].long()

    @staticmethod
    def _imbalance(
        streams: zip,
    ):
        for stream in streams:
            frames = [frame[:-1]
                      for frame in list(itertools.chain(*stream[0]))]
            labels = [label[-1][0][0][0].item()
                      for label in list(itertools.chain(*stream[0]))]
            yield torch.stack(frames).float(), torch.Tensor(labels).long()

    def __iter__(self):
        self.__streams = zip(
            *tuple(map(self._get_stream_loaders, self.__datasets)))
        if self.balanced:
            return self._balanced(self.__streams, self.__batch_idx, len(self.__datasets) > 2)
        return self._imbalance(self.__streams)


if __name__ == '__main__':
    non_flicker_dir = "../data/no_flicker"
    flicker1_dir = "../data/flicker1"
    flicker2_dir = "../data/flicker2"
    flicker3_dir = "../data/flicker3"
    flicker4_dir = "../data/flicker4"

    non_flicker_files = [os.path.join(non_flicker_dir, path)
                         for path in os.listdir(non_flicker_dir)]
    flicker1_files = [os.path.join(flicker1_dir, path)
                      for path in os.listdir(flicker1_dir)]
    flicker2_files = [os.path.join(flicker2_dir, path)
                      for path in os.listdir(flicker2_dir)]
    flicker3_files = [os.path.join(flicker3_dir, path)
                      for path in os.listdir(flicker3_dir)]
    flicker4_files = [os.path.join(flicker4_dir, path)
                      for path in os.listdir(flicker4_dir)]

    labels = json.load(open("../data/multi_label.json", "r"))
    batch_size = 4
    non_flickers = VideoDataSet.split_datasets(
        non_flicker_files[:12], labels=labels, class_size=1, max_workers=8, undersample=0)
    flicker1 = VideoDataSet.split_datasets(
        flicker1_files[:4]+flicker4_files[:4], labels=labels, class_size=1, max_workers=1, oversample=True)
    flicker2 = VideoDataSet.split_datasets(
        flicker2_files[:8], labels=labels, class_size=1, max_workers=1, oversample=True)
    flicker3 = VideoDataSet.split_datasets(
        flicker3_files[:8], labels=labels, class_size=1, max_workers=1, oversample=True)

    loader = MultiStreamer(
        non_flickers, batch_size=batch_size)  # , flicker1, flicker2, flicker3,

    for i in range(2):
        print(f"{i} WTF")
        for inputs, labels in loader:
            print(inputs.shape, labels)

    # test_loader()
