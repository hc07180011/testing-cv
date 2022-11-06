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


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)


class RandomDataset(torch.utils.data.IterableDataset):
    """
    https://pytorch.org/vision/main/auto_examples/plot_video_api.html
    """

    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(RandomDataset).__init__()
        self.samples = get_samples(root)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - \
                (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output


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
        if label < 0:
            yield video
            return
        pad = np.zeros((video.shape[0]+1, *video.shape[1:]))
        pad[-1].fill(label)
        yield pad

    def _get_stream(self, vid_lst: list,) -> itertools.chain.from_iterable:
        # print(not self.oversample and not self.undersample)
        if not self.oversample and not self.undersample:
            # print("WTF")
            mapping = map(lambda f: self._load(
                f, self.__labels[f.split("/", 3)[-1].replace(".mp4", "")]), vid_lst)
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
    """

    def __init__(
        self,
        *args: Tuple[IterableDataset],
        batch_size: int,
        multiclass: bool,
        binary: bool,
    ) -> None:
        self.binary = binary
        self.multiclass = multiclass
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
    def _binary(
        streams: zip,
        idx: list,
    ):
        """
        USAGE: 

        batch_size = 4
        non_flicker_train = VideoDataSet.split_datasets(
            non_flicker_train, class_size=batch_size//2, max_workers=1, undersample=len(flicker_train))
        flicker_train = VideoDataSet.split_datasets(
            flicker_train, class_size=batch_size//2, max_workers=1)

        ds_train = MultiStreamer(non_flicker_train, flicker_train, batch_size)
        """
        for f, n in streams:
            random.shuffle(idx)
            f_vid, n_vid = list(itertools.chain(*f)), list(itertools.chain(*n))
            f_labels, n_labels = torch.ones(
                len(f_vid)), torch.zeros(len(n_vid))
            inputs, labels = torch.stack(
                f_vid+n_vid), torch.cat([f_labels, n_labels])
            print(inputs.shape, labels.shape)
            yield inputs[idx].float(), labels[idx].long()

    @staticmethod
    def _multi(
        streams: zip,
        idx: list,
    ):
        """
        USAGE :

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
        """
        for stream in streams:
            random.shuffle(idx)
            stream = list(itertools.chain(
                *tuple(map(lambda s: list(itertools.chain(*s)), stream))
            ))
            inputs = torch.stack(stream)
            labels = torch.arange(len(stream))
            yield inputs[idx].float(), labels[idx].long()

    @staticmethod
    def _imbalance(
        streams: zip,
    ):
        for stream in streams:
            frames = [frame[:-1]
                      for frame in list(itertools.chain(*stream))]
            labels = [label[-1][0][0][0].item()
                      for label in list(itertools.chain(*stream))]
            yield torch.cat(frames).float(), torch.Tensor(labels).long()

    def __iter__(self):
        self.__streams = zip(
            *tuple(map(self._get_stream_loaders, self.__datasets)))
        if self.multiclass:
            return self._multi(self.__streams, self.__batch_idx)
        if self.binary:
            return self._binary(self.__streams, self.__batch_idx)
        return self._imbalance(self.__streams)


class TestDS(IterableDataset):
    def __init__(
        self,
        data_list: list,
        batch_size: int
    ) -> None:
        self.data_list = data_list
        self.batch_size = batch_size

    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        labels = json.load(open("../data/multi_label.json", "r"))
        label = labels[data.split("/", 3)[-1].replace(".mp4", "")]
        video = skvideo.io.vread(data)
        pad = np.zeros((video.shape[0]+1, *video.shape[1:]))
        pad[-1].fill(label)
        # print(pad[-1])
        # print(pad.shape, video.shape)
        yield pad

    def get_stream(self, data_list):
        # print(itertools.chain.from_iterable(map(self.process_data, data_list)))
        return itertools.chain.from_iterable(map(self.process_data, data_list))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    @classmethod
    def split_datasets(cls, data_list, batch_size, max_workers):
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size//num_workers
        return[cls(data_list, batch_size=split_size) for _ in range(num_workers)]


class TestMulti:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None) for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            # print(*batch_parts)
            parts = [frame[:-1]
                     for frame in list(itertools.chain(*batch_parts))]
            labels = [label[-1][0][0][0].item()
                      for label in list(itertools.chain(*batch_parts))]
            yield torch.cat(parts), labels


def test_loader() -> None:
    # data = [
    #     list(range(i*10, i*10+5))for i in range(10)
    # ]
    data = [os.path.join("../data/flicker1", f)
            for f in os.listdir("../data/flicker1")][:20]
    ds = TestDS.split_datasets(data_list=data, batch_size=2, max_workers=1)
    loader = TestMulti(ds)
    for out in loader:
        print("WTF")
        print(out[0].shape, out[1])


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
        non_flicker_files[:8], labels=labels, class_size=2, max_workers=1, undersample=8)
    flicker1 = VideoDataSet.split_datasets(
        flicker1_files[:4]+flicker4_files[:4], labels=labels, class_size=2, max_workers=1, oversample=True)
    flicker2 = VideoDataSet.split_datasets(
        flicker2_files[:8], labels=labels, class_size=2, max_workers=1, oversample=True)
    flicker3 = VideoDataSet.split_datasets(
        flicker3_files[:8], labels=labels, class_size=2, max_workers=1, oversample=True)

    loader = MultiStreamer(non_flickers,
                           flicker1, batch_size=batch_size, multiclass=False, binary=True)  # flicker2, flicker3

    # non_flickers = VideoDataSet.split_datasets(
    #     non_flicker_files[:8]+flicker1_files[:8], class_size=4, max_workers=8)

    # loader = MultiStreamer(non_flickers, batch_size=4,
    #                        multiclass=False, binary=False)
    for i in range(2):
        print(f"{i} WTF")
        for inputs, labels in loader:
            print(inputs.shape, labels)

    # test_loader()
