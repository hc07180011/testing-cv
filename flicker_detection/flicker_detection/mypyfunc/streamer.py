import os
import random
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
        class_size: int,
        oversample: bool,
        undersample: int,
    ) -> None:
        self.vid_lst = vid_lst
        self.batch_size = class_size
        self.oversample = oversample
        self.undersample = undersample

    @property
    def shuffled_data_list(self):
        return random.sample(self.vid_lst, len(self.vid_lst))

    @staticmethod
    def _load(vid: str) -> np.ndarray:
        yield skvideo.io.vread(vid)

    def _get_stream(self, vid_lst: list,) -> itertools.chain.from_iterable:
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
            class_size=class_size,
            oversample=oversample,
            undersample=undersample)
            for lst in split_lst]

    def __iter__(self) -> itertools.chain.from_iterable:
        return self._get_streams()


class MultiStreamer(object):
    """
    https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    TO DO implement multiprocess labeling from the DataLoader class 
    """

    def __init__(
        self,
        *args: Tuple[IterableDataset],
        batch_size: int,
        multiclass: bool,
    ) -> None:
        self.multiclass = multiclass
        self.batch_size = batch_size
        self.batch_idx = list(range(batch_size))

        self.no_streams = len(args)
        self.streams = zip(*tuple(map(self._get_stream_loaders, args)))

    @staticmethod
    def _get_stream_loaders(vid_lst: list):
        random.shuffle(vid_lst)
        return zip(*[
            DataLoader(ds, num_workers=1, batch_size=None, pin_memory=True)
            for ds in vid_lst
        ])

    def _binary(self):
        f_stream, n_stream = self._get_stream_loaders(
            self.flickers), self._get_stream_loaders(self.non_flickers)
        for f, n in zip(f_stream, n_stream):
            random.shuffle(self.batch_idx)
            f_vid, n_vid = list(itertools.chain(*f)), list(itertools.chain(*n))
            f_labels, n_labels = torch.ones(
                len(f_vid)), torch.zeros(len(n_vid))
            inputs, labels = torch.stack(
                f_vid+n_vid), torch.cat([f_labels, n_labels])
            yield inputs[self.batch_idx].float(), labels[self.batch_idx].long()

    def _multi(self):
        for stream in self.streams:
            random.shuffle(self.batch_idx)
            stream = list(itertools.chain(
                *tuple(map(lambda s: list(itertools.chain(*s)), stream))
            ))
            inputs = torch.stack(stream)
            labels = torch.arange(self.batch_size)
            yield inputs[self.batch_idx].float(), labels[self.batch_idx].long()

    def __iter__(self):
        if self.multiclass:
            return self._multi()
        return self._binary()


if __name__ == '__main__':
    # work with shorter videos with smaller samples to debug

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
                           flicker1, flicker2, flicker3, flicker4, batch_size=batch_size,multiclass=True)
    for i in range(2):
        print(f"{i} WTF")
        for inputs, labels in loader:
            print(inputs.shape, labels)
            pass
