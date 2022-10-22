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
        cycle: bool,
    ) -> None:
        self.vid_lst = vid_lst
        self.batch_size = class_size
        self.cycle = cycle

    @property
    def shuffled_data_list(self):
        return random.sample(self.vid_lst, len(self.vid_lst))

    @staticmethod
    def _load(vid: str) -> np.ndarray:
        yield skvideo.io.vread(vid)

    def _get_stream(self, vid_lst: list,) -> itertools.chain.from_iterable:
        return itertools.chain.from_iterable(map(self._load, itertools.cycle(vid_lst) if self.cycle else vid_lst))

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
        cycle: bool
    ) -> list:
        for n in range(max_workers, 0, -1):
            if class_size % n == 0:
                num_workers = n
                break
        split_lst = np.split(np.array(vid_lst), num_workers)
        return [cls(lst.tolist(), class_size=class_size//num_workers, cycle=cycle)
                for lst in split_lst]

    def __iter__(self) -> itertools.chain.from_iterable:
        return self._get_streams()


class MultiStreamer(object):
    def __init__(
        self,
        non_flickers: IterableDataset,
        flickers: IterableDataset,
        batch_size: int
    ) -> None:
        self.non_flickers = non_flickers
        self.flickers = flickers
        self.batch_size = batch_size
        self.batch_idx = list(range(batch_size))

    @staticmethod
    def _get_stream_loaders(vid_lst: list):
        random.shuffle(vid_lst)
        return zip(*[
            DataLoader(ds, num_workers=1, batch_size=None, pin_memory=True)
            for ds in vid_lst
        ])

    def __iter__(self):
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


if __name__ == '__main__':
    # work with shorter videos with smaller samples to debug

    non_flicker_dir = "../data/10_frame_data/meta-data"
    flicker_dir = "../data/10_frame_data/flicker-chunks"
    non_flicker_files = [os.path.join(non_flicker_dir, path)
                         for path in os.listdir(non_flicker_dir)]
    flicker_files = [os.path.join(flicker_dir, path)
                     for path in os.listdir(flicker_dir)]

    batch_size = 4
    non_flickers = VideoDataSet.split_datasets(
        non_flicker_files[:12], class_size=batch_size//2, max_workers=4, cycle=False)
    flickers = VideoDataSet.split_datasets(
        flicker_files[:4], class_size=batch_size//2, max_workers=4, cycle=True)

    loader = MultiStreamer(non_flickers, flickers, batch_size)
    for i in range(2):
        for inputs, labels in loader:
            print(inputs.shape, labels.shape)
