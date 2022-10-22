import os
import random
import itertools
import numpy as np
import skvideo.io
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader

from typing import Tuple, Callable


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
        batch_size: int
    ) -> None:
        self.vid_lst = vid_lst
        self.batch_size = batch_size

    @property
    def shuffled_data_list(self):
        return random.sample(self.vid_lst, len(self.vid_lst))

    @staticmethod
    def _load(vid_lst: list) -> np.ndarray:
        for vid_name in vid_lst:
            worker = torch.utils.data.get_worker_info()
            worker_id = id(worker) if worker is not None else -1,
            yield worker_id, skvideo.io.vread(vid_name)

    def _get_stream(self, vid_lst: list) -> list:
        return itertools.chain.from_iterable(map(self._load, itertools.cycle(vid_lst)))

    def _get_streams(self) -> zip:
        return zip(*[self._get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    @classmethod
    def split_datasets(
        loader: DataLoader,
        vid_lst: list,
        batch_size: int,
        max_workers: int
    ) -> list:
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        return [loader(vid_lst, batch_size=batch_size//num_workers) for _ in range(num_workers)]

    def __iter__(self) -> itertools.chain.from_iterable:
        return self._get_stream()


class MultiStreamer(object):
    def __init__(
        self,
        datasets: list
    ) -> None:
        self.datasets = datasets

    def _get_stream_loaders(self):
        return zip(*[DataLoader(ds, num_workers=1, batch_size=None) for ds in self.datasets])

    def __iter__(self):
        for batch in self._get_stream_loaders():
            yield list(itertools.chain(*batch))


if __name__ == '__main__':
    # dataset = RandomDataset("../data/10_frame_data",
    #                         epoch_size=None, frame_transform=None)
    # loader = DataLoader(dataset, batch_size=12)
    # data = {"video": [], 'start': [], 'end': [], 'tensorsize': []}
    # for batch in loader:
    #     for i in range(len(batch['path'])):
    #         data['video'].append(batch['path'][i])
    #         data['start'].append(batch['start'][i].item())
    #         data['end'].append(batch['end'][i].item())
    #         data['tensorsize'].append(batch['video'][i].size())
    # print(data)

    video_dir = "../data/10_frame_data/flicker-chunks"
    video_files = [os.path.join(video_dir, path)
                   for path in os.listdir(video_dir)]
    datasets = VideoDataSet.split_datasets(
        video_files, batch_size=4, max_workers=1)
    loader = MultiStreamer(datasets)

    for batch in loader:
        print(type(batch))
