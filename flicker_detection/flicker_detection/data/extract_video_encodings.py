import os
import av
import gc
import json
import cv2
import numpy as np
import pandas as pd
import skvideo.io
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from typing import Tuple


def get_pts(
    src: str,
    dst: str = 'pts_encodings',
    map_src: str = 'mapping.json',
) -> None:
    mapping = {
        code: num
        for num, code in json.load(open(map_src, "r")).items()
    }
    for vid in os.listdir(src):
        if os.path.exists(os.path.join(dst, "{}".format(mapping[vid.split(".mp4")[0].replace(" ", "")]))):
            continue
        fh = av.open(os.path.join(src, vid))
        video = fh.streams.video[0]
        decoded = tuple(fh.decode(video))
        print(f"{vid} duration: {float(video.duration*video.time_base)}")
        pts_interval = np.array((0,) + tuple(
            float(decoded[i+1].pts*video.time_base -
                  decoded[i].pts*video.time_base)
            for i in range(0, len(decoded)-1, 1)
        ))
        std_arr = ((pts_interval - pts_interval.mean(axis=0)) /
                   pts_interval.std(axis=0))
        np.save(os.path.join(
            dst, f'{mapping[vid.split(".mp4")[0].replace(" ","")]}.npy'), std_arr)
        gc.collect()


def test_frame_extraction(inpath: str, outpath: str) -> None:
    container = av.open(inpath)
    vidstream = container.streams.video[0]
    for frame in container.decode(video=0):
        fts = float(frame.pts*vidstream.time_base)
        frame.to_image()\
            .save("{}_{}_{:.2f}.jpg".format(outpath, int(frame.index), fts))


def mov_dif_aug(
    src: str,
    dst: str,
    res: tuple = (200, 200)
) -> None:
    """
    http://www.scikit-video.org/stable/io.html
    https://github.com/dmlc/decord
    https://ottverse.com/change-resolution-resize-scale-video-using-ffmpeg/
    normalize frames later
    """
    for vid in os.listdir(src):
        if os.path.exists(os.path.join(dst, vid.split('.mp4')[0]+'.npy')):
            continue

        mvd = skvideo.io.vread(os.path.join(src, vid))
        frames = np.array([
            cv2.resize(frame, dsize=res, interpolation=cv2.INTER_CUBIC)
            for frame in mvd
        ])
        nvm = frames.copy()
        # nvm = np.array([
        #     cv2.normalize(img, None, alpha=0, beta=1,
        #                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #     for img in mvd])
        frames = np.apply_along_axis(
            lambda f: (f*(255/f.max())).astype(np.uint16),
            axis=0, arr=np.diff(frames, axis=0).astype(np.uint16))

        stacked = np.array([
            np.hstack((norm, mov))
            for norm, mov in zip(nvm, frames)
        ])
        np.save(
            os.path.join(dst, vid.split('.mp4')[0]),
            stacked
        )
        skvideo.io.vwrite(os.path.join("augmented", vid), stacked)
        gc.collect()


def preprocessing(
    label_path: str,
    data_dir: str,
    cache_path: str,
) -> Tuple[np.ndarray, np.ndarray]:

    if os.path.exists("/{}.npz".format(cache_path)):
        __cache__ = np.load("/{}.npz".format(cache_path), allow_pickle=True)
        return tuple(__cache__[k] for k in __cache__)

    raw_labels = json.load(open(label_path, "r"))

    embedding_path_list = sorted([
        x for x in os.listdir(data_dir)
        if x.split("reduced_")[1].replace(".npy", "") in raw_labels
    ])

    embedding_list_train, embedding_list_test, _, _ = train_test_split(
        embedding_path_list,
        # dummy buffer just to split embedding_path_list
        list(range(len(embedding_path_list))),
        test_size=0.1,
        random_state=42
    )
    false_positives_vid = [
        'reduced_17271FQCB00002_video_6',
        'reduced_video_0B061FQCB00136_barbet_07-07-2022_00-05-51-678',
        'reduced_video_0B061FQCB00136_barbet_07-07-2022_00-12-11-280',
        'reduced_video_0B061FQCB00136_barbet_07-21-2022_15-37-32-891',
        'reduced_video_0B061FQCB00136_barbet_07-21-2022_14-17-42-501',
        'reduced_video_03121JEC200057_sunfish_07-06-2022_23-18-35-286'
    ]
    embedding_list_test += false_positives_vid
    embedding_list_val = embedding_list_test

    embedding_list_train = list(
        set(embedding_list_train) - set(embedding_list_test))

    length = max([len(embedding_list_test), len(
        embedding_list_val), len(embedding_list_train)])
    pd.DataFrame({
        "train": tuple(embedding_list_train) + ("",) * (length - len(embedding_list_train)),
        "val": tuple(embedding_list_val) + ("",) * (length - len(embedding_list_val)),
        "test": tuple(embedding_list_test) + ("",) * (length - len(embedding_list_test))
    }).to_csv("{}.csv".format(cache_path))

    np.savez(cache_path, embedding_list_train,
             embedding_list_val, embedding_list_test)


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--label_path', type=str, default="new_label.json",
                        help='path of json that store the labeled frames')
    parser.add_argument('--mapping_path', type=str, default="mapping.json",
                        help='path of json that maps encrpypted video file name to simple naming')
    parser.add_argument('--data_dir', type=str, default="meta_data",
                        help='directory of extracted feature embeddings')
    parser.add_argument('--cache_path', type=str, default="../.cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--videos_path', type=str, default="lower_res",
                        help='src directory to extract embeddings from')
    parser.add_argument(
        "-train", "--train", action="store_true",
        default=False,
        help="Whether to do training"
    )
    parser.add_argument(
        "-test", "--test", action="store_true",
        default=False,
        help="Whether to do testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = command_arg()
    videos_path, label_path, mapping_path, data_path, cache_path = args.videos_path, args.label_path, args.mapping_path, args.data_dir, args.cache_path
    mov_dif_aug(
        videos_path,
        data_path,
        res=(180, 380)
    )
    preprocessing(
        label_path,
        data_path,
        cache_path,
    )
