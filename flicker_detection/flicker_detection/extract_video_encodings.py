import os
import av
import gc
import json
import logging
import tqdm
import cv2
import numpy as np
import pandas as pd
import skvideo.io
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from typing import Tuple
from mypyfunc.logger import init_logger


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


def flicker_chunk(
    src: str,
    dst: str,
    labels: dict
) -> None:
    for chunk in os.listdir(src):
        frame_idx, vid_name = chunk.replace(".mp4", "").split("_", 1)
        if int(frame_idx) in labels[vid_name]:
            logging.debug(
                f"{os.path.join(src, chunk)} - {os.path.join(dst, chunk)}")
            os.replace(os.path.join(src, chunk), os.path.join(dst, chunk))


def mov_dif_aug(
    src: str,
    dst: str,
    labels: dict,
    chunk_size: int,
    shape: tuple
) -> None:
    """
    http://www.scikit-video.org/stable/io.html
    https://github.com/dmlc/decord
    https://ottverse.com/change-resolution-resize-scale-video-using-ffmpeg/
    normalize frames later
    ffmpeg -i 0096.mp4 -vf scale=-1:512 frame_%d.jpg
    """
    dst_vid = [vid.split("_", 1)[1].replace(".mp4", "")
               for vid in os.listdir(dst)]
    w_chunk = np.zeros((chunk_size,)+shape, dtype=np.uint8)
    for vid in tqdm.tqdm(os.listdir(src)):
        if vid.replace(".mp4", "").replace("reduced_", "") in dst_vid:
            continue
        # frames = skvideo.io.vread(os.path.join(src, vid)).astype(np.uint8)

        cur = 0
        vidcap = cv2.VideoCapture(os.path.join(src, vid))
        success, frame = vidcap.read()
        while success:
            w_chunk[cur % chunk_size] = frame
            cur += 1
            idx = [i % chunk_size for i in range(cur-chunk_size, cur)]
            if cur in labels[vid.replace("reduced_", "").replace(".mp4", "")]:
                idx = [i % chunk_size for i in range(
                    cur-chunk_size//2,
                    cur+1+chunk_size//2
                )]

            mov = np.apply_along_axis(
                lambda f: (f*(255/f.max())).astype(np.uint8),
                axis=0, arr=np.diff(w_chunk[idx], axis=0).astype(np.uint8)
            )
            stacked = np.array([
                np.hstack((norm, mov))
                for norm, mov in zip(w_chunk[idx], mov)
            ])
            skvideo.io.vwrite(
                os.path.join(dst, f"{cur}_"+vid.replace("reduced_", "")),
                stacked
            )
            success, frame = vidcap.read()
        gc.collect()


def preprocessing(
    flicker_dir: str,
    non_flicker_dir: str,
    cache_path: str,
) -> Tuple[np.ndarray, np.ndarray]:

    if os.path.exists("/{}.npz".format(cache_path)):
        __cache__ = np.load("/{}.npz".format(cache_path), allow_pickle=True)
        return tuple(__cache__[k] for k in __cache__)

    false_positives_vid = [
        '17271FQCB00002_video_6',
        'video_0B061FQCB00136_barbet_07-07-2022_00-05-51-678',
        'video_0B061FQCB00136_barbet_07-07-2022_00-12-11-280',
        'video_0B061FQCB00136_barbet_07-21-2022_15-37-32-891',
        'video_0B061FQCB00136_barbet_07-21-2022_14-17-42-501',
        'video_03121JEC200057_sunfish_07-06-2022_23-18-35-286'
    ]
    flicker_lst = os.listdir(flicker_dir)
    non_flicker_lst = [
        x for x in os.listdir(non_flicker_dir)
        if x.replace(".mp4", "").split("_")[-1] not in false_positives_vid
    ]
    logging.debug(f"{non_flicker_lst}")
    fp_test = list(set(os.listdir(non_flicker_dir)) - set(non_flicker_lst))
    logging.debug(f"{len(non_flicker_lst)} - {len(fp_test)}")

    flicker_train, flicker_test, _, _ = train_test_split(
        flicker_lst,
        # dummy buffer just to split embedding_path_list
        list(range(len(flicker_lst))),
        test_size=0.1,
        random_state=42
    )
    non_flicker_train, non_flicker_test, _, _ = train_test_split(
        non_flicker_lst,
        # dummy buffer just to split embedding_path_list
        list(range(len(non_flicker_lst))),
        test_size=0.1,
        random_state=42
    )

    length = max([
        len(fp_test),
        len(flicker_train),
        len(flicker_test),
        len(non_flicker_train),
        len(non_flicker_test)
    ])
    pd.DataFrame({
        "flicker_train": tuple(flicker_train) + ("",) * (length - len(flicker_train)),
        "non_flicker_train": tuple(non_flicker_train) + ("",) * (length - len(non_flicker_train)),
        "flicker_test": tuple(flicker_test) + ("",) * (length - len(flicker_test)),
        "non_flicker_test": tuple(non_flicker_test) + ("",) * (length - len(non_flicker_test)),
        "fp_test": tuple(fp_test) + ("",) * (length - len(fp_test)),
    }).to_csv("{}.csv".format(cache_path))

    np.savez(cache_path, flicker_train, non_flicker_train,
             fp_test, flicker_test, non_flicker_test)


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--label_path', type=str, default="data/new_label.json",
                        help='path of json that store the labeled frames')
    parser.add_argument('--mapping_path', type=str, default="data/mapping.json",
                        help='path of json that maps encrpypted video file name to simple naming')
    parser.add_argument('--flicker_dir', type=str, default="data/flicker-chunks",
                        help='directory of flicker videos')
    parser.add_argument('--non_flicker_dir', type=str, default="data/meta-data",
                        help='directory of flicker videos')
    parser.add_argument('--cache_path', type=str, default=".cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--videos_path', type=str, default="data/lower_res",
                        help='src directory to extract embeddings from')
    parser.add_argument(
        "-preprocess", "--preprocess", action="store_true",
        default=False,
        help="Whether to do training"
    )
    parser.add_argument(
        "-split", "--split", action="store_true",
        default=False,
        help="Whether to do testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    take difference between images and then vectorize or difference between vectors is also fine(standard for motion detection),
    key is rapid changes between change, normalize them between 0 - 255,
    use the difference of consecutive frames as data, or both (just concatenate the embeddings)
    train end to end, integrate cnn with lstm, and do back prop for same loss function
    smaller windows of variable frame rate should have few percent performance boost
    need to verify smote
    sliding window each frame is a data point
    do not delete flicker frames for non flickers data points
    divide by 255 to get range of 0,1 normalization(known cv preprocess, may not affect), multiply everything by 255 to rescale it and take floor/ ceeling
    include flicker frames in non flicker video data ponts as well because testing data will not have data label
    traiing should be as close as possible to testing(otherwise causes domain shifts network will not perform well)
    just oversample by drawing to mini batch just make sure epochs dont have repeating minibatch
    Use torch data loader
    Anomaly detection
    find state of art and compare for paper

    25471 : 997
    """
    init_logger()
    args = command_arg()
    videos_path, label_path, mapping_path, flicker_path, non_flicker_path, cache_path = args.videos_path, args.label_path, args.mapping_path, args.flicker_dir, args.non_flicker_dir, args.cache_path
    labels = json.load(open(label_path, "r"))

    # flicker_chunk(non_flicker_path, "data/flicker_chunks", labels)

    if args.preprocess:
        mov_dif_aug(
            videos_path,
            non_flicker_path,
            labels,
            chunk_size=31,
            shape=(360, 180, 3)
        )

    if args.split:
        preprocessing(
            flicker_path,
            non_flicker_path,
            cache_path,
        )
