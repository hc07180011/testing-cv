import os
import av
import gc
import json
import logging
import tqdm
import cv2
import random
import itertools
import numpy as np
import pandas as pd
import skvideo.io
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from collections import Counter
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


def multi_flicker_storage(
    src: str,
    dst: Tuple[str, str, str, str],
    labels: dict
) -> None:
    for chunk in os.listdir(src):
        vid_name = chunk.replace(".mp4", "")
        if labels.get(vid_name):
            logging.debug(
                f"{os.path.join(src, chunk)} - {os.path.join(dst[labels[vid_name]-1], chunk)}")
            os.replace(os.path.join(src, chunk), os.path.join(
                dst[labels[vid_name]-1], chunk))


def mov_dif_aug(
    src: str,
    dst: str,
    chunk_size: int,
    shape: tuple
) -> None:
    """
    http://www.scikit-video.org/stable/io.html
    https://github.com/dmlc/decord
    https://stackoverflow.com/questions/22994189/clean-way-to-fill-third-dimension-of-numpy-array
    https://ottverse.com/change-resolution-resize-scale-video-using-ffmpeg/
    ffmpeg -i 0096.mp4 -vf scale=-1:512 frame_%d.jpg
    """
    dst_vid = [vid.split("_", 1)[1].replace(".mp4", "")
               for vid in os.listdir(dst)]
    w_chunk = np.zeros((chunk_size,)+shape, dtype=np.uint8)
    for vid in tqdm.tqdm(os.listdir(src)):
        if  vid.replace(".mp4", "").replace("reduced_", "") in dst_vid:
            continue

        cur = 0
        vidcap = cv2.VideoCapture(os.path.join(src, vid))
        success, frame = vidcap.read()
        ms = vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000
        w_chunk = np.full_like(w_chunk,frame)
        while success:
            w_chunk[cur % chunk_size] = frame
            cur += 1
            idx = [i % chunk_size for i in range(cur-chunk_size, cur)]

            mov = np.apply_along_axis(
                lambda f: (f*(255/f.max())).astype(np.uint8),
                axis=0, arr=np.diff(w_chunk[idx], axis=0).astype(np.uint8)
            )
            w_chunk[idx] = cv2.normalize(
                w_chunk[idx],
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )
            stacked = np.array([
                np.hstack((norm, mov))
                for norm, mov in zip(w_chunk[idx], mov)
            ])
            skvideo.io.vwrite(
                os.path.join(dst, f"{cur}_frame_{int(ms)}_sec_"+vid.replace("reduced_", "")),
                stacked
            )
            success, frame = vidcap.read()
            ms = vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000
        
        gc.collect()


def preprocessing(
    flicker_dir: Tuple[str, str, str, str],
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
    # logging.debug(set(false_positives_vid)-set([f.split('_',1)[-1].replace('.mp4','') for f in os.listdir('data/no_flicker')]))
    flicker_lst = list(itertools.chain(
        *list(map(lambda f: os.listdir(f), flicker_dir))))
    non_flicker_lst = [
        x for x in os.listdir(non_flicker_dir)
        if x.replace(".mp4", "").split("_", 1)[-1] not in false_positives_vid
    ]
    fp = list(set(os.listdir(non_flicker_dir)) - set(non_flicker_lst))

    random.seed(42)
    random.shuffle(non_flicker_lst)
    random.shuffle(flicker_lst)
    random.shuffle(fp)
    non_flicker_train = non_flicker_lst[:int(len(non_flicker_lst)*0.8)]
    non_flicker_test = non_flicker_lst[int(len(non_flicker_lst)*0.8):]
    flicker_train = flicker_lst[:int(len(flicker_lst)*0.8)]
    flicker_test = flicker_lst[int(len(flicker_lst)*0.8):]
    fp_train = fp[:int(len(fp)*0.8)]
    fp_test = fp[int(len(fp)*0.8):]

    length = max([
        len(flicker_train),
        len(flicker_test),
        len(fp_train),#+non_flicker_train
        len(fp_test)#+non_flicker_test
    ])
    pd.DataFrame({
        "flicker_train": tuple(flicker_train) + ("",) * (length - len(flicker_train)),
        "non_flicker_train": tuple(fp_train) + ("",) * (length - len(fp_train)),#+non_flicker_train
        "flicker_test": tuple(flicker_test) + ("",) * (length - len(flicker_test)),
        "non_flicker_test": tuple(fp_test) + ("",) * (length - len(fp_test)),#+non_flicker_test
    }).to_csv("{}.csv".format(cache_path))

    logging.debug(f"{len(fp_train)} - {len(fp_test)}") #non_flicker_train +non_flicker_test+ <- bring back
    
    np.savez(cache_path, flicker_train, fp_train, flicker_test, fp_test)#+non_flicker_train+non_flicker_test


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--flicker1', type=str, default="data/flicker1",
                        help='directory of flicker1')
    parser.add_argument('--flicker2', type=str, default="data/flicker2",
                        help='directory of flicker2')
    parser.add_argument('--flicker3', type=str, default="data/flicker3",
                        help='directory of flicker3')
    parser.add_argument('--flicker4', type=str, default="data/flicker4",
                        help='directory of flicker4')
    parser.add_argument('--meta_data_dir', type=str, default="data/show-data",
                        help='directory of flicker videos')
    parser.add_argument('--cache_path', type=str, default=".cache/train_test",
                        help='directory of miscenllaneous information')
    parser.add_argument('--videos_path', type=str, default="data/reduced-data",
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

      Predictive Modeling in Biomedicine
    said doesn't have to be flicker, but just general state of the art
    said can consider data augmentation to generate flickers? then consider publish dataset
    computation can be registered for google UR
    reduce computation can also be research
    use TPUS
    
    multiclass increase batch size
    can improve gpu memory usage
    big changes but rare occurance outlier dataset
    use logging time stamps match with video time stamps?
    
    if use transformers, take the largest frame rate as default tensor size , problem lower rates need to pad them
    resnet50
    
    """
    init_logger()
    args = command_arg()
    videos_path, flicker1_path, flicker2_path, flicker3_path, flicker4_path, meta_data_path, cache_path =\
        args.videos_path, args.flicker1, args.flicker2, args.flicker3, args.flicker4, args.meta_data_dir, args.cache_path

    if args.preprocess:
        mov_dif_aug(
            videos_path,
            meta_data_path,
            chunk_size=11,
            shape=(360, 180, 3)
        )

    if args.split:
        preprocessing(
            (flicker1_path, flicker2_path, flicker3_path, flicker4_path),
            meta_data_path,
            cache_path,
        )
    # flicker_chunk(non_flicker_path, flicker_path, labels)
    # multi_flicker_storage(
    #     flicker_path,
    #     ("data/flicker1", "data/flicker2", "data/flicker3", "data/flicker4"),
    #     json.load(open("data/multi_label.json", "r"))
    # )
