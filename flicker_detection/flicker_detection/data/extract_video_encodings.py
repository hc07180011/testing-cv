import os
import av
import gc
import json
import cv2
import decord
import numpy as np
import skvideo.io
import torchvision.transforms as transforms
from decord import VideoLoader, VideoReader, cpu, gpu


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


def mov_dif_aug(src: str, dst: str) -> None:
    """
    http://www.scikit-video.org/stable/io.html
    https://github.com/dmlc/decord
    """
    for vid in os.listdir(src):
        if os.path.exists(os.path.join(dst, "mvd_"+vid)):
            continue
        writer = skvideo.io.FFmpegWriter(os.path.join(dst, "mvd_"+vid))
        mvd = skvideo.io.vread(os.path.join(src, vid))
        mvd = np.array([
            cv2.resize(frame, dsize=(
                *np.array(mvd.shape[1:3:])[::-1]//2,), interpolation=cv2.INTER_CUBIC)
            for frame in mvd
        ])
        mvd = np.diff(mvd, axis=0).astype(np.uint8)
        mvd = np.apply_along_axis(
            lambda frame: (frame*(255/frame.max())).astype(np.uint8),
            axis=0, arr=mvd)
        for frame in mvd:
            writer.writeFrame(frame)
        writer.close()
        # skvideo.io.vwrite(os.path.join(dst, "mvd_"+vid), mvd)
        gc.collect()


def norm_aug(src: str, dst: str) -> None:
    for vid in os.listdir(src):
        if os.path.exists(os.path.join(dst, "nmv_"+vid)):
            continue
        writer = skvideo.io.FFmpegWriter(os.path.join(dst, "nmv_"+vid))
        nmv = skvideo.io.vread(os.path.join(src, vid)).astype(np.uint8)
        nmv = np.array([
            cv2.resize(frame, dsize=(
                *np.array(nmv.shape[1:3])[::-1]//2,), interpolation=cv2.INTER_CUBIC)
            for frame in nmv
        ])
        nmv = np.apply_along_axis(
            lambda frame: (frame - frame.mean())/frame.std().astype(np.uint8),
            axis=0, arr=nmv)
        for frame in nmv:
            writer.writeFrame(frame)
        writer.close()
        # skvideo.io.vwrite(os.path.join(dst, "nmv_"+vid), nmv)
        gc.collect()


if __name__ == "__main__":
    src = 'flicker-detection/'
    # get_pts(src)
    # test_frame_extraction(
    #     'flicker-detection/00_flicker_issue_00_00_18.304 - 00_00_18.606_b1d8b1fc-a81d-4ab6-bbc9-d9e8f6e072dd.mp4',
    #     'test_frames/00_flicker_issue_00_00_18.304 - 00_00_18.606_b1d8b1fc-a81d-4ab6-bbc9-d9e8f6e072dd'
    # )
    mov_dif_aug(
        'flicker-detection/',
        'augmented/'
    )
    # norm_aug(
    #     'flicker-detection/',
    #     'augmented/'
    # )
