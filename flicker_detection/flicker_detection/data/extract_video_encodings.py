import os
import av
import gc
import numpy as np


def get_pts(src: str, dst: str = 'pts_encodings') -> None:
    for vid in os.listdir(src):
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
        np.save(os.path.join(dst, f'{vid}.npy'), std_arr)
        gc.collect()


def test_frame_extraction(inpath: str, outpath: str) -> None:
    container = av.open(inpath)
    vidstream = container.streams.video[0]
    for frame in container.decode(video=0):
        # print(f"{int(frame.index)} - {float(frame.pts*vidstream.time_base)}")
        fts = float(frame.pts*vidstream.time_base)
        frame.to_image()\
            .save("{}_{}_{:.2f}.jpg".format(outpath, int(frame.index), fts))


if __name__ == "__main__":
    src = 'flicker-detection/'
    get_pts(src)
    # test_frame_extraction('flicker-detection/0145.mp4', 'test_frames/0145')
