import os
import json
import av
from fractions import Fraction


def get_pts(src: str) -> None:
    pts = {}
    for vid in os.listdir(src):
        fh = av.open(os.path.join(src, vid))
        video = fh.streams.video[0]
        print(f"{vid} duration: {float(video.duration*video.time_base)}")
        pts[vid] = tuple(
            float(frame.pts*video.time_base) for frame in fh.decode(video))
        json.dump(pts, open("pts.json", "w"))


def test_frame_extraction(inpath: str, outpath: str) -> None:
    container = av.open(inpath)
    vidstream = container.streams.video[0]
    for frame in container.decode(video=0):
        # print(f"{int(frame.index)} - {float(frame.pts*vidstream.time_base)}")
        frame.to_image()\
            .save("{}_{}_{:.2f}.jpg".format(outpath, int(frame.index), float(frame.pts*vidstream.time_base)))


if __name__ == "__main__":
    src = 'flicker-detection/'
    # get_pts(src)
    test_frame_extraction('flicker-detection/0145.mp4', 'test_frames/0145')
