import os
import av
import gc
import json
import numpy as np


def get_pts(
    src: str,
    dst: str = 'pts_encodings',
    map_src: str = 'mapping.json',
    context_dim: int = 15,
) -> None:
    mapping = {
        code: num
        for num, code in json.load(open(map_src, "r")).items()
    }
    unique = ()
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
        #unique += (*tuple(i for i in pts_interval if i not in unique),) 
        std_arr = ((pts_interval - pts_interval.mean(axis=0)) /
                   pts_interval.std(axis=0))
        # bag_of_words = np.array([
        #     (std_arr[i-context_dim:i].tolist() +
        #      std_arr[i+1:i+context_dim].tolist(), std_arr[i])
        #     for i in range(context_dim, std_arr.size-context_dim)
        # ])
        #print(len(unique))
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


if __name__ == "__main__":
    src = 'flicker-detection/'
    get_pts(src)
    # test_frame_extraction('flicker-detection/0145.mp4', 'test_frames/0145')
