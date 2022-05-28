import os
import numpy as np
from imgaug import augmenters as iaa
from vid_aug import read_vid, write_vid


def crop_flip_blur():
    return iaa.Sequential([
        # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        # blur images with a sigma of 0 to 3.0
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])


def random_im_aug(buf, seq, portion_size=(200, 200), fc=0):
    rng = np.random.RandomState(1000)
    trigger = rng.choice([0, 1], size=buf.shape[0])  # p=[.1, .9]
    while (fc < buf.shape[0]):
        if trigger[fc] == 1:
            y1 = rng.randint(
                0, buf.shape[2]-portion_size[0]-1, size=1, dtype=np.int64)[0]
            x1 = rng.randint(
                0, buf.shape[1]-portion_size[1]-1, size=1, dtype=np.int64)[0]
            x2, y2 = x1+portion_size[0]-1, y1+portion_size[1]-1
            buf[fc][x1:x2, y1:y2] = seq(images=buf[fc][x1:x2, y1:y2])
        fc += 1
    return buf


if __name__ == "__main__":
    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')

    for vid in os.listdir(videos_root):

        frames, npframes, writer = read_vid(
            videos_root+'/'+vid, outstr=videos_root+f'/{vid[:-4]}_imaug.mp4')
        imaug = random_im_aug(npframes, crop_flip_blur())
        write_vid(writer, npframes)
