import os
import cv2
import numpy as np
from vidaug import augmentors as va
from imgaug import augmenters as iaa
from PIL import Image, ImageSequence


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


def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames


def read_vid(path, outstr='output.mp4', fcc=cv2.VideoWriter_fourcc(*'mp4v'), fc=0, ret=True):
    vidcap = cv2.VideoCapture(path)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        outstr, fcc, fps, (frameWidth, frameHeight), False)
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    while (fc < frameCount and ret):
        ret, buf[fc] = vidcap.read()
        fc += 1
    vidcap.release()
    frames = [Image.fromarray(frame.astype('uint8')) for frame in buf]
    return frames, buf, writer


def compose(aug, random=False, sometimes=lambda aug: va.Sometimes(0.5, aug)):
    return va.Sequential(map(sometimes, aug)) if random else va.Sequential(aug)


def affine_aug():
    return [va.RandomCrop(size=(1520, 720)), va.RandomRotate(degrees=10), va.RandomRotate(degrees=90),
            va.RandomResize(rate=0.2), va.RandomTranslate(x=10, y=10), va.RandomShear(x=10, y=10)]


def intensity_aug(value=10):
    return [va.InvertColor(), va.Add(value=value),
            va.Salt(), va.Pepper(),
            va.Multiply(value=value)]


def flip_aug():
    return [va.HorizontalFlip(), va.VerticalFlip()]


def geometric_aug():
    return [va.GaussianBlur(sigma=0.25), va.ElasticTransformation(), va.PiecewiseAffineTransform()]


if __name__ == "__main__":
    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')
    frames, npframes, writer = read_vid(videos_root+'/0000.mp4')
    npframes = random_im_aug(npframes, crop_flip_blur())

    for idx, aug in enumerate(affine_aug()):
        video_aug = aug(frames)
        video_aug[0].save(
            f'0000_affine{idx}.gif', save_all=True, append_images=video_aug, loop=0)

    for idx, aug in enumerate(intensity_aug()):
        video_aug = aug(frames)
        video_aug[0].save(
            f'0000_intensity{idx}.gif', save_all=True, append_images=video_aug, loop=0)

    for idx, aug in enumerate(flip_aug()):
        video_aug = aug(frames)
        video_aug[0].save(
            f'0000_flip{idx}.gif', save_all=True, append_images=video_aug, loop=0)
