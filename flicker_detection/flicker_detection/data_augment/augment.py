import os
import random
import cv2
import numpy as np
from vidaug import augmentors as va
from imgaug import augmenters as iaa
from PIL import Image, ImageSequence


def crop_flip_blue():
    return iaa.Sequential([
        # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        # blur images with a sigma of 0 to 3.0
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])


def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames


def vid_to_frames(path, seq, portion_size=(200, 200), fcc=cv2.VideoWriter_fourcc(*'mp4v'), fc=0, ret=True):
    vidcap = cv2.VideoCapture(path)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter('output.mp4', fcc, fps,
                             (frameWidth, frameHeight), False)

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    np.random.seed(1000)
    trigger = np.random.choice([0, 1], size=frameCount)  # p=[.1, .9]
    while (fc < frameCount and ret):
        ret, buf[fc] = vidcap.read()
        if trigger[fc] == 1:
            y1 = np.random.randint(
                0, frameWidth-portion_size[0]-1, size=1, dtype=np.int64)[0]
            x1 = np.random.randint(
                0, frameHeight-portion_size[1]-1, size=1, dtype=np.int64)[0]
            x2, y2 = x1+portion_size[0]-1, y1+portion_size[1]-1
            buf[fc][x1:x2, y1:y2] = seq(images=buf[fc][x1:x2, y1:y2])
        fc += 1
    vidcap.release()
    return buf, writer


if __name__ == "__main__":
    """
    frames = gif_loader("../videos/salt.gif")
    sometimes = lambda aug: va.Sometimes(1, aug) # Used to apply augmentor with 100% probability
    seq = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
        sometimes(va.HorizontalFlip()) # horizontally flip the video with 100% probability
    ])   
    video_aug = seq(frames)
    video_aug[0].save("out.gif", save_all=True, append_images=video_aug[1:], duration=10
    """

    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')

    seq = crop_flip_blue()
    # path = "/content/drive/MyDrive/google-cv/flicker-detection/0000.mp4"
    frames, writer = vid_to_frames(videos_root+'/0000.mp4', seq)

    for frame in frames:
        writer.write(frame.astype('uint8'))
    writer.release()

    for idx, frame in enumerate(frames):
        im = Image.fromarray(frame)
        im.save(f'out{idx}.png')
