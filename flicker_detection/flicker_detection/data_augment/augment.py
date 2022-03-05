import os
import cv2
import numpy as np
from vidaug import augmentors as va
from imgaug import augmenters as iaa
from PIL import Image, ImageSequence


def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames


def vid_to_frames(path):
    vidcap = cv2.VideoCapture(path)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(
        *'mp4v'), 30, (frameWidth, frameHeight))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc, ret = 0, True
    while (fc < frameCount and ret):
        ret, buf[fc] = vidcap.read()
        fc += 1
    vidcap.release()
    return buf, writer


# Used to apply augmentor with 50% probability
def sometimes(aug):
    return va.Sometimes(0.5, aug)


def crop_flip_blur():
    return iaa.Sequential([
        # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        # blur images with a sigma of 0 to 3.0
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])


if __name__ == "__main__":
    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')

    seq = va.Sequential([
        # randomly resize frames
        # va.RandomResize(rate=0.2),
        # randomly crop video with a size of (240 x 180)
        # va.RandomCrop(size=(240, 180)),
        # randomly rotates the video with a degree randomly choosen from [-10, 10]
        va.RandomRotate(degrees=10),
        # horizontally flip the video with 50% probability
        sometimes(va.HorizontalFlip())
    ])
    for vid in os.listdir(videos_root):
        path = os.path.join(
            videos_root, videos_root+'/'+'0000.mp4')
        print(path)
        # 'video' should be either a list of images from type of numpy array or PIL images
        frames, out = vid_to_frames(path)
        print("augmenting video")
        video_aug = seq(frames[:10])

        for idx, frame in enumerate(video_aug):
            print(idx)
            cv2.imwrite(f'output{idx}.jpeg', frame)
            out.write(frame.astype('uint8'))
        out.release()
        break
