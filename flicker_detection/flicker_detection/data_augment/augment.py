import os
import cv2
import numpy as np
from vidaug import augmentors as va


def vid_to_frames(path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    frames, ishape = [], None
    while success:
        success, image = vidcap.read()
        frames.append(image if image is not None else np.zeros(ishape))
        ishape = image.shape if image is not None else ishape
    return np.array(frames)


# Used to apply augmentor with 50% probability
def sometimes(aug):
    return va.Sometimes(0.5, aug)


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
    fourcc, fps = cv2.VideoWriter_fourcc(*'mp4v'), 30
    for vid in os.listdir(videos_root):
        path = os.path.join(
            videos_root, videos_root+'/'+vid)
        # 'video' should be either a list of images from type of numpy array or PIL images
        video = vid_to_frames(path)
        video_aug = seq(video)
        writer = cv2.VideoWriter(os.getcwd()+'/../data/augmented/aug_' +
                                 vid, fourcc, fps, (video_aug[0].shape[0], video_aug[0].shape[1]))
        print(video_aug[0].astype(np.uint8))
        for frame in video_aug:
            print(frame.shape)
            writer.write(frame.astype(np.uint8))
        writer.release()
        break
