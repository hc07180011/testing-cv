import os
import cv2
import numpy as np
from vidaug import augmentors as va
from imgaug import augmenters as iaa
from PIL import Image, ImageSequence
from multiprocessing import Process, Queue


def gif_loader(path, modality="RGB"):
    frames = []
    with open(path, 'rb') as f:
        with Image.open(f) as video:
            index = 1
            for frame in ImageSequence.Iterator(video):
                frames.append(frame.convert(modality))
                index += 1
        return frames


def read_vid(path, outstr='output.mp4', fcc=cv2.VideoWriter_fourcc(*'MJPG'), fc=0, ret=True):
    vidcap = cv2.VideoCapture(path)
    if (vidcap.isOpened() == False):
        print("Error reading video file")

    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    size = (int(vidcap.get(3)), int(vidcap.get(4)))
    writer = cv2.VideoWriter(outstr,
                             fcc,
                             fps, size)

    buf = np.empty((frameCount, size[1], size[0], 3), np.dtype('uint8'))
    while (fc < frameCount and ret):
        ret, buf[fc] = vidcap.read()
        fc += 1
    vidcap.release()
    frames = [Image.fromarray(frame.astype('uint8')) for frame in buf]
    return frames, buf, writer


def write_vid(writer, frames):
    for frame in frames:
        writer.write(frame)
    writer.release()


def compose(aug, random=False, sometimes=lambda aug: va.Sometimes(0.5, aug)):
    return va.Sequential(map(sometimes, aug)) if random else va.Sequential(aug)


def affine_aug():
    return [va.RandomCrop(size=(1520, 720)), *tuple(va.RandomRotate(degrees=i*10) for i in range(36)),
            va.RandomResize(rate=0.2), va.RandomTranslate(x=10, y=10), va.RandomShear(x=10, y=10)]


def intensity_aug(value=10):
    return [va.InvertColor(), va.Add(value=value),
            va.Salt(), va.Pepper(),
            va.Multiply(value=value)]


def flip_aug():
    return [va.HorizontalFlip(), va.VerticalFlip()]


def geometric_aug():
    return [va.GaussianBlur(sigma=0.25), va.ElasticTransformation(), va.PiecewiseAffineTransform()]


def producer(frames, aug_lst, q):
    for aug in aug_lst:
        video_aug = aug(frames)
        q.put(video_aug)


def consumer(q, i):
    while True:
        if not q.empty():
            vidaug = q.get()
            vidaug[0].save(
                f'{vid[:-4]}_vidaug{i}.gif', save_all=True, append_images=vidaug, loop=0)
            i += 1


if __name__ == "__main__":
    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')

    # for vid in os.listdir(videos_root):

    #     frames, npframes, writer = read_vid(
    #         videos_root+'/'+vid, outstr=videos_root+f'/{vid[:-4]}_imaug.mp4')
    #     imaug = random_im_aug(npframes, crop_flip_blur())
    #     write_vid(writer, npframes)

    for vid in os.listdir(videos_root):

        frames, npframes, writer = read_vid(
            videos_root+'/'+vid, outstr=videos_root+f'/{vid[:-4]}_imaug.mp4')
        print("WTF")
        q = Queue()
        augmentors = affine_aug()+intensity_aug()+flip_aug()+geometric_aug()
        for i in range(0, len(augmentors), os.cpu_count()-2):

            aug_lst = augmentors[i:i+os.cpu_count()-6]

            p = tuple(Process(target=producer, args=(frames, aug_lst, q))
                      for _ in range(os.cpu_count()-6))

            c = tuple(Process(target=consumer, args=(q, i))
                      for _ in range(os.cpu_count()-10))

            # for c_ in c:
            #     c_.daemmon = True

            for p_ in p:
                p_.start()

            for c_ in c:
                c_.start()

            for p_ in p:
                p_.join()

            for c_ in c:
                c_.join()
