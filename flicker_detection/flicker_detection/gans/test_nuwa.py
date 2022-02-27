import os
import itertools
import logging
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.data.dataloader import DataLoader
# from video_dataset import VideoFrameDataset, ImglistToTensor
from torchvision.io import VideoReader
from nuwa_pytorch import VQGanVAE, NUWA
# from mpl_toolkits.axes_grid1 import ImageGrid


def example_read_video(video_object, start=0, end=None, read_video=True, read_audio=True):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_frames = torch.empty(0)
    video_pts = []
    if read_video:
        video_object.set_current_stream("video")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)

    audio_frames = torch.empty(0)
    audio_pts = []
    if read_audio:
        video_object.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
            frames.append(frame['data'])
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)

    return video_frames, audio_frames, (video_pts, audio_pts), video_object.get_metadata()


if __name__ == "__main__":
    """
    https://github.com/lucidrains/nuwa-pytorch
    """
    from sys import path
    from pathlib import Path
    from os.path import dirname as dir
    googlecv = Path().parent.absolute().parent
    path = path.append(dir(str(googlecv)+"/"))
    __package__ = "mypyfunc"
    from mypyfunc.logger import init_logger

    init_logger()

    vae = VQGanVAE(
        dim=512,
        # default is 3, but can be changed to any value for the training of the segmentation masks (sketches)
        channels=3,
        image_size=256,           # image size
        num_layers=3,             # number of downsampling layers
        num_conv_blocks=2,        # number of convnext blocks
        vq_codebook_size=8192,    # codebook size
        vq_decay=0.8              # codebook exponential decay
    )

    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')

    logging.info("Extracting VAE encoder")
    recon_images = []
    for vid in os.listdir(videos_root):
        # video_object = VideoReader(videos_root+'/'+vid, 'video')
        vidcap = cv2.VideoCapture(os.path.join(
            videos_root, videos_root+'/'+vid))
        success, image = vidcap.read()

        while success:
            success, image = vidcap.read()
            frame = torch.Tensor(np.array(cv2.resize(image, (256, 256))
                                          )).reshape(1, 3, 256, 256)

            logging.info("tensor shape: {}".format(
                frame.shape))

            loss = vae(frame, return_loss=True)
            loss.backward()

            discr_loss = vae(frame,
                             return_discr_loss=True)
            discr_loss.backward()
            # generated images
            recon_images.extend(vae(frame))

        break

    logging.info("Done extracting VAE encoder")

    logging.info("initializing NUWA")

    nuwa = NUWA(
        vae=vae,
        dim=512,
        text_num_tokens=20000,                # number of text tokens
        text_enc_depth=12,                    # text encoder depth
        text_enc_heads=8,                     # number of attention heads for encoder
        # max sequence length of text conditioning tokens (keep at 256 as in paper, or shorter, if your text is not that long)
        text_max_seq_len=256,
        max_video_frames=10,                  # number of video frames
        image_size=256,                       # size of each frame of video
        dec_depth=64,                         # video decoder depth
        dec_heads=8,                          # number of attention heads in decoder
        # reversible networks - from reformer, decoupling memory usage from depth
        dec_reversible=True,
        enc_reversible=True,                  # reversible encoders, if you need it
        attn_dropout=0.05,                    # dropout for attention
        ff_dropout=0.05,                      # dropout for feedforward
        # kernel size of the sparse 3dna attention. can be a single value for frame, height, width, or different values (to simulate axial attention, etc)
        sparse_3dna_kernel_size=(5, 3, 3),
        # cycle dilation of 3d conv attention in decoder, for more range
        sparse_3dna_dilation=(1, 2, 4),
        # cheap relative positions for sparse 3dna transformer, by shifting along spatial dimensions by one
        shift_video_tokens=True
    ).cuda()

    loss = nuwa(video=recon_images, return_loss=True)
    loss.backward()

    video = nuwa.generate(video=recon_images)
